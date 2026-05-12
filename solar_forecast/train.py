"""Train a single LightGBM CI model with walk-forward cross-validation."""

import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Optional

from solar_forecast.clearsky import get_clearsky
from solar_forecast.fetch_actuals import fetch_era5, fetch_entsoe
from solar_forecast.features import build_features
import config

_MODEL_PATH = config.MODEL_PATH

# ci excluded — it's ERA5-derived during training and would cause leakage;
# lag/rolling features of ci are included since they're shifted (no direct leakage).
FEATURE_COLS = [
    "poa_nwp",
    "sin_elevation",
    "cos_azimuth",
    "is_daytime",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "cloud_cover",
    "cloud_cover_sq",
    "temperature_2m",
    "shortwave_radiation",
    "ci_lag1",
    "ci_lag2",
    "ci_rolling_mean_3h",
    "cloud_change_1h",
    "cloud_rolling_mean_3h",
    "member_id",
]

DEFAULT_PARAMS: dict = config.LGBM_PARAMS


def _target_ci(
    actuals: pd.Series,
    clearsky: pd.DataFrame,
    installed_capacity_mw: "float | pd.Series",
    stc_irradiance: float = 1000.0,
) -> pd.Series:
    """
    Compute actual clearness index = actual_mw / clearsky_mw.

    installed_capacity_mw may be a scalar (fixed) or a monthly pd.Series
    (time-varying). When a Series is passed it is forward-filled to the
    hourly actuals index so every timestamp gets the capacity that was
    installed at that point in time.
    """
    cs_poa = clearsky["poa_clearsky"].reindex(actuals.index)
    is_day = clearsky["is_daytime"].reindex(actuals.index)

    if isinstance(installed_capacity_mw, pd.Series):
        cap = installed_capacity_mw.tz_convert(actuals.index.tz)
        cap = cap.reindex(actuals.index.union(cap.index)).ffill().reindex(actuals.index)
    else:
        cap = installed_capacity_mw

    clearsky_mw = (cs_poa / stc_irradiance) * cap
    ci = (actuals / clearsky_mw).clip(0, 1.1)
    return ci.where(is_day, 0.0).rename("ci_actual")


def prepare_training_data(
    lat: float,
    lon: float,
    start: str,
    end: str,
    installed_capacity_mw: float,
    area: str = "DE",
    tz: str = "Europe/Vienna",
    altitude: float = 500,
    surface_tilt: float = 30,
    surface_azimuth: float = 180,
    entsoe_api_key: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetch ERA5 + ENTSO-E actuals, build features, compute CI target.

    Returns
    -------
    X : pd.DataFrame  — FEATURE_COLS, daytime rows with valid actuals only.
    y : pd.Series     — actual CI aligned to X.
    """
    print("  Fetching ERA5...")
    era5 = fetch_era5(lat, lon, start, end, tz=tz)
    clearsky = get_clearsky(
        lat, lon, era5.index,
        altitude=altitude,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
    )
    feats = build_features(era5, clearsky, member_id=-1)

    print("  Fetching ENTSO-E actuals...")
    actuals = fetch_entsoe(area, start, end, api_key=entsoe_api_key)
    actuals = actuals.tz_convert(tz).reindex(feats.index)

    try:
        from solar_forecast.capacity import capacity_timeseries
        cap = capacity_timeseries(tz="UTC")
        print(f"  Time-varying capacity: {cap.min():.0f}–{cap.max():.0f} MW")
    except FileNotFoundError:
        cap = installed_capacity_mw
        print(f"  MaStR DB not available, using fixed capacity: {cap:.0f} MW")

    y = _target_ci(actuals, clearsky, cap)

    valid = y.notna() & feats["is_daytime"].astype(bool)
    return feats.loc[valid, FEATURE_COLS], y[valid]


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    gap_months: int = 6,
    min_train_months: int = 12,
    params: Optional[dict] = None,
) -> list[dict]:
    """
    Walk-forward CV with a gap between training and validation.

    For each validation month, training uses all months up to
    (val_month - gap_months), skipping the gap entirely.
    Example: gap=6 → train through month 12, validate month 18.

    Returns
    -------
    list of dicts: val_month, rmse, n_val, best_iteration.
    """
    _params = {**DEFAULT_PARAMS, **(params or {})}
    months = sorted(X.index.to_period("M").unique())
    results = []

    for val_month in months:
        train_end = val_month - gap_months
        n_train_months = sum(1 for m in months if m <= train_end)
        if n_train_months < min_train_months:
            continue

        train_mask = X.index.to_period("M") <= train_end
        val_mask = X.index.to_period("M") == val_month
        if not val_mask.any():
            continue

        model = lgb.LGBMRegressor(**_params)
        model.fit(
            X[train_mask], y[train_mask],
            eval_set=[(X[val_mask], y[val_mask])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(False),
            ],
        )

        preds = model.predict(X[val_mask]).clip(0, 1.1)
        rmse = float(np.sqrt(np.mean((preds - y[val_mask].values) ** 2)))
        best_iter = model.best_iteration_ or _params["n_estimators"]
        results.append({
            "val_month": str(val_month),
            "rmse": rmse,
            "n_val": int(val_mask.sum()),
            "best_iteration": best_iter,
        })
        print(f"  val={val_month} | RMSE={rmse:.4f} | iters={best_iter} | n={val_mask.sum()}")

    return results


def train(
    lat: float,
    lon: float,
    start: str,
    end: str,
    installed_capacity_mw: float,
    area: str = "DE",
    tz: str = "Europe/Vienna",
    altitude: float = 500,
    surface_tilt: float = 30,
    surface_azimuth: float = 180,
    entsoe_api_key: Optional[str] = None,
    params: Optional[dict] = None,
    run_cv: bool = True,
    save_path: Optional[Path] = None,
) -> lgb.LGBMRegressor:
    """
    Full training pipeline: fetch → walk-forward CV → final model on all data.

    The CV determines the optimal number of boosting iterations; the final
    model is then retrained on the full dataset using that iteration count.

    Returns
    -------
    lgb.LGBMRegressor  — fitted model, also saved to models/lgbm_ci.pkl.
    """
    print(f"Preparing training data ({start} → {end})...")
    X, y = prepare_training_data(
        lat, lon, start, end, installed_capacity_mw,
        area=area, tz=tz, altitude=altitude,
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        entsoe_api_key=entsoe_api_key,
    )
    print(f"  {len(X)} daytime rows loaded.")

    _params = {**DEFAULT_PARAMS, **(params or {})}

    if run_cv:
        print("Running walk-forward CV...")
        cv_results = walk_forward_cv(X, y, params=_params)
        if cv_results:
            mean_rmse = float(np.mean([r["rmse"] for r in cv_results]))
            mean_iters = int(np.mean([r["best_iteration"] for r in cv_results]))
            print(f"CV summary: mean RMSE={mean_rmse:.4f}, mean best_iters={mean_iters}")
            _params["n_estimators"] = mean_iters
        else:
            print("  Not enough data for CV folds — skipping.")

    print("Training final model on all data...")
    final_model = lgb.LGBMRegressor(**_params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(100)])

    out_path = save_path or _MODEL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"Model saved → {out_path}")

    return final_model


def load_model(path: Optional[Path] = None) -> lgb.LGBMRegressor:
    p = path or _MODEL_PATH
    with open(p, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM CI model.")
    parser.add_argument("--lat", type=float, default=config.LAT)
    parser.add_argument("--lon", type=float, default=config.LON)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--capacity-mw", type=float, default=config.CAPACITY_MW)
    parser.add_argument("--area", default=config.AREA, help="ENTSO-E area: DE or AT")
    parser.add_argument("--no-cv", action="store_true", help="Skip walk-forward CV")
    args = parser.parse_args()

    train(
        lat=args.lat,
        lon=args.lon,
        start=args.start,
        end=args.end,
        installed_capacity_mw=args.capacity_mw,
        area=args.area,
        entsoe_api_key=os.environ.get("ENTSOE_API_KEY"),
        run_cv=not args.no_cv,
    )
