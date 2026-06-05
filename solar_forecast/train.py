"""Train a single LightGBM CI model with walk-forward cross-validation."""

import os
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Optional

from solar_forecast.clearsky import get_clearsky, temperature_efficiency
from solar_forecast.fetch_actuals import fetch_era5, fetch_entsoe
from solar_forecast.features import build_features
import config


def _ar1_noise(
    n: int,
    rho: float,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """AR(1) noise: x[t] = rho*x[t-1] + sqrt(1-rho²)*N(0,std). Unit variance at stationarity."""
    innovations = rng.standard_normal(n) * (std * np.sqrt(1.0 - rho ** 2))
    noise = np.empty(n)
    noise[0] = rng.standard_normal() * std
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + innovations[i]
    return noise


def _inject_nwp_noise(
    era5: pd.DataFrame,
    rng: np.random.Generator,
    radiation_rel_std: float = 0.15,
    cloud_abs_std: float = 15.0,
    temperature_abs_std: float = 1.5,
    ar1_rho: float = 0.70,
) -> pd.DataFrame:
    """
    Add temporally-correlated noise to ERA5 variables to simulate NWP forecast errors.

    Noise parameters are calibrated to ECMWF IFS day-ahead errors over Central Europe:
    radiation ~15% relative RMSE, cloud cover ~15 pp RMSE, temperature ~1.5°C RMSE.
    AR(1) rho=0.70 captures the persistence of NWP errors across consecutive hours.

    Radiation uses multiplicative noise (errors scale with irradiance level).
    Cloud cover and temperature use additive noise.
    """
    n = len(era5)
    noisy = era5.copy()

    for col in ("shortwave_radiation", "direct_radiation", "diffuse_radiation"):
        if col in noisy.columns:
            factor = np.clip(1.0 + _ar1_noise(n, ar1_rho, radiation_rel_std, rng), 0.0, 2.5)
            noisy[col] = np.clip(noisy[col].values * factor, 0.0, None)

    for col in ("cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"):
        if col in noisy.columns:
            noisy[col] = np.clip(
                noisy[col].values + _ar1_noise(n, ar1_rho, cloud_abs_std, rng), 0.0, 100.0
            )

    if "temperature_2m" in noisy.columns:
        noisy["temperature_2m"] = (
            noisy["temperature_2m"].values + _ar1_noise(n, ar1_rho, temperature_abs_std, rng)
        )

    return noisy

_MODEL_PATH = config.MODEL_PATH
_QUANTILE_MODEL_PATH = config.QUANTILE_MODEL_PATH
QUANTILE_LEVELS = [(i + 1) / 101 for i in range(100)]

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
    temperature_2m: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute actual clearness index = actual_mw / clearsky_mw_corrected.

    When temperature_2m is supplied, clearsky_mw is multiplied by the Faiman
    panel temperature efficiency factor so the CI target is temperature-neutral
    (clouds only). The ensemble must apply the same correction at inference time.

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
    if temperature_2m is not None:
        eta = temperature_efficiency(cs_poa, temperature_2m.reindex(actuals.index))
        clearsky_mw = clearsky_mw * eta

    ci = (actuals / clearsky_mw).clip(0, 1.1)
    return ci.where(is_day, 0.0).rename("ci_actual")


def _aggregate_centroids_era5(
    centroids: list[dict],
    start: str,
    end: str,
    tz: str = "Europe/Vienna",
    altitude: float = 500,
    surface_tilt: float = 30,
    surface_azimuth: float = 180,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch ERA5 + clearsky at each centroid and return capacity-weighted aggregates.

    Each weather variable becomes a weighted average across centroids.
    poa_clearsky is also weighted so that: weighted_avg × total_capacity = sum of
    per-centroid clearsky_mw, keeping CI targets consistent with _target_ci.
    """
    era5_agg = None
    cs_num_agg = None  # numeric clearsky cols: poa_clearsky, apparent_elevation, azimuth
    is_daytime_agg = None

    for i, c in enumerate(centroids):
        print(f"  [{i + 1}/{len(centroids)}] lat={c['lat']}, lon={c['lon']}, w={c['weight']:.3f}")
        if i > 0:
            time.sleep(65)  # stay within Open-Meteo free-tier 1 req/min rate limit
        era5_i = fetch_era5(c["lat"], c["lon"], start, end, tz=tz)
        cs_i = get_clearsky(
            c["lat"], c["lon"], era5_i.index,
            altitude=altitude,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
        )
        w = c["weight"]
        cs_num_i = cs_i[["poa_clearsky", "apparent_elevation", "azimuth"]] * w

        if era5_agg is None:
            era5_agg = era5_i * w
            cs_num_agg = cs_num_i
            is_daytime_agg = cs_i["is_daytime"].copy()
        else:
            era5_agg = era5_agg + era5_i * w
            cs_num_agg = cs_num_agg + cs_num_i
            is_daytime_agg = is_daytime_agg | cs_i["is_daytime"]

    cs_num_agg["is_daytime"] = is_daytime_agg
    cs_num_agg.attrs["surface_tilt"] = surface_tilt
    cs_num_agg.attrs["surface_azimuth"] = surface_azimuth

    return era5_agg, cs_num_agg


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
    n_noise_augments: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetch ERA5 + ENTSO-E actuals, build features, compute CI target.

    n_noise_augments noisy copies of the ERA5 features are appended to the clean
    copy to teach the model to handle imperfect NWP inputs at inference time.
    The CI target is always computed from clean ERA5 (ground truth).

    Returns
    -------
    X : pd.DataFrame  — FEATURE_COLS, daytime rows with valid actuals only.
    y : pd.Series     — actual CI aligned to X.
    """
    if config.CENTROIDS:
        print(f"  Aggregating ERA5 across {len(config.CENTROIDS)} centroids...")
        era5, clearsky = _aggregate_centroids_era5(
            config.CENTROIDS, start, end,
            tz=tz, altitude=altitude,
            surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        )
    else:
        print("  Fetching ERA5 (single point)...")
        era5 = fetch_era5(lat, lon, start, end, tz=tz)
        clearsky = get_clearsky(
            lat, lon, era5.index,
            altitude=altitude,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
        )

    print("  Fetching ENTSO-E actuals...")
    actuals = fetch_entsoe(area, start, end, api_key=entsoe_api_key)
    actuals = actuals.tz_convert(tz)

    try:
        from solar_forecast.capacity import capacity_timeseries
        cap = capacity_timeseries(tz="UTC")
        cap = cap * (installed_capacity_mw / cap.max())
        print(f"  Time-varying capacity (AC-scaled): {cap.min():.0f}–{cap.max():.0f} MW")
    except FileNotFoundError:
        cap = installed_capacity_mw
        print(f"  MaStR DB not available, using fixed capacity: {cap:.0f} MW")

    # Build clean features and compute CI target from clean ERA5
    rng = np.random.default_rng(42)
    feats_clean = build_features(era5, clearsky, member_id=-1)
    feats_clean["member_id"] = rng.integers(0, 100, size=len(feats_clean))

    actuals_aligned = actuals.reindex(feats_clean.index)
    y = _target_ci(actuals_aligned, clearsky, cap, temperature_2m=era5["temperature_2m"])
    valid = y.notna() & feats_clean["is_daytime"].astype(bool)

    X_clean = feats_clean.loc[valid, FEATURE_COLS]
    y_clean = y[valid]

    if n_noise_augments == 0:
        return X_clean, y_clean

    # Augment with n_noise_augments noisy copies of features; targets are unchanged
    print(f"  Building {n_noise_augments} noise-augmented copies (NWP error simulation)...")
    all_X = [X_clean]
    all_y = [y_clean]
    for _ in range(n_noise_augments):
        era5_noisy = _inject_nwp_noise(era5, rng)
        feats_noisy = build_features(era5_noisy, clearsky, member_id=-1)
        feats_noisy["member_id"] = rng.integers(0, 50, size=len(feats_noisy))
        all_X.append(feats_noisy.loc[valid, FEATURE_COLS])
        all_y.append(y_clean)

    return pd.concat(all_X), pd.concat(all_y)


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
    quantile: bool = False,
    save_path: Optional[Path] = None,
    n_noise_augments: int = 4,
) -> lgb.LGBMRegressor:
    """
    Full training pipeline: fetch → walk-forward CV → final model on all data.

    The CV determines the optimal number of boosting iterations; the final
    model is then retrained on the full dataset using that iteration count.

    If quantile=True, also trains 50 quantile models using the same best_iters.

    Returns
    -------
    lgb.LGBMRegressor  — fitted CI model, also saved to models/lgbm_ci.pkl.
    """
    print(f"Preparing training data ({start} → {end})...")
    X, y = prepare_training_data(
        lat, lon, start, end, installed_capacity_mw,
        area=area, tz=tz, altitude=altitude,
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        entsoe_api_key=entsoe_api_key,
        n_noise_augments=n_noise_augments,
    )
    print(f"  {len(X)} daytime rows loaded ({n_noise_augments + 1}x augmented).")

    _params = {**DEFAULT_PARAMS, **(params or {})}
    best_iters = _params["n_estimators"]

    if run_cv:
        print("Running walk-forward CV...")
        cv_results = walk_forward_cv(X, y, params=_params)
        if cv_results:
            mean_rmse = float(np.mean([r["rmse"] for r in cv_results]))
            best_iters = int(np.mean([r["best_iteration"] for r in cv_results]))
            print(f"CV summary: mean RMSE={mean_rmse:.4f}, mean best_iters={best_iters}")
            _params["n_estimators"] = best_iters
        else:
            print("  Not enough data for CV folds — skipping.")

    print("Training final CI model on all data...")
    final_model = lgb.LGBMRegressor(**_params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(100)])

    out_path = save_path or _MODEL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"Model saved → {out_path}")

    if quantile:
        print(f"Training 100 quantile models (n_estimators={best_iters})...")
        train_quantile(X, y, best_iters=best_iters, params=params)

    return final_model


def load_model(path: Optional[Path] = None) -> lgb.LGBMRegressor:
    p = path or _MODEL_PATH
    with open(p, "rb") as f:
        return pickle.load(f)


def train_quantile(
    X: pd.DataFrame,
    y: pd.Series,
    best_iters: int,
    quantile_levels: Optional[list] = None,
    params: Optional[dict] = None,
    save_path: Optional[Path] = None,
) -> list:
    """
    Train one LightGBM quantile model per quantile level.

    Uses best_iters from walk_forward_cv (via train()) as n_estimators so
    each model trains for the same number of trees without early stopping.

    Parameters
    ----------
    best_iters : int
        Number of boosting rounds — take from CV summary of train().
    quantile_levels : list of float, optional
        Defaults to QUANTILE_LEVELS (50 evenly spaced quantiles).

    Returns
    -------
    list of lgb.LGBMRegressor, one per quantile level, in order.
    """
    levels = quantile_levels or QUANTILE_LEVELS
    base_params = {**DEFAULT_PARAMS, **(params or {})}
    base_params.pop("metric", None)

    models = []
    for i, q in enumerate(levels):
        q_params = {
            **base_params,
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "n_estimators": best_iters,
        }
        model = lgb.LGBMRegressor(**q_params)
        model.fit(X, y, callbacks=[lgb.log_evaluation(False)])
        models.append(model)
        if (i + 1) % 10 == 0:
            print(f"  Trained {i + 1}/{len(levels)} quantile models")

    out_path = save_path or _QUANTILE_MODEL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(models, f)
    print(f"Quantile models saved → {out_path}")
    return models


def load_quantile_models(path: Optional[Path] = None) -> list:
    p = path or _QUANTILE_MODEL_PATH
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
    parser.add_argument("--quantile", action="store_true", help="Also train 100 quantile models")
    parser.add_argument("--augments", type=int, default=4,
                        help="Noise-augmented ERA5 copies added to training (default: 4)")
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
        quantile=args.quantile,
        n_noise_augments=args.augments,
    )
