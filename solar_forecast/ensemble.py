"""Apply trained LightGBM CI model across all NWP ensemble members → MW trajectories."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Optional

from solar_forecast.clearsky import get_clearsky, temperature_efficiency
from solar_forecast.fetch_nwp import fetch_nwp
from solar_forecast.features import build_features
from solar_forecast.train import FEATURE_COLS, QUANTILE_LEVELS, load_model, load_quantile_models, _inject_nwp_noise
import config

_STC_IRRADIANCE = config.STC_IRRADIANCE


def _pivot_nwp(nwp_raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Pivot fetch_nwp output from {variable: wide_df} to {member_name: single_member_df}.

    fetch_nwp returns: variable → DataFrame(times × members)
    build_features expects: member_name → DataFrame(times × variables)
    """
    first_var = next(iter(nwp_raw))
    members = nwp_raw[first_var].columns
    index = nwp_raw[first_var].index
    return {
        member: pd.DataFrame(
            {var: nwp_raw[var][member] for var in nwp_raw},
            index=index,
        )
        for member in members
    }


def run_ensemble(
    model: lgb.LGBMRegressor,
    nwp_raw: dict[str, pd.DataFrame],
    clearsky: pd.DataFrame,
    clearsky_15min: pd.DataFrame,
    installed_capacity_mw: float,
    stc_irradiance: float = _STC_IRRADIANCE,
) -> pd.DataFrame:
    """
    Apply the CI model to every ensemble member and convert to MW.

    Parameters
    ----------
    model : lgb.LGBMRegressor
        Trained CI model from train.py.
    nwp_raw : dict[str, pd.DataFrame]
        Output of fetch_nwp() — {variable: DataFrame(times × members)}.
    clearsky : pd.DataFrame
        Hourly clearsky (for feature engineering).
    clearsky_15min : pd.DataFrame
        15-min clearsky (for physics-based intra-hour ramp scaling).
    installed_capacity_mw : float
        Total installed solar capacity used to scale CI → MW.
    stc_irradiance : float
        Reference irradiance (W/m²) at which panels produce rated capacity.

    Returns
    -------
    pd.DataFrame
        Shape (timesteps_15min, n_members), columns member_00…member_N, values in MW.
        Nighttime rows are zero.
    """
    nwp_by_member = _pivot_nwp(nwp_raw)
    base_clearsky_mw_15min = (clearsky_15min["poa_clearsky"] / stc_irradiance) * installed_capacity_mw

    trajectories = {}
    for member_name, nwp_df in nwp_by_member.items():
        try:
            member_id = int(member_name.split("_")[-1])
        except (ValueError, AttributeError):
            member_id = -1

        feats = build_features(nwp_df, clearsky, member_id=member_id)
        ci_hourly = pd.Series(
            model.predict(feats[FEATURE_COLS]).clip(0, 1.1), index=feats.index
        )
        ci_15min = (
            ci_hourly.resample("15min").asfreq()
            .reindex(clearsky_15min.index)
            .interpolate("linear")
            .ffill().bfill()
            .clip(0, 1.1)
        )

        clearsky_mw_15min = base_clearsky_mw_15min.copy()
        if "temperature_2m" in nwp_df.columns:
            eta_15min = (
                temperature_efficiency(clearsky["poa_clearsky"], nwp_df["temperature_2m"])
                .resample("15min").asfreq()
                .reindex(clearsky_15min.index)
                .interpolate("linear")
                .ffill().bfill()
            )
            clearsky_mw_15min = clearsky_mw_15min * eta_15min

        mw = ci_15min * clearsky_mw_15min
        trajectories[member_name] = mw.where(clearsky_15min["is_daytime"], 0.0)

    return pd.DataFrame(trajectories)


def run_quantile_ensemble(
    models: list,
    nwp_raw: dict[str, pd.DataFrame],
    clearsky: pd.DataFrame,
    clearsky_15min: pd.DataFrame,
    installed_capacity_mw: float,
    stc_irradiance: float = _STC_IRRADIANCE,
) -> pd.DataFrame:
    """
    Apply N quantile CI models to produce N MW trajectories.

    Iterates over models (not NWP members). When len(models) > n_nwp_members,
    NWP members are cycled so each model gets a real weather input while still
    spanning the full quantile range. With 100 models and 50 NWP members each
    NWP member is used twice, paired with two different quantile levels.

    Returns
    -------
    pd.DataFrame
        Shape (timesteps_15min, len(models)), columns member_000…member_N, values in MW.
    """
    nwp_by_member = list(_pivot_nwp(nwp_raw).values())
    n_nwp = len(nwp_by_member)
    base_clearsky_mw_15min = (clearsky_15min["poa_clearsky"] / stc_irradiance) * installed_capacity_mw

    trajectories = {}
    for i, model in enumerate(models):
        nwp_df = nwp_by_member[i % n_nwp]
        feats = build_features(nwp_df, clearsky, member_id=i)
        ci_hourly = pd.Series(
            model.predict(feats[FEATURE_COLS]).clip(0, 1.1), index=feats.index
        )
        ci_15min = (
            ci_hourly.resample("15min").asfreq()
            .reindex(clearsky_15min.index)
            .interpolate("linear")
            .ffill().bfill()
            .clip(0, 1.1)
        )

        clearsky_mw_15min = base_clearsky_mw_15min.copy()
        if "temperature_2m" in nwp_df.columns:
            eta_15min = (
                temperature_efficiency(clearsky["poa_clearsky"], nwp_df["temperature_2m"])
                .resample("15min").asfreq()
                .reindex(clearsky_15min.index)
                .interpolate("linear")
                .ffill().bfill()
            )
            clearsky_mw_15min = clearsky_mw_15min * eta_15min

        mw = ci_15min * clearsky_mw_15min
        trajectories[f"member_{i:03d}"] = mw.where(clearsky_15min["is_daytime"], 0.0)

    return pd.DataFrame(trajectories)


def forecast(
    lat: float,
    lon: float,
    installed_capacity_mw: float,
    forecast_days: int = 7,
    model: Optional[lgb.LGBMRegressor] = None,
    model_path: Optional[Path] = None,
    quantile_models: Optional[list] = None,
    altitude: float = 500,
    surface_tilt: float = 30,
    surface_azimuth: float = 180,
    nwp_model: str = "ecmwf_ifs025",
) -> pd.DataFrame:
    """
    End-to-end forecast for a single location.

    Fetches NWP → computes clearsky → applies ensemble model.

    Parameters
    ----------
    lat, lon : float
        Location coordinates.
    installed_capacity_mw : float
        Installed solar capacity at this location in MW.
    forecast_days : int
        Number of days ahead to forecast.
    model : lgb.LGBMRegressor, optional
        Pre-loaded model. If None, loads from model_path or default path.
    model_path : Path, optional
        Override path for loading the model artifact.

    Returns
    -------
    pd.DataFrame
        Shape (timesteps, n_members) in MW.
    """
    nwp_raw = fetch_nwp(lat, lon, forecast_days=forecast_days, model=nwp_model)
    times = nwp_raw[next(iter(nwp_raw))].index
    clearsky = get_clearsky(
        lat, lon, times,
        altitude=altitude,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
    )
    times_15min = pd.date_range(
        start=times[0],
        end=times[-1] + pd.Timedelta(minutes=45),
        freq="15min",
        tz=times.tz,
    )
    clearsky_15min = get_clearsky(
        lat, lon, times_15min,
        altitude=altitude,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
    )

    if quantile_models is not None:
        return run_quantile_ensemble(quantile_models, nwp_raw, clearsky, clearsky_15min, installed_capacity_mw)

    if model is None:
        model = load_model(model_path)
    return run_ensemble(model, nwp_raw, clearsky, clearsky_15min, installed_capacity_mw)


def forecast_country(
    model: Optional[lgb.LGBMRegressor] = None,
    model_path: Optional[Path] = None,
    quantile_models: Optional[list] = None,
    forecast_days: int = config.FORECAST_DAYS,
    nwp_model: str = "ecmwf_ifs025",
) -> pd.DataFrame:
    """
    Run ensemble forecast aggregated across all centroids in config.CENTROIDS.

    If config.CENTROIDS is None, falls back to a single-point forecast using
    config.LAT / config.LON / config.CAPACITY_MW.

    Each centroid contributes: forecast(lat, lon, CAPACITY_MW × weight).
    Outputs are summed to produce the country-level MW trajectories.

    Returns
    -------
    pd.DataFrame
        Shape (timesteps, n_members) in MW.
    """
    if quantile_models is None and model is None:
        model = load_model(model_path)

    if not config.CENTROIDS:
        return forecast(
            lat=config.LAT,
            lon=config.LON,
            installed_capacity_mw=config.CAPACITY_MW,
            forecast_days=forecast_days,
            model=model,
            quantile_models=quantile_models,
            nwp_model=nwp_model,
        )

    total: Optional[pd.DataFrame] = None
    for i, c in enumerate(config.CENTROIDS):
        print(f"Centroid {i + 1}/{len(config.CENTROIDS)}: lat={c['lat']:.4f}, lon={c['lon']:.4f}, weight={c['weight']:.3f}")
        traj = forecast(
            lat=c["lat"],
            lon=c["lon"],
            installed_capacity_mw=config.CAPACITY_MW * c["weight"],
            forecast_days=forecast_days,
            model=model,
            quantile_models=quantile_models,
            nwp_model=nwp_model,
        )
        total = traj if total is None else total + traj

    return total


def backtest(
    start: str,
    end: str,
    lat: float = config.LAT,
    lon: float = config.LON,
    installed_capacity_mw: float = config.CAPACITY_MW,
    area: str = config.AREA,
    tz: str = config.TZ,
    altitude: float = config.ALTITUDE,
    surface_tilt: float = config.SURFACE_TILT,
    surface_azimuth: float = config.SURFACE_AZIMUTH,
    n_members: int = 50,
    model: Optional[lgb.LGBMRegressor] = None,
    model_path: Optional[Path] = None,
    use_quantile: bool = False,
    quantile_models: Optional[list] = None,
    entsoe_api_key: Optional[str] = None,
    use_smard: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run a historical backtest using ERA5 reanalysis as pseudo-NWP input.

    ERA5 is broadcast into n_members identical copies. With use_quantile=True,
    member i is run through quantile model i, giving proper probabilistic spread
    from quantile regression rather than member_id feature variation.

    Returns
    -------
    trajectories : pd.DataFrame
        Shape (timesteps, n_members) in MW.
    actuals : pd.Series
        Solar generation in MW, aligned to trajectories index.
        15-min SMARD actuals by default; pass use_smard=False for hourly ENTSO-E.
    """
    import os
    from solar_forecast.fetch_actuals import fetch_era5, fetch_entsoe, fetch_smard

    era5 = fetch_era5(lat, lon, start, end, tz=tz)
    clearsky = get_clearsky(
        lat, lon, era5.index,
        altitude=altitude,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
    )
    times_15min = pd.date_range(
        start=era5.index[0],
        end=era5.index[-1] + pd.Timedelta(minutes=45),
        freq="15min",
        tz=era5.index.tz,
    )
    clearsky_15min = get_clearsky(
        lat, lon, times_15min,
        altitude=altitude,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
    )

    rng = np.random.default_rng(42)
    noisy_members = [_inject_nwp_noise(era5, rng) for _ in range(n_members)]
    nwp_raw = {
        var: pd.DataFrame(
            {f"member_{m:02d}": noisy_members[m][var] for m in range(n_members)},
        )
        for var in era5.columns
    }

    if use_quantile:
        if quantile_models is None:
            quantile_models = load_quantile_models()
        trajectories = run_quantile_ensemble(quantile_models, nwp_raw, clearsky, clearsky_15min, installed_capacity_mw)
    else:
        if model is None:
            model = load_model(model_path)
        trajectories = run_ensemble(model, nwp_raw, clearsky, clearsky_15min, installed_capacity_mw)

    if use_smard:
        actuals = fetch_smard(start, end, tz=tz)
    else:
        token = entsoe_api_key or os.environ.get("ENTSOE_API_KEY")
        actuals = fetch_entsoe(area, start, end, api_key=token)
    actuals = actuals.tz_convert(tz).reindex(
        trajectories.index, method=None if use_smard else "ffill"
    )

    return trajectories, actuals


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ensemble forecast for a single location.")
    parser.add_argument("--lat", type=float, default=config.LAT)
    parser.add_argument("--lon", type=float, default=config.LON)
    parser.add_argument("--capacity-mw", type=float, default=config.CAPACITY_MW)
    parser.add_argument("--days", type=int, default=config.FORECAST_DAYS)
    args = parser.parse_args()

    trajectories = forecast(
        lat=args.lat,
        lon=args.lon,
        installed_capacity_mw=args.capacity_mw,
        forecast_days=args.days,
    )

    print(f"Output shape: {trajectories.shape}")
    print(f"Columns: {list(trajectories.columns)}")
    print(trajectories.describe().round(1))
