"""Apply trained LightGBM CI model across all NWP ensemble members → MW trajectories."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Optional

from solar_forecast.clearsky import get_clearsky
from solar_forecast.fetch_nwp import fetch_nwp
from solar_forecast.features import build_features
from solar_forecast.train import FEATURE_COLS, load_model
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
        Output of get_clearsky() for the same location and time range.
    installed_capacity_mw : float
        Total installed solar capacity used to scale CI → MW.
    stc_irradiance : float
        Reference irradiance (W/m²) at which panels produce rated capacity.

    Returns
    -------
    pd.DataFrame
        Shape (timesteps, n_members), columns member_00…member_N, values in MW.
        Nighttime rows are zero.
    """
    nwp_by_member = _pivot_nwp(nwp_raw)
    clearsky_mw = (clearsky["poa_clearsky"] / stc_irradiance) * installed_capacity_mw

    trajectories = {}
    for member_name, nwp_df in nwp_by_member.items():
        try:
            member_id = int(member_name.split("_")[-1])
        except (ValueError, AttributeError):
            member_id = -1

        feats = build_features(nwp_df, clearsky, member_id=member_id)
        ci_pred = model.predict(feats[FEATURE_COLS]).clip(0, 1.1)
        mw = pd.Series(ci_pred, index=feats.index) * clearsky_mw
        trajectories[member_name] = mw.where(clearsky["is_daytime"], 0.0)

    return pd.DataFrame(trajectories)


def forecast(
    lat: float,
    lon: float,
    installed_capacity_mw: float,
    forecast_days: int = 7,
    model: Optional[lgb.LGBMRegressor] = None,
    model_path: Optional[Path] = None,
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
    if model is None:
        model = load_model(model_path)

    nwp_raw = fetch_nwp(lat, lon, forecast_days=forecast_days, model=nwp_model)
    times = nwp_raw[next(iter(nwp_raw))].index
    clearsky = get_clearsky(
        lat, lon, times,
        altitude=altitude,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
    )

    return run_ensemble(model, nwp_raw, clearsky, installed_capacity_mw)


def forecast_country(
    model: Optional[lgb.LGBMRegressor] = None,
    model_path: Optional[Path] = None,
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
    if model is None:
        model = load_model(model_path)

    if not config.CENTROIDS:
        return forecast(
            lat=config.LAT,
            lon=config.LON,
            installed_capacity_mw=config.CAPACITY_MW,
            forecast_days=forecast_days,
            model=model,
            nwp_model=nwp_model,
        )

    total: Optional[pd.DataFrame] = None
    for i, c in enumerate(config.CENTROIDS):
        print(f"  Centroid {i+1}/{len(config.CENTROIDS)}: lat={c['lat']}, lon={c['lon']}, weight={c['weight']:.3f}")
        traj = forecast(
            lat=c["lat"],
            lon=c["lon"],
            installed_capacity_mw=config.CAPACITY_MW * c["weight"],
            forecast_days=forecast_days,
            model=model,
            nwp_model=nwp_model,
        )
        total = traj if total is None else total + traj

    return total


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
