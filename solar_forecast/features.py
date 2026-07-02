# solar_forecast/features.py
"""Feature engineering for probabilistic solar forecasting (POA/POA clear-sky index)."""

import numpy as np
import pandas as pd
from pvlib.irradiance import get_total_irradiance

_NWP_REQUIRED = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
    "temperature_2m",
]
_CLEARSKY_REQUIRED = ["poa_clearsky", "apparent_elevation", "azimuth", "is_daytime"]


def _validate_inputs(nwp: pd.DataFrame, clearsky: pd.DataFrame) -> None:
    for col in _NWP_REQUIRED:
        if col not in nwp.columns:
            raise ValueError(f"nwp missing required column: {col!r}")
    for col in _CLEARSKY_REQUIRED:
        if col not in clearsky.columns:
            raise ValueError(f"clearsky missing required column: {col!r}")
    for attr in ("surface_tilt", "surface_azimuth"):
        if attr not in clearsky.attrs:
            raise ValueError(f"clearsky.attrs missing: {attr!r} — use get_clearsky() to produce clearsky")
    if nwp.index.tz is None:
        raise ValueError("nwp index must be timezone-aware")
    if clearsky.index.tz is None:
        raise ValueError("clearsky index must be timezone-aware")
    if not nwp.index.equals(clearsky.index):
        raise ValueError("nwp and clearsky must have identical indices")


def build_features(
    nwp: pd.DataFrame,
    clearsky: pd.DataFrame,
    ci_clip: tuple = (0.0, 1.1),
) -> pd.DataFrame:
    """
    Build model features for a single NWP member.

    Parameters
    ----------
    nwp : pd.DataFrame
        Single-member NWP data. Required columns: shortwave_radiation,
        direct_radiation, diffuse_radiation, cloud_cover, temperature_2m.
        Must have a timezone-aware DatetimeIndex.
    clearsky : pd.DataFrame
        Output of clearsky.get_clearsky(). Must carry surface_tilt and
        surface_azimuth in df.attrs.
    ci_clip : tuple
        (min, max) bounds applied to the clear-sky index.

    Returns
    -------
    pd.DataFrame
        One row per timestamp, all feature columns.
    """
    _validate_inputs(nwp, clearsky)

    idx = nwp.index
    surface_tilt = clearsky.attrs["surface_tilt"]
    surface_azimuth = clearsky.attrs["surface_azimuth"]

    # --- POA from NWP ---
    # Open-Meteo direct_radiation is horizontal-projected beam → recover true DNI.
    # Clamp cos(zenith) to 0.05 to avoid near-horizon division blow-up.
    solar_zenith = 90.0 - clearsky["apparent_elevation"]
    cos_zenith = np.cos(np.deg2rad(solar_zenith)).clip(lower=0.05)
    dni_nwp = nwp["direct_radiation"] / cos_zenith

    poa_components = get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dhi=nwp["diffuse_radiation"],
        dni=dni_nwp,
        ghi=nwp["shortwave_radiation"],
        solar_zenith=solar_zenith,
        solar_azimuth=clearsky["azimuth"],
    )
    poa_nwp = poa_components["poa_global"].clip(lower=0)

    # --- Clear-Sky Index ---
    is_daytime = clearsky["is_daytime"]
    ci = (poa_nwp / clearsky["poa_clearsky"]).clip(*ci_clip)
    ci = ci.where(is_daytime, 0.0)

    # --- Solar geometry features ---
    elevation = clearsky["apparent_elevation"].clip(lower=0)
    azimuth = clearsky["azimuth"]
    sin_elevation = np.sin(np.deg2rad(elevation))
    cos_azimuth = np.cos(np.deg2rad(azimuth - 180))

    # --- Cyclic time encodings ---
    hour = idx.hour + idx.minute / 60.0
    doy = idx.day_of_year
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)

    # --- Cloud features ---
    cloud_cover = nwp["cloud_cover"] / 100.0
    cloud_cover_sq = cloud_cover ** 2

    # --- Lag & rolling (shift(1) ensures no leakage) ---
    ci_lag1 = ci.shift(1)
    ci_lag2 = ci.shift(2)
    ci_rolling_mean_3h = ci.rolling(3).mean().shift(1)
    cloud_change_1h = cloud_cover.diff(1)
    cloud_rolling_mean_3h = cloud_cover.rolling(3).mean().shift(1)

    return pd.DataFrame(
        {
            "ci": ci,
            "poa_nwp": poa_nwp,
            "solar_elevation": elevation,
            "solar_azimuth": azimuth,
            "sin_elevation": sin_elevation,
            "cos_azimuth": cos_azimuth,
            "is_daytime": is_daytime.astype(float),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
            "cloud_cover": cloud_cover,
            "cloud_cover_sq": cloud_cover_sq,
            "temperature_2m": nwp["temperature_2m"],
            "shortwave_radiation": nwp["shortwave_radiation"],
            "ci_lag1": ci_lag1,
            "ci_lag2": ci_lag2,
            "ci_rolling_mean_3h": ci_rolling_mean_3h,
            "cloud_change_1h": cloud_change_1h,
            "cloud_rolling_mean_3h": cloud_rolling_mean_3h,
        },
        index=idx,
    )
