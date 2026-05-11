# solar_forecast/clearsky.py
"""Clear-sky solar irradiance estimation using pvlib."""

import pandas as pd
import pvlib
from pvlib.irradiance import get_total_irradiance


def get_clearsky(
    lat: float,
    lon: float,
    times: pd.DatetimeIndex,
    altitude: float = 500,
    model: str = "ineichen",
    surface_tilt: float = 30,
    surface_azimuth: float = 180,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    lat: float
    lon: float
    times: pd.DatetimeIndex
    altitude: float
    model: str
    surface_tilt: float
        Panel tilt from horizontal in degrees. Default 30.
    surface_azimuth: float
        Panel azimuth in degrees (180 = south-facing). Default 180.

    Returns
    -------
    pd.DataFrame
        Columns: ghi, dni, dhi, apparent_elevation, azimuth, is_daytime, poa_clearsky.
        attrs: surface_tilt, surface_azimuth.
    """
    if times.tz is None:
        raise ValueError("times must be timezone aware, use tz.localize()")

    location = pvlib.location.Location(
        latitude=lat,
        longitude=lon,
        altitude=altitude,
        tz=str(times.tz),
    )

    clearsky = location.get_clearsky(times, model=model)
    solar_pos = location.get_solarposition(times)

    #ghi =  Global Horizontal Irradiance (W/m²) — total on a flat surface
    #dni = Direct Normal Irradiance (W/m²) — beam radiation
    #dhi = Diffuse Horizontal Irradiance (W/m²) — scattered sky light
    #apparent_elevation Sun angle above horizon in degrees (accounts for atmospheric refraction)
    #azimuth = Sun compass irection in degrees
    result = clearsky[['ghi', 'dni', 'dhi']].copy()
    result["apparent_elevation"] = solar_pos["apparent_elevation"]
    result["azimuth"] = solar_pos["azimuth"]
    result["is_daytime"] = result["apparent_elevation"] > 0

    # Zero out irradiance at night (pvlib occasionally returns tiny positives)
    for col in ["ghi", "dni", "dhi"]:
        result.loc[~result["is_daytime"], col] = 0.0

    poa = get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dhi=clearsky["dhi"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
    )
    result["poa_clearsky"] = poa["poa_global"].clip(lower=0)
    result.loc[~result["is_daytime"], "poa_clearsky"] = 0.0

    result.attrs["surface_tilt"] = surface_tilt
    result.attrs["surface_azimuth"] = surface_azimuth

    return result
