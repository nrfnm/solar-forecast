# solar_forecast/clearsky.py

import pandas as pd
import pvlib

def get_clearsky(
    lat: float,
    lon: float,
    times: pd.DatetimeIndex,
    altitude: float = 500,
    model: str = "ineichen"
) -> pd.DataFrame:
    """

    Parameters
    ----------
    lat: float
    lon: float
    times: pd.DatetimeIndex
    altitude: float
    model: str


    Returns
    -------

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

    return result