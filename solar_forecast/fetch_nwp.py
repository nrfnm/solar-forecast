# solar_forecast/fetch_nwp.py

import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "data" / "nwp"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

NWP_VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "temperature_2m",
]

def _get_client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession(CACHE_DIR/ ".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)

def fetch_nwp(
        lat: float,
        lon: float,
        forecast_days: int,
        model: str = 'ecmwf_ifs025',
) -> dict[str, pd.DataFrame]:
    """
Fetch ensemble NWP forecasts from Open-Meteo.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    forecast_days : int, optional
        Number of days to fetch. Default is 2.
    model : str, optional
        Open-Meteo ensemble model. Default is 'ecmwf_ifs025' (50 members).

    Returns
    -------
    dict[str, pd.DataFrame]
        One key per NWP variable. Each DataFrame has shape
        (n_hours, n_members) with a timezone-aware DatetimeIndex.
    """

    client = _get_client()

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": NWP_VARIABLES,
        "models": model,
        "forecast_days": forecast_days,
    }

    responses = client.weather_api(
        "https://ensemble-api.open-meteo.com/v1/ensemble", params=params
    )

    response = responses[0]
    hourly = response.Hourly()

    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_convert("Europe/Vienna")

    result = {}

    for i, var in enumerate(NWP_VARIABLES):
        values = hourly.Variables(i).ValuesAsNumpy()
        n_members = values.shape[0]//len(times)
        result[var] = pd.DataFrame(
            values.reshape(n_members, len(times)).T,
            index=times,
            columns=[f"member_{m:02d}" for m in range(n_members)],
        )

    return result
