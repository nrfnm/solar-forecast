# solar_forecast/fetch_nwp.py

import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "data" / "nwp"
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

    # SDK stores one entry per (variable, member) pair; group by Variable() code
    # in order of first appearance, which matches the NWP_VARIABLES request order.
    var_data: dict[int, list[tuple[int, np.ndarray]]] = {}
    var_order: list[int] = []
    for i in range(hourly.VariablesLength()):
        entry = hourly.Variables(i)
        code = entry.Variable()
        if code not in var_data:
            var_data[code] = []
            var_order.append(code)
        var_data[code].append((entry.EnsembleMember(), entry.ValuesAsNumpy()))

    result = {}
    for var_name, code in zip(NWP_VARIABLES, var_order):
        members = sorted(var_data[code], key=lambda x: x[0])
        result[var_name] = pd.DataFrame(
            np.stack([vals for _, vals in members], axis=1),
            index=times,
            columns=[f"member_{mid:02d}" for mid, _ in members],
        )

    return result
