# solar_forecast/fetch_actuals.py
"""Fetch historical training data: ERA5 reanalysis inputs + ENTSO-E solar generation targets."""

import os
import requests
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET
from retry_requests import retry

_CACHE_DIR = Path(__file__).parent.parent / "data" / "era5_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
ENTSOE_AREAS = {
    "DE": "10Y1001A1001A82H",
    "AT": "10YAT-APG------L",
}

ERA5_VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
    "temperature_2m",
]


def _get_om_client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession(
        _CACHE_DIR / ".era5_cache", expire_after=-1  # historical data doesn't change
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_era5(
    lat: float,
    lon: float,
    start: str,
    end: str,
    tz: str = "Europe/Vienna",
) -> pd.DataFrame:
    """
    Fetch ERA5 reanalysis via Open-Meteo Historical Archive API.

    Parameters
    ----------
    lat, lon : float
    start, end : str
        "YYYY-MM-DD" inclusive date range.
    tz : str
        Target timezone for the returned index. Default "Europe/Vienna".

    Returns
    -------
    pd.DataFrame
        Hourly data with columns matching the single-member output of fetch_nwp:
        shortwave_radiation, direct_radiation, diffuse_radiation, cloud_cover,
        temperature_2m. Index is timezone-aware. Compatible with build_features(member_id=-1).
    """
    client = _get_om_client()
    responses = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ERA5_VARIABLES,
            "timezone": tz,
        },
    )
    response = responses[0]
    hourly = response.Hourly()

    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_convert(tz)

    return pd.DataFrame(
        {var: hourly.Variables(i).ValuesAsNumpy() for i, var in enumerate(ERA5_VARIABLES)},
        index=times,
    )


def _parse_entsoe_xml(xml_text: str) -> pd.Series:
    """Parse ENTSO-E ActualGenerationPerProductionType XML into hourly UTC MW Series."""
    ns = {"e": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}
    root = ET.fromstring(xml_text)

    freq_map = {"PT15M": 15, "PT30M": 30, "PT60M": 60}
    all_timestamps: list[pd.Timestamp] = []
    all_values: list[float] = []

    for ts_elem in root.findall(".//e:TimeSeries", ns):
        period = ts_elem.find("e:Period", ns)
        if period is None:
            continue
        start_str = period.find("e:timeInterval/e:start", ns).text
        resolution = period.find("e:resolution", ns).text
        freq_min = freq_map.get(resolution, 60)
        origin = pd.to_datetime(start_str, utc=True)

        for point in period.findall("e:Point", ns):
            pos = int(point.find("e:position", ns).text)
            qty_elem = point.find("e:quantity", ns)
            if qty_elem is None or qty_elem.text is None:
                continue
            all_timestamps.append(origin + pd.Timedelta(minutes=freq_min * (pos - 1)))
            all_values.append(float(qty_elem.text))

    if not all_timestamps:
        return pd.Series(dtype=float, name="solar_mw")

    raw = pd.Series(
        all_values,
        index=pd.DatetimeIndex(all_timestamps, tz="UTC"),
        name="solar_mw",
        dtype=float,
    )
    # Sum contributions from multiple TimeSeries at the same timestamp (multiple bidding zones),
    # then average sub-hourly intervals to get mean MW per hour.
    raw = raw.groupby(raw.index).sum()
    return raw.sort_index().resample("h").mean()


def fetch_entsoe(
    area: str,
    start: str,
    end: str,
    api_key: str | None = None,
) -> pd.Series:
    """
    Fetch solar generation actuals (B16) from ENTSO-E Transparency Platform.

    Parameters
    ----------
    area : str
        Country code: "DE" or "AT".
    start, end : str
        "YYYY-MM-DD" inclusive date range. Chunked into yearly batches internally.
    api_key : str, optional
        ENTSO-E security token; falls back to ENTSOE_API_KEY env var.

    Returns
    -------
    pd.Series
        Hourly solar generation in MW, UTC DatetimeIndex, name="solar_mw".
    """
    token = api_key or os.environ.get("ENTSOE_API_KEY")
    if not token:
        raise ValueError(
            "ENTSO-E API key required. Pass api_key= or set ENTSOE_API_KEY env var."
        )

    domain = ENTSOE_AREAS.get(area.upper())
    if domain is None:
        raise ValueError(f"Unknown area {area!r}. Available: {list(ENTSOE_AREAS)}")

    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

    chunks: list[pd.Series] = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + pd.DateOffset(years=1), end_dt)
        resp = requests.get(
            ENTSOE_URL,
            params={
                "securityToken": token,
                "documentType": "A75",
                "processType": "A16",
                "psrType": "B16",
                "in_Domain": domain,
                "periodStart": chunk_start.strftime("%Y%m%d%H%M"),
                "periodEnd": chunk_end.strftime("%Y%m%d%H%M"),
            },
            timeout=30,
        )
        resp.raise_for_status()
        chunks.append(_parse_entsoe_xml(resp.text))
        chunk_start = chunk_end

    if not chunks:
        return pd.Series(dtype=float, name="solar_mw")

    combined = pd.concat(chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.loc[start_dt : end_dt - pd.Timedelta(hours=1)]
