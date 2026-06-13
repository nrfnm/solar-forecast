# solar_forecast/fetch_actuals.py
"""Fetch historical training data: IFS historical forecast inputs + ENTSO-E solar generation targets."""

import os
import time
from zoneinfo import ZoneInfo

import requests
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET
from retry_requests import retry

_RATE_LIMIT_WAIT = 65  # seconds to wait after hitting Open-Meteo per-minute rate limit

_SMARD_SOLAR_MODULE_ID = 125   # Photovoltaik: Realisierte Erzeugung DE-LU
_SMARD_SOLAR_REGION = "DE-LU"
_SMARD_MWH_TO_MW = 4.0        # SMARD values are MWh/quarter-hour → MW average

_CACHE_DIR = Path(__file__).parent.parent / "data" / "ifs_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
ENTSOE_AREAS = {
    "DE": "10Y1001A1001A82H",
    "AT": "10YAT-APG------L",
}


def _get_om_client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession(
        _CACHE_DIR / ".ifs_cache", expire_after=-1  # historical data doesn't change
    )
    # Only retries HTTP-level errors (4xx/5xx); application-level rate limits (200 + error JSON)
    # are handled by the _rate_limited_weather_api wrapper below.
    retry_session = retry(cache_session, retries=3, backoff_factor=1.0)
    return openmeteo_requests.Client(session=retry_session)


def _rate_limited_weather_api(client, url: str, params: dict, max_retries: int = 5) -> list:
    """Call client.weather_api with automatic retry on Open-Meteo rate-limit responses."""
    for attempt in range(max_retries):
        try:
            return client.weather_api(url, params=params)
        except Exception as exc:
            msg = str(exc).lower()
            if "rate" in msg or "limit exceeded" in msg or "429" in msg:
                wait = _RATE_LIMIT_WAIT * (attempt + 1)
                print(f"Open-Meteo rate limit hit (attempt {attempt + 1}/{max_retries}) — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Open-Meteo rate limit not cleared after {max_retries} retries")


def fetch_ifs_historical(
    lat: float,
    lon: float,
    start: str,
    end: str,
    tz: str = "Europe/Vienna",
) -> pd.DataFrame:
    """
    Fetch deterministic ECMWF IFS historical forecasts from Open-Meteo.

    Available since 2024-02-03. Returns hourly NWP_VARIABLES (shortwave/direct/diffuse
    radiation, cloud_cover, cloud_cover_low/mid/high, temperature_2m). Index is
    timezone-aware. Compatible with build_features(member_id=-1).
    """
    from solar_forecast.fetch_nwp import NWP_VARIABLES

    client = _get_om_client()
    responses = _rate_limited_weather_api(
        client,
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": NWP_VARIABLES,
            "models": "ecmwf_ifs025",
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
        {var: hourly.Variables(i).ValuesAsNumpy() for i, var in enumerate(NWP_VARIABLES)},
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


_SMARD_INDEX_URL = (
    "https://www.smard.de/app/chart_data/{module_id}/{region}/index_{resolution}.json"
)
_SMARD_DATA_URL = (
    "https://www.smard.de/app/chart_data/{module_id}/{region}"
    "/{module_id}_{region}_{resolution}_{timestamp}.json"
)


def fetch_smard(
    start: str,
    end: str,
    tz: str = "Europe/Berlin",
) -> pd.Series:
    """
    Fetch quarter-hourly solar generation actuals from SMARD (DE-LU, module 125).

    No API key required. Returns 15-min UTC pd.Series in MW.
    Uses the chart_data JSON API (weekly files). SMARD values are MWh per
    quarter-hour interval; multiply by 4 to get average MW.

    Parameters
    ----------
    start, end : str
        "YYYY-MM-DD" inclusive date range.
    tz : str
        Local timezone for boundary interpretation. Default "Europe/Berlin".

    Returns
    -------
    pd.Series
        15-min solar generation in MW, UTC DatetimeIndex, name="solar_mw".
    """
    local_tz = ZoneInfo(tz)
    start_ms = int(pd.Timestamp(start).tz_localize(local_tz).timestamp() * 1000)
    end_ms = int(
        (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(local_tz).timestamp() * 1000
    )

    index_url = _SMARD_INDEX_URL.format(
        module_id=_SMARD_SOLAR_MODULE_ID,
        region=_SMARD_SOLAR_REGION,
        resolution="quarterhour",
    )
    resp = requests.get(index_url, timeout=20)
    resp.raise_for_status()
    week_timestamps = resp.json()["timestamps"]

    # keep weeks that overlap with [start_ms, end_ms)
    overlapping = [t for t in week_timestamps if t < end_ms and t + 7 * 24 * 3600 * 1000 > start_ms]

    all_points: list[tuple[int, float]] = []
    for week_ts in overlapping:
        data_url = _SMARD_DATA_URL.format(
            module_id=_SMARD_SOLAR_MODULE_ID,
            region=_SMARD_SOLAR_REGION,
            resolution="quarterhour",
            timestamp=week_ts,
        )
        r = requests.get(data_url, timeout=20)
        r.raise_for_status()
        for ts_ms, val in r.json().get("series", []):
            if val is not None and start_ms <= ts_ms < end_ms:
                all_points.append((ts_ms, float(val)))

    if not all_points:
        return pd.Series(dtype=float, name="solar_mw")

    all_points.sort(key=lambda x: x[0])
    index = pd.to_datetime([t for t, _ in all_points], unit="ms", utc=True)
    values = [v * _SMARD_MWH_TO_MW for _, v in all_points]
    return pd.Series(values, index=index, name="solar_mw", dtype=float)
