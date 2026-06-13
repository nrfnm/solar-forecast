"""Tests for solar_forecast.fetch_actuals."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from solar_forecast.fetch_actuals import (
    ENTSOE_AREAS,
    fetch_ifs_historical,
    fetch_entsoe,
    _parse_entsoe_xml,
)
from solar_forecast.fetch_nwp import NWP_VARIABLES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_HOURS = 24
START_TS = 1_719_000_000  # 2024-06-21 22:00 UTC
END_TS = START_TS + N_HOURS * 3600


def _make_mock_archive_client():
    """Mock openmeteo client returning a single-member (1D) time series."""
    mock_var = MagicMock()
    mock_var.ValuesAsNumpy.return_value = np.zeros(N_HOURS)

    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = START_TS
    mock_hourly.TimeEnd.return_value = END_TS
    mock_hourly.Interval.return_value = 3600
    mock_hourly.Variables.return_value = mock_var

    mock_response = MagicMock()
    mock_response.Hourly.return_value = mock_hourly

    mock_client = MagicMock()
    mock_client.weather_api.return_value = [mock_response]
    return mock_client


# Minimal XML with 4 hourly points
SAMPLE_XML_60MIN = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <Period>
      <timeInterval><start>2024-06-21T00:00Z</start></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>0.0</quantity></Point>
      <Point><position>2</position><quantity>100.0</quantity></Point>
      <Point><position>3</position><quantity>200.0</quantity></Point>
      <Point><position>4</position><quantity>0.0</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

# 4 × 15-min points that all fall in the first hour → hourly mean = 110.0
SAMPLE_XML_15MIN = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <Period>
      <timeInterval><start>2024-06-21T00:00Z</start></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>80.0</quantity></Point>
      <Point><position>2</position><quantity>100.0</quantity></Point>
      <Point><position>3</position><quantity>120.0</quantity></Point>
      <Point><position>4</position><quantity>140.0</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

# Two TimeSeries for the same hour: 100 MW + 50 MW → sum = 150 MW
SAMPLE_XML_TWO_TS = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <Period>
      <timeInterval><start>2024-06-21T00:00Z</start></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>100.0</quantity></Point>
    </Period>
  </TimeSeries>
  <TimeSeries>
    <Period>
      <timeInterval><start>2024-06-21T00:00Z</start></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>50.0</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
</GL_MarketDocument>"""


# ---------------------------------------------------------------------------
# fetch_ifs_historical
# ---------------------------------------------------------------------------

class TestFetchIfsHistorical:
    @pytest.fixture(autouse=True)
    def mock_client(self):
        with patch("solar_forecast.fetch_actuals._get_om_client", return_value=_make_mock_archive_client()):
            yield

    def test_returns_dataframe(self):
        result = fetch_ifs_historical(48.2, 16.3, "2024-06-21", "2024-06-21")
        assert isinstance(result, pd.DataFrame)

    def test_all_columns_present(self):
        result = fetch_ifs_historical(48.2, 16.3, "2024-06-21", "2024-06-21")
        assert set(NWP_VARIABLES).issubset(set(result.columns))

    def test_correct_length(self):
        result = fetch_ifs_historical(48.2, 16.3, "2024-06-21", "2024-06-21")
        assert len(result) == N_HOURS

    def test_index_is_timezone_aware(self):
        result = fetch_ifs_historical(48.2, 16.3, "2024-06-21", "2024-06-21")
        assert result.index.tz is not None

    def test_api_called_with_historical_forecast_url_and_correct_params(self):
        mock_client = _make_mock_archive_client()
        with patch("solar_forecast.fetch_actuals._get_om_client", return_value=mock_client):
            fetch_ifs_historical(48.2, 16.3, "2024-06-01", "2024-06-30", tz="Europe/Vienna")

        url, params = (
            mock_client.weather_api.call_args.args[0],
            mock_client.weather_api.call_args.kwargs["params"],
        )
        assert "historical-forecast-api.open-meteo.com" in url
        assert params["latitude"] == 48.2
        assert params["longitude"] == 16.3
        assert params["start_date"] == "2024-06-01"
        assert params["end_date"] == "2024-06-30"
        assert params["models"] == "ecmwf_ifs025"
        assert set(params["hourly"]) == set(NWP_VARIABLES)


# ---------------------------------------------------------------------------
# _parse_entsoe_xml
# ---------------------------------------------------------------------------

class TestParseEntsoeXml:
    def test_returns_series(self):
        assert isinstance(_parse_entsoe_xml(SAMPLE_XML_60MIN), pd.Series)

    def test_series_name(self):
        assert _parse_entsoe_xml(SAMPLE_XML_60MIN).name == "solar_mw"

    def test_utc_index(self):
        result = _parse_entsoe_xml(SAMPLE_XML_60MIN)
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_hourly_frequency(self):
        result = _parse_entsoe_xml(SAMPLE_XML_60MIN)
        diffs = result.index.to_series().diff().dropna().dt.total_seconds()
        assert (diffs == 3600).all()

    def test_values_correct_60min(self):
        result = _parse_entsoe_xml(SAMPLE_XML_60MIN)
        np.testing.assert_array_equal(result.values, [0.0, 100.0, 200.0, 0.0])

    def test_15min_resampled_to_hourly_mean(self):
        result = _parse_entsoe_xml(SAMPLE_XML_15MIN)
        assert len(result) == 1
        np.testing.assert_allclose(result.iloc[0], (80 + 100 + 120 + 140) / 4)

    def test_two_timeseries_summed(self):
        result = _parse_entsoe_xml(SAMPLE_XML_TWO_TS)
        assert len(result) == 1
        np.testing.assert_allclose(result.iloc[0], 150.0)

    def test_empty_xml_returns_empty_series(self):
        result = _parse_entsoe_xml(EMPTY_XML)
        assert len(result) == 0
        assert result.name == "solar_mw"


# ---------------------------------------------------------------------------
# fetch_entsoe
# ---------------------------------------------------------------------------

class TestFetchEntsoe:
    @pytest.fixture
    def mock_response(self):
        resp = MagicMock()
        resp.text = SAMPLE_XML_60MIN
        resp.raise_for_status = MagicMock()
        return resp

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            fetch_entsoe("DE", "2024-06-01", "2024-06-01")

    def test_raises_on_unknown_area(self):
        with pytest.raises(ValueError, match="Unknown area"):
            fetch_entsoe("XX", "2024-06-01", "2024-06-01", api_key="tok")

    def test_returns_series(self, mock_response):
        with patch("solar_forecast.fetch_actuals.requests.get", return_value=mock_response):
            result = fetch_entsoe("DE", "2024-06-21", "2024-06-21", api_key="tok")
        assert isinstance(result, pd.Series)

    def test_series_name(self, mock_response):
        with patch("solar_forecast.fetch_actuals.requests.get", return_value=mock_response):
            result = fetch_entsoe("DE", "2024-06-21", "2024-06-21", api_key="tok")
        assert result.name == "solar_mw"

    def test_utc_index(self, mock_response):
        with patch("solar_forecast.fetch_actuals.requests.get", return_value=mock_response):
            result = fetch_entsoe("DE", "2024-06-21", "2024-06-21", api_key="tok")
        assert str(result.index.tz) == "UTC"

    def test_correct_domain_and_psr_type(self, mock_response):
        with patch("solar_forecast.fetch_actuals.requests.get", return_value=mock_response) as mock_get:
            fetch_entsoe("DE", "2024-06-21", "2024-06-21", api_key="tok")
        params = mock_get.call_args.kwargs["params"]
        assert params["in_Domain"] == ENTSOE_AREAS["DE"]
        assert params["psrType"] == "B16"
        assert params["documentType"] == "A75"

    def test_env_api_key_used(self, mock_response, monkeypatch):
        monkeypatch.setenv("ENTSOE_API_KEY", "env_token")
        with patch("solar_forecast.fetch_actuals.requests.get", return_value=mock_response) as mock_get:
            fetch_entsoe("AT", "2024-06-21", "2024-06-21")
        assert mock_get.call_args.kwargs["params"]["securityToken"] == "env_token"

    def test_multi_year_chunks_into_two_requests(self, mock_response):
        with patch("solar_forecast.fetch_actuals.requests.get", return_value=mock_response) as mock_get:
            fetch_entsoe("DE", "2023-01-01", "2024-12-31", api_key="tok")
        assert mock_get.call_count == 2

    def test_area_is_case_insensitive(self, mock_response):
        with patch("solar_forecast.fetch_actuals.requests.get", return_value=mock_response):
            result = fetch_entsoe("de", "2024-06-21", "2024-06-21", api_key="tok")
        assert isinstance(result, pd.Series)
