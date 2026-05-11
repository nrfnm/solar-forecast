"""Tests for solar_forecast.fetch_nwp."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from solar_forecast.fetch_nwp import NWP_VARIABLES, fetch_nwp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_HOURS = 48
N_MEMBERS = 3
START_TS = 1_719_000_000  # 2024-06-21 22:00 UTC (arbitrary fixed point)
END_TS = START_TS + N_HOURS * 3600


def _make_mock_client():
    """Build a mock openmeteo client that returns N_MEMBERS × N_HOURS data."""
    mock_var = MagicMock()
    mock_var.ValuesAsNumpy.return_value = np.zeros(N_HOURS * N_MEMBERS)

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


# ---------------------------------------------------------------------------
# NWP_VARIABLES contract
# ---------------------------------------------------------------------------

class TestNwpVariables:
    REQUIRED = {
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
        "temperature_2m",
    }

    def test_contains_features_required_by_features_module(self):
        assert self.REQUIRED.issubset(set(NWP_VARIABLES))

    def test_is_list_of_strings(self):
        assert isinstance(NWP_VARIABLES, list)
        assert all(isinstance(v, str) for v in NWP_VARIABLES)

    def test_no_duplicates(self):
        assert len(NWP_VARIABLES) == len(set(NWP_VARIABLES))


# ---------------------------------------------------------------------------
# fetch_nwp — output structure (mocked API)
# ---------------------------------------------------------------------------

class TestFetchNwp:
    @pytest.fixture(autouse=True)
    def mock_client(self):
        with patch("solar_forecast.fetch_nwp._get_client", return_value=_make_mock_client()):
            yield

    def test_result_is_dict(self):
        result = fetch_nwp(48.2, 16.3, forecast_days=2)
        assert isinstance(result, dict)

    def test_returns_all_variables(self):
        result = fetch_nwp(48.2, 16.3, forecast_days=2)
        assert set(result.keys()) == set(NWP_VARIABLES)

    def test_each_value_is_dataframe(self):
        result = fetch_nwp(48.2, 16.3, forecast_days=2)
        for var, df in result.items():
            assert isinstance(df, pd.DataFrame), f"{var} is not a DataFrame"

    def test_dataframe_has_member_columns(self):
        result = fetch_nwp(48.2, 16.3, forecast_days=2)
        for var, df in result.items():
            assert df.shape[1] == N_MEMBERS, f"{var} has wrong number of member columns"
            expected_cols = [f"member_{m:02d}" for m in range(N_MEMBERS)]
            assert list(df.columns) == expected_cols

    def test_index_is_timezone_aware(self):
        result = fetch_nwp(48.2, 16.3, forecast_days=2)
        for var, df in result.items():
            assert df.index.tz is not None, f"{var} index has no timezone"

    def test_index_has_correct_length(self):
        result = fetch_nwp(48.2, 16.3, forecast_days=2)
        for var, df in result.items():
            assert len(df) == N_HOURS, f"{var} has wrong number of rows"

    def test_api_called_with_correct_params(self):
        mock_client_instance = _make_mock_client()
        with patch("solar_forecast.fetch_nwp._get_client", return_value=mock_client_instance):
            fetch_nwp(48.2, 16.3, forecast_days=3, model="ecmwf_ifs025")

        call_kwargs = mock_client_instance.weather_api.call_args
        params = call_kwargs.kwargs["params"]
        assert params["latitude"] == 48.2
        assert params["longitude"] == 16.3
        assert params["forecast_days"] == 3
        assert params["models"] == "ecmwf_ifs025"
        assert set(params["hourly"]) == set(NWP_VARIABLES)
