"""Shared fixtures for the solar_forecast test suite."""

import numpy as np
import pandas as pd
import pytest

from solar_forecast.clearsky import get_clearsky

LAT, LON = 48.2, 16.3  # Vienna


@pytest.fixture
def times():
    """48 hourly timestamps starting at midnight, Vienna time.

    Summer solstice → sunrise ~05:00, sunset ~21:00, so both daytime and
    nighttime rows are present in every test that uses this fixture.
    """
    return pd.date_range("2024-06-21", periods=48, freq="h", tz="Europe/Vienna")


@pytest.fixture
def clearsky_df(times):
    """Real pvlib clear-sky output for Vienna, default tilt/azimuth."""
    return get_clearsky(LAT, LON, times)


@pytest.fixture
def nwp_df(times):
    """Synthetic single-member NWP DataFrame with all required columns."""
    rng = np.random.default_rng(42)
    n = len(times)
    return pd.DataFrame(
        {
            "shortwave_radiation": rng.uniform(0, 800, n),
            "direct_radiation": rng.uniform(0, 600, n),
            "diffuse_radiation": rng.uniform(0, 200, n),
            "cloud_cover": rng.uniform(0, 100, n),
            "temperature_2m": rng.uniform(10, 30, n),
        },
        index=times,
    )
