"""Tests for solar_forecast.clearsky."""

import numpy as np
import pandas as pd
import pytest

from solar_forecast.clearsky import get_clearsky

LAT, LON = 48.2, 16.3

EXPECTED_COLUMNS = {
    "ghi", "dni", "dhi",
    "apparent_elevation", "azimuth",
    "is_daytime", "poa_clearsky",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_raises_on_naive_index(times):
    naive = times.tz_localize(None)
    with pytest.raises(ValueError, match="timezone aware"):
        get_clearsky(LAT, LON, naive)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_output_columns(clearsky_df):
    assert EXPECTED_COLUMNS.issubset(set(clearsky_df.columns))


def test_output_index_matches_input(times, clearsky_df):
    assert clearsky_df.index.equals(times)


def test_attrs_stored_with_defaults(clearsky_df):
    assert clearsky_df.attrs["surface_tilt"] == 30
    assert clearsky_df.attrs["surface_azimuth"] == 180


def test_attrs_stored_with_custom_values(times):
    result = get_clearsky(LAT, LON, times, surface_tilt=15, surface_azimuth=90)
    assert result.attrs["surface_tilt"] == 15
    assert result.attrs["surface_azimuth"] == 90


# ---------------------------------------------------------------------------
# Physical correctness
# ---------------------------------------------------------------------------

def test_poa_clearsky_nonnegative(clearsky_df):
    assert (clearsky_df["poa_clearsky"] >= 0).all()


def test_irradiance_nonnegative(clearsky_df):
    for col in ("ghi", "dni", "dhi"):
        assert (clearsky_df[col] >= 0).all(), f"{col} has negative values"


def test_nighttime_irradiance_is_zero(clearsky_df):
    night = ~clearsky_df["is_daytime"]
    assert night.any(), "fixture should contain at least one nighttime row"
    for col in ("ghi", "dni", "dhi", "poa_clearsky"):
        assert (clearsky_df.loc[night, col] == 0).all(), \
            f"{col} is non-zero at night"


def test_is_daytime_consistent_with_elevation(clearsky_df):
    expected = clearsky_df["apparent_elevation"] > 0
    pd.testing.assert_series_equal(clearsky_df["is_daytime"], expected, check_names=False)


def test_daytime_has_positive_poa(clearsky_df):
    day = clearsky_df["is_daytime"]
    assert day.any(), "fixture should contain at least one daytime row"
    assert (clearsky_df.loc[day, "poa_clearsky"] > 0).all()


def test_custom_tilt_changes_poa(times):
    flat = get_clearsky(LAT, LON, times, surface_tilt=0)
    tilted = get_clearsky(LAT, LON, times, surface_tilt=45)
    # For a south-facing panel in northern hemisphere, 45° tilt ≠ 0° tilt
    assert not flat["poa_clearsky"].equals(tilted["poa_clearsky"])


def test_ghi_gte_dhi_during_day(clearsky_df):
    day = clearsky_df["is_daytime"]
    assert (clearsky_df.loc[day, "ghi"] >= clearsky_df.loc[day, "dhi"]).all()
