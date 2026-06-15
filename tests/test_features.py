"""Tests for solar_forecast.features."""

import numpy as np
import pandas as pd
import pytest

from solar_forecast.features import _validate_inputs, build_features

EXPECTED_COLUMNS = {
    "ci", "poa_nwp",
    "solar_elevation", "solar_azimuth", "sin_elevation", "cos_azimuth", "is_daytime",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "cloud_cover", "cloud_cover_sq", "temperature_2m", "shortwave_radiation",
    "ci_lag1", "ci_lag2", "ci_rolling_mean_3h",
    "cloud_change_1h", "cloud_rolling_mean_3h",
}


# ---------------------------------------------------------------------------
# _validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_missing_nwp_column(self, nwp_df, clearsky_df):
        bad = nwp_df.drop(columns=["cloud_cover"])
        with pytest.raises(ValueError, match="nwp missing required column"):
            _validate_inputs(bad, clearsky_df)

    def test_missing_clearsky_column(self, nwp_df, clearsky_df):
        bad = clearsky_df.drop(columns=["poa_clearsky"])
        with pytest.raises(ValueError, match="clearsky missing required column"):
            _validate_inputs(nwp_df, bad)

    def test_missing_surface_tilt_attr(self, nwp_df, clearsky_df):
        bad = clearsky_df.copy()
        bad.attrs = {"surface_azimuth": 180}
        with pytest.raises(ValueError, match="surface_tilt"):
            _validate_inputs(nwp_df, bad)

    def test_missing_surface_azimuth_attr(self, nwp_df, clearsky_df):
        bad = clearsky_df.copy()
        bad.attrs = {"surface_tilt": 30}
        with pytest.raises(ValueError, match="surface_azimuth"):
            _validate_inputs(nwp_df, bad)

    def test_naive_nwp_index(self, nwp_df, clearsky_df):
        bad = nwp_df.copy()
        bad.index = nwp_df.index.tz_localize(None)
        with pytest.raises(ValueError, match="timezone-aware"):
            _validate_inputs(bad, clearsky_df)

    def test_mismatched_indices(self, nwp_df, clearsky_df):
        shifted = nwp_df.shift(1).dropna()
        with pytest.raises(ValueError, match="identical indices"):
            _validate_inputs(shifted, clearsky_df)

    def test_valid_inputs_do_not_raise(self, nwp_df, clearsky_df):
        _validate_inputs(nwp_df, clearsky_df)  # must not raise


# ---------------------------------------------------------------------------
# build_features — structure
# ---------------------------------------------------------------------------

class TestBuildFeaturesStructure:
    def test_all_columns_present(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        assert EXPECTED_COLUMNS.issubset(set(feat.columns))

    def test_output_length(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        assert len(feat) == len(nwp_df)

    def test_index_preserved(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        pd.testing.assert_index_equal(feat.index, nwp_df.index)


# ---------------------------------------------------------------------------
# build_features — correctness
# ---------------------------------------------------------------------------

class TestBuildFeaturesCorrectness:
    def test_cloud_cover_normalized(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        assert feat["cloud_cover"].between(0.0, 1.0).all()

    def test_cloud_cover_sq_equals_square(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        np.testing.assert_allclose(feat["cloud_cover_sq"], feat["cloud_cover"] ** 2)

    def test_ci_zero_at_night(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        night = clearsky_df["is_daytime"] == False  # noqa: E712
        assert night.any()
        assert (feat.loc[night, "ci"] == 0.0).all()

    def test_ci_within_clip_bounds(self, times, clearsky_df):
        n = len(times)
        # Use extreme radiation to push raw CI well above the clip ceiling
        nwp = pd.DataFrame(
            {
                "shortwave_radiation": np.full(n, 5000.0),
                "direct_radiation": np.full(n, 4000.0),
                "diffuse_radiation": np.full(n, 1000.0),
                "cloud_cover": np.zeros(n),
                "temperature_2m": np.full(n, 20.0),
            },
            index=times,
        )
        feat = build_features(nwp, clearsky_df, ci_clip=(0.0, 1.1))
        assert feat["ci"].max() <= 1.1 + 1e-9
        assert feat["ci"].min() >= 0.0 - 1e-9

    def test_poa_nwp_nonnegative(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        assert (feat["poa_nwp"] >= 0).all()

    def test_solar_elevation_clipped_at_zero(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        assert (feat["solar_elevation"] >= 0).all()

    def test_hour_sin_cos_unit_circle(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        r2 = feat["hour_sin"] ** 2 + feat["hour_cos"] ** 2
        np.testing.assert_allclose(r2, 1.0, atol=1e-12)

    def test_doy_sin_cos_unit_circle(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        r2 = feat["doy_sin"] ** 2 + feat["doy_cos"] ** 2
        np.testing.assert_allclose(r2, 1.0, atol=1e-12)

    def test_is_daytime_matches_clearsky(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        expected = clearsky_df["is_daytime"].astype(float)
        pd.testing.assert_series_equal(feat["is_daytime"], expected, check_names=False)


# ---------------------------------------------------------------------------
# build_features — lag correctness
# ---------------------------------------------------------------------------

class TestBuildFeaturesLags:
    def test_ci_lag1_is_shift1(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        pd.testing.assert_series_equal(
            feat["ci_lag1"].iloc[1:].reset_index(drop=True),
            feat["ci"].shift(1).iloc[1:].reset_index(drop=True),
            check_names=False,
        )

    def test_ci_lag2_is_shift2(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        pd.testing.assert_series_equal(
            feat["ci_lag2"].iloc[2:].reset_index(drop=True),
            feat["ci"].shift(2).iloc[2:].reset_index(drop=True),
            check_names=False,
        )

    def test_ci_rolling_mean_3h_is_lagged(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        expected = feat["ci"].rolling(3).mean().shift(1)
        pd.testing.assert_series_equal(
            feat["ci_rolling_mean_3h"], expected, check_names=False
        )

    def test_cloud_rolling_mean_3h_is_lagged(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        expected = feat["cloud_cover"].rolling(3).mean().shift(1)
        pd.testing.assert_series_equal(
            feat["cloud_rolling_mean_3h"], expected, check_names=False
        )

    def test_no_leakage_first_rows_nan(self, nwp_df, clearsky_df):
        feat = build_features(nwp_df, clearsky_df)
        assert pd.isna(feat["ci_lag1"].iloc[0])
        assert pd.isna(feat["ci_lag2"].iloc[0])
        assert pd.isna(feat["ci_lag2"].iloc[1])


