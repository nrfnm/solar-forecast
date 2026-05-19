"""Tests for train.py: _ar1_noise, _inject_nwp_noise, _target_ci, walk_forward_cv, FEATURE_COLS."""

import numpy as np
import pandas as pd
import pytest

from solar_forecast.train import (
    FEATURE_COLS,
    _ar1_noise,
    _inject_nwp_noise,
    _target_ci,
    walk_forward_cv,
)
from solar_forecast.fetch_actuals import ERA5_VARIABLES

TZ = "Europe/Vienna"


# ---------------------------------------------------------------------------
# Helpers

def _make_era5(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(10)
    idx = pd.date_range("2023-06-01 06:00", periods=n, freq="h", tz=TZ)
    data = {
        "shortwave_radiation": rng.uniform(0, 800, n),
        "direct_radiation": rng.uniform(0, 600, n),
        "diffuse_radiation": rng.uniform(0, 300, n),
        "cloud_cover": rng.uniform(0, 100, n),
        "temperature_2m": rng.uniform(5, 35, n),
    }
    return pd.DataFrame(data, index=idx)


def _make_clearsky(idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(idx)
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "poa_clearsky": rng.uniform(0, 900, n),
            "apparent_elevation": rng.uniform(0, 70, n),
            "azimuth": rng.uniform(90, 270, n),
            "is_daytime": True,
        },
        index=idx,
    )
    df.attrs["surface_tilt"] = 30
    df.attrs["surface_azimuth"] = 180
    return df


# ---------------------------------------------------------------------------

class TestAr1Noise:
    def test_output_length(self):
        rng = np.random.default_rng(0)
        noise = _ar1_noise(100, rho=0.7, std=1.0, rng=rng)
        assert len(noise) == 100

    def test_mean_near_zero(self):
        rng = np.random.default_rng(0)
        noise = _ar1_noise(10_000, rho=0.7, std=1.0, rng=rng)
        assert abs(noise.mean()) < 0.1

    def test_std_within_tolerance(self):
        rng = np.random.default_rng(0)
        std = 2.5
        noise = _ar1_noise(10_000, rho=0.7, std=std, rng=rng)
        assert abs(noise.std() / std - 1.0) < 0.3

    def test_rho_zero_is_roughly_uncorrelated(self):
        rng = np.random.default_rng(0)
        n = 5000
        noise = _ar1_noise(n, rho=0.0, std=1.0, rng=rng)
        autocorr = np.corrcoef(noise[:-1], noise[1:])[0, 1]
        assert abs(autocorr) < 0.1

    def test_deterministic_with_same_seed(self):
        n = 50
        noise1 = _ar1_noise(n, rho=0.5, std=1.0, rng=np.random.default_rng(99))
        noise2 = _ar1_noise(n, rho=0.5, std=1.0, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(noise1, noise2)


class TestInjectNwpNoise:
    def test_returns_same_shape_and_columns(self):
        era5 = _make_era5()
        rng = np.random.default_rng(0)
        noisy = _inject_nwp_noise(era5, rng)
        assert noisy.shape == era5.shape
        assert list(noisy.columns) == list(era5.columns)

    def test_radiation_nonnegative(self):
        era5 = _make_era5()
        rng = np.random.default_rng(0)
        noisy = _inject_nwp_noise(era5, rng)
        for col in ("shortwave_radiation", "direct_radiation", "diffuse_radiation"):
            assert noisy[col].min() >= 0.0

    def test_cloud_cover_clipped_to_100(self):
        era5 = _make_era5()
        rng = np.random.default_rng(0)
        noisy = _inject_nwp_noise(era5, rng)
        assert noisy["cloud_cover"].max() <= 100.0
        assert noisy["cloud_cover"].min() >= 0.0

    def test_output_differs_from_input(self):
        era5 = _make_era5()
        rng = np.random.default_rng(0)
        noisy = _inject_nwp_noise(era5, rng)
        assert not np.allclose(noisy.values, era5.values)

    def test_temperature_column_preserved(self):
        era5 = _make_era5()
        rng = np.random.default_rng(0)
        noisy = _inject_nwp_noise(era5, rng)
        assert "temperature_2m" in noisy.columns


class TestTargetCi:
    def test_returns_series_named_ci_actual(self):
        era5 = _make_era5(48)
        cs = _make_clearsky(era5.index)
        actuals = pd.Series(np.full(48, 500.0), index=era5.index)
        ci = _target_ci(actuals, cs, installed_capacity_mw=1000.0)
        assert isinstance(ci, pd.Series)
        assert ci.name == "ci_actual"

    def test_nighttime_rows_are_zero(self):
        era5 = _make_era5(24)
        cs = _make_clearsky(era5.index)
        cs["is_daytime"] = False
        actuals = pd.Series(np.full(24, 200.0), index=era5.index)
        ci = _target_ci(actuals, cs, installed_capacity_mw=1000.0)
        assert (ci == 0.0).all()

    def test_daytime_values_in_clip_range(self):
        era5 = _make_era5(48)
        cs = _make_clearsky(era5.index)
        cs["poa_clearsky"] = 500.0
        cs["is_daytime"] = True
        actuals = pd.Series(np.full(48, 400.0), index=era5.index)
        ci = _target_ci(actuals, cs, installed_capacity_mw=1000.0)
        assert ci.min() >= 0.0
        assert ci.max() <= 1.1

    def test_scalar_capacity_works(self):
        era5 = _make_era5(10)
        cs = _make_clearsky(era5.index)
        actuals = pd.Series(np.full(10, 100.0), index=era5.index)
        ci = _target_ci(actuals, cs, installed_capacity_mw=500.0)
        assert len(ci) == 10

    def test_series_capacity_works(self):
        era5 = _make_era5(24)
        cs = _make_clearsky(era5.index)
        actuals = pd.Series(np.full(24, 100.0), index=era5.index)
        # monthly capacity series covering the test period
        cap = pd.Series(
            [1000.0, 1010.0],
            index=pd.date_range("2023-06-01", periods=2, freq="MS", tz="UTC"),
            name="capacity_mw",
        )
        ci = _target_ci(actuals, cs, installed_capacity_mw=cap)
        assert len(ci) == 24
        assert ci.isna().sum() == 0


class TestWalkForwardCv:
    def _make_xy(self, n_months: int = 20) -> tuple:
        """Synthetic X (FEATURE_COLS) and y (CI) over n_months, 100 rows/month."""
        rng = np.random.default_rng(5)
        n = n_months * 100
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz=TZ)
        X = pd.DataFrame(
            rng.uniform(0, 1, (n, len(FEATURE_COLS))),
            columns=FEATURE_COLS,
            index=idx,
        )
        y = pd.Series(rng.uniform(0, 1, n), index=idx, name="ci_actual")
        return X, y

    def test_returns_list_of_dicts(self):
        X, y = self._make_xy(20)
        results = walk_forward_cv(X, y)
        assert isinstance(results, list)

    def test_each_dict_has_required_keys(self):
        X, y = self._make_xy(20)
        results = walk_forward_cv(X, y)
        for r in results:
            assert {"val_month", "rmse", "n_val", "best_iteration"} == set(r.keys())

    def test_returns_empty_for_insufficient_data(self):
        X, y = self._make_xy(n_months=5)  # fewer than gap(6)+min_train(12)=18 months
        results = walk_forward_cv(X, y, gap_months=6, min_train_months=12)
        assert results == []

    def test_rmse_is_positive(self):
        X, y = self._make_xy(20)
        results = walk_forward_cv(X, y)
        for r in results:
            assert r["rmse"] > 0


class TestFeatureCols:
    def test_is_list_of_strings(self):
        assert isinstance(FEATURE_COLS, list)
        assert all(isinstance(c, str) for c in FEATURE_COLS)

    def test_no_duplicates(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_does_not_contain_ci_leakage(self):
        # 'ci' itself would be leakage; lag/rolling variants are allowed
        assert "ci" not in FEATURE_COLS

    def test_contains_member_id(self):
        assert "member_id" in FEATURE_COLS

    def test_contains_lag_and_rolling(self):
        assert "ci_lag1" in FEATURE_COLS
        assert "ci_rolling_mean_3h" in FEATURE_COLS
