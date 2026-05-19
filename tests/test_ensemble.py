"""Tests for ensemble.py: _pivot_nwp, run_ensemble, run_quantile_ensemble."""

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

from solar_forecast.clearsky import get_clearsky
from solar_forecast.ensemble import _pivot_nwp, run_ensemble, run_quantile_ensemble
from solar_forecast.train import FEATURE_COLS

LAT, LON = 48.2, 16.3   # Vienna
TZ = "Europe/Vienna"
N_MEMBERS = 3
NWP_VARS = [
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "cloud_cover", "temperature_2m",
]
CAPACITY_MW = 1000.0


# ---------------------------------------------------------------------------
# Fixtures

@pytest.fixture(scope="session")
def tiny_model() -> lgb.LGBMRegressor:
    """Tiny LGBMRegressor trained on synthetic data — fast (<0.1 s)."""
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.uniform(0, 1, (n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = pd.Series(rng.uniform(0, 1, n))
    model = lgb.LGBMRegressor(n_estimators=10, num_leaves=4, verbose=-1, n_jobs=1)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def hourly_times():
    return pd.date_range("2024-06-21", periods=48, freq="h", tz=TZ)


@pytest.fixture(scope="module")
def clearsky_hourly(hourly_times):
    return get_clearsky(LAT, LON, hourly_times)


@pytest.fixture(scope="module")
def clearsky_15min(hourly_times):
    idx = pd.date_range(
        start=hourly_times[0],
        end=hourly_times[-1] + pd.Timedelta(minutes=45),
        freq="15min",
        tz=TZ,
    )
    return get_clearsky(LAT, LON, idx)


@pytest.fixture(scope="module")
def nwp_raw(hourly_times) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = len(hourly_times)
    members = [f"member_{i:02d}" for i in range(N_MEMBERS)]
    return {
        var: pd.DataFrame(
            {m: rng.uniform(0, 800 if "radiation" in var else 100, n) for m in members},
            index=hourly_times,
        )
        for var in NWP_VARS
    }


# ---------------------------------------------------------------------------

class TestPivotNwp:
    def test_returns_dict_keyed_by_members(self, nwp_raw):
        result = _pivot_nwp(nwp_raw)
        members = list(nwp_raw[NWP_VARS[0]].columns)
        assert set(result.keys()) == set(members)

    def test_each_value_has_variable_columns(self, nwp_raw):
        result = _pivot_nwp(nwp_raw)
        first_member = next(iter(result.values()))
        assert set(NWP_VARS).issubset(first_member.columns)

    def test_index_preserved(self, nwp_raw, hourly_times):
        result = _pivot_nwp(nwp_raw)
        first_member = next(iter(result.values()))
        assert list(first_member.index) == list(hourly_times)

    def test_member_count_matches_input(self, nwp_raw):
        result = _pivot_nwp(nwp_raw)
        n_members_in = len(nwp_raw[NWP_VARS[0]].columns)
        assert len(result) == n_members_in


class TestRunEnsemble:
    def test_output_shape(self, tiny_model, nwp_raw, clearsky_hourly, clearsky_15min):
        out = run_ensemble(tiny_model, nwp_raw, clearsky_hourly, clearsky_15min, CAPACITY_MW)
        n_15min = len(clearsky_15min)
        assert out.shape == (n_15min, N_MEMBERS)

    def test_columns_are_member_names(self, tiny_model, nwp_raw, clearsky_hourly, clearsky_15min):
        out = run_ensemble(tiny_model, nwp_raw, clearsky_hourly, clearsky_15min, CAPACITY_MW)
        expected = [f"member_{i:02d}" for i in range(N_MEMBERS)]
        assert list(out.columns) == expected

    def test_values_nonnegative(self, tiny_model, nwp_raw, clearsky_hourly, clearsky_15min):
        out = run_ensemble(tiny_model, nwp_raw, clearsky_hourly, clearsky_15min, CAPACITY_MW)
        assert out.values.min() >= 0.0

    def test_nighttime_rows_are_zero(self, tiny_model, nwp_raw, clearsky_hourly, clearsky_15min):
        out = run_ensemble(tiny_model, nwp_raw, clearsky_hourly, clearsky_15min, CAPACITY_MW)
        night_mask = ~clearsky_15min["is_daytime"].values
        if night_mask.any():
            night_values = out.values[night_mask]
            assert night_values.sum() == 0.0

    def test_output_bounded_by_capacity(self, tiny_model, nwp_raw, clearsky_hourly, clearsky_15min):
        out = run_ensemble(tiny_model, nwp_raw, clearsky_hourly, clearsky_15min, CAPACITY_MW)
        # CI is clipped to 1.1 → max MW ≤ (poa/STC) × capacity × 1.1
        # at STC poa=1000 → max possible ≈ CAPACITY_MW × 1.1
        assert out.values.max() <= CAPACITY_MW * 1.2


class TestRunQuantileEnsemble:
    def test_output_shape_with_more_models_than_members(
        self, tiny_model, nwp_raw, clearsky_hourly, clearsky_15min
    ):
        n_models = N_MEMBERS * 2  # more models than NWP members
        models = [tiny_model] * n_models
        out = run_quantile_ensemble(models, nwp_raw, clearsky_hourly, clearsky_15min, CAPACITY_MW)
        assert out.shape == (len(clearsky_15min), n_models)

    def test_column_names_are_member_000_format(
        self, tiny_model, nwp_raw, clearsky_hourly, clearsky_15min
    ):
        n_models = 4
        models = [tiny_model] * n_models
        out = run_quantile_ensemble(models, nwp_raw, clearsky_hourly, clearsky_15min, CAPACITY_MW)
        expected = [f"member_{i:03d}" for i in range(n_models)]
        assert list(out.columns) == expected

    def test_cycles_nwp_members(self, tiny_model, nwp_raw, clearsky_hourly, clearsky_15min):
        # 6 models, 3 NWP members → cycling: member 3 reuses NWP of member 0
        # member_id feature differs (0 vs 3), so outputs are NOT identical,
        # but both must still respect the nighttime-zero constraint.
        n_models = 6
        models = [tiny_model] * n_models
        out = run_quantile_ensemble(models, nwp_raw, clearsky_hourly, clearsky_15min, CAPACITY_MW)
        assert out.shape[1] == n_models
        night_mask = ~clearsky_15min["is_daytime"].values
        if night_mask.any():
            assert out["member_000"].values[night_mask].sum() == 0.0
            assert out["member_003"].values[night_mask].sum() == 0.0

    def test_values_nonnegative(self, tiny_model, nwp_raw, clearsky_hourly, clearsky_15min):
        models = [tiny_model] * N_MEMBERS
        out = run_quantile_ensemble(models, nwp_raw, clearsky_hourly, clearsky_15min, CAPACITY_MW)
        assert out.values.min() >= 0.0
