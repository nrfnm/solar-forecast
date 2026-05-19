"""Tests for evaluate.py: crps_score, pit_values, evaluate."""

import numpy as np
import pandas as pd
import pytest

from solar_forecast.evaluate import crps_score, evaluate, pit_values

TZ = "UTC"
N = 200
N_MEMBERS = 20
RNG = np.random.default_rng(0)


def _make_index(n: int = N) -> pd.DatetimeIndex:
    return pd.date_range("2023-06-01", periods=n, freq="h", tz=TZ)


def _make_obs(n: int = N) -> pd.Series:
    return pd.Series(RNG.uniform(0, 1000, n), index=_make_index(n))


def _make_forecasts(obs: pd.Series, spread: float = 50.0) -> pd.DataFrame:
    arr = obs.values[:, None] + RNG.normal(0, spread, (len(obs), N_MEMBERS))
    return pd.DataFrame(
        np.clip(arr, 0, None),
        index=obs.index,
        columns=[f"m{i:02d}" for i in range(N_MEMBERS)],
    )


# ---------------------------------------------------------------------------

class TestCrpsScore:
    def test_returns_required_keys(self):
        obs = _make_obs()
        fct = _make_forecasts(obs)
        result = crps_score(obs, fct)
        assert set(result.keys()) == {"crps_mean", "crps_climatology", "crps_skill", "n_samples"}

    def test_n_samples_correct(self):
        obs = _make_obs()
        fct = _make_forecasts(obs)
        result = crps_score(obs, fct)
        assert result["n_samples"] == N

    def test_n_samples_excludes_nan_observations(self):
        obs = _make_obs()
        obs.iloc[0] = np.nan
        fct = _make_forecasts(obs)
        result = crps_score(obs, fct)
        assert result["n_samples"] == N - 1

    def test_skill_zero_when_ensemble_equals_climatology(self):
        obs = _make_obs()
        constant = obs.mean()
        fct = pd.DataFrame(
            np.full((N, N_MEMBERS), constant),
            index=obs.index,
            columns=[f"m{i}" for i in range(N_MEMBERS)],
        )
        result = crps_score(obs, fct)
        assert abs(result["crps_skill"]) < 0.05

    def test_skill_positive_when_ensemble_is_better(self):
        obs = _make_obs()
        fct = _make_forecasts(obs, spread=5.0)
        result = crps_score(obs, fct)
        assert result["crps_skill"] > 0

    def test_crps_mean_nonnegative(self):
        obs = _make_obs()
        fct = _make_forecasts(obs)
        result = crps_score(obs, fct)
        assert result["crps_mean"] >= 0


class TestPitValues:
    def test_returns_array_in_unit_interval(self):
        obs = _make_obs()
        fct = _make_forecasts(obs)
        pit = pit_values(obs, fct)
        assert pit.min() >= 0.0
        assert pit.max() <= 1.0

    def test_length_matches_valid_rows(self):
        obs = _make_obs()
        fct = _make_forecasts(obs)
        pit = pit_values(obs, fct)
        assert len(pit) == N

    def test_obs_below_all_members_gives_zero(self):
        idx = _make_index(10)
        obs = pd.Series(np.full(10, 0.0), index=idx)
        fct = pd.DataFrame(
            np.full((10, N_MEMBERS), 100.0),
            index=idx,
            columns=[f"m{i}" for i in range(N_MEMBERS)],
        )
        pit = pit_values(obs, fct)
        assert np.all(pit == 0.0)

    def test_obs_above_all_members_gives_one(self):
        idx = _make_index(10)
        obs = pd.Series(np.full(10, 1e6), index=idx)
        fct = pd.DataFrame(
            np.full((10, N_MEMBERS), 0.0),
            index=idx,
            columns=[f"m{i}" for i in range(N_MEMBERS)],
        )
        pit = pit_values(obs, fct)
        assert np.all(pit == 1.0)

    def test_uniform_for_ideal_ensemble(self):
        # If obs is drawn uniformly from the sorted ensemble, PIT should be near-uniform
        rng = np.random.default_rng(42)
        n = 500
        n_mem = 50
        idx = pd.date_range("2023-01-01", periods=n, freq="h", tz=TZ)
        raw = rng.uniform(0, 1000, (n, n_mem))
        # for each row draw obs as the median member
        rank = rng.integers(0, n_mem, n)
        sorted_raw = np.sort(raw, axis=1)
        obs_vals = sorted_raw[np.arange(n), rank]
        obs = pd.Series(obs_vals, index=idx)
        fct = pd.DataFrame(raw, index=idx, columns=[f"m{i}" for i in range(n_mem)])
        pit = pit_values(obs, fct)
        # uniform distribution has mean 0.5, std 1/sqrt(12)
        assert abs(pit.mean() - 0.5) < 0.1
        assert abs(pit.std() - 1 / np.sqrt(12)) < 0.1


class TestEvaluate:
    _EXPECTED_KEYS = {
        "crps_mean", "crps_climatology", "crps_skill",
        "n_samples", "pit_mean", "pit_std", "pit_bias", "dispersion_ratio",
    }

    def test_returns_all_keys(self):
        obs = _make_obs()
        fct = _make_forecasts(obs)
        result = evaluate(obs, fct, daytime_only=False)
        assert self._EXPECTED_KEYS == set(result.keys())

    def test_daytime_only_true_drops_zero_rows(self):
        obs = _make_obs()
        fct = _make_forecasts(obs)
        # zero out the last 20 rows
        fct.iloc[-20:] = 0.0
        result_day = evaluate(obs, fct, daytime_only=True)
        result_all = evaluate(obs, fct, daytime_only=False)
        assert result_day["n_samples"] < result_all["n_samples"]

    def test_daytime_only_false_includes_all_rows(self):
        obs = _make_obs()
        fct = _make_forecasts(obs)
        result = evaluate(obs, fct, daytime_only=False)
        assert result["n_samples"] == N

    def test_pit_bias_zero_for_unbiased_ensemble(self):
        rng = np.random.default_rng(7)
        idx = _make_index(500)
        obs_vals = rng.uniform(100, 800, 500)
        spread = rng.normal(0, 30, (500, N_MEMBERS))
        fct_vals = np.clip(obs_vals[:, None] + spread, 0, None)
        obs = pd.Series(obs_vals, index=idx)
        fct = pd.DataFrame(fct_vals, index=idx, columns=[f"m{i}" for i in range(N_MEMBERS)])
        result = evaluate(obs, fct, daytime_only=False)
        assert abs(result["pit_bias"]) < 0.15
