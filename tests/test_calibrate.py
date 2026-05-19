"""Tests for calibrate.py: apply_spread_correction, apply_emos, fit, fit_emos, save/load_params."""

import json
import numpy as np
import pandas as pd
import pytest

from solar_forecast import calibrate

TZ = "UTC"
N_DAY = 200
N_MEMBERS = 50
RNG = np.random.default_rng(1)


def _make_index(n: int = N_DAY) -> pd.DatetimeIndex:
    return pd.date_range("2023-06-01 06:00", periods=n, freq="h", tz=TZ)


def _make_daytime_ensemble(n: int = N_DAY, centre: float = 500.0, spread: float = 80.0) -> tuple:
    """Returns (obs, forecasts) both daytime (all members > 0)."""
    obs_vals = RNG.uniform(centre * 0.8, centre * 1.2, n)
    noise = RNG.normal(0, spread, (n, N_MEMBERS))
    fct_vals = np.clip(obs_vals[:, None] + noise, 0, None)
    idx = _make_index(n)
    obs = pd.Series(obs_vals, index=idx)
    fct = pd.DataFrame(fct_vals, index=idx, columns=[f"m{i:02d}" for i in range(N_MEMBERS)])
    return obs, fct


def _add_night_rows(obs: pd.Series, fct: pd.DataFrame, n_night: int = 20) -> tuple:
    """Append n_night all-zero rows to simulate nighttime."""
    night_idx = pd.date_range("2023-06-01 00:00", periods=n_night, freq="h", tz=TZ)
    night_obs = pd.Series(np.zeros(n_night), index=night_idx)
    night_fct = pd.DataFrame(
        np.zeros((n_night, N_MEMBERS)),
        index=night_idx,
        columns=fct.columns,
    )
    return pd.concat([obs, night_obs]), pd.concat([fct, night_fct])


# ---------------------------------------------------------------------------

class TestApplySpreadCorrection:
    def test_preserves_index_and_columns(self):
        _, fct = _make_daytime_ensemble()
        params = {"bias_factor": 1.0, "spread_factor": 1.0}
        out = calibrate.apply_spread_correction(fct, params=params)
        assert list(out.index) == list(fct.index)
        assert list(out.columns) == list(fct.columns)

    def test_values_nonnegative(self):
        _, fct = _make_daytime_ensemble()
        params = {"bias_factor": 0.1, "spread_factor": 5.0}
        out = calibrate.apply_spread_correction(fct, params=params)
        assert out.values.min() >= 0.0

    def test_identity_when_both_factors_one(self):
        _, fct = _make_daytime_ensemble()
        params = {"bias_factor": 1.0, "spread_factor": 1.0}
        out = calibrate.apply_spread_correction(fct, params=params)
        np.testing.assert_allclose(out.values, fct.values, rtol=1e-6)

    def test_bias_factor_doubles_mean(self):
        _, fct = _make_daytime_ensemble()
        params = {"bias_factor": 2.0, "spread_factor": 1.0}
        out = calibrate.apply_spread_correction(fct, params=params)
        # ensemble mean should be roughly doubled
        in_mean = fct.mean(axis=1).mean()
        out_mean = out.mean(axis=1).mean()
        assert abs(out_mean / in_mean - 2.0) < 0.05

    def test_spread_factor_doubles_deviations(self):
        _, fct = _make_daytime_ensemble()
        params = {"bias_factor": 1.0, "spread_factor": 2.0}
        out = calibrate.apply_spread_correction(fct, params=params)
        # per-row std should approximately double (ignoring zero-clip effects)
        in_std = fct.std(axis=1).mean()
        out_std = out.std(axis=1).mean()
        assert abs(out_std / in_std - 2.0) < 0.2

    def test_zero_rows_stay_zero(self):
        _, fct = _make_daytime_ensemble()
        # insert an all-zero row (nighttime)
        fct.iloc[0] = 0.0
        params = {"bias_factor": 2.0, "spread_factor": 3.0}
        out = calibrate.apply_spread_correction(fct, params=params)
        # row 0 mean was 0 → corrected mean 0 → spread 0 → still 0
        assert out.iloc[0].sum() == 0.0


class TestApplyEmos:
    _PARAMS = {"emos": {"a": 0.0, "b": 1.0, "c": 10.0, "d": 0.01}}

    def test_output_shape(self):
        _, fct = _make_daytime_ensemble()
        out = calibrate.apply_emos(fct, params=self._PARAMS)
        assert out.shape == (N_DAY, 100)

    def test_column_names(self):
        _, fct = _make_daytime_ensemble()
        out = calibrate.apply_emos(fct, params=self._PARAMS)
        assert list(out.columns) == [f"member_{i:03d}" for i in range(100)]

    def test_nighttime_rows_are_zero(self):
        _, fct = _make_daytime_ensemble(n=50)
        night_idx = pd.date_range("2023-07-01 00:00", periods=10, freq="h", tz=TZ)
        night_fct = pd.DataFrame(
            np.zeros((10, N_MEMBERS)), index=night_idx, columns=fct.columns
        )
        full_fct = pd.concat([fct, night_fct])
        out = calibrate.apply_emos(full_fct, params=self._PARAMS)
        night_out = out.loc[night_idx]
        assert night_out.values.sum() == 0.0

    def test_values_nonnegative(self):
        _, fct = _make_daytime_ensemble()
        out = calibrate.apply_emos(fct, params=self._PARAMS)
        assert out.values.min() >= 0.0

    def test_falls_back_to_spread_correction_without_emos_key(self):
        _, fct = _make_daytime_ensemble()
        params = {"bias_factor": 1.0, "spread_factor": 1.0}
        out = calibrate.apply_emos(fct, params=params)
        # falls back → output shape should match input shape
        assert out.shape == fct.shape


class TestFit:
    def test_raises_on_all_nan_observations(self):
        _, fct = _make_daytime_ensemble()
        obs = pd.Series(np.full(N_DAY, np.nan), index=fct.index)
        with pytest.raises(ValueError, match="No valid"):
            calibrate.fit(obs, fct, save_path=None)

    def test_returns_bias_and_spread_keys(self, tmp_path):
        obs, fct = _make_daytime_ensemble()
        params = calibrate.fit(obs, fct, save_path=tmp_path / "cal.json")
        assert "bias_factor" in params
        assert "spread_factor" in params

    def test_bias_factor_near_one_when_unbiased(self, tmp_path):
        obs, fct = _make_daytime_ensemble()
        params = calibrate.fit(obs, fct, save_path=tmp_path / "cal.json")
        assert 0.5 <= params["bias_factor"] <= 2.0

    def test_spread_factor_in_valid_range(self, tmp_path):
        obs, fct = _make_daytime_ensemble()
        params = calibrate.fit(obs, fct, save_path=tmp_path / "cal.json")
        assert 0.2 <= params["spread_factor"] <= 3.0

    def test_saves_to_path(self, tmp_path):
        obs, fct = _make_daytime_ensemble()
        path = tmp_path / "cal.json"
        calibrate.fit(obs, fct, save_path=path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert "bias_factor" in loaded


class TestFitEmos:
    def test_returns_emos_key(self, tmp_path):
        obs, fct = _make_daytime_ensemble()
        params = calibrate.fit_emos(obs, fct, save_path=tmp_path / "emos.json")
        assert "emos" in params

    def test_emos_params_keys(self, tmp_path):
        obs, fct = _make_daytime_ensemble()
        params = calibrate.fit_emos(obs, fct, save_path=tmp_path / "emos.json")
        assert set(params["emos"].keys()) == {"a", "b", "c", "d"}

    def test_c_and_d_positive(self, tmp_path):
        obs, fct = _make_daytime_ensemble()
        params = calibrate.fit_emos(obs, fct, save_path=tmp_path / "emos.json")
        assert params["emos"]["c"] > 0
        assert params["emos"]["d"] > 0


class TestSaveLoadParams:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "params.json"
        original = {"bias_factor": 1.23, "spread_factor": 0.87}
        calibrate.save_params(original, path)
        loaded = calibrate.load_params(path)
        assert loaded == original

    def test_round_trip_with_emos(self, tmp_path):
        path = tmp_path / "params.json"
        original = {"emos": {"a": 0.1, "b": 0.95, "c": 5.0, "d": 0.02}}
        calibrate.save_params(original, path)
        loaded = calibrate.load_params(path)
        assert loaded == original

    def test_load_raises_when_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            calibrate.load_params(tmp_path / "nonexistent.json")
