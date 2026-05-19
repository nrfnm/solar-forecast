"""Tests for capacity.py: get_k_centroids, capacity_timeseries (CSV fallback)."""

import numpy as np
import pandas as pd
import pytest

from solar_forecast import capacity as cap_module
from solar_forecast.capacity import get_k_centroids, capacity_timeseries


# ---------------------------------------------------------------------------
# Helpers

def _make_units(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Synthetic solar unit DataFrame matching MaStR column names."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Breitengrad": rng.uniform(47.0, 55.0, n),   # Germany lat range
            "Laengengrad": rng.uniform(6.0, 15.0, n),    # Germany lon range
            "Nettonennleistung": rng.uniform(5.0, 500.0, n),  # kW
        }
    )


# ---------------------------------------------------------------------------

class TestGetKCentroids:
    def test_returns_dataframe_with_required_columns(self):
        df = _make_units()
        result = get_k_centroids(df, k=3)
        assert set(result.columns) == {"lat", "lon", "weight"}

    def test_n_rows_equals_k(self):
        df = _make_units()
        for k in (1, 3, 5):
            result = get_k_centroids(df, k=k)
            assert len(result) == k

    def test_weights_sum_to_one(self):
        df = _make_units()
        result = get_k_centroids(df, k=5)
        assert abs(result["weight"].sum() - 1.0) < 1e-6

    def test_single_cluster_weight_is_one(self):
        df = _make_units()
        result = get_k_centroids(df, k=1)
        assert abs(result["weight"].iloc[0] - 1.0) < 1e-6

    def test_centroid_lat_lon_in_data_range(self):
        df = _make_units()
        result = get_k_centroids(df, k=4)
        assert result["lat"].between(47.0, 55.0).all()
        assert result["lon"].between(6.0, 15.0).all()

    def test_deterministic_with_same_random_state(self):
        df = _make_units()
        r1 = get_k_centroids(df, k=3, random_state=42)
        r2 = get_k_centroids(df, k=3, random_state=42)
        pd.testing.assert_frame_equal(r1, r2)


class TestCapacityTimeseriesCsvFallback:
    def _write_csv(self, path, tz="UTC"):
        idx = pd.date_range("2020-01-01", periods=12, freq="MS", tz=tz)
        s = pd.Series(np.cumsum(np.random.default_rng(0).uniform(100, 500, 12)),
                      index=idx, name="capacity_mw")
        s.to_csv(path)
        return s

    def test_returns_series_named_capacity_mw(self, tmp_path, monkeypatch):
        csv = tmp_path / "capacity_timeseries.csv"
        self._write_csv(csv)
        monkeypatch.setattr(cap_module, "_CSV_PATH", csv)
        monkeypatch.setattr(cap_module, "_DB_PATH", tmp_path / "nonexistent.sqlite")
        result = capacity_timeseries(tz="UTC")
        assert result.name == "capacity_mw"

    def test_index_is_timezone_aware(self, tmp_path, monkeypatch):
        csv = tmp_path / "capacity_timeseries.csv"
        self._write_csv(csv, tz="UTC")
        monkeypatch.setattr(cap_module, "_CSV_PATH", csv)
        monkeypatch.setattr(cap_module, "_DB_PATH", tmp_path / "nonexistent.sqlite")
        result = capacity_timeseries(tz="UTC")
        assert result.index.tz is not None

    def test_raises_when_neither_db_nor_csv_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cap_module, "_CSV_PATH", tmp_path / "missing.csv")
        monkeypatch.setattr(cap_module, "_DB_PATH", tmp_path / "missing.sqlite")
        with pytest.raises(FileNotFoundError):
            capacity_timeseries()
