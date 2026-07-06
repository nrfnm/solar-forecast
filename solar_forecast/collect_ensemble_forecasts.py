#!/usr/bin/env python3
"""
Archive the live 50-member NWP ensemble forecast for later EMOS training.

Open-Meteo's ensemble API only serves the *current* forecast run — past
50-member runs cannot be retrieved. The historical-forecast API returns only
the deterministic IFS member. To calibrate EMOS on *authentic* ensemble
spread (instead of the synthetic AR(1) noise used offline), we therefore have
to snapshot the live ensemble every day and build the archive going forward.

Each run fetches the raw NWP ensemble (all members, all variables) at every
configured centroid and writes one Parquet file per collection day:

    data/nwp_archive/<YYYY-MM-DD>.parquet

Raw NWP is stored (rather than the derived power trajectories) because it is
the genuinely irrecoverable input and is model-agnostic: EMOS — and any
future LightGBM model — can be re-fit from it without re-archiving.

Schedule this at the *same* time as the live day-ahead submission so the
archived lead times match the ones seen in production.

Usage:
    python -m solar_forecast.collect_ensemble_forecasts
    python -m solar_forecast.collect_ensemble_forecasts --force   # overwrite today
    python -m solar_forecast.collect_ensemble_forecasts --forecast-days 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import config
from solar_forecast.fetch_nwp import fetch_nwp


def _centroids() -> list[dict]:
    """Configured centroids, or the single fallback point from config."""
    if config.CENTROIDS:
        return config.CENTROIDS
    return [{"lat": config.LAT, "lon": config.LON, "weight": 1.0}]


def collect(forecast_days: int) -> pd.DataFrame:
    """
    Fetch the current ensemble run at every centroid and return a tidy frame.

    Columns: collected_at, valid_time, lead_hours, centroid, lat, lon, weight,
             variable, member, value (float32).
    """
    collected_at = pd.Timestamp.now(tz="UTC")
    frames: list[pd.DataFrame] = []

    for cid, c in enumerate(_centroids()):
        var_frames = fetch_nwp(c["lat"], c["lon"], forecast_days=forecast_days)
        for variable, df in var_frames.items():
            # df: index = valid_time (tz-aware), columns = member_00 .. member_49
            long = df.reset_index(names="valid_time").melt(
                id_vars="valid_time", var_name="member", value_name="value"
            )
            long["member"] = long["member"].str.removeprefix("member_").astype("int16")
            long["centroid"] = cid
            long["lat"] = c["lat"]
            long["lon"] = c["lon"]
            long["weight"] = c.get("weight", 1.0)
            long["variable"] = variable
            frames.append(long)

    out = pd.concat(frames, ignore_index=True)
    out["collected_at"] = collected_at
    valid_utc = out["valid_time"].dt.tz_convert("UTC")
    out["lead_hours"] = (
        (valid_utc - collected_at.floor("h")) // pd.Timedelta(hours=1)
    ).astype("int32")
    out["value"] = out["value"].astype("float32")

    return out[[
        "collected_at", "valid_time", "lead_hours", "centroid",
        "lat", "lon", "weight", "variable", "member", "value",
    ]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive the live NWP ensemble forecast for EMOS training."
    )
    parser.add_argument(
        "--forecast-days", type=int, default=config.FORECAST_DAYS,
        help=f"Horizon in days (default: {config.FORECAST_DAYS}).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite today's archive file if it already exists.",
    )
    args = parser.parse_args()

    archive_dir: Path = config.FORECAST_ARCHIVE_PATH
    archive_dir.mkdir(parents=True, exist_ok=True)

    run_date = pd.Timestamp.now(tz="UTC").date()
    out_path = archive_dir / f"{run_date}.parquet"
    if out_path.exists() and not args.force:
        print(f"Already archived: {out_path} (use --force to overwrite)")
        return

    df = collect(args.forecast_days)
    df.to_parquet(out_path, compression="zstd", index=False)

    size_kb = out_path.stat().st_size / 1024
    print(
        f"Archived {len(df):,} rows "
        f"({df['centroid'].nunique()} centroids × "
        f"{df['variable'].nunique()} vars × "
        f"{df['member'].nunique()} members × "
        f"{df['valid_time'].nunique()} hours) "
        f"→ {out_path} ({size_kb:.0f} KB)"
    )


if __name__ == "__main__":
    main()
