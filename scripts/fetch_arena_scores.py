#!/usr/bin/env python3
"""
Fetch and freeze official Energy Arena scores + forecast values for a challenge.

Pulls GET /api/v1/leaderboard/timeseries once for two participants (default:
spinner=19 vs. NaiveBaseline=1) over a window, then writes:

  1. a raw JSON snapshot (the paper's frozen "data of record")
  2. daily metrics per participant as CSV (CRPS, WIS, coverage, ... — whatever
     the platform exposes in available_metrics)
  3. a daily skill CSV: CRPS_me vs. CRPS_baseline and skill = 1 - me/baseline
  4. per-15-min ground truth + each participant's forecast values as parquet
     (for fan-chart / calibration plots)

The leaderboard endpoints are public; ARENA_API_KEY is sent if available.

Usage:
    python scripts/fetch_arena_scores.py
    python scripts/fetch_arena_scores.py --start 2026-06-01 --end 2026-07-04
    python scripts/fetch_arena_scores.py --me 19 --baseline 1 --challenge-id 16
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import config

_OUT_DIR = _REPO_ROOT / "data" / "eval_cache"
_DEFAULT_API_BASE = "https://api.energy-arena.org"
_TZ_LOCAL = "Europe/Berlin"  # DE_LU delivery zone


def _load_env(name: str) -> str:
    """Read a variable from the environment or the repo .env file."""
    import os

    val = os.environ.get(name, "").strip()
    if val:
        return val
    env_path = _REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _get(obj: dict, *keys: str, default: Any = None) -> Any:
    """Return the first present key from obj (tolerant to schema key naming)."""
    for k in keys:
        if k in obj:
            return obj[k]
    return default


def fetch_timeseries(
    api_base: str,
    challenge_id: str,
    area: str,
    participant_ids: str,
    window_start: str,
    window_end: str,
    api_key: str = "",
) -> dict:
    """Call GET /api/v1/leaderboard/timeseries and return the parsed JSON body."""
    url = f"{api_base.rstrip('/')}/api/v1/leaderboard/timeseries"
    params = {
        "challenge_id": challenge_id,
        "area": area,
        "participant_ids": participant_ids,
        "window_start": window_start,
        "window_end": window_end,
    }
    headers = {"X-API-Key": api_key} if api_key else None
    resp = requests.get(url, params=params, headers=headers, timeout=45)
    resp.raise_for_status()
    return resp.json()


def _metrics_frame(body: dict) -> pd.DataFrame:
    """participant_metrics[].points[] → tidy DataFrame with one metric column each."""
    rows = []
    for series in _get(body, "participant_metrics", default=[]) or []:
        pid = _get(series, "participant_id")
        pname = _get(series, "participant_name")
        for pt in _get(series, "points", default=[]) or []:
            metrics = _get(pt, "metrics", default={}) or {}
            rows.append(
                {
                    "target_start": _get(pt, "target_start", "ts"),
                    "participant_id": pid,
                    "participant_name": pname,
                    **{k: float(v) for k, v in metrics.items()},
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["target_start", "participant_id"]).reset_index(drop=True)
    return df


def _skill_frame(metrics: pd.DataFrame, me: int, baseline: int, metric: str = "CRPS") -> pd.DataFrame:
    """Per target_start: CRPS_me, CRPS_baseline, skill = 1 - me/baseline."""
    if metrics.empty or metric not in metrics.columns:
        return pd.DataFrame(columns=["target_start", f"{metric}_me", f"{metric}_baseline", "skill"])
    me_s = metrics[metrics.participant_id == me].set_index("target_start")[metric]
    base_s = metrics[metrics.participant_id == baseline].set_index("target_start")[metric]
    out = pd.DataFrame({f"{metric}_me": me_s, f"{metric}_baseline": base_s}).dropna()
    out["skill"] = 1.0 - out[f"{metric}_me"] / out[f"{metric}_baseline"]
    return out.reset_index()


def _actuals_series(body: dict) -> pd.Series:
    """ground_truth[] → tz-aware Series of actual MW at 15-min resolution."""
    gt = _get(body, "ground_truth", default=[]) or []
    if not gt:
        return pd.Series(dtype=float, name="actual")
    idx = pd.to_datetime([_get(p, "ts") for p in gt], utc=True).tz_convert(_TZ_LOCAL)
    vals = [float(_get(p, "value")) for p in gt]
    return pd.Series(vals, index=idx, name="actual").sort_index()


def _forecast_frame(body: dict, pid: int) -> pd.DataFrame:
    """participants[].points[] for one id → (ts × members) MW DataFrame."""
    for series in _get(body, "participants", "participant_series", default=[]) or []:
        if _get(series, "participant_id") != pid:
            continue
        pts = _get(series, "points", default=[]) or []
        if not pts:
            return pd.DataFrame()
        idx = pd.to_datetime([_get(p, "ts") for p in pts], utc=True).tz_convert(_TZ_LOCAL)
        rows = [[float(v) for v in _get(p, "values", default=[]) or []] for p in pts]
        n_members = max((len(r) for r in rows), default=0)
        cols = [f"member_{i:03d}" for i in range(n_members)]
        return pd.DataFrame(rows, index=idx, columns=cols).sort_index()
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Energy Arena scores + values for the paper.")
    parser.add_argument("--challenge-id", default="16", help="Challenge id (default: 16, solar ensemble)")
    parser.add_argument("--area", default="DE_LU", help="Bidding zone (default: DE_LU)")
    parser.add_argument("--me", type=int, default=19, help="Your participant id (default: 19 = spinner)")
    parser.add_argument("--baseline", type=int, default=1, help="Baseline participant id (default: 1)")
    parser.add_argument("--participants", default=None,
                        help="Comma-separated participant ids to fetch (default: me,baseline)")
    parser.add_argument("--start", default=None, help="Window start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Window end date YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--api-base", default=None, help="Override ARENA_API_BASE_URL")
    parser.add_argument("--out-dir", default=str(_OUT_DIR), help="Output directory")
    args = parser.parse_args()

    end = args.end or str(date.today() + timedelta(days=1))
    api_base = args.api_base or _load_env("ARENA_API_BASE_URL") or _DEFAULT_API_BASE
    api_key = _load_env("ARENA_API_KEY")
    if args.participants:
        ids = [int(x) for x in args.participants.split(",") if x.strip() != ""]
    else:
        ids = [args.me, args.baseline]
    participant_ids = ",".join(str(i) for i in ids)
    window_start = f"{args.start}T00:00:00Z"
    window_end = f"{end}T00:00:00Z"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"ch{args.challenge_id}_{'-'.join(str(i) for i in ids)}_{args.start}_{end}"

    print(f"Fetching timeseries: challenge {args.challenge_id} / {args.area} / ids {participant_ids}")
    print(f"  window {window_start} → {window_end}")
    body = fetch_timeseries(
        api_base, args.challenge_id, args.area, participant_ids,
        window_start, window_end, api_key=api_key,
    )

    # 1) raw snapshot (frozen data of record)
    raw_path = out_dir / f"arena_ts_{slug}.json"
    raw_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(f"  raw snapshot → {raw_path}")

    avail = _get(body, "available_metrics", default=[])
    print(f"  available_metrics: {avail}")

    # 2) daily metrics
    metrics = _metrics_frame(body)
    metrics_path = out_dir / f"arena_metrics_{slug}.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"  daily metrics ({len(metrics)} rows) → {metrics_path}")

    # 3) daily skill (CRPS me vs. baseline)
    skill = _skill_frame(metrics, args.me, args.baseline)
    skill_path = out_dir / f"arena_skill_{slug}.csv"
    skill.to_csv(skill_path, index=False)
    print(f"  daily skill ({len(skill)} days) → {skill_path}")

    # 4) per-15-min actuals + forecasts
    actuals = _actuals_series(body)
    if not actuals.empty:
        actuals.to_frame().to_parquet(out_dir / f"arena_actuals_{slug}.parquet")
        print(f"  actuals ({len(actuals)} slots) → arena_actuals_{slug}.parquet")
    labels = {args.me: "me", args.baseline: "baseline"}
    for pid in ids:
        label = labels.get(pid, f"p{pid}")
        fc = _forecast_frame(body, pid)
        if not fc.empty:
            fc.to_parquet(out_dir / f"arena_forecast_{label}_p{pid}_{slug}.parquet")
            print(f"  forecast {label} p{pid} ({fc.shape[0]}×{fc.shape[1]}) → arena_forecast_{label}_p{pid}_{slug}.parquet")

    # summary
    if not skill.empty:
        print("\n=== Window summary ===")
        print(f"  days scored:       {len(skill)}")
        print(f"  mean CRPS (me):    {skill['CRPS_me'].mean():.2f}")
        print(f"  mean CRPS (base):  {skill['CRPS_baseline'].mean():.2f}")
        print(f"  mean daily skill:  {skill['skill'].mean():+.3f}")
        print(f"  pooled skill:      {1.0 - skill['CRPS_me'].mean() / skill['CRPS_baseline'].mean():+.3f}")
    else:
        print("\nNo overlapping CRPS days for me vs. baseline — check window/ids "
              "(participant_metrics may be empty for very recent, not-yet-scored days).")


if __name__ == "__main__":
    main()
