"""
Solar-forecast integration for energy-arena-participate-working.

Prereqs:
    pip install -e ../solar-forecast          # local (editable)
    pip install git+https://github.com/you/solar-forecast.git  # VM / CI

solar-forecast/pipeline.py is not inside the installable package, so we add
the repo root to sys.path as a fallback. This works whether both repos are
cloned side-by-side (the normal layout) or solar_forecast is pip-installed.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --- make solar-forecast importable ----------------------------------------
_SF_ROOT = Path(__file__).resolve().parent.parent / "solar-forecast"
if _SF_ROOT.exists() and str(_SF_ROOT) not in sys.path:
    sys.path.insert(0, str(_SF_ROOT))

import pipeline as sf_pipeline  # solar-forecast/pipeline.py
# ---------------------------------------------------------------------------


_SOLAR_DE_LU_CHALLENGE_IDS = {"4", "10", "16"}  # Solar Generation DE_LU: point, quantile, ensemble
_GERMANY_AREA_PREFIX = "DE"  # area codes: "DE_LU", "DE", etc.


def transform_payload(
    payload: dict,
    *,
    target_date: date,
    target_start: datetime,
    challenge_id: str,
    area: str,
    entsoe_api_key: str,
    api_base: str,
    data_source: str,
    challenge_context,
    challenge_detail: dict,
    forecast_objective: str,
    tz_name: str,
) -> dict:
    if str(challenge_id) not in _SOLAR_DE_LU_CHALLENGE_IDS:
        return payload  # not a solar DE_LU challenge — use baseline as-is
    if not area.startswith(_GERMANY_AREA_PREFIX):
        return payload  # model configured for Germany only; other regions use baseline

    # Run the full solar-forecast pipeline for the target day.
    # Returns pd.DataFrame (timesteps × 50 members), hourly UTC index, MW.
    trajectories = sf_pipeline.run(run_date=target_date.isoformat(), save=False)

    traj_utc = trajectories.copy()
    traj_utc.index = traj_utc.index.tz_convert("UTC")

    n_slots = len(payload["values"])
    # For a 24-hour calendar day: hourly → step=1h, quarter-hourly → step=15min.
    # Solar-forecast is hourly; if the challenge is finer-grained we upsample.
    step = timedelta(hours=24) / n_slots
    start_utc = target_start.astimezone(timezone.utc)

    traj_step = traj_utc.index[1] - traj_utc.index[0] if len(traj_utc) > 1 else timedelta(hours=1)
    if step < traj_step:
        # Upsample trajectories to the challenge resolution via forward-fill.
        full_range = pd.date_range(
            start=traj_utc.index[0],
            periods=len(traj_utc) * int(traj_step / step),
            freq=step,
            tz="UTC",
        )
        traj_utc = traj_utc.reindex(full_range).ffill()

    for i in range(n_slots):
        ts = pd.Timestamp(start_utc + i * step)
        if ts not in traj_utc.index:
            # Slot falls outside the forecast window — keep baseline value.
            continue

        member_values = np.maximum(0.0, traj_utc.loc[ts].values.astype(float))

        if forecast_objective == "ensemble":
            n = challenge_context.max_ensemble_size
            if len(member_values) < n:
                member_values = np.quantile(member_values, np.linspace(0, 1, n))
            payload["values"][i] = [float(v) for v in member_values[:n]]

        elif forecast_objective == "quantile":
            qs = challenge_context.probabilistic_quantiles
            payload["values"][i] = [float(v) for v in np.quantile(member_values, qs)]

        else:  # point
            payload["values"][i] = float(np.median(member_values))

    return payload
