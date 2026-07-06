"""Evaluate probabilistic solar forecasts: CRPS, skill score, PIT."""

import glob
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import properscoring as ps

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from solar_forecast.benchmark import naive_benchmark
from solar_forecast.fetch_actuals import fetch_entsoe


def crps_score(
    observations: pd.Series,
    forecasts: pd.DataFrame,
    baseline: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute CRPS and a skill score relative to a reference forecast.

    By default the reference is the naive historical-ensemble benchmark, built from
    ENTSOE for the forecast's target day. If ``baseline`` is given, the skill score is
    computed against that ensemble instead — matching the competition's CRPS skill
    definition.

    Parameters
    ----------
    observations : pd.Series
        Actual solar generation in MW, aligned to forecasts index.
    forecasts : pd.DataFrame
        Ensemble trajectories, shape (timesteps, n_members), values in MW.
    baseline : pd.DataFrame, optional
        Reference ensemble (timesteps, n_members) in MW. When None, the naive
        benchmark is built from the forecast index.

    Returns
    -------
    dict with keys: crps_mean, crps_benchmark, crps_skill, n_samples.
    """
    if baseline is None:
        baseline = naive_benchmark(forecasts.index, str(forecasts.index.tz))

    obs, fct = observations.align(forecasts, join="inner", axis=0)
    obs_vals = obs.values
    fct_vals = fct.values

    base_vals = baseline.reindex(obs.index).values

    valid = (
        ~np.isnan(obs_vals)
        & ~np.isnan(fct_vals).any(axis=1)
        & ~np.isnan(base_vals).any(axis=1)
    )
    obs_vals = obs_vals[valid]
    fct_vals = fct_vals[valid]
    base_vals = base_vals[valid]

    crps_mean = float(ps.crps_ensemble(obs_vals, fct_vals).mean())
    crps_ref = float(ps.crps_ensemble(obs_vals, base_vals).mean())
    skill = float(1.0 - crps_mean / crps_ref) if crps_ref > 0 else 0.0

    return {
        "crps_mean": crps_mean,
        "crps_benchmark": crps_ref,
        "crps_skill": skill,
        "n_samples": len(obs_vals),
    }


def pit_values(
    observations: pd.Series,
    forecasts: pd.DataFrame,
) -> np.ndarray:
    """
    Compute PIT (Probability Integral Transform) values.

    For each timestep, compute the fraction of ensemble members that
    fall below the observation. A well-calibrated ensemble produces
    uniformly distributed PIT values over [0, 1].

    Returns
    -------
    np.ndarray of floats in [0, 1].
    """
    obs, fct = observations.align(forecasts, join="inner", axis=0)
    obs_vals = obs.values
    fct_vals = fct.values

    valid = ~np.isnan(obs_vals) & ~np.isnan(fct_vals).any(axis=1)
    obs_vals = obs_vals[valid]
    fct_vals = fct_vals[valid]

    return np.array([np.mean(fct_vals[i] < obs_vals[i]) for i in range(len(obs_vals))])


def evaluate(
    observations: pd.Series,
    forecasts: pd.DataFrame,
    daytime_only: bool = True,
    baseline: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Full evaluation: CRPS, skill score, and PIT-based calibration diagnostics.

    Parameters
    ----------
    observations : pd.Series
        Actual solar generation in MW.
    forecasts : pd.DataFrame
        Ensemble trajectories, shape (timesteps, n_members), values in MW.
    daytime_only : bool
        Exclude rows where all ensemble members are zero (nighttime).
    baseline : pd.DataFrame, optional
        Reference ensemble for the skill score (e.g. the naive benchmark). When
        None, the naive benchmark is built from the forecast index.

    Returns
    -------
    dict with all metrics. Key fields:
        crps_mean        — lower is better
        crps_skill       — positive = better than the benchmark
        pit_bias         — positive = model underforecasts (obs in upper tail)
        dispersion_ratio — >1 underdispersed (too narrow), <1 overdispersed (too wide), 1.0 = perfect
    """
    if daytime_only:
        is_day = forecasts.max(axis=1) > 0
        observations = observations[is_day]
        forecasts = forecasts[is_day]
        if baseline is not None:
            baseline = baseline.reindex(forecasts.index)

    scores = crps_score(observations, forecasts, baseline=baseline)
    pit = pit_values(observations, forecasts)

    pit_std = float(np.std(pit))
    pit_mean = float(np.mean(pit))
    uniform_std = 1.0 / np.sqrt(12)

    return {
        **scores,
        "pit_mean": pit_mean,
        "pit_std": pit_std,
        "pit_bias": pit_mean - 0.5,
        "dispersion_ratio": pit_std / uniform_std,
    }


def _target_date(path: str):
    """Target calendar date encoded in the payload filename (after ``__``)."""
    stem = Path(path).name.split("__")[1][13:]  # 2026-04-30T00_00_00_02_00
    return datetime.strptime(stem[:10], "%Y-%m-%d").date()


def _submission_dt(path: str) -> datetime:
    """Leading submission timestamp from the filename (YYYYMMDD_HHMMSS_micro)."""
    return datetime.strptime(Path(path).name.split("__")[0][:22], "%Y%m%d_%H%M%S_%f")


def load_submission(path: str, tz: str = config.TZ) -> pd.DataFrame:
    """Load an ensemble payload JSON as a ``(96, n_members)`` MW DataFrame in ``tz``."""
    with open(path) as fh:
        payload = json.load(fh)
    vals = np.asarray(payload["values"], dtype=float)
    idx = pd.date_range(payload["target_start"], periods=len(vals), freq="15min").tz_convert(tz)
    return pd.DataFrame(vals, index=idx)


def evaluate_submissions(
    folder: str,
    start: str,
    end: str,
    dry_run: bool = False,
    tz: str = config.TZ,
    max_missing: int = 8,
) -> pd.DataFrame:
    """
    Score every submission in ``folder`` against the naive benchmark.

    Keeps the latest submission per target day whose target date falls in
    ``[start, end]`` and whose dry-run flag matches ``dry_run``, then computes CRPS
    and the skill score against the naive historical-ensemble benchmark (built
    automatically inside :func:`evaluate`).

    ENTSO-E actuals are frequently missing a few quarter-hours. Up to ``max_missing``
    gaps per day are interpolated; days with more missing slots (or whose benchmark
    can't be built) are skipped with a warning.

    Returns
    -------
    pd.DataFrame
        One row per scored target day: target_date, crps_mean, crps_benchmark,
        crps_skill, n_samples.
    """
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()

    files = [
        f
        for f in glob.glob(str(Path(folder) / "*.json"))
        if ("dry_run" in f) == dry_run and start_date <= _target_date(f) <= end_date
    ]

    # Keep only the latest submission per target day.
    latest: dict = {}
    for f in files:
        d = _target_date(f)
        if d not in latest or _submission_dt(f) > _submission_dt(latest[d]):
            latest[d] = f

    rows = []
    for d in sorted(latest):
        forecast = load_submission(latest[d], tz=tz)
        actuals = (
            fetch_entsoe(config.AREA, str(d), str(d), resample_hourly=False)
            .tz_convert(tz)
            .reindex(forecast.index)
        )
        n_missing = int(actuals.isna().sum())
        if n_missing > max_missing:
            warnings.warn(f"{d}: {n_missing} missing obs slots (> {max_missing}); day skipped.")
            continue
        actuals = actuals.interpolate(limit_direction="both")
        try:
            metrics = evaluate(actuals, forecast, daytime_only=False)  # baseline auto-built = naive benchmark
        except ValueError as exc:
            warnings.warn(f"{d}: benchmark unavailable ({exc}); day skipped.")
            continue
        rows.append(
            {
                "target_date": d,
                "crps_mean": round(metrics["crps_mean"], 4),
                "crps_benchmark": round(metrics["crps_benchmark"], 4),
                "crps_skill": round(metrics["crps_skill"], 4),
                "n_samples": metrics["n_samples"],
            }
        )

    return pd.DataFrame(
        rows,
        columns=["target_date", "crps_mean", "crps_benchmark", "crps_skill", "n_samples"],
    )


def print_summary(metrics: dict) -> None:
    print(f"  CRPS:             {metrics['crps_mean']:.2f} MW")
    print(f"  CRPS benchmark:   {metrics['crps_benchmark']:.2f} MW")
    print(f"  CRPS skill:       {metrics['crps_skill']:+.3f}  (>0 = better than benchmark)")
    print(f"  PIT bias:         {metrics['pit_bias']:+.3f}  (>0 = underforecast)")
    print(f"  Dispersion ratio: {metrics['dispersion_ratio']:.3f}  (1.0 = perfect)")
    print(f"  N samples:        {metrics['n_samples']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score a folder of submission payloads against the naive benchmark."
    )
    parser.add_argument("--folder", required=True, help="Folder of submission *.json payloads")
    parser.add_argument("--start", required=True, help="Target start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Target end date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Score dry-run payloads instead of live")
    parser.add_argument("--max-missing", type=int, default=8,
                        help="Max interpolatable missing slots/day before skipping (default 8)")
    args = parser.parse_args()

    results = evaluate_submissions(
        args.folder, args.start, args.end, dry_run=args.dry_run, max_missing=args.max_missing
    )
    if results.empty:
        print("No matching submissions with published actuals.")
    else:
        print(results.to_string(index=False))
        print(
            f"\nMean CRPS:           {results['crps_mean'].mean():.4f}  (n={len(results)} days)"
        )
        print(f"Mean CRPS benchmark: {results['crps_benchmark'].mean():.4f}")
        print(f"Mean CRPS skill:     {results['crps_skill'].mean():+.4f}")
