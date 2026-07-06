# solar_forecast/benchmark.py
"""Naive historical-ensemble benchmark (the competition's scoring reference).

No ML, no weather model. The assumption: solar generation on the target day looks
like generation on recent days at the same clock times. For the target day we gather
the last 10 historical days (Day −FIRST_LOOKBACK_DAY … back), re-align each to the
target day's clock times via a whole-day forward shift, then tile those 10 days
column-wise up to the platform's 100 members. Small per-day gaps (≤ ``max_missing``
slots) are interpolated; a historical day with more missing slots raises.

Data source is ENTSO-E "Solar - Actual Aggregated" (B16) at native 15-min — the same
dataset the platform uses for internal scoring (NOT the SMARD public mirror).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from solar_forecast.fetch_actuals import fetch_entsoe

N_HISTORICAL_DAYS = 10
N_MEMBERS = 100
FIRST_LOOKBACK_DAY = 2
MAX_MISSING_SLOTS = 8  # per historical day: interpolate up to this many gaps, else raise

def naive_benchmark(
    index: pd.DatetimeIndex,
    tz: str,
    area: str = config.AREA,
    max_missing: int = MAX_MISSING_SLOTS,
) -> pd.DataFrame:
    local_index = index.tz_convert(tz)
    target_date = local_index[0].date()

    last_lookback = FIRST_LOOKBACK_DAY + N_HISTORICAL_DAYS - 1
    start = target_date - pd.Timedelta(days=last_lookback)
    end = target_date - pd.Timedelta(days=FIRST_LOOKBACK_DAY)
    hist = fetch_entsoe(area, str(start), str(end), resample_hourly=False).tz_convert(tz)

    samples = []
    for i in range(N_HISTORICAL_DAYS):
        lookback = FIRST_LOOKBACK_DAY + i
        shifted = hist.copy()
        shifted.index = shifted.index + pd.Timedelta(days=lookback)
        member = shifted.reindex(local_index)
        n_missing = int(member.isna().sum())
        if n_missing > max_missing:
            day = target_date - pd.Timedelta(days=lookback)
            raise ValueError(
                f"historical day {day} has {n_missing} missing slots (> {max_missing})"
            )
        # interpolate the few remaining gaps so the ensemble is complete
        samples.append(member.interpolate(limit_direction="both").to_numpy())

    sample_matrix = np.column_stack(samples)  # (timesteps, N_HISTORICAL_DAYS)
    reps = int(np.ceil(N_MEMBERS / N_HISTORICAL_DAYS))
    out = np.tile(sample_matrix, (1, reps))[:, :N_MEMBERS]

    return pd.DataFrame(out, index=index,
                        columns=[f"member_{i:03d}" for i in range(N_MEMBERS)])


def platform_naive_benchmark(
    index: pd.DatetimeIndex,
    tz: str,
    area: str = config.AREA,
    max_missing: int = MAX_MISSING_SLOTS,
) -> pd.DataFrame:
    """Faithful copy of energy-arena's internal NaiveBenchmark ensemble.

    Same 10 recent days (t-2 … t-11) as :func:`naive_benchmark`, but instead of
    tiling them evenly, the ensemble is padded to 100 members by repeating the
    OLDEST day (t-11) 90×. This mirrors the platform's ``_attach_objective_values``
    ensemble branch (which pads with ``values[-1]`` under a live ``history_count``
    of 10). The oldest-day spike makes this benchmark deliberately weak — use it
    only to reproduce the platform's skill-score reference, not as a real forecast.
    """
    local_index = index.tz_convert(tz)
    target_date = local_index[0].date()

    last_lookback = FIRST_LOOKBACK_DAY + N_HISTORICAL_DAYS - 1
    start = target_date - pd.Timedelta(days=last_lookback)
    end = target_date - pd.Timedelta(days=FIRST_LOOKBACK_DAY)
    hist = fetch_entsoe(area, str(start), str(end), resample_hourly=False).tz_convert(tz)

    samples = []
    for i in range(N_HISTORICAL_DAYS):
        lookback = FIRST_LOOKBACK_DAY + i  # 2 … 11 → newest … oldest
        shifted = hist.copy()
        shifted.index = shifted.index + pd.Timedelta(days=lookback)
        member = shifted.reindex(local_index)
        n_missing = int(member.isna().sum())
        if n_missing > max_missing:
            day = target_date - pd.Timedelta(days=lookback)
            raise ValueError(
                f"historical day {day} has {n_missing} missing slots (> {max_missing})"
            )
        # interpolate the few remaining gaps so the ensemble is complete
        samples.append(member.interpolate(limit_direction="both").to_numpy())

    sample_matrix = np.column_stack(samples)  # (timesteps, N_HISTORICAL_DAYS), col -1 = oldest (t-11)
    pad = np.repeat(sample_matrix[:, [-1]], N_MEMBERS - N_HISTORICAL_DAYS, axis=1)  # 90× oldest day
    out = np.column_stack([sample_matrix, pad])  # (timesteps, N_MEMBERS)

    return pd.DataFrame(out, index=index,
                        columns=[f"member_{i:03d}" for i in range(N_MEMBERS)])


def _quarter_hourly_index(target_date: str, tz: str) -> pd.DatetimeIndex:
    """96-slot 15-min index for the local calendar day ``target_date``."""
    start = pd.Timestamp(target_date, tz=tz)
    return pd.date_range(start=start, periods=96, freq="15min", tz=tz)


if __name__ == "__main__":
    import argparse

    from solar_forecast.evaluate import crps_score

    parser = argparse.ArgumentParser(description="Build and score the naive benchmark for a day.")
    parser.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    args = parser.parse_args()

    index = _quarter_hourly_index(args.date, config.TZ)
    benchmark = naive_benchmark(index, config.TZ)
    print(f"Benchmark for {args.date}: {benchmark.shape[1]} members × {benchmark.shape[0]} slots")

    # ENTSO-E actuals for the target day: same dataset the platform scores against.
    actuals = (
        fetch_entsoe(config.AREA, args.date, args.date, resample_hourly=False)
        .tz_convert(config.TZ)
        .reindex(index)
    )
    if actuals.isna().all():
        print("Target day has no ENTSO-E actuals yet — cannot score.")
    else:
        scores = crps_score(actuals, benchmark)
        print(f"Benchmark CRPS: {scores['crps_mean']:.2f} MW  (n={scores['n_samples']})")
