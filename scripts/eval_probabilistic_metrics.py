"""
Full probabilistic-forecast metric table for cached energy-arena ensembles.

Computes, per participant, over a window (with day-persistence imputation of
missing days): deterministic errors, CRPS + normalizations, pinball/WIS,
interval coverage & sharpness, PIT-based calibration, and the multivariate
Energy Score. Comparison without relying on a skill score.

Usage:
  python scripts/eval_probabilistic_metrics.py \
      --ts data/eval_cache/arena_ts_ch16_all_2026-06-01_2026-07-07.json \
      --start 2026-06-07 --end 2026-07-06 \
      --out paper/results_ch16_metrics.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import properscoring as ps

sys.path.insert(0, str(Path(__file__).parent.parent))

TZ = "Europe/Berlin"
QLEVELS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
WIS_ALPHAS = np.array([0.1, 0.2, 0.5])            # central intervals 90/80/50 %
COVER_LEVELS = [0.5, 0.8, 0.9]
DAY_FROM, DAY_TO = "05:00", "21:00"


def load(ts_path):
    d = json.load(open(ts_path))
    gt = pd.Series({pd.Timestamp(x["ts"]): x["value"] for x in d["ground_truth"]}).sort_index()
    gt.index = gt.index.tz_convert(TZ)
    return d["participants"], gt


def impute_day_persistence(fc, grid):
    """Reindex to grid; fill missing whole days with same-time-of-day neighbour."""
    fc = fc[~fc.index.duplicated()].reindex(grid)
    return fc.groupby(fc.index.time, group_keys=False).apply(lambda g: g.ffill().bfill())


def pinball(o, q, tau):
    d = o - q
    return np.maximum(tau * d, (tau - 1) * d)


def wis(o, Q):
    """Weighted interval score from quantile forecasts Q (T × len(QLEVELS))."""
    med = Q[:, np.where(QLEVELS == 0.5)[0][0]]
    total = 0.5 * np.abs(o - med)
    for a in WIS_ALPHAS:
        lo = Q[:, np.argmin(np.abs(QLEVELS - a / 2))]
        hi = Q[:, np.argmin(np.abs(QLEVELS - (1 - a / 2)))]
        below = (o < lo) * (2 / a) * (lo - o)
        above = (o > hi) * (2 / a) * (o - hi)
        total = total + (a / 2) * ((hi - lo) + below + above)
    return total / (len(WIS_ALPHAS) + 0.5)


def pit_calibration(o, F, mask):
    """PIT via mid-rank on masked steps; return reliability (hist L1 dev) + coverage errors."""
    o_m, F_m = o[mask], F[mask]
    less = (F_m < o_m[:, None]).sum(1)
    equal = (F_m == o_m[:, None]).sum(1)
    pit = (less + 0.5 * equal) / F_m.shape[1]
    nb = 10
    hist, _ = np.histogram(pit, bins=nb, range=(0, 1))
    reliability = np.abs(hist / hist.sum() - 1 / nb).sum()   # 0 = perfectly uniform
    return reliability, pit.mean()


def energy_score(o, F, dates):
    """Mean daily multivariate Energy Score over full-day trajectories."""
    scores = []
    pos = np.arange(len(o))
    for _, grp in pd.Series(pos).groupby(np.asarray(dates)):
        idx = grp.values
        y = o[idx]                       # (D,)
        X = F[idx]                       # (D, M)
        M = X.shape[1]
        term1 = np.linalg.norm(X - y[:, None], axis=0).mean()
        diff = X[:, :, None] - X[:, None, :]
        term2 = np.linalg.norm(diff, axis=0).mean()
        scores.append(term1 - 0.5 * term2)
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser(description="Full probabilistic metric table")
    ap.add_argument("--ts", default="data/eval_cache/arena_ts_ch16_all_2026-06-01_2026-07-07.json")
    ap.add_argument("--start", default="2026-06-07")
    ap.add_argument("--end", default="2026-07-06")
    ap.add_argument("--out", help="Optional CSV output path")
    args = ap.parse_args()

    participants, gt = load(args.ts)
    start = pd.Timestamp(args.start, tz=TZ)
    end = pd.Timestamp(args.end, tz=TZ)
    grid = pd.date_range(start, end, freq="15min", tz=TZ, inclusive="left")
    obs = gt.reindex(gt.index.union(grid)).interpolate("time").reindex(grid)
    o = obs.values
    day = np.zeros(len(grid), bool)
    day[obs.index.indexer_between_time(DAY_FROM, DAY_TO)] = True
    dates = obs.index.normalize()
    mo, mod, pk = obs.mean(), obs[day].mean(), obs.max()

    rows = []
    for p in participants:
        idx = pd.to_datetime([e["ts"] for e in p["ensemble_points"]]).tz_convert(TZ)
        fc = pd.DataFrame(np.array([e["values"] for e in p["ensemble_points"]], float), index=idx).sort_index()
        imputed = int((~fc[~fc.index.duplicated()].reindex(grid).iloc[:, 0].notna()).sum())
        fc = impute_day_persistence(fc, grid)
        F = fc.values
        med = np.median(F, axis=1)
        mean = F.mean(1)
        Q = np.quantile(F, QLEVELS, axis=1).T

        crps = ps.crps_ensemble(o, F)
        rel, pit_mean = pit_calibration(o, F, day)

        row = dict(
            participant=p["participant_name"],
            imputed_steps=imputed,
            # deterministic
            MAE_median=np.abs(o - med).mean(),
            RMSE_mean=np.sqrt(((o - mean) ** 2).mean()),
            Bias_mean=(mean - o).mean(),
            # CRPS + normalizations
            CRPS=crps.mean(),
            CRPS_day=crps[day].mean(),
            nCRPS_mean_pct=crps.mean() / mo * 100,
            nCRPS_peak_pct=crps.mean() / pk * 100,
            # quantile / interval scores
            Pinball_mean=np.mean([pinball(o, Q[:, i], t) for i, t in enumerate(QLEVELS)]),
            WIS=wis(o, Q).mean(),
            # sharpness (interval width) + coverage
        )
        for lv in COVER_LEVELS:
            lo = np.quantile(F, (1 - lv) / 2, axis=1)
            hi = np.quantile(F, 1 - (1 - lv) / 2, axis=1)
            cov = ((o >= lo) & (o <= hi))[day].mean() * 100
            row[f"cover{int(lv*100)}_pct"] = cov
            row[f"width{int(lv*100)}_day"] = (hi - lo)[day].mean()
        row["reliability_L1"] = rel        # 0 = perfectly calibrated PIT
        row["PIT_mean"] = pit_mean          # ~0.5 if unbiased
        row["EnergyScore"] = energy_score(o, F, dates)
        rows.append(row)

    res = pd.DataFrame(rows).set_index("participant").sort_values("CRPS")
    pd.set_option("display.width", 260, "display.max_columns", 40)
    print(f"Window {start.date()} → {end.date()}  n={len(grid)} steps")
    print(f"mean_obs={mo:.0f} MW  mean_obs_day={mod:.0f} MW  peak_obs={pk:.0f} MW\n")
    print(res.round(2).T.to_string())

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.round(4).to_csv(args.out)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
