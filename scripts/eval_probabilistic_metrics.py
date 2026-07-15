"""
Probabilistic-forecast comparison table for cached energy-arena ensembles.

Primary scores (CRPS, WIS, RMSE, Energy Score, Variogram Score, all-slot
coverage) are taken directly from the platform's own per-day metrics
(``participant_metrics`` in the snapshot) and averaged over the window's
delivery days — no local recomputation, no imputation (each participant is
scored on the days it actually submitted; see the ``n_days`` column).

PIT calibration is not exposed by the API and is computed locally from the
archived ensembles, restricted to daylight slots (ensemble spread > 0).

Usage:
  python scripts/eval_probabilistic_metrics.py \
      --ts data/eval_cache/arena_ts_ch16_1-2-8-17-19-20-26_2026-06-06_2026-07-07.json \
      --start 2026-06-07 --end 2026-07-08 \
      --out paper/results_ch16_metrics.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

TZ = "Europe/Berlin"
COVER_LEVELS = [0.5, 0.9]


def load(ts_path):
    """Return (participants, ground_truth, api_metrics).

    api_metrics maps participant name -> DataFrame of the platform's per-day
    metrics, indexed by target_start (UTC).
    """
    d = json.load(open(ts_path))
    gt = pd.Series({pd.Timestamp(x["ts"]): x["value"] for x in d["ground_truth"]}).sort_index()
    gt.index = gt.index.tz_convert(TZ)
    api = {}
    for pm in d.get("participant_metrics", []):
        recs = {pd.Timestamp(p["target_start"]): p["metrics"] for p in pm["points"]}
        api[pm["participant_name"]] = pd.DataFrame(recs).T.sort_index()
    return d["participants"], gt, api


def daylight_cov(F, o, day, lv):
    """Empirical coverage of the central lv-interval on daylight slots (submitted days)."""
    Fd, od = F[day], o[day]
    lo = np.quantile(Fd, (1 - lv) / 2, axis=1)
    hi = np.quantile(Fd, 1 - (1 - lv) / 2, axis=1)
    return ((od >= lo) & (od <= hi)).mean() * 100


def pit_calibration(o, F, mask):
    """PIT via mid-rank on masked steps; return reliability (hist L1 dev) + mean."""
    o_m, F_m = o[mask], F[mask]
    less = (F_m < o_m[:, None]).sum(1)
    equal = (F_m == o_m[:, None]).sum(1)
    pit = (less + 0.5 * equal) / F_m.shape[1]
    nb = 10
    hist, _ = np.histogram(pit, bins=nb, range=(0, 1))
    reliability = np.abs(hist / hist.sum() - 1 / nb).sum()   # 0 = perfectly uniform
    return reliability, pit.mean()


def main():
    ap = argparse.ArgumentParser(description="API-based probabilistic metric table")
    ap.add_argument("--ts", default="data/eval_cache/arena_ts_ch16_1-2-8-17-19-20-26_2026-06-06_2026-07-07.json")
    ap.add_argument("--start", default="2026-06-07")
    ap.add_argument("--end", default="2026-07-08")
    ap.add_argument("--out", help="Optional CSV output path")
    ap.add_argument("--impute", action="store_true",
                    help="Fill each participant's non-submitted delivery days with the "
                         "baseline's daily metrics (skill 0) so all are scored over the "
                         "full window; default off = score on submitted days only.")
    ap.add_argument("--baseline", default="NaiveBenchmark",
                    help="Participant whose daily metrics impute missing days (--impute).")
    args = ap.parse_args()

    participants, gt, api = load(args.ts)
    start = pd.Timestamp(args.start, tz=TZ)
    end = pd.Timestamp(args.end, tz=TZ)
    grid = pd.date_range(start, end, freq="15min", tz=TZ, inclusive="left")
    deliver_dates = pd.Index(grid.normalize().unique())
    obs = gt.reindex(gt.index.union(grid)).interpolate("time").reindex(grid)
    o = obs.values
    mo, mod, pk = obs.mean(), obs[obs > 0].mean(), obs.max()

    def window_days(name):
        """Participant's daily API metrics, indexed by Berlin delivery date, in window."""
        am = api[name].copy()
        am.index = am.index.tz_convert(TZ).normalize()
        return am[am.index.isin(deliver_dates)]

    if args.baseline not in api:
        raise SystemExit(f"--baseline {args.baseline!r} not in snapshot")
    base_daily = window_days(args.baseline)   # daily baseline CRPS, for skill + imputation

    rows = []
    for p in participants:
        name = p["participant_name"]
        if name not in api:
            continue

        # --- platform (API) scores: mean over the window's delivery days ---
        am_w = window_days(name)
        n_sub = len(am_w)
        n_imputed = 0
        if args.impute:
            full = am_w.reindex(deliver_dates)
            miss = full.index[full["CRPS"].isna()]
            full.loc[miss] = base_daily.reindex(miss)     # missing day := baseline (skill 0)
            am_w, n_imputed = full, len(miss)
        row = dict(
            participant=name,
            n_days=n_sub,
            n_imputed=n_imputed,
            CRPS=am_w["CRPS"].mean(),
            WIS=am_w["WIS"].mean(),
            RMSE_mean=am_w["RMSE_MEAN"].mean(),
            EnergyScore=am_w["ENERGY_SCORE"].mean(),
            VariogramScore=am_w["VARIOGRAM_SCORE"].mean(),
            cover50_pct=am_w["COVERAGE_50"].mean(),   # all slots (platform)
            cover90_pct=am_w["COVERAGE_90"].mean(),
        )
        # CRPS skill vs baseline, over the same days this participant is scored on
        base_crps = base_daily["CRPS"].reindex(am_w.index).mean()
        row["skill"] = 1 - row["CRPS"] / base_crps

        # --- daylight coverage + PIT: local, from the archived ensembles (submitted days) ---
        idx = pd.to_datetime([e["ts"] for e in p["ensemble_points"]]).tz_convert(TZ)
        fc = pd.DataFrame(np.array([e["values"] for e in p["ensemble_points"]], float), index=idx).sort_index()
        fc = fc[~fc.index.duplicated()].reindex(grid)     # NaN on days not submitted
        F = fc.values
        valid = fc.iloc[:, 0].notna().values
        day = np.zeros(len(F), bool)
        day[valid] = F[valid].max(axis=1) > 0             # daylight = non-zero spread
        for lv in COVER_LEVELS:
            row[f"cover{int(lv * 100)}_day"] = daylight_cov(F, o, day, lv)
        row["reliability_L1"], row["PIT_mean"] = pit_calibration(o, F, day)
        rows.append(row)

    res = pd.DataFrame(rows).set_index("participant").sort_values("CRPS")
    pd.set_option("display.width", 260, "display.max_columns", 40)
    print(f"Window {start.date()} → {end.date()}  ({len(deliver_dates)} delivery days)")
    print(f"mean_obs={mo:.0f} MW  mean_obs_day={mod:.0f} MW  peak_obs={pk:.0f} MW")
    agg = f"full window, missing days imputed with '{args.baseline}'" if args.impute else "submitted days only, no imputation"
    print(f"Scores: platform API, mean over {agg}")
    print("PIT: local, daylight slots only\n")
    print(res.round(2).T.to_string())

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.round(4).to_csv(args.out)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
