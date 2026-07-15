"""
Per-day PIT (Probability Integral Transform) diagrams for a cached energy-arena
ensemble. One histogram per day, night excluded, so day-to-day calibration
regimes (under/over-dispersion, systematic bias) are visible at a glance.

PIT is the mid-rank of the observation within the ensemble (ties split evenly),
so a well-calibrated forecast gives a flat histogram. The dashed line marks the
uniform target; the per-panel number is the reliability L1 deviation from it
(0 = perfectly uniform). PIT mean > 0.5 = obs sits in the upper tail =
underforecast; < 0.5 = overforecast.

Usage:
  python scripts/plot_pit_by_day.py \
      --ts data/eval_cache/arena_ts_ch16_1-8-17-19_2026-06-06_2026-07-08.json \
      --participant OpenForecast --start 2026-06-07 --end 2026-07-08 \
      --output paper/figures/pit_by_day
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

TZ = "Europe/Berlin"
DAY_FROM, DAY_TO = "05:00", "21:00"   # night excluded (matches eval_probabilistic_metrics)
NBINS = 10


def load(ts_path: str, participant: str):
    d = json.load(open(ts_path))
    gt = pd.Series({pd.Timestamp(x["ts"]): x["value"] for x in d["ground_truth"]}).sort_index()
    gt.index = gt.index.tz_convert(TZ)

    p = next(x for x in d["participants"] if x["participant_name"] == participant)
    ens = p["ensemble_points"]
    idx = pd.to_datetime([e["ts"] for e in ens]).tz_convert(TZ)
    arr = np.array([e["values"] for e in ens], dtype=float)  # (T, n_members)
    fc = pd.DataFrame(arr, index=idx).sort_index()
    return fc, gt


def pit_midrank(obs: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Mid-rank PIT: fraction of members below obs + half the ties."""
    less = (F < obs[:, None]).sum(1)
    equal = (F == obs[:, None]).sum(1)
    return (less + 0.5 * equal) / F.shape[1]


def reliability_l1(pit: np.ndarray) -> float:
    """L1 deviation of the PIT histogram from uniform (0 = perfectly flat)."""
    hist, _ = np.histogram(pit, bins=NBINS, range=(0, 1))
    return float(np.abs(hist / hist.sum() - 1 / NBINS).sum())


def draw_panel(ax, pit, label):
    """Draw one day's PIT histogram (relative frequency) onto ax."""
    ax.hist(pit, bins=NBINS, range=(0, 1), weights=np.full(len(pit), 1 / len(pit)),
            color="#3a86ff", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axhline(1 / NBINS, color="#e63946", ls="--", lw=1.0)
    rel = reliability_l1(pit)
    ax.set_title(f"{label}\nPIT̄={pit.mean():.2f}  L1={rel:.2f}  n={len(pit)}", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.30, ax.get_ylim()[1]))
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description="Per-day PIT diagrams (night excluded)")
    ap.add_argument("--ts", default="data/eval_cache/arena_ts_ch16_1-8-17-19_2026-06-06_2026-07-08.json")
    ap.add_argument("--participant", default="OpenForecast")
    ap.add_argument("--start", default="2026-06-07", help="First day (inclusive), YYYY-MM-DD")
    ap.add_argument("--end", default="2026-07-08", help="Last day (inclusive), YYYY-MM-DD")
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--output", help="Output stem; writes <stem>.pdf and <stem>.svg")
    args = ap.parse_args()

    lo = pd.Timestamp(args.start, tz=TZ)
    hi = pd.Timestamp(args.end, tz=TZ) + pd.Timedelta(days=1)   # end inclusive

    fc, gt = load(args.ts, args.participant)
    common = fc.index.intersection(gt.index)
    fc, obs = fc.loc[common], gt.loc[common]

    # night excluded, then window
    day_pos = obs.index.indexer_between_time(DAY_FROM, DAY_TO)
    mask = np.zeros(len(obs), bool)
    mask[day_pos] = True
    mask &= (obs.index >= lo) & (obs.index < hi)

    fc, obs = fc[mask], obs[mask]
    F, o = fc.values, obs.values
    pit_all = pit_midrank(o, F)

    days = sorted(set(obs.index.normalize()))
    day_norm = obs.index.normalize()
    ncols = args.ncols
    nrows = int(np.ceil(len(days) / ncols))

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 1.9 * nrows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, day in zip(axes, days):
        draw_panel(ax, pit_all[day_norm == day], day.strftime("%a %d.%m"))
    for ax in axes[len(days):]:
        ax.axis("off")

    fig.suptitle(
        f"Per-day PIT · {args.participant} · {args.start} – {args.end} "
        f"(daytime {DAY_FROM}–{DAY_TO}, {len(days)} days)\n"
        f"overall PIT̄={pit_all.mean():.3f}  reliability-L1={reliability_l1(pit_all):.3f}",
        fontsize=11, fontweight="semibold",
    )
    for ax in axes.reshape(nrows, ncols)[-1]:
        ax.set_xlabel("PIT", fontsize=8)

    if args.output:
        stem = Path(args.output)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg"):
            fig.savefig(stem.with_suffix(f".{ext}"))
            print(f"Saved {stem.with_suffix('.' + ext)}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
