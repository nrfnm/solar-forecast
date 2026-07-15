"""
Single aggregated PIT (Probability Integral Transform) histogram for a cached
energy-arena ensemble, daytime slots only (night excluded).

PIT is the mid-rank of the observation within the ensemble (ties split evenly),
so a well-calibrated forecast gives a flat histogram. The dashed line marks the
uniform target; the reported number is the reliability L1 deviation from it
(0 = perfectly uniform). PIT mean > 0.5 = obs sits in the upper tail =
underforecast; < 0.5 = overforecast. A U-shape = under-dispersion, a hump =
over-dispersion.

Usage:
  python scripts/plot_pit_daytime.py \
      --ts data/eval_cache/arena_ts_ch16_1-2-8-17-19-20-26_2026-06-06_2026-07-07.json \
      --participant OpenForecast --start 2026-06-07 --end 2026-07-08 \
      --output paper/figures/pit_daytime
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
DAY_FROM, DAY_TO = "05:00", "21:00"   # night excluded (matches plot_pit_by_day)
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
    fc = fc[~fc.index.duplicated()]
    return fc, gt


def pit_midrank(obs: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Mid-rank PIT: fraction of members below obs + half the ties."""
    less = (F < obs[:, None]).sum(1)
    equal = (F == obs[:, None]).sum(1)
    return (less + 0.5 * equal) / F.shape[1]


def main():
    ap = argparse.ArgumentParser(description="Aggregated daytime PIT histogram (night excluded)")
    ap.add_argument("--ts", default="data/eval_cache/arena_ts_ch16_1-2-8-17-19-20-26_2026-06-06_2026-07-07.json")
    ap.add_argument("--participant", default="spinner")
    ap.add_argument("--start", default="2026-06-07", help="First day (inclusive), YYYY-MM-DD")
    ap.add_argument("--end", default="2026-07-08", help="Last day (inclusive), YYYY-MM-DD")
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
    pit = pit_midrank(obs.values, fc.values)
    ndays = len(set(obs.index.normalize()))

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.hist(pit, bins=NBINS, range=(0, 1), weights=np.full(len(pit), 1 / len(pit)),
            color="#3a86ff", alpha=0.8, edgecolor="white", linewidth=0.6)
    ax.axhline(1 / NBINS, color="#e63946", ls="--", lw=1.2, label="uniform")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.20, ax.get_ylim()[1]))
    ax.set_xlabel("PIT")
    ax.set_ylabel("relative frequency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        f"Daytime PIT · {args.participant} · {args.start} – {args.end}\n"
        f"(daytime {DAY_FROM}–{DAY_TO}, {ndays} days)",
        fontsize=10,
    )

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
