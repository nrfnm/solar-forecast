"""
Fan chart over the sustained clear-sky spell where the forecast systematically
under-predicts (actual peak above the ensemble P90). Data: cached energy-arena
challenge-16 timeseries (ground truth + 100-member ensemble, MW).

Usage:
  # single participant
  python scripts/plot_underforecast_fan.py \
      --ts data/eval_cache/arena_ts_ch16_19-1_2026-06-01_2026-07-04.json \
      --participant spinner --start 2026-06-22 --end 2026-06-28 \
      --output paper/figures/underforecast_fan

  # two participants stacked into one file (top-to-bottom in the given order)
  python scripts/plot_underforecast_fan.py \
      --participant spinner NaiveBenchmark --start 2026-06-07 --end 2026-06-11 \
      --output paper/figures/regime_change_fan_combined
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

TZ = "Europe/Berlin"


def load(ts_path: str, participant: str):
    d = json.load(open(ts_path))
    gt = pd.Series({pd.Timestamp(x["ts"]): x["value"] for x in d["ground_truth"]}).sort_index()
    gt.index = gt.index.tz_convert(TZ)

    p = next(x for x in d["participants"] if x["participant_name"] == participant)
    ens = p["ensemble_points"]
    idx = pd.to_datetime([e["ts"] for e in ens]).tz_convert(TZ)
    arr = np.array([e["values"] for e in ens], dtype=float)  # (T, n_members)
    fc = pd.DataFrame(arr, index=idx)
    return fc, gt


def draw_fan(ax, t, q, act, title, legend=False, scale=1000.0):
    """Draw one fan chart (values scaled MW -> GW) onto ax."""
    band = "#3a86ff"
    ax.fill_between(t, q[5] / scale, q[95] / scale, color=band, alpha=0.12, label="P05–P95")
    ax.fill_between(t, q[10] / scale, q[90] / scale, color=band, alpha=0.18, label="P10–P90")
    ax.fill_between(t, q[25] / scale, q[75] / scale, color=band, alpha=0.30, label="P25–P75")
    ax.plot(t, q[50] / scale, color="#1d4ed8", lw=1.6, label="Median (P50)")

    ax.plot(t, act / scale, color="#e63946", lw=1.8, label="Ist (SMARD)", zorder=6)
    # mark where actual exceeds the P90 band -> systematic under-forecast
    under = act > q[90]
    ax.fill_between(t, q[90] / scale, act / scale, where=under,
                    color="#e63946", alpha=0.22, label="Ist über P90 (Underforecast)")

    ax.set_title(title)
    ax.set_ylabel("Solarerzeugung  [GW]")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(loc="upper left", frameon=False, ncol=2)


def main():
    ap = argparse.ArgumentParser(description="Fan chart of systematic under-forecast days")
    ap.add_argument("--ts", default="data/eval_cache/arena_ts_ch16_19-1_2026-06-01_2026-07-04.json")
    ap.add_argument("--participant", nargs="+", default=["spinner"],
                    help="One or more participant names. Multiple names are stacked vertically.")
    ap.add_argument("--start", default="2026-06-22", help="First day (inclusive), YYYY-MM-DD")
    ap.add_argument("--end", default="2026-06-28", help="Last day (exclusive), YYYY-MM-DD")
    ap.add_argument("--output", help="Output stem; writes <stem>.pdf and <stem>.svg")
    ap.add_argument("--title", help="Custom title (only used for a single participant; "
                                    "default: '<participant> · <start> – <end>')")
    args = ap.parse_args()

    lo = pd.Timestamp(args.start, tz=TZ)
    hi = pd.Timestamp(args.end, tz=TZ)
    end_label = (pd.Timestamp(args.end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    participants = args.participant
    n = len(participants)

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "semibold"})
    fig, axes = plt.subplots(n, 1, figsize=(12, 4.0 * n if n > 1 else 5.5),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    t = None
    for i, (ax, name) in enumerate(zip(axes, participants)):
        fc, gt = load(args.ts, name)
        fc = fc[(fc.index >= lo) & (fc.index < hi)]
        common = fc.index.intersection(gt.index)
        fc, act = fc.loc[common], gt.loc[common]
        t = fc.index
        q = {p: fc.quantile(p / 100, axis=1) for p in (5, 10, 25, 50, 75, 90, 95)}

        title = args.title if (args.title and n == 1) else f"{name} · {args.start} – {end_label}"
        draw_fan(ax, t, q, act, title, legend=(i == 0))

    # shared date axis on the bottom panel only
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m", tz=t.tz))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(tz=t.tz))
    axes[-1].set_xlabel(f"Lokalzeit ({TZ})")

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
