"""
Clear-sky GHI and theoretical PV power (per MWp) at summer vs winter solstice
for a single location — illustrates the pure solar-geometry effect (amplitude +
day length) that the clear-sky normalization factors out before any ML is applied.

Usage examples:
  # Karlsruhe, save vector output (PDF + SVG next to the given stem)
  python scripts/plot_ghi_solstice.py --output paper/figures/ghi_solstice

  # Different location / year, show interactively
  python scripts/plot_ghi_solstice.py --lat 52.52 --lon 13.40 --alt 34 --year 2024
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Ensure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from solar_forecast.clearsky import get_clearsky


def main():
    parser = argparse.ArgumentParser(description="Clear-sky GHI at both solstices")
    parser.add_argument("--lat", type=float, default=49.01, help="Latitude (default: Karlsruhe)")
    parser.add_argument("--lon", type=float, default=8.40, help="Longitude (default: Karlsruhe)")
    parser.add_argument("--alt", type=float, default=115, help="Altitude in m (default: Karlsruhe)")
    parser.add_argument("--year", type=int, default=2024, help="Year for the solstice dates")
    parser.add_argument("--tz", default="Europe/Berlin", help="Local timezone for the x-axis")
    parser.add_argument("--label", default="Karlsruhe (49,0°N / 8,4°E)", help="Location label in the title")
    parser.add_argument("--output", help="Output stem; writes <stem>.pdf and <stem>.svg. Omit to show interactively.")
    args = parser.parse_args()

    STC = 1000.0  # W/m² reference irradiance -> power per 1 MWp installed
    # Sunset palette: warm summer, cool winter
    days = {
        "Sommersonnenwende (21. Jun)": (f"{args.year}-06-21", "#e76f51"),
        "Wintersonnenwende (21. Dez)": (f"{args.year}-12-21", "#2a9d8f"),
    }

    plt.rcParams.update({
        "font.size": 11,
        "axes.edgecolor": "#8d99ae",
        "axes.linewidth": 0.8,
        "axes.titleweight": "semibold",
    })

    fig, (ax_ghi, ax_pv) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                        constrained_layout=True)

    for label, (day, color) in days.items():
        times = pd.date_range(f"{day} 00:00", f"{day} 23:45", freq="15min", tz=args.tz)
        cs = get_clearsky(args.lat, args.lon, times, altitude=args.alt)
        x = cs.index.hour + cs.index.minute / 60.0  # local hour-of-day

        daylight_h = cs["is_daytime"].sum() * 0.25
        ghi = cs["ghi"]
        power = cs["poa_clearsky"] / STC  # MW per MWp installed

        for ax, y, peak_unit in ((ax_ghi, ghi, "W/m²"), (ax_pv, power, "MW/MWp")):
            ax.plot(x, y, color=color, lw=2.4,
                    label=f"{label}\n{daylight_h:.1f} h Tag · Peak {y.max():.0f} {peak_unit}"
                          if peak_unit == "W/m²"
                          else f"{label}\n{daylight_h:.1f} h Tag · Peak {y.max():.2f} {peak_unit}")
            ax.fill_between(x, y, color=color, alpha=0.12)

    for ax in (ax_ghi, ax_pv):
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.30)  # Kopffreiheit oben -> Legende überlappt die Kurve nicht
        ax.legend(frameon=False, loc="upper left", fontsize=9)

    ax_ghi.set_title(f"Solargeometrie-Effekt bei wolkenlosem Himmel — {args.label}")
    ax_ghi.set_ylabel("Clear-Sky GHI  [W/m²]")
    ax_pv.set_ylabel("Theor. Max-Leistung  [MW pro MWp]")
    ax_pv.set_xlabel(f"Lokalzeit ({args.tz})")

    if args.output:
        stem = Path(args.output)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg"):
            path = stem.with_suffix(f".{ext}")
            fig.savefig(path)
            print(f"Saved {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
