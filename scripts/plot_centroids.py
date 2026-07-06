"""
Capacity-weighted centroids (config.CENTROIDS) on a map of Germany.

Each centroid is a representative NWP grid point; the bubble size and colour
encode its capacity weight (the fraction of installed solar it stands in for).
The outline is loaded from data/germany.geojson (cached Natural-shape GeoJSON).

Usage examples:
  # Save vector output (PDF + SVG next to the given stem)
  python scripts/plot_centroids.py --output paper/figures/centroids

  # Show interactively
  python scripts/plot_centroids.py
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Ensure the package / config is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

_GEOJSON = Path(__file__).parent.parent / "data" / "germany.geojson"


def main():
    parser = argparse.ArgumentParser(description="Plot capacity-weighted centroids over Germany")
    parser.add_argument("--geojson", default=str(_GEOJSON), help="Germany outline GeoJSON (WGS84)")
    parser.add_argument("--output", help="Output stem; writes <stem>.pdf and <stem>.svg. Omit to show interactively.")
    args = parser.parse_args()

    centroids = config.CENTROIDS
    lats = np.array([c["lat"] for c in centroids])
    lons = np.array([c["lon"] for c in centroids])
    weights = np.array([c["weight"] for c in centroids])

    germany = gpd.read_file(args.geojson)

    plt.rcParams.update({
        "font.size": 11,
        "axes.edgecolor": "#8d99ae",
        "axes.linewidth": 0.8,
        "axes.titleweight": "semibold",
    })

    fig, ax = plt.subplots(figsize=(7.5, 9), constrained_layout=True)

    # Country fill + border for geographic context
    germany.plot(ax=ax, facecolor="#f2f4f7", edgecolor="#8d99ae", linewidth=1.0, zorder=1)

    # Bubble size ∝ weight; colour also encodes weight (viridis)
    sizes = 120 + weights / weights.max() * 2200
    sc = ax.scatter(
        lons, lats, s=sizes, c=weights, cmap="viridis",
        alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3,
    )

    # Annotate each centroid with its weight in percent
    for lon, lat, w in zip(lons, lats, weights):
        ax.annotate(
            f"{w * 100:.1f}%", (lon, lat),
            textcoords="offset points", xytext=(0, 0),
            ha="center", va="center", fontsize=7.5, fontweight="semibold",
            color="white", zorder=4,
        )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Kapazitätsgewicht", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Correct aspect for the mean latitude so shapes are not distorted
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lats.mean())))
    ax.set_xlabel("Längengrad [°E]")
    ax.set_ylabel("Breitengrad [°N]")
    ax.set_title(
        f"Kapazitätsgewichtete Gitterpunkte (k={len(centroids)})\n"
        "Blasengröße & -farbe ∝ Anteil installierter PV-Leistung"
    )
    ax.grid(linestyle="--", alpha=0.3, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

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
