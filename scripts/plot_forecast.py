"""
Plot ensemble forecast trajectories with optional actuals overlay.

Usage examples:
  # Live forecast for today, show interactively
  python scripts/plot_forecast.py

  # Load a saved submission JSON (energy-arena payload format)
  python scripts/plot_forecast.py --input submitted_payloads/challenge_16/foo.json --actuals

  # Load saved pipeline parquet
  python scripts/plot_forecast.py --input data/forecasts/2024-06-15.parquet --actuals --output forecast.png

  # Show individual member traces on top of bands
  python scripts/plot_forecast.py --members
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Ensure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_json(path: str) -> pd.DataFrame:
    with open(path) as f:
        payload = json.load(f)

    target_start = pd.Timestamp(payload["target_start"])
    raw = payload["values"]  # list of n_slots items
    n_slots = len(raw)

    freq = pd.Timedelta(hours=24) / n_slots
    index = pd.date_range(start=target_start, periods=n_slots, freq=freq)

    if isinstance(raw[0], list):
        arr = np.array(raw, dtype=float)          # (n_slots, n_members)
        cols = [f"member_{i:03d}" for i in range(arr.shape[1])]
    else:
        arr = np.array(raw, dtype=float).reshape(-1, 1)
        cols = ["member_000"]

    df = pd.DataFrame(arr, index=index, columns=cols)
    challenge_id = payload.get("challenge_id", "?")
    print(f"Loaded JSON payload (challenge {challenge_id}): {df.shape[0]} slots × {df.shape[1]} members, start {target_start}")
    return df


def load_forecast(args) -> pd.DataFrame:
    if args.input:
        path = Path(args.input)
        if path.suffix.lower() == ".json":
            return _load_json(args.input)
        df = pd.read_parquet(args.input)
        print(f"Loaded forecast from {args.input}: {df.shape}")
        return df

    print("Running live forecast (this may take a minute)...")
    from solar_forecast.ensemble import forecast_country
    from solar_forecast.train import load_quantile_models

    try:
        quantile_models = load_quantile_models()
        df = forecast_country(quantile_models=quantile_models, forecast_days=args.forecast_days)
    except FileNotFoundError:
        print("Quantile models not found, falling back to mean model.")
        df = forecast_country(forecast_days=args.forecast_days)

    return df


def filter_to_date(df: pd.DataFrame, date: str) -> pd.DataFrame:
    idx = pd.to_datetime(date).date()
    mask = df.index.date == idx
    filtered = df[mask]
    if filtered.empty:
        raise ValueError(f"No data for {date}. Available range: {df.index[0]} – {df.index[-1]}")
    return filtered


def fetch_actuals(date: str, forecast_days: int) -> pd.Series:
    from solar_forecast.fetch_actuals import fetch_smard
    from config import TZ

    start = date
    end = (pd.Timestamp(date) + pd.Timedelta(days=forecast_days)).strftime("%Y-%m-%d")
    print(f"Fetching SMARD actuals {start} → {end}...")
    try:
        actuals = fetch_smard(start=start, end=end, tz=TZ)
        return actuals
    except Exception as e:
        print(f"Warning: could not fetch actuals ({e})")
        return None


def plot(df: pd.DataFrame, actuals: pd.Series | None, title: str, args):
    member_cols = df.columns.tolist()
    values = df[member_cols].values  # shape (T, N)

    quantiles = {
        "p10": np.percentile(values, 10, axis=1),
        "p25": np.percentile(values, 25, axis=1),
        "p50": np.percentile(values, 50, axis=1),
        "p75": np.percentile(values, 75, axis=1),
        "p90": np.percentile(values, 90, axis=1),
    }
    t = df.index

    fig, ax = plt.subplots(figsize=(12, 5))

    # Individual member traces
    if args.members:
        for col in member_cols:
            ax.plot(t, df[col].values, color="#4c8cbf", alpha=0.06, linewidth=0.5)

    # Shaded bands
    ax.fill_between(t, quantiles["p10"], quantiles["p90"],
                    color="#4c8cbf", alpha=0.18, label="P10–P90")
    ax.fill_between(t, quantiles["p25"], quantiles["p75"],
                    color="#4c8cbf", alpha=0.32, label="P25–P75")

    # Median
    ax.plot(t, quantiles["p50"], color="#1a5fa8", linewidth=1.8, label="Median")

    # Actuals
    if actuals is not None:
        # Align to forecast index window
        actuals_aligned = actuals.reindex(t, method="nearest", tolerance=pd.Timedelta("8min"))
        ax.plot(t, actuals_aligned.values, color="#e05c2a", linewidth=1.6,
                label="Actuals (SMARD)", zorder=5)

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%b %d"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate(rotation=0, ha="center")

    n_members = len(member_cols)
    capacity_gw = values.max() / 1000
    ax.set_ylabel("Solar Generation (MW)")
    ax.set_title(title or f"Solar Ensemble Forecast  ·  {n_members} members  ·  peak ~{capacity_gw:.1f} GW")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot solar ensemble forecast")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: first day in forecast)")
    parser.add_argument("--input", help="Load forecast from .parquet instead of running live")
    parser.add_argument("--actuals", action="store_true", help="Fetch and overlay SMARD actuals")
    parser.add_argument("--members", action="store_true", help="Draw individual member traces")
    parser.add_argument("--forecast-days", type=int, default=2, help="Forecast horizon in days (live mode, default 2)")
    parser.add_argument("--output", help="Save plot to file (PNG/PDF/SVG)")
    parser.add_argument("--title", help="Custom plot title")
    args = parser.parse_args()

    df = load_forecast(args)

    if args.date:
        df = filter_to_date(df, args.date)
        plot_date = args.date
    else:
        plot_date = df.index[0].strftime("%Y-%m-%d")

    actuals = None
    if args.actuals:
        n_days = max(1, int(len(df) * 15 / 60 / 24) + 1)
        actuals = fetch_actuals(plot_date, n_days)
        if actuals is not None:
            actuals = actuals[actuals.index >= df.index[0]]
            actuals = actuals[actuals.index <= df.index[-1]]

    plot(df, actuals, args.title, args)


if __name__ == "__main__":
    main()
