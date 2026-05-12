"""Daily forecast pipeline: fetch NWP → ensemble → (calibrate) → (evaluate) → save."""

import os
import pandas as pd
from pathlib import Path
from datetime import date
from typing import Optional

import config
from solar_forecast.ensemble import forecast
from solar_forecast.fetch_actuals import fetch_entsoe
from solar_forecast.evaluate import evaluate, print_summary
from solar_forecast.calibrate import apply_spread_correction

_OUTPUT_DIR = Path(__file__).parent / "data" / "forecasts"


def run(
    run_date: Optional[str] = None,
    lat: float = config.LAT,
    lon: float = config.LON,
    installed_capacity_mw: float = config.CAPACITY_MW,
    forecast_days: int = config.FORECAST_DAYS,
    spread_factor: float = 1.0,
    entsoe_api_key: Optional[str] = None,
    evaluate_actuals: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run the daily forecast pipeline for a single location.

    Parameters
    ----------
    run_date : str, optional
        ISO date string (YYYY-MM-DD). Defaults to today.
    spread_factor : float
        Multiplicative spread correction. Use 1.0 (no correction) until
        calibrate.py has been run on a validation period to estimate it.
    evaluate_actuals : bool
        If True, fetch ENTSO-E actuals for the forecast window and print
        CRPS + PIT metrics. Only meaningful for past dates.
    save : bool
        Write trajectories to data/forecasts/<run_date>.parquet.

    Returns
    -------
    pd.DataFrame
        Shape (timesteps, n_members) in MW.
    """
    run_date = run_date or str(date.today())
    print(f"=== Pipeline run: {run_date} ===")

    print("Fetching NWP and running ensemble model...")
    trajectories = forecast(
        lat=lat,
        lon=lon,
        installed_capacity_mw=installed_capacity_mw,
        forecast_days=forecast_days,
    )
    print(f"  Output: {trajectories.shape[0]} timesteps × {trajectories.shape[1]} members")

    if spread_factor != 1.0:
        print(f"  Applying spread correction (factor={spread_factor:.3f})...")
        trajectories = apply_spread_correction(trajectories, spread_factor)

    if evaluate_actuals:
        token = entsoe_api_key or os.environ.get("ENTSOE_API_KEY")
        if not token:
            print("  Skipping evaluation — ENTSOE_API_KEY not set.")
        else:
            start = str(trajectories.index[0].date())
            end = str(trajectories.index[-1].date())
            print(f"  Fetching actuals ({start} → {end})...")
            actuals = fetch_entsoe(config.AREA, start, end, api_key=token)
            actuals = actuals.tz_convert(trajectories.index.tz).reindex(trajectories.index)
            metrics = evaluate(actuals, trajectories)
            print("\n  Evaluation:")
            print_summary(metrics)

    if save:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _OUTPUT_DIR / f"{run_date}.parquet"
        trajectories.to_parquet(out_path)
        print(f"\n  Saved → {out_path}")

    return trajectories


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run daily solar forecast pipeline.")
    parser.add_argument("--date", default=None, help="Run date YYYY-MM-DD (default: today)")
    parser.add_argument("--lat", type=float, default=config.LAT)
    parser.add_argument("--lon", type=float, default=config.LON)
    parser.add_argument("--capacity-mw", type=float, default=config.CAPACITY_MW)
    parser.add_argument("--days", type=int, default=config.FORECAST_DAYS)
    parser.add_argument("--spread-factor", type=float, default=1.0,
                        help="Spread correction factor (default: 1.0 = no correction)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Fetch actuals and compute CRPS + PIT (past dates only)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    run(
        run_date=args.date,
        lat=args.lat,
        lon=args.lon,
        installed_capacity_mw=args.capacity_mw,
        forecast_days=args.days,
        spread_factor=args.spread_factor,
        entsoe_api_key=os.environ.get("ENTSOE_API_KEY"),
        evaluate_actuals=args.evaluate,
        save=not args.no_save,
    )
