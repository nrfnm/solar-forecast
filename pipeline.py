"""Daily forecast pipeline: fetch NWP → ensemble → (calibrate) → (evaluate) → save."""

import os
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import config
from solar_forecast.ensemble import forecast_country
from solar_forecast.fetch_actuals import fetch_entsoe
from solar_forecast.evaluate import evaluate, print_summary
from solar_forecast.calibrate import apply_spread_correction
from solar_forecast.train import load_quantile_models

_OUTPUT_DIR = Path(__file__).parent / "data" / "forecasts"


def run(
    run_date: Optional[str] = None,
    forecast_days: int = config.FORECAST_DAYS,
    entsoe_api_key: Optional[str] = None,
    evaluate_actuals: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run the daily forecast pipeline using the quantile ensemble model.

    Uses config.CENTROIDS for spatial aggregation if set, otherwise falls
    back to the single point config.LAT / config.LON.

    Parameters
    ----------
    run_date : str, optional
        ISO date string (YYYY-MM-DD). Defaults to today.
    evaluate_actuals : bool
        If True, fetch ENTSO-E actuals for the forecast window and print
        CRPS + PIT metrics. Only meaningful for past dates.
    save : bool
        Write trajectories to data/forecasts/<run_date>.parquet.

    Returns
    -------
    pd.DataFrame
        Shape (timesteps, 100) in MW — 100 quantile trajectories, calibrated.
    """
    # run_date is the TARGET date (day being forecast/evaluated).
    # Default to tomorrow so the daily production run always targets the next day.
    run_date = run_date or str(date.today() + timedelta(days=1))
    n_centroids = len(config.CENTROIDS) if config.CENTROIDS else 1
    print(f"=== Pipeline run: {run_date} ({n_centroids} centroid(s)) ===")

    print("Loading quantile models...")
    quantile_models = load_quantile_models()
    print(f"  {len(quantile_models)} quantile models loaded")

    print("Fetching NWP and running quantile ensemble...")
    trajectories = forecast_country(quantile_models=quantile_models, forecast_days=forecast_days)
    print(f"  Output: {trajectories.shape[0]} timesteps × {trajectories.shape[1]} members")

    print("  Applying calibration...")
    trajectories = apply_spread_correction(trajectories)

    # run_date IS the target date — select it directly from the NWP trajectories.
    tz = trajectories.index.tz
    target_start_ts = pd.Timestamp(run_date).tz_localize(tz)
    day_mask = (trajectories.index >= target_start_ts) & (
        trajectories.index < target_start_ts + pd.Timedelta(hours=24)
    )
    submission = trajectories[day_mask].clip(lower=0)
    print(
        f"  Submission format: {len(submission)} timesteps × {submission.shape[1]} scenarios "
        f"(target {target_start_ts.date()})"
    )
    #TODO replace evaluation block
    today_in_tz = pd.Timestamp.now(tz=tz).date()
    if evaluate_actuals and target_start_ts.date() >= today_in_tz:
        print("  Skipping evaluation — target day has not yet passed.")
        evaluate_actuals = False
    elif evaluate_actuals and target_start_ts.date() < today_in_tz:
        print(
            "  Skipping evaluation — live NWP has no data for past dates. "
            "Use scripts/backtest.py for historical evaluation."
        )
        evaluate_actuals = False

    if evaluate_actuals:
        token = entsoe_api_key or os.environ.get("ENTSOE_API_KEY")
        if not token:
            print("  Skipping evaluation — ENTSOE_API_KEY not set.")
        else:
            target_date_str = str(target_start_ts.date())
            print(f"  Fetching actuals for target day ({target_date_str})...")
            actuals = fetch_entsoe(config.AREA, target_date_str, target_date_str, api_key=token)
            actuals = actuals.tz_convert(submission.index.tz).reindex(submission.index)
            metrics = evaluate(actuals, submission)
            print("\n  Evaluation:")
            print_summary(metrics)

    if save:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _OUTPUT_DIR / f"{run_date}.parquet"
        submission.to_parquet(out_path)
        print(f"\n  Saved → {out_path}")

    return submission


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run daily solar forecast pipeline.")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--days", type=int, default=config.FORECAST_DAYS)
    parser.add_argument("--evaluate", action="store_true",
                        help="Fetch actuals and compute CRPS + PIT (past dates only)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    run(
        run_date=args.date,
        forecast_days=args.days,
        entsoe_api_key=os.environ.get("ENTSOE_API_KEY"),
        evaluate_actuals=args.evaluate,
        save=not args.no_save,
    )
