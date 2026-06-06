"""
Retrain LightGBM + refit EMOS using IFS historical forecasts instead of ERA5.

Replaces the ERA5-based training pipeline with deterministic ECMWF IFS historical
data (Open-Meteo Historical Forecast API, available since 2024-02-03). This closes
the training/inference distribution gap — the model trains on inputs with realistic
IFS errors rather than near-perfect reanalysis.

Data split:
    2024-02-03 ──────────── <backtest-start> │ <backtest-start> ──── <backtest-end>
              LightGBM training               │   EMOS backtest (held-out)

Usage:
    python scripts/retrain_with_ifs.py
    python scripts/retrain_with_ifs.py --train-start 2024-06-01
    python scripts/retrain_with_ifs.py --emos-only   # skip retrain, refit EMOS only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from solar_forecast.train import train
from solar_forecast.ensemble import backtest
from solar_forecast.calibrate import fit_emos, apply_emos
from solar_forecast.evaluate import evaluate, print_summary

IFS_START = "2024-02-03"   # earliest available ecmwf_ifs025 on historical-forecast-api


def main(
    train_start: str,
    backtest_start: str,
    backtest_end: str,
    emos_only: bool,
) -> None:
    if not emos_only:
        print(f"\n=== Step 1: Retrain LightGBM on IFS historical ({train_start} → {backtest_start}) ===")
        train(
            lat=config.LAT,
            lon=config.LON,
            start=train_start,
            end=backtest_start,
            installed_capacity_mw=config.CAPACITY_MW,
            use_ifs_historical=True,
        )
    else:
        print("Skipping LightGBM retrain (--emos-only).")

    print(f"\n=== Step 2: Backtest with IFS historical ({backtest_start} → {backtest_end}) ===")
    trajectories, actuals = backtest(
        start=backtest_start,
        end=backtest_end,
        use_ifs_historical=True,
        use_smard=True,
    )
    print(f"  Trajectories: {trajectories.shape}, actuals: {actuals.shape}")
    print(f"  Daytime actuals available: {actuals.notna().sum()} slots")

    print("\n=== Step 3: Refit EMOS ===")
    params = fit_emos(actuals, trajectories)
    print(f"  New params: {params}")

    print("\n=== Step 4: Evaluate calibrated vs raw ===")
    calibrated = apply_emos(trajectories, params)
    print("Raw backtest:")
    print_summary(evaluate(actuals, trajectories))
    print("Calibrated:")
    print_summary(evaluate(actuals, calibrated))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain with IFS historical data.")
    parser.add_argument(
        "--train-start", default=IFS_START,
        help=f"Start of LightGBM training window (default: {IFS_START})",
    )
    parser.add_argument(
        "--backtest-start", default="2025-09-01",
        help="Start of held-out EMOS backtest period",
    )
    parser.add_argument(
        "--backtest-end", default="2026-05-31",
        help="End of EMOS backtest period",
    )
    parser.add_argument(
        "--emos-only", action="store_true",
        help="Skip LightGBM retrain, only refit EMOS on the backtest window",
    )
    args = parser.parse_args()
    main(args.train_start, args.backtest_start, args.backtest_end, args.emos_only)
