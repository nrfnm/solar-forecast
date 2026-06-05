"""
Backtest the trained ensemble model against held-out ENTSO-E actuals.

Modes:
  ensemble  — CI model with member_id spread (default, optimistic CRPS)
  quantile  — 50 quantile models paired with 50 ERA5 members (proper spread)
  both      — run both and compare

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --start 2024-01-01 --end 2024-06-30 --mode quantile
    python scripts/backtest.py --mode both
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solar_forecast.ensemble import backtest
from solar_forecast.calibrate import fit_emos as fit_calibration, apply_emos as apply_calibration
from solar_forecast.evaluate import evaluate, print_summary


def run(start, end, mode, save):
    if mode in ("ensemble", "both"):
        print(f"\n--- CI ensemble ---")
        traj, actuals = backtest(start=start, end=end, use_quantile=False)
        traj.to_parquet(save.replace(".parquet", "_ensemble.parquet"))

        print("\nRaw:")
        metrics = evaluate(actuals, traj)
        print_summary(metrics)

        print("\nCalibrated:")
        params = fit_calibration(actuals, traj)
        traj_cal = apply_calibration(traj, params)
        print_summary(evaluate(actuals, traj_cal))

    if mode in ("quantile", "both"):
        print(f"\n--- Quantile ensemble ---")
        traj_q, actuals = backtest(start=start, end=end, use_quantile=True)
        traj_q.to_parquet(save.replace(".parquet", "_quantile.parquet"))

        print("\nRaw:")
        metrics_q = evaluate(actuals, traj_q)
        print_summary(metrics_q)

        print("\nCalibrated:")
        params_q = fit_calibration(actuals, traj_q)
        traj_q_cal = apply_calibration(traj_q, params_q)
        print_summary(evaluate(actuals, traj_q_cal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument("--mode", default="ensemble", choices=["ensemble", "quantile", "both"])
    parser.add_argument("--save", default="data/processed/backtest.parquet")
    args = parser.parse_args()

    run(args.start, args.end, args.mode, args.save)
