"""
Held-out evaluation using IFS historical as input and ENTSO-E as target.

Inputs come from the Open-Meteo Historical Forecast API (ecmwf_ifs025), so
errors match live IFS inference rather than near-perfect reanalysis.

Usage:
    python scripts/evaluate_holdout.py --train-end 2023-12-31 --eval-start 2024-01-01 --eval-end 2024-12-31
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from solar_forecast.train import prepare_training_data, load_model, FEATURE_COLS
from solar_forecast.evaluate import evaluate, print_summary


def run_holdout_eval(
    eval_start: str,
    eval_end: str,
    capacity_mw: float = config.CAPACITY_MW,
    area: str = config.AREA,
    tz: str = config.TZ,
):
    print(f"Loading trained model...")
    model = load_model()

    print(f"Fetching held-out data ({eval_start} → {eval_end})...")
    X, y_actual = prepare_training_data(
        lat=config.LAT,
        lon=config.LON,
        start=eval_start,
        end=eval_end,
        installed_capacity_mw=capacity_mw,
        area=area,
        tz=tz,
        entsoe_api_key=os.environ.get("ENTSOE_API_KEY"),
    )

    print("Running predictions...")
    ci_pred = model.predict(X[FEATURE_COLS]).clip(0, 1.1)

    # Build a pseudo-ensemble: single deterministic prediction repeated
    # across 50 columns so evaluate() can compute CRPS.
    # This gives a lower bound on CRPS (zero spread = worst probabilistic score).
    forecasts_det = pd.DataFrame(
        np.tile(ci_pred[:, None], 50),
        index=X.index,
        columns=[f"member_{i:02d}" for i in range(50)],
    )

    # Convert CI predictions to MW for evaluation
    from solar_forecast.clearsky import get_clearsky
    from solar_forecast.fetch_actuals import fetch_ifs_historical
    ifs = fetch_ifs_historical(config.LAT, config.LON, eval_start, eval_end, tz=tz)
    clearsky = get_clearsky(config.LAT, config.LON, ifs.index)
    clearsky_mw = (clearsky["poa_clearsky"] / 1000.0) * capacity_mw

    forecasts_mw = forecasts_det.multiply(
        clearsky_mw.reindex(X.index), axis=0
    )
    actuals_mw = y_actual * clearsky_mw.reindex(X.index)

    print(f"\n{'='*50}")
    print(f"Held-out evaluation: {eval_start} → {eval_end}")
    print(f"  N daytime samples: {len(X)}")
    print(f"  (IFS historical inputs — errors match live NWP)")
    print(f"{'='*50}")
    metrics = evaluate(actuals_mw, forecasts_mw)
    print_summary(metrics)

    # Also print plain RMSE in MW
    ensemble_mean_mw = forecasts_mw.mean(axis=1)
    rmse_mw = float(np.sqrt(np.mean((ensemble_mean_mw - actuals_mw) ** 2)))
    mae_mw = float(np.mean(np.abs(ensemble_mean_mw - actuals_mw)))
    print(f"  RMSE:             {rmse_mw:.0f} MW")
    print(f"  MAE:              {mae_mw:.0f} MW")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-start", default="2024-01-01")
    parser.add_argument("--eval-end", default="2024-12-31")
    parser.add_argument("--capacity-mw", type=float, default=config.CAPACITY_MW)
    args = parser.parse_args()

    run_holdout_eval(
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        capacity_mw=args.capacity_mw,
    )
