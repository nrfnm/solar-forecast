"""Calibrate ensemble spread using RMSE/spread ratio correction."""

import os
import numpy as np
import pandas as pd
from typing import Optional

from solar_forecast.evaluate import evaluate, print_summary


def estimate_spread_factor(
    observations: pd.Series,
    forecasts: pd.DataFrame,
) -> float:
    """
    Estimate a multiplicative spread correction factor.

    Uses the RMSE/spread ratio: a well-calibrated ensemble has
    RMSE(ensemble_mean) ≈ mean(ensemble_std). If RMSE > spread,
    the ensemble is under-dispersed and needs widening (factor > 1).

    Parameters
    ----------
    observations : pd.Series
        Actual solar generation in MW.
    forecasts : pd.DataFrame
        Ensemble trajectories, shape (timesteps, n_members), values in MW.

    Returns
    -------
    float
        factor > 1 → spread too narrow (under-dispersed).
        factor < 1 → spread too wide (over-dispersed).
        factor = 1 → no correction needed.
    """
    obs, fct = observations.align(forecasts, join="inner", axis=0)
    obs_vals = obs.values
    fct_vals = fct.values

    ensemble_mean = fct_vals.mean(axis=1)
    ensemble_std = fct_vals.std(axis=1)

    rmse = float(np.sqrt(np.mean((ensemble_mean - obs_vals) ** 2)))
    mean_spread = float(np.mean(ensemble_std))

    if mean_spread < 1e-6:
        return 1.0

    return rmse / mean_spread


def apply_spread_correction(
    forecasts: pd.DataFrame,
    factor: float,
) -> pd.DataFrame:
    """
    Scale ensemble spread around the per-timestep median.

    calibrated = median + factor × (forecasts - median)

    Parameters
    ----------
    forecasts : pd.DataFrame
        Ensemble trajectories, shape (timesteps, n_members), values in MW.
    factor : float
        Spread correction factor from estimate_spread_factor().

    Returns
    -------
    pd.DataFrame
        Calibrated forecasts clipped to [0, ∞), same shape/index/columns.
    """
    median = forecasts.median(axis=1).values[:, None]
    calibrated = median + factor * (forecasts.values - median)
    return pd.DataFrame(
        np.clip(calibrated, 0, None),
        index=forecasts.index,
        columns=forecasts.columns,
    )


def calibrate(
    observations: pd.Series,
    forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    Estimate spread factor and return calibrated forecasts.

    Returns
    -------
    calibrated_forecasts : pd.DataFrame
    factor : float
    """
    factor = estimate_spread_factor(observations, forecasts)
    return apply_spread_correction(forecasts, factor), factor


if __name__ == "__main__":
    import argparse
    from solar_forecast.fetch_actuals import fetch_entsoe

    parser = argparse.ArgumentParser(description="Estimate spread correction factor from saved forecasts.")
    parser.add_argument("--forecast", required=True, help="Path to forecast .parquet file")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--area", default="DE")
    args = parser.parse_args()

    forecasts = pd.read_parquet(args.forecast)
    actuals = fetch_entsoe(args.area, args.start, args.end, api_key=os.environ.get("ENTSOE_API_KEY"))
    actuals = actuals.tz_convert(forecasts.index.tz).reindex(forecasts.index)

    factor = estimate_spread_factor(actuals, forecasts)
    print(f"Spread correction factor: {factor:.4f}")
    print(f"  {'Under-dispersed — widening spread' if factor > 1 else 'Over-dispersed — narrowing spread'}")

    calibrated, _ = calibrate(actuals, forecasts)
    print("\nBefore calibration:")
    print_summary(evaluate(actuals, forecasts))
    print("\nAfter calibration:")
    print_summary(evaluate(actuals, calibrated))
