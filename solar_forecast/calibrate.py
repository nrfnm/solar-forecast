"""Calibrate ensemble forecasts: bias correction + spread correction."""

import json
import os
import numpy as np
import pandas as pd
import properscoring as ps
from pathlib import Path
from typing import Optional

import config
from solar_forecast.evaluate import evaluate, print_summary

def fit(
    observations: pd.Series,
    forecasts: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Estimate bias and spread correction factors from a held-out backtest.

    bias_factor   = mean(actuals) / mean(ensemble_mean)    — daytime only
    spread_factor = argmin CRPS over grid search [0.2, 3.0] — directly targets CRPS

    Returns
    -------
    dict with keys: bias_factor, spread_factor.
    """
    is_day = forecasts.max(axis=1) > 0
    obs = observations[is_day]
    fct = forecasts[is_day]

    obs_aligned, fct_aligned = obs.align(fct, join="inner", axis=0)
    obs_vals = obs_aligned.values
    fct_vals = fct_aligned.values

    valid = ~np.isnan(obs_vals)
    obs_vals = obs_vals[valid]
    fct_vals = fct_vals[valid]

    if len(obs_vals) == 0:
        raise ValueError("No valid (non-NaN) observations after alignment — check actuals data.")

    # Step 1: estimate bias from mean ratio
    ensemble_mean = fct_vals.mean(axis=1)
    bias_factor = float(obs_vals.mean() / ensemble_mean.mean()) if ensemble_mean.mean() > 0 else 1.0

    # Step 2: apply bias correction first, then estimate spread from corrected forecasts.
    # Computing spread on the biased distribution inflates the PIT std and gives a
    # wrong spread_factor — applying both corrections jointly then worsens dispersion.
    corrected_mean = ensemble_mean * bias_factor
    spread = fct_vals - ensemble_mean[:, None]
    fct_bias_corrected = corrected_mean[:, None] + spread

    # Grid search over spread_factor to directly minimise CRPS on the bias-corrected ensemble.
    # PIT std is a poor proxy for CRPS when the distribution is bounded below at zero because
    # expanding spread causes asymmetric clipping, shifting the effective mean and inflating CRPS.
    spread_factors = np.linspace(0.2, 3.0, 57)
    best_sf, best_crps = 1.0, float("inf")
    for sf in spread_factors:
        trial = np.clip(corrected_mean[:, None] + spread * sf, 0, None)
        c = float(ps.crps_ensemble(obs_vals, trial).mean())
        if c < best_crps:
            best_crps, best_sf = c, sf
    spread_factor = float(best_sf)

    params = {"bias_factor": round(bias_factor, 6), "spread_factor": round(spread_factor, 6)}
    save_params(params, save_path or config.CALIBRATION_PATH)
    return params


def apply_spread_correction(
    forecasts: pd.DataFrame,
    params: Optional[dict] = None,
    load_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Apply bias and spread correction to ensemble trajectories.

    calibrated = (ensemble_mean × bias_factor) + (spread × spread_factor)
    """
    if params is None:
        params = load_params(load_path or config.CALIBRATION_PATH)

    bias_factor = params["bias_factor"]
    spread_factor = params["spread_factor"]

    mean = forecasts.mean(axis=1).values[:, None]
    corrected_mean = mean * bias_factor
    spread = forecasts.values - mean
    calibrated = corrected_mean + spread * spread_factor
    return pd.DataFrame(
        np.clip(calibrated, 0, None),
        index=forecasts.index,
        columns=forecasts.columns,
    )


def save_params(params: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Calibration params saved → {path}")
    print(f"  bias_factor:   {params['bias_factor']:.4f}")
    print(f"  spread_factor: {params['spread_factor']:.4f}")


def load_params(path: Optional[Path] = None) -> dict:
    p = Path(path or config.CALIBRATION_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"Calibration file not found at {p}. Run calibrate.fit() first."
        )
    with open(p) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit calibration from saved backtest.")
    parser.add_argument("--forecast", required=True, help="Path to backtest .parquet file")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--area", default="DE")
    args = parser.parse_args()

    from solar_forecast.fetch_actuals import fetch_entsoe

    forecasts = pd.read_parquet(args.forecast)
    actuals = fetch_entsoe(args.area, args.start, args.end, api_key=os.environ.get("ENTSOE_API_KEY"))
    actuals = actuals.tz_convert(forecasts.index.tz).reindex(forecasts.index)

    params = fit(actuals, forecasts)

    calibrated = apply_spread_correction(forecasts, params)

    print("Before calibration:")
    print_summary(evaluate(actuals, forecasts))
    print("After calibration:")
    print_summary(evaluate(actuals, calibrated))
