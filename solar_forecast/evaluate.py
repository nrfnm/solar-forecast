"""Evaluate probabilistic solar forecasts: CRPS, skill score, PIT."""

import os
import numpy as np
import pandas as pd
import properscoring as ps
from typing import Optional


def crps_score(
    observations: pd.Series,
    forecasts: pd.DataFrame,
) -> dict:
    """
    Compute CRPS and skill score vs a climatological baseline.

    The climatological baseline is the ensemble mean treated as a
    single deterministic forecast, representing the naive "always
    predict the average" strategy.

    Parameters
    ----------
    observations : pd.Series
        Actual solar generation in MW, aligned to forecasts index.
    forecasts : pd.DataFrame
        Ensemble trajectories, shape (timesteps, n_members), values in MW.

    Returns
    -------
    dict with keys: crps_mean, crps_climatology, crps_skill, n_samples.
    """
    obs, fct = observations.align(forecasts, join="inner", axis=0)
    obs_vals = obs.values
    fct_vals = fct.values

    crps_vals = ps.crps_ensemble(obs_vals, fct_vals)
    crps_mean = float(crps_vals.mean())

    clim_val = obs_vals.mean()
    clim = np.full_like(fct_vals, clim_val)
    crps_clim = float(ps.crps_ensemble(obs_vals, clim).mean())

    skill = float(1.0 - crps_mean / crps_clim) if crps_clim > 0 else 0.0

    return {
        "crps_mean": crps_mean,
        "crps_climatology": crps_clim,
        "crps_skill": skill,
        "n_samples": len(obs_vals),
    }


def pit_values(
    observations: pd.Series,
    forecasts: pd.DataFrame,
) -> np.ndarray:
    """
    Compute PIT (Probability Integral Transform) values.

    For each timestep, compute the fraction of ensemble members that
    fall below the observation. A well-calibrated ensemble produces
    uniformly distributed PIT values over [0, 1].

    Returns
    -------
    np.ndarray of floats in [0, 1].
    """
    obs, fct = observations.align(forecasts, join="inner", axis=0)
    obs_vals = obs.values
    fct_vals = fct.values
    return np.array([np.mean(fct_vals[i] < obs_vals[i]) for i in range(len(obs_vals))])


def evaluate(
    observations: pd.Series,
    forecasts: pd.DataFrame,
    daytime_only: bool = True,
) -> dict:
    """
    Full evaluation: CRPS, skill score, and PIT-based calibration diagnostics.

    Parameters
    ----------
    observations : pd.Series
        Actual solar generation in MW.
    forecasts : pd.DataFrame
        Ensemble trajectories, shape (timesteps, n_members), values in MW.
    daytime_only : bool
        Exclude rows where all ensemble members are zero (nighttime).

    Returns
    -------
    dict with all metrics. Key fields:
        crps_mean        — lower is better
        crps_skill       — positive = better than climatology
        pit_bias         — positive = model overforecasts
        dispersion_ratio — >1 overdispersed, <1 underdispersed, 1.0 = perfect
    """
    if daytime_only:
        is_day = forecasts.max(axis=1) > 0
        observations = observations[is_day]
        forecasts = forecasts[is_day]

    scores = crps_score(observations, forecasts)
    pit = pit_values(observations, forecasts)

    pit_std = float(np.std(pit))
    pit_mean = float(np.mean(pit))
    uniform_std = 1.0 / np.sqrt(12)

    return {
        **scores,
        "pit_mean": pit_mean,
        "pit_std": pit_std,
        "pit_bias": pit_mean - 0.5,
        "dispersion_ratio": pit_std / uniform_std,
    }


def print_summary(metrics: dict) -> None:
    print(f"  CRPS:             {metrics['crps_mean']:.2f} MW")
    print(f"  CRPS climatology: {metrics['crps_climatology']:.2f} MW")
    print(f"  CRPS skill:       {metrics['crps_skill']:+.3f}  (>0 = better than climatology)")
    print(f"  PIT bias:         {metrics['pit_bias']:+.3f}  (>0 = overforecast)")
    print(f"  Dispersion ratio: {metrics['dispersion_ratio']:.3f}  (1.0 = perfect)")
    print(f"  N samples:        {metrics['n_samples']}")


if __name__ == "__main__":
    import argparse
    from solar_forecast.fetch_actuals import fetch_entsoe

    parser = argparse.ArgumentParser(description="Evaluate saved forecasts against ENTSO-E actuals.")
    parser.add_argument("--forecast", required=True, help="Path to forecast .parquet file")
    parser.add_argument("--start", required=True, help="Actuals start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Actuals end date YYYY-MM-DD")
    parser.add_argument("--area", default="DE")
    args = parser.parse_args()

    forecasts = pd.read_parquet(args.forecast)
    actuals = fetch_entsoe(args.area, args.start, args.end, api_key=os.environ.get("ENTSOE_API_KEY"))
    actuals = actuals.tz_convert(forecasts.index.tz).reindex(forecasts.index)

    metrics = evaluate(actuals, forecasts)
    print_summary(metrics)
