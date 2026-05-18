"""Calibrate ensemble forecasts: bias correction + spread correction."""

import json
import os
import numpy as np
import pandas as pd
import properscoring as ps
from pathlib import Path
from typing import Optional
from scipy.optimize import minimize
from scipy.stats import truncnorm

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


def fit_emos(
    observations: pd.Series,
    forecasts: pd.DataFrame,
    save_path: Optional[Path] = None,
    n_quant: int = 50,
) -> dict:
    """
    Fit EMOS parameters by minimising CRPS of Truncated Normal distribution.

    μ_EMOS = a + b · ens_mean
    σ_EMOS = sqrt(c + d · ens_var)

    Fit on daytime rows only (where ensemble max > 0). Parameters c and d are
    log-transformed during optimisation to enforce non-negativity.

    CRPS is approximated with n_quant evenly-spaced quantiles of the TruncNormal
    distribution (vectorised via truncnorm.ppf), then evaluated with ps.crps_ensemble.
    This avoids a closed-form formula while remaining deterministic and fast.

    Returns
    -------
    dict — {"emos": {"a": float, "b": float, "c": float, "d": float}}
    """
    is_day = forecasts.max(axis=1) > 0
    obs_s, fct_s = observations[is_day].align(forecasts[is_day], join="inner", axis=0)
    obs_vals = obs_s.values
    fct_vals = fct_s.values

    valid = ~np.isnan(obs_vals)
    obs_vals = obs_vals[valid]
    fct_vals = fct_vals[valid]

    if len(obs_vals) == 0:
        raise ValueError("No valid daytime observations to fit EMOS — check actuals data.")

    ens_mean = fct_vals.mean(axis=1)
    ens_var = fct_vals.var(axis=1)

    # Fixed quantile levels, shape (1, n_quant) — broadcast over timesteps
    u_fixed = np.linspace(0.01, 0.99, n_quant)[None, :]

    def objective(params):
        a, b, log_c, log_d = params
        c, d = np.exp(log_c), np.exp(log_d)
        mu = a + b * ens_mean          # shape (T,)
        sigma = np.sqrt(c + d * ens_var).clip(1e-3)
        a_std = (-mu / sigma)[:, None]  # lower truncation in std space, shape (T, 1)
        # truncnorm.ppf broadcasts (T, 1) a_std with (1, n_quant) u_fixed → (T, n_quant)
        samples = truncnorm.ppf(u_fixed, a=a_std, b=np.inf,
                                loc=mu[:, None], scale=sigma[:, None])
        return float(ps.crps_ensemble(obs_vals, samples).mean())

    x0 = [0.0, 1.0, 0.0, -2.3]
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-7})

    a, b, log_c, log_d = result.x
    params = {
        "emos": {
            "a": round(float(a), 6),
            "b": round(float(b), 6),
            "c": round(float(np.exp(log_c)), 6),
            "d": round(float(np.exp(log_d)), 6),
        }
    }
    save_params(params, save_path or config.CALIBRATION_PATH)
    return params


def apply_emos(
    forecasts: pd.DataFrame,
    params: Optional[dict] = None,
    load_path: Optional[Path] = None,
    n_samples: int = 100,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Replace ensemble trajectories with samples from EMOS-fitted Truncated Normal.

    Daytime rows: sample n_samples from TruncNormal(a + b·ens_mean, sqrt(c + d·ens_var), lower=0).
    Nighttime rows: keep as zeros.
    Falls back to apply_spread_correction() if params contain no "emos" key.
    """
    if params is None:
        params = load_params(load_path or config.CALIBRATION_PATH)

    if "emos" not in params:
        return apply_spread_correction(forecasts, params)

    p = params["emos"]
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]

    ens = forecasts.values
    ens_mean = ens.mean(axis=1)
    ens_var = ens.var(axis=1)

    mu = a + b * ens_mean
    sigma = np.sqrt(c + d * ens_var).clip(1e-3)
    is_day = ens.max(axis=1) > 0

    rng = np.random.default_rng(rng_seed)
    out = np.zeros((len(forecasts), n_samples))

    for t in np.where(is_day)[0]:
        a_std = (0.0 - mu[t]) / sigma[t]
        out[t] = truncnorm.rvs(
            a=a_std, b=np.inf,
            loc=mu[t], scale=sigma[t],
            size=n_samples,
            random_state=rng,
        )

    cols = (
        forecasts.columns
        if n_samples == len(forecasts.columns)
        else [f"member_{i:03d}" for i in range(n_samples)]
    )
    return pd.DataFrame(out, index=forecasts.index, columns=cols)


def save_params(params: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Calibration params saved → {path}")
    if "bias_factor" in params:
        print(f"  bias_factor:   {params['bias_factor']:.4f}")
    if "spread_factor" in params:
        print(f"  spread_factor: {params['spread_factor']:.4f}")
    if "emos" in params:
        p = params["emos"]
        print(f"  emos: a={p['a']:.4f}  b={p['b']:.4f}  c={p['c']:.4f}  d={p['d']:.4f}")


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
