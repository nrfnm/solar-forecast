# ☀️ solar-forecast

Probabilistic day-ahead **solar power forecasting** for Germany, built for the
forecasting challenge at [energy-arena.org](https://energy-arena.org).

Developed as part of a seminar at Karlsruher Institut für Technologie
([KIT](https://kit.edu)), Institut für Industriebetriebslehre und Industrielle
Produktion ([IIP](https://www.iip.kit.edu)).

The model does not predict raw power directly. It predicts the **Clear-Sky Index**

```
CI = P_actual / P_clearsky
```

with a LightGBM model, then multiplies the predicted CI back by a pvlib
clear-sky power curve and the installed capacity. Factoring out the
deterministic solar geometry first removes most seasonal bias and leaves the ML
model to learn only the residual (cloud attenuation, temperature effects). The
trained model is applied across all **50 Open-Meteo NWP ensemble members** to
produce 50 calibrated MW trajectories.

```
Open-Meteo Ensemble (50 members)
        │
        ▼
  pvlib clear-sky power  ──►  CI features  ──►  LightGBM  ──►  50 MW trajectories
```

---

## 1. Installation

Requires **Python ≥ 3.12**.

```bash
git clone <this-repo> solar-forecast
cd solar-forecast

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"      # library + dev tools (pytest, jupyter, matplotlib)
```

## 2. Configuration

### API key — `.env`

Training targets and evaluation actuals come from the ENTSO-E Transparency
Platform, which needs a free API token.

```bash
cp .env.example .env
# edit .env and set ENTSOE_API_KEY
```

Register at [transparency.entsoe.eu](https://transparency.entsoe.eu) →
*My Account → Security Token*. The token is read from the `ENTSOE_API_KEY`
environment variable.

**This `.env` is required in the solar-forecast repo itself** — including when
the forecast is driven from the energy-arena submission repo (see §7). The
submission repo has its own separate configuration; it does **not** replace this
one. Keep an `.env` here as well.

The Open-Meteo NWP and IFS-historical APIs need **no key** on the free tier.

### Model settings — `config.py`

Model, physics, and geometry settings live in [`config.py`](config.py):

| Setting | Meaning |
|---|---|
| `LAT`, `LON`, `ALTITUDE` | Single fallback grid point (used when `CENTROIDS` is `None`) |
| `SURFACE_TILT`, `SURFACE_AZIMUTH` | Panel geometry for the pvlib POA calculation |
| `CAPACITY_MW` | Total installed DE solar capacity — converts CI to MW (**verify before use**) |
| `CENTROIDS` | Capacity-weighted grid points for spatial aggregation (see §6) |
| `LGBM_PARAMS` | LightGBM hyperparameters |
| `FORECAST_DAYS` | Forecast horizon in days |

Arena-specific runtime values (`area`, `challenge_id`, `api_base`, ENTSO-E key
for the live submission) are **not** set here — they are supplied by the
submission repo and passed into `pipeline.run()`. See §7.

---

## 3. End-to-end workflow

The modules run in a fixed order — each stage feeds the next. All commands are
run from the repo root with the venv active.

### Stage A — Train the model

Trains the global LightGBM CI model on ECMWF IFS-historical inputs (Open-Meteo
`historical-forecast-api`) and ENTSO-E solar generation targets, with
walk-forward CV. Writes `models/lgbm_ci.pkl`.

```bash
python -m solar_forecast.train --start 2020-01-01 --end 2023-12-31

# also train the 100 quantile CI models (writes models/lgbm_quantile.pkl):
python -m solar_forecast.train --start 2020-01-01 --end 2023-12-31 --quantile
```

Useful flags: `--area {DE,AT}`, `--capacity-mw`, `--no-cv`,
`--augments N` (noise-augmented IFS copies, default 4).

> A pre-trained `models/lgbm_ci.pkl` and `models/lgbm_quantile.pkl` ship with
> the repo, so you can skip this step to just run or evaluate the shipped model.

### Stage B — Daily forecast

Fetches the live 50-member NWP ensemble, runs the CI ensemble, applies
calibration, and writes `data/forecasts/<date>.parquet` (timesteps × 50
scenarios, in MW).

```bash
python pipeline.py --date 2026-07-16            # target date (default: tomorrow)
python pipeline.py --date 2026-07-16 --quantile # use the 100 quantile models
python pipeline.py --date 2026-07-16 --no-save  # print only, don't write
```

`scripts/run_daily.sh` wraps this for a local cron job (targets tomorrow).

### Stage C — Fit calibration (optional, run once)

EMOS/spread calibration fitted on a backtest window; writes
`models/calibration.json`, which `pipeline.py` then applies automatically.

```bash
# 1. produce a backtest parquet
python scripts/backtest.py --start 2024-01-01 --end 2024-06-30 --mode ensemble \
    --save data/processed/backtest.parquet

# 2. fit calibration against actuals for that window
python -m solar_forecast.calibrate --forecast data/processed/backtest.parquet \
    --start 2024-01-01 --end 2024-06-30
```

---

## 4. Reproducing the paper results

The evaluation snapshot used for the seminar paper is **committed** under
`data/eval_cache/` (Challenge 16, DE_LU, window 2026-06-07 … 2026-07-07, seven
participants). No network access or API key is needed to regenerate the metric
tables from it:

```bash
python scripts/eval_probabilistic_metrics.py
# → prints the CRPS / WIS / skill table from the frozen snapshot
```

Add `--out results.csv` to save the table. The frozen output tables the paper
cites live in [`paper/results_ch16_metrics.csv`](paper/results_ch16_metrics.csv)
and [`paper/results_ch16_window.csv`](paper/results_ch16_window.csv).

To **re-fetch** a fresh snapshot from the arena API instead of using the cache
(needs arena API access), use `scripts/fetch_arena_scores.py`
(`--challenge-id`, `--start`, `--end`, `--participants`).

### Diagnostic plots

```bash
python scripts/plot_pit_daytime.py       # PIT calibration histogram
python scripts/plot_underforecast_fan.py
python scripts/plot_forecast.py
```

### Notebooks

Exploratory analysis lives in [`notebooks/`](notebooks/): data exploration,
feature engineering, and calibration analysis. Launch with `jupyter lab`.

---

## 5. Tests

```bash
pytest tests/
```

Covers `clearsky`, `features`, `fetch_actuals`, and `fetch_nwp`.

---

## 6. Spatial clustering (grid points)

Instead of one grid point, the forecast aggregates over capacity-weighted
centroids `P_country = Σᵢ wᵢ · Pᵢ`. Installed-capacity data is sourced via the
**open-mastr** package from the Marktstammdatenregister (MaStR); missing GPS
coordinates are filled from a PLZ→coordinate map sourced from
[WZBSocialScienceCenter](https://github.com/WZBSocialScienceCenter/plz_geocoord).

Regenerate the centroids and paste the printed list into `config.CENTROIDS`:

```bash
python -m solar_forecast.capacity --k 15        # k-means centroids
python -m solar_forecast.capacity --k 15 --force-download  # re-download MaStR
```

Set `config.CENTROIDS = None` to fall back to the single `LAT`/`LON` point.

---

## 7. energy-arena integration

This repo is **platform-agnostic** — it returns 50 trajectories as a DataFrame
and knows nothing about the competition payload format. The integration lives in
[`custom_model.py`](custom_model.py), which adapts the trajectories into the
arena submission payload and passes the arena runtime config (area, challenge id,
API base, ENTSO-E key) into `pipeline.run()`.

Setup:

1. Clone the energy-arena
   [starter repo](https://github.com/zubasa107/energy-arena-participate)
   **side by side** with this repo (same parent directory) — `custom_model.py`
   resolves the sibling `../solar-forecast` on `sys.path`.
2. Set up the starter repo per its docs (its own API credentials and config).
3. **Copy [`custom_model.py`](custom_model.py) from this repo into the starter
   repo.** It is the bridge between the two repos and must live there to run.
4. Ensure this repo still has its own `.env` (see §2) — the pipeline reads it.
5. Automate on a VM (e.g. cron) as described in the starter repo.

To enable proper EMOS calibration later, archive the live NWP ensemble once per
day at submission time with [`run_collect_ensemble.sh`](run_collect_ensemble.sh)
(Open-Meteo only serves the current run, so the archive can only grow forward).

---

## 8. Repository layout

```
solar_forecast/
├── clearsky.py      # pvlib Ineichen clear-sky POA power
├── fetch_nwp.py     # Open-Meteo 50-member ensemble (cached)
├── fetch_actuals.py # IFS historical + ENTSO-E solar actuals
├── features.py      # CI, cyclic encodings, solar geometry, lags
├── train.py         # LightGBM + walk-forward CV  (CLI)
├── ensemble.py      # apply model across 50 members → MW trajectories
├── calibrate.py     # PIT/EMOS spread correction  (CLI)
├── evaluate.py      # CRPS, skill score, PIT       (CLI)
├── capacity.py      # MaStR k-means centroids       (CLI)
└── collect_ensemble_forecasts.py  # daily NWP archiver (CLI)

config.py            # coordinates, capacity, centroids, hyperparameters
pipeline.py          # daily run: fetch → forecast → calibrate → save (CLI)
custom_model.py      # adapter — copy into the energy-arena starter repo
scripts/             # backtest, evaluation, plotting, arena fetch
data/eval_cache/     # frozen arena snapshot for reproducing paper results
models/              # trained artifacts (lgbm_ci.pkl, lgbm_quantile.pkl)
```

## Data sources

| Data | Source | Use |
|---|---|---|
| NWP ensemble (50 members) | [Open-Meteo Ensemble API](https://open-meteo.com/en/docs/ensemble-api) | Inference inputs |
| Historical NWP | [IFS historical via Open-Meteo](https://open-meteo.com/en/docs/historical-forecast-api) (ecmwf_ifs025) | Training inputs |
| Generation actuals | [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | Training targets |
| Installed capacity | [Marktstammdatenregister](https://www.marktstammdatenregister.de) (via open-mastr) | Spatial weights |
| Clear-sky model | [pvlib (Ineichen)](https://pvlib-python.readthedocs.io) | Physics baseline |

## License

See [`LICENSE`](LICENSE).
