# ☀️ solar-forecast

Probabilistic day-ahead **solar power forecasting** for Germany, built for the
forecasting challenge at [energy-arena.org](https://energy-arena.org).

Developed as part of a seminar at Karlsruher Institut für Technologie
([KIT](https://kit.edu)), Institut für Industriebetriebslehre und Industrielle
Produktion ([IIP](https://www.iip.kit.edu)).

The model does not predict raw power directly. It predicts the **Clear-Sky
Index** `k_PV` — the ratio of actual power to the clear-sky power a clear day
would yield:

```
k_PV = P_actual / P_clearsky
```

A LightGBM model predicts `k_PV`, which is then multiplied back by a pvlib
clear-sky power curve and the installed capacity to recover MW. Factoring out
the deterministic solar geometry first removes most seasonal bias and leaves the
ML model to learn only the residual (cloud attenuation, temperature effects).
The model is applied across all **50 Open-Meteo NWP ensemble members** to
produce 50 calibrated MW trajectories.

```
        Open-Meteo NWP ensemble  —  50 members
        (irradiance, cloud cover, temperature)
                        │
                        ▼
   features   (pvlib clear-sky power P_clearsky, k_PV, time, lags)
                        │
                        ▼
        LightGBM   ──►   predicted  k_PV     (per member)
                        │
                        ▼
   power  =  k_PV × P_clearsky × installed capacity   (back to MW)
                        │
                        ▼
          EMOS calibration (trunc-normal + ECC)
                        │
                        ▼
   50 calibrated MW trajectories   (hourly, day-ahead)
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
| `CAPACITY_MW` | Total installed DE solar capacity — converts `k_PV` to MW (**verify before use**) |
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

**Running from scratch** (another machine, no shipped artifacts). Follow these
steps in order; each links to its section below:

1. **Centroids** (Stage 0) — generate the capacity-weighted grid points and
   paste them into `config.CENTROIDS`. Do this **first**: training aggregates IFS
   across exactly these points.
2. **Train + fit EMOS** (Stage A) — one command,
   `python scripts/retrain_with_ifs.py`, retrains the `k_PV` model on IFS
   historical and refits EMOS → `models/lgbm_ci.pkl` + `models/calibration.json`.
3. **Daily forecast** (Stage B) — verify a single run produces trajectories.
4. **Automate** ([§7](#7-energy-arena-integration)) — wire the daily run into
   the energy-arena submission repo (cron on a VM).



### Stage 0 — Generate centroids

Derive the capacity-weighted grid points the forecast aggregates over
(`P_country = Σᵢ wᵢ · Pᵢ`) from Marktstammdatenregister (MaStR) installed-capacity
data, then paste the printed list into `config.CENTROIDS`. Do this **before**
training — Stage A aggregates IFS across exactly these points. See
[§6](#6-spatial-clustering-grid-points) for the data sources and details.

```bash
python -m solar_forecast.capacity --k 15                   # k-means centroids
python -m solar_forecast.capacity --k 15 --force-download  # re-download MaStR first
# → copy the printed list into config.CENTROIDS
```

Skip this to keep the committed default `config.CENTROIDS`, or set
`config.CENTROIDS = None` to fall back to the single `LAT`/`LON` point.

### Stage A — Train the model and fit EMOS

`scripts/retrain_with_ifs.py` runs the whole training + calibration pipeline in
one command, anchored on ECMWF IFS-historical inputs (Open-Meteo
`historical-forecast-api`) and ENTSO-E / SMARD solar actuals, with a sensible
train/backtest split. This is the recommended setup path:

```bash
python scripts/retrain_with_ifs.py
#  Step 1  train LightGBM   (2024-02-03 → backtest-start)  → models/lgbm_ci.pkl
#  Step 2  backtest         (backtest-start → backtest-end)
#  Step 3  refit EMOS       (fit_emos on the held-out window) → models/calibration.json
#  Step 4  print raw vs. calibrated backtest metrics
```

Flags: `--train-start` (default `2024-02-03`, earliest IFS date),
`--backtest-start` (default `2025-09-01`), `--backtest-end` (default
`2026-05-31`), `--emos-only` (skip the LightGBM retrain and only refit EMOS on
the backtest window — use this after tweaking calibration).

Training aggregates IFS across `config.CENTROIDS`, so those must be set first
(step 1 of the checklist above).

<details>
<summary>Lower-level entry points (finer control)</summary>

```bash
# train only  (add --quantile for the 100 quantile k_PV models → models/lgbm_quantile.pkl)
python -m solar_forecast.train --start 2024-02-03 --end 2025-09-01
#   useful flags: --area {DE,AT}, --capacity-mw, --no-cv, --augments N (default 4)

# backtest + fit EMOS only
python scripts/backtest.py --start 2025-09-01 --end 2026-05-31 --mode ensemble \
    --save data/processed/backtest.parquet
```
</details>


### Stage B — Daily forecast

Fetches the live 50-member NWP ensemble, runs the `k_PV` ensemble, applies
calibration, and writes `data/forecasts/<date>.parquet` (timesteps × 50
scenarios, in MW).

```bash
python pipeline.py --date 2026-07-16            # target date (default: tomorrow)
python pipeline.py --date 2026-07-16 --quantile # use the 100 quantile models
python pipeline.py --date 2026-07-16 --no-save  # print only, don't write
```

`scripts/run_daily.sh` wraps this for a local cron job (targets tomorrow).

### Stage C — Calibration details

Stage A already fits the calibration; this explains what it does. The pipeline
calibrates via **EMOS** (`apply_emos`): a truncated-normal distribution
`μ = a + b·ens_mean`, `σ² = c + d·ens_var` fitted by CRPS minimisation, with the
NWP member ranks mapped through it by Ensemble Copula Coupling (ECC) to keep each
trajectory temporally coherent. The fitted `a, b, c, d` live in
`models/calibration.json`, which `pipeline.py` loads and applies automatically.

To refit calibration only (e.g. after a new backtest window) without retraining
LightGBM, use `python scripts/retrain_with_ifs.py --emos-only`.

A simpler **bias + spread** correction (single spread factor, no per-timestep
`σ`) is also available via the `calibrate` CLI. `apply_emos` falls back to it
automatically when `calibration.json` carries no `emos` key:

```bash
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

This repo is **platform-agnostic** — it returns the trajectories as a DataFrame
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

### NWP Ensemble Crawler
To enable proper EMOS calibration later, archive the live NWP ensemble once per
day at submission time with [`run_collect_ensemble.sh`](run_collect_ensemble.sh), optimally with a cronjob
(Open-Meteo only serves the current run, so the archive can only grow forward).

---

## 8. Repository layout

```
solar_forecast/
├── clearsky.py      # pvlib Ineichen clear-sky POA power
├── fetch_nwp.py     # Open-Meteo 50-member ensemble (cached)
├── fetch_actuals.py # IFS historical + ENTSO-E solar actuals
├── features.py      # k_PV, cyclic encodings, solar geometry, lags
├── train.py         # LightGBM + walk-forward CV  (CLI)
├── ensemble.py      # apply model across 50 members → MW trajectories
├── calibrate.py     # EMOS (trunc-normal + ECC) + bias/spread fallback  (CLI)
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
