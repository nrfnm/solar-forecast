"""Central configuration for solar-forecast."""

from pathlib import Path

# --- Paths ---
MODEL_PATH = Path(__file__).parent / "models" / "lgbm_ci.pkl"

# --- Location ---
# Single representative centroid for Germany (used until spatial aggregation is implemented)
LAT = 51.0
LON = 10.0
ALTITUDE = 500        # metres above sea level
SURFACE_TILT = 30     # panel tilt from horizontal, degrees
SURFACE_AZIMUTH = 180 # south-facing

# --- Market area ---
AREA = "DE"           # ENTSO-E area code
TZ = "Europe/Vienna"

# --- Capacity ---
# Total installed solar capacity in MW.
# Used to convert dimensionless CI predictions to MW: MW = CI × (poa / 1000) × CAPACITY_MW
# Source: Bundesnetzagentur / ENTSO-E ~65 GWp installed in Germany as of end-2023.
# Update this when retraining on more recent data.
CAPACITY_MW = 65_000.0

# --- Physics ---
STC_IRRADIANCE = 1000.0  # W/m² — irradiance at which panels produce rated capacity

# --- Forecast ---
FORECAST_DAYS = 7

# --- LightGBM hyperparameters ---
LGBM_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbose": -1,
}
