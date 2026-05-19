"""Central configuration for solar-forecast."""

from pathlib import Path

# --- Paths ---
MODEL_PATH = Path(__file__).parent / "models" / "lgbm_ci.pkl"
QUANTILE_MODEL_PATH = Path(__file__).parent / "models" / "lgbm_quantile.pkl"
CALIBRATION_PATH = Path(__file__).parent / "models" / "calibration.json"

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
# Source: Bundesnetzagentur Marktstammdatenregister, ~120 GWp installed in Germany as of mid-2026
# (99 GW end-2024, growing ~18 GW/year). Verify at bundesnetzagentur.de before retraining.
CAPACITY_MW = 123000.0

# --- Physics ---
STC_IRRADIANCE = 1000.0  # W/m² — irradiance at which panels produce rated capacity

# --- Forecast ---
FORECAST_DAYS = 2

# --- Spatial centroids ---
# Capacity-weighted grid points derived from MaStR k-means clustering.
# Generate with: python solar_forecast/capacity.py --k 5
# Then paste the printed output here as a list of dicts.
# Set to None to fall back to the single point (LAT, LON) above.
CENTROIDS: list[dict] | None = [
    {"lat": 48.3727, "lon": 8.7665,  "weight": 0.0761},
    {"lat": 53.4822, "lon": 11.2423, "weight": 0.0281},
    {"lat": 52.4754, "lon": 8.0128,  "weight": 0.0804},
    {"lat": 50.9910, "lon": 12.3929, "weight": 0.0548},
    {"lat": 48.7082, "lon": 12.3664, "weight": 0.1153},
    {"lat": 51.3123, "lon": 6.9151,  "weight": 0.0775},
    {"lat": 51.9337, "lon": 13.9104, "weight": 0.0673},
    {"lat": 49.8602, "lon": 7.1676,  "weight": 0.0475},
    {"lat": 48.3294, "lon": 10.5864, "weight": 0.0911},
    {"lat": 49.8430, "lon": 8.8287,  "weight": 0.0726},
    {"lat": 54.0169, "lon": 9.4878,  "weight": 0.0495},
    {"lat": 51.9404, "lon": 11.5152, "weight": 0.0506},
    {"lat": 51.7023, "lon": 9.5917,  "weight": 0.0534},
    {"lat": 53.4259, "lon": 13.0132, "weight": 0.0510},
    {"lat": 49.8022, "lon": 10.6873, "weight": 0.0848},
]

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
