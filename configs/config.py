"""
Configuration module for Mumbai Urban Heat Island PINN and Equity-Aware Optimization.
Defines geospatial bounds, feature mappings, physical constants, and paths.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DATA_DIR = DATA_DIR / "vector"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

# Ensure all result and data directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DATA_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Geospatial Bounding Box: Mumbai Metropolitan Region
# Coordinates in WGS84 (EPSG:4326)
MUMBAI_BOUNDS = {
    "min_lon": 72.75,
    "max_lon": 73.02,
    "min_lat": 18.88,
    "max_lat": 19.32,
}

# Projected Coordinate Reference System: UTM Zone 43N
CRS_WGS84 = "EPSG:4326"
CRS_UTM43N = "EPSG:32643"

# Features Definition
FEATURE_COLUMNS = [
    "ndvi",              # Normalized Difference Vegetation Index
    "ndbi",              # Normalized Difference Built-up Index
    "ndwi",              # Normalized Difference Water Index
    "mndwi",             # Modified Normalized Difference Water Index
    "albedo",            # Surface Shortwave Albedo (MCD43A3)
    "emissivity",        # Surface Emissivity (Landsat ST_EMIS)
    "elevation",         # SRTM Elevation (m)
    "slope",             # Terrain Slope (degrees)
    "dist_to_sea_km",    # Euclidean distance to Arabian Sea coast (km)
    "building_density",  # Open Buildings fractional density
    "building_height",   # Open Buildings height (m)
    "u_wind",            # ERA5 u-component of 10m wind (m/s)
    "v_wind",            # ERA5 v-component of 10m wind (m/s)
    "air_temp_c",        # ERA5 2m Air Temperature (°C)
    "solar_radiation",   # ERA5 Downward Solar Radiation (W/m²)
]

TARGET_COLUMN = "lst_c"  # Land Surface Temperature in °C

# Socioeconomic / Vulnerability Columns
EQUITY_COLUMNS = [
    "population_100m",    # WorldPop count per pixel
    "vulnerability_index" # Composite Urban Heat Vulnerability Index (0 to 1)
]

# Physical Default Constants for PINN
PHYSICS_CONFIG = {
    "alpha_init": 0.05,       # Initial effective thermal diffusivity
    "lambda_exch": 0.15,      # Convective exchange rate with ambient air
    "kd_wind_init": 8.0,      # Sea breeze inland decay length scale (km)
    "lambda_physics": 0.25,   # Weight of PDE residual in PINN loss
    "staged_epochs": {
        "stage1": 150,        # Data loss only (benchmark MLP)
        "stage2": 250,        # Add diffusion loss
        "stage3": 350         # Add sea-breeze advection + day/night switch
    },
    "batch_size": 2048,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5
}

# Scenario Simulation Parameters
SCENARIO_CONFIG = {
    "greening_ndvi_boost": 0.20,      # Simulated +0.20 NDVI increase in target pixels
    "cool_roof_albedo_boost": 0.20,   # Simulated +0.20 Albedo increase on roofs
    "cost_green_per_sqm": 450.0,      # Estimated INR per m² for urban tree/pocket park greening
    "cost_cool_roof_per_sqm": 120.0,  # Estimated INR per m² for high-albedo roof coating
}

# NSGA-II Optimization Parameters
OPTIMIZATION_CONFIG = {
    "pop_size": 80,
    "n_gen": 60,
    "budget_cap_inr_cr": 25.0,  # Budget cap in Crore INR (e.g., 25 Cr)
    "grid_res_m": 500           # Grid block resolution for decision variables
}
