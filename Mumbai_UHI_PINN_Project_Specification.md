# Project Specification Document
## Physics-Informed Neural Network for Urban Heat Prediction and Equity-Aware Cooling Recommendations — Mumbai

**Prepared for:** FF-180 Group 16, B.Tech CSE (AIML) — Prajwal Kadam, Atharva Kangralkar
**Guide:** Mrs. Grishma Bobhate
**Document purpose:** A single, self-contained reference that a new reader — a guide, an examiner, or a new team member — can use to understand the entire project without any other context: what is being built, why, with what data, through what pipeline, measured by what metrics, and within what timeline.
**Timeframe covered:** 1-month (4-week) reduced-scope execution plan.
**Important note on accuracy:** every dataset ID, band name, and numeric scale factor below was checked against Google Earth Engine's public catalog or the provider's own documentation at the time of writing. Catalog entries change over time (new versions, renamed bands, deprecations). **Before writing code against any ID in this document, re-check it at the live source URL given in the table.** Where a fact could not be independently re-verified in this session, it is explicitly marked "**UNVERIFIED — confirm before use**" rather than stated as fact.

---

## 1. Problem Statement

Mumbai experiences a strong Urban Heat Island (UHI) effect: built-up, low-vegetation areas run several degrees hotter than green or coastal areas, and the effect is worse for low-income, high-density wards with less tree cover and weaker access to cooling infrastructure. Standard heat-mapping approaches (pure machine learning on satellite temperature) fit the data well but do not respect the physics of heat transfer, do not generalize well to areas or times with sparse data, and rarely translate a heat map into an actionable, resource-constrained set of interventions.

## 2. Objective

Build a pipeline that:
1. Predicts Land Surface Temperature (LST) across Mumbai using a **Physics-Informed Neural Network (PINN)** that embeds an **advection-diffusion heat equation** — diffusion plus a directional, wind-driven **sea-breeze advection term** (not merely a surface-energy-balance sign constraint) — and a **day/night source-term switch**, so that Mumbai's coastal geography and diurnal cycle are represented explicitly rather than left for the network to infer from data alone. See Section 17 for why this specific formulation is the project's point of difference from comparable public projects.
2. Explains **why** locations are hot using SHAP (SHapley Additive exPlanations) attribution.
3. Simulates **what-if cooling scenarios** (adding vegetation, raising roof albedo) by editing input features and re-running the trained model.
4. Recommends **where to intervene** using NSGA-II (a multi-objective genetic optimizer) that balances cooling benefit, cost, and the vulnerability of the population affected — an equity-aware recommendation, not just a temperature-minimizing one.

## 3. What Success Looks Like (defined up front, so the project cannot be judged a failure just because the PINN doesn't "beat" the baseline on RMSE)

A successful outcome is **any one or more** of the following, and all should be attempted:
- The PINN achieves LST prediction error (RMSE) competitive with (not necessarily better than) the Random Forest/XGBoost baseline.
- The PINN generalizes better than the baseline to a held-out geographic area it was not trained on (a stronger claim than raw accuracy).
- The PINN's predictions are physically consistent: the residual of the heat-diffusion equation, evaluated at the network's own output, is small.
- The sea-breeze term measurably improves predictions near the coast relative to a version of the PINN without it (an ablation study).
- SHAP attribution and the NSGA-II Pareto front produce a defensible, equity-aware shortlist of intervention locations.

---

## 4. One-Month Scope: What Is In, What Is Cut

| Component | In this 1-month build | Reason |
|---|---|---|
| LST target, NDVI/NDBI/NDWI, distance to sea, elevation, WorldPop | **In** | Available in one Earth Engine export; no extra effort |
| RF / XGBoost baseline + SHAP | **In** | Fast, gives real results in week 1–2 |
| PINN — daytime, diffusion + sea-breeze term | **In**, built in 3 stages, stop at whichever stage is stable | This is the project's novelty; everything else is secondary |
| Day/night switch (night regime via MODIS) | **In only if the daytime PINN is stable by week 3** | Otherwise documented as designed but not fully trained — be explicit about this in the report |
| Scenario simulation (NDVI/albedo edits) | **In, simple version** | Needed to feed the optimizer |
| NSGA-II optimization | **In, at ward or grid-block level** | Small optimization problem, tractable with Pymoo in a day |
| Land-cover classification | **Cut.** Use ESA WorldCover directly | A trained CNN/RF classifier is not needed to hit the objective |
| ECOSTRESS | **Cut** | Coverage over Mumbai is unconfirmed and integration cost is high for the time available |
| UT-GLOBUS | **Cut** | Source availability unverified (see Section 6, row 16); use Open Buildings Temporal or GHSL instead |
| Per-scene ERA5 extraction | **Simplified** to a seasonal composite, or ERA5 pulled only for the specific scene dates actually used | Full per-scene time series is a "future work" item |
| CPCB validation | **In, as a light check only**, not a primary metric | Station data quality/coverage for Mumbai must be verified first (Section 6, row 13) |
| Streamlit dashboard | **Optional / only if time remains** | Static figures and a notebook are sufficient for the demo |

---

## 5. End-to-End Data and Model Flow

```
                         ┌─────────────────────────────┐
                         │   GOOGLE EARTH ENGINE        │
                         │  (raster acquisition layer)  │
                         └───────────────┬───────────────┘
                                         │
        ┌──────────────┬────────────────┼───────────────┬───────────────┐
        ▼              ▼                ▼               ▼               ▼
   Landsat 8/9     Sentinel-2      ERA5-Land Hourly   MODIS (LST-night, WorldPop /
   (LST target,    (NDVI, NDBI,    (wind u/v, air     albedo)          building /
   emissivity)     NDWI, MNDWI)    temp, radiation)                    terrain / water
        │              │                │               │               │
        └──────────────┴────────────────┴───────────────┴───────────────┘
                                         │
                                         ▼
                      ┌───────────────────────────────────────┐
                      │   30 m CO-REGISTERED FEATURE STACK     │
                      │   (single multi-band GeoTIFF, UTM 43N) │
                      └───────────────────┬─────────────────────┘
                                         │
                     ┌───────────────────┼────────────────────┐
                     ▼                                        ▼
        ┌─────────────────────────┐               ┌─────────────────────────┐
        │  BASELINE MODELS         │               │  PHYSICS-INFORMED NN     │
        │  Random Forest, XGBoost  │               │  (x, y, t, features)→LST │
        │  → LST prediction        │               │  loss = data + physics   │
        └────────────┬─────────────┘               │  residual (heat eqn +    │
                     │                              │  sea-breeze term)        │
                     │                              └────────────┬─────────────┘
                     └───────────────────┬───────────────────────┘
                                         ▼
                          ┌───────────────────────────┐
                          │   SHAP EXPLAINABILITY      │
                          │   (driver ranking, both    │
                          │    baseline and PINN)      │
                          └─────────────┬───────────────┘
                                         ▼
                      ┌────────────────────────────────────┐
                      │  SCENARIO SIMULATION                │
                      │  Edit NDVI (greening) / albedo      │
                      │  (cool roofs) → re-run trained model│
                      │  → ΔLST maps                        │
                      └─────────────────┬────────────────────┘
                                         ▼
        ┌─────────────────────────────────────────────────────────┐
        │  NSGA-II MULTI-OBJECTIVE OPTIMIZATION                    │
        │  Objectives: maximize cooling gain, minimize cost,       │
        │  maximize vulnerability-weighted benefit                 │
        │  Inputs: ΔLST per ward/grid-block, cost per intervention,│
        │  WorldPop + Census vulnerability weight                  │
        │  Output: Pareto front of recommended interventions       │
        └─────────────────────────┬─────────────────────────────────┘
                                         ▼
                      ┌────────────────────────────────────┐
                      │  VALIDATION                         │
                      │  - vs CPCB stations (if usable)     │
                      │  - heat-equation residual check     │
                      │  - held-out spatial area check      │
                      └─────────────────┬────────────────────┘
                                         ▼
                      ┌────────────────────────────────────┐
                      │  REPORT + DEMO (dashboard optional) │
                      └────────────────────────────────────┘
```

---

## 6. Complete Dataset List

Every row states: what it contains, exactly how and why it is used, its native format, and its access link. GEE = accessible inside Google Earth Engine (free for this use, see Section 7).

| # | Dataset | Provider / GEE ID | What it contains | Spatial / temporal resolution | Role — how & why it's used | Format as obtained | Link |
|---|---|---|---|---|---|---|---|
| 1 | **Landsat 8 Collection 2 Level-2** | USGS/NASA — `LANDSAT/LC08/C02/T1_L2` | `ST_B10`: surface temperature, DN → Kelvin via `DN × 0.00341802 + 149.0`. `ST_EMIS`: emissivity (`DN × 0.0001`). `ST_QA`: LST uncertainty in K (`DN × 0.01`). `QA_PIXEL`: bitmask for cloud/shadow/cirrus/snow. `SR_B1–B7`: surface reflectance (`DN × 0.0000275 − 0.2`). | 30 m; revisit ≈16 days; daytime overpass ≈10:30 local time; data from 2013–present | **This is your target variable.** LST is what the baseline and PINN both predict. Emissivity is a physics-loss input (affects the surface energy balance). Reflectance bands support any spectral index you compute directly from Landsat. | GeoTIFF (via GEE export) | https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2 |
| 2 | **Landsat 9 Collection 2 Level-2** | USGS/NASA — `LANDSAT/LC09/C02/T1_L2` | Identical band structure to row 1. | 30 m; operational from late 2021; combined with Landsat 8 gives ≈8-day effective revisit | **Doubles the number of usable clear-sky LST scenes.** Given Mumbai's monsoon cloud cover, this materially increases how much training data the PINN has. | GeoTIFF | https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2 |
| 3 | **Sentinel-2 Level-2A (Harmonized)** | ESA/Copernicus — `COPERNICUS/S2_SR_HARMONIZED` | Reflectance bands B2 (blue), B3 (green), B4 (red), B8 (NIR), B11/B12 (SWIR), scaled by 10000; `SCL` scene-classification band (values include 3=cloud shadow, 8/9=cloud medium/high probability, 10=thin cirrus, 11=snow/ice) used for masking. | 10 m (B2/B3/B4/B8), 20 m (B11/B12); revisit ≈5 days; data from 2017-03-28, but **full global coverage was not reached until later in 2018 — prefer 2019 onward for reliability** | Computes **NDVI, NDBI, NDWI, MNDWI** (formulas in Section 8). These are model input features **and** the direct "dial" used in the greening scenario (raise NDVI on a pixel and re-predict). | GeoTIFF (via GEE export) | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED |
| 4 | **ERA5-Land Hourly (reanalysis)** | ECMWF/Copernicus — `ECMWF/ERA5_LAND/HOURLY` | `temperature_2m` (K), `dewpoint_temperature_2m` (K), `u_component_of_wind_10m`, `v_component_of_wind_10m` (m/s), `surface_solar_radiation_downwards_hourly` (J/m², accumulated), `surface_pressure`. | ≈9–11 km grid (native resolution 11,132 m); hourly; 1950–near present (few months' lag) | **Meteorological forcing for the physics loss.** Wind (u, v) is the direct input to the **sea-breeze term** you are adding to the heat equation. Air temperature is used in the boundary/exchange term. Because ERA5-Land is coarse relative to your 30 m grid, treat it as (nearly) constant across the city per time step and vary it in time, not in space. | Raster (via GEE) | https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY |
| 5 | **MODIS Aqua Daily LST (MYD11A1)** | NASA — `MODIS/061/MYD11A1` | `LST_Day_1km`, `LST_Night_1km` (Kelvin, `DN × 0.02`); associated QC bands. | 1 km; daily; from 2002-07-04 | **Only source of night-time LST in this pipeline** (Landsat and Sentinel-2 are daytime-only). Used as a coarse constraint/second regime for the day/night source-term switch in the PINN — not a per-pixel ground truth at 30 m, because of the 1 km/30 m resolution mismatch. | GeoTIFF (via GEE export) | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD11A1 |
| 6 | **MODIS Albedo (MCD43A3)** | NASA — `MODIS/061/MCD43A3` | `Albedo_WSA_shortwave` (white-sky shortwave albedo, `DN × 0.001`). | 500 m; daily; from 2000-02-16 | Surface-energy-balance input (absorbed solar radiation depends on albedo). This is also the **"dial" for the cool-roof scenario**: raise albedo on built-up pixels and re-predict LST. | GeoTIFF | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A3 |
| 7 | **ESA WorldCover v200** | ESA — `ESA/WorldCover/v200` | 11-class land-cover map (tree cover, shrubland, grassland, cropland, built-up, bare/sparse vegetation, snow/ice, permanent water, herbaceous wetland, mangroves, moss/lichen). | 10 m; single epoch, year 2021 | Land-cover class as a **direct model feature**, and as the **feasibility mask** for interventions (you cannot "green" a water pixel or a road). | GeoTIFF (via GEE export, resample to grid using mode) | https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200 |
| 8 | **JRC Global Surface Water** | EC Joint Research Centre — `JRC/GSW1_4/GlobalSurfaceWater` | `occurrence` band: % of time (1984–2021) a pixel was classed as water, 0–100. | 30 m; static (multi-year composite) | Used to build a **sea/water mask**, from which **distance to the coast** is computed (e.g., via a distance transform). Distance to sea is the key spatial variable that scales the sea-breeze term's strength in the PINN. **Caveat: this mask also includes rivers, creeks (e.g. Mithi River) and lakes — clip to a hand-drawn coastline polygon if you need distance to *open sea* specifically, not to any water body.** | GeoTIFF | https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater |
| 9 | **SRTM Digital Elevation Model** | USGS/NASA — `USGS/SRTMGL1_003` | `elevation` band, metres. | 30 m; static (mission flown in 2000) | Elevation and (derived) slope as model features. Effect is expected to be small for Mumbai except near hills (e.g. Malabar Hill, Powai) and may weakly modulate the sea breeze. | GeoTIFF | https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003 |
| 10 | **WorldPop Population (100 m, unconstrained)** | WorldPop / GEE — `WorldPop/GP/100m/pop` | `population` band: estimated population per 100 m pixel, filtered by `country` and `year`. | 100 m; annual estimates | **Not a model input.** Used only at the optimization stage, as (or as part of) the **vulnerability weight**: cooling benefit is multiplied by exposed population to make the NSGA-II objective equity-aware. | GeoTIFF | https://developers.google.com/earth-engine/datasets/catalog/WorldPop_GP_100m_pop |
| 11 | **Open Buildings 2.5D Temporal V1** | Google Research — `GOOGLE/Research/open-buildings-temporal/v1` | `building_presence`, `building_fractional_count` (a per-pixel building-density index), `building_height` (metres, range 0–100). | Pixel size 4 m; annual layers 2016–2023; coverage: Africa, South Asia, South-East Asia, Latin America, Caribbean (India falls under South Asia, but **confirm Mumbai specifically has non-empty coverage before relying on it** — coverage within a covered region is not guaranteed to be complete everywhere) | Building height and density features (urban form), and a cross-check / fallback for UT-GLOBUS (row 16, whose availability is unverified). Also an intervention-feasibility input (dense built pixels are candidates for roof-based interventions, not ground-level greening). | GeoTIFF/raster (via GEE), or direct Cloud Storage download for non-GEE users | https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1 |
| 12 | **OpenStreetMap (roads, parks, water, land use)** | OSM contributors | Vector layers: road centrelines (by class), park/green-space polygons, water body polygons, general land-use polygons. | Vector; resolution effectively unlimited; continuously and irregularly updated by volunteers, so completeness varies by area | Road-density and distance-to-green-space **features**, and a secondary source for the **intervention feasibility map** (e.g. don't recommend greening in the middle of a road). | Vector: `.osm.pbf` (raw) or GeoJSON/GeoPackage extracted via a tool | Data: https://download.geofabrik.de/asia/india.html · API/library: https://www.openstreetmap.org (or the `osmnx` Python package) |
| 13 | **CPCB Continuous Ambient Air Quality Monitoring Stations (CAAQMS)** | Central Pollution Control Board | Hourly pollutant concentrations; a subset of stations also report meteorological parameters including ambient temperature, humidity, and wind — **which stations report temperature must be checked individually**. | Point stations, hourly | **Independent validation only — never used for training.** Predicted LST/near-surface temperature at station coordinates is compared against the station's recorded value. **Verified caveat: historically, availability has been unreliable** — a 2019 report found only 3 of 16 Mumbai Metropolitan Region stations provided real-time public data; by 2026 Mumbai has an expanded network (28 CAAQMS across MPCB, IITM and BMC per a 2026 news report), but per-station data completeness and whether each one logs temperature must still be checked before you rely on it for validation. | CSV (portal download; there are per-request date-range limits) | https://airquality.cpcb.gov.in/ccr/ (historical bulk access: https://app.cpcbccr.com/ccr/#/login) |
| 14 | **Census of India 2011, ward/town-level tables** | Government of India, Office of the Registrar General & Census Commissioner | Population, households, and related tables at sub-ward ("town") granularity, joinable up to BMC ward level. | Tabular; single epoch, **2011 — over a decade old at project time** | Base socio-economic layer for the vulnerability weight in NSGA-II, alongside WorldPop. Its age is a genuine limitation — state this explicitly in your report; do not present the vulnerability numbers as current. | CSV/Excel table | https://censusindia.gov.in/ |
| 15 | **Mumbai ward boundaries** | No single confirmed authoritative open-GIS source found in this session | Polygon boundaries for BMC's administrative wards (commonly referenced as A, B, C, …, ward names), used to aggregate pixel-level results to a policy-relevant unit. | Vector polygons | Spatial aggregation unit for the equity-weighted optimization and for reporting recommendations at a scale a municipal reader recognizes. | Shapefile/GeoJSON, **if a reliable one can be found** | **UNVERIFIED — confirm before use.** The BMC Election Commission site (https://www.bmcelections.com/bmc-new-ward-maps/) hosts ward maps but these may not be machine-readable GIS files. A community-maintained, unofficial GeoJSON derived from Census 2011 ward-level data exists at https://github.com/mickeykedia/Mumbai-Population-Map — treat it as a starting point only, verify it against a current BMC source, and if you cannot verify current boundaries, **fall back to aggregating by regular grid blocks instead of wards** and say so in the report. |
| 16 | **UT-GLOBUS (building height/urban-canopy dataset)** | Referenced in the original proposal | Building footprints with heights, and urban-canopy parameters, for a large number of global cities. | Vector/gridded (exact packaging not re-confirmed) | Was proposed as a building-height source; **cut from the 1-month scope** (Section 4) in favour of Open Buildings Temporal (row 11) or GHSL (row 18). | Unconfirmed | **UNVERIFIED — confirm before use.** Search "UT-GLOBUS dataset download" directly and re-check its current hosting location, licence, and Mumbai coverage before using it. Do not rely on any link for this dataset stated in earlier drafts of this project's chat history without re-checking it yourself. |
| 17 | **ECOSTRESS LST/Emissivity** | NASA JPL, via NASA AppEEARS | LST and emissivity from the ISS-mounted ECOSTRESS instrument, at irregular acquisition times (day and night, since the ISS is not sun-synchronous). | ≈70 m effective resolution (tiled L2T product); irregular revisit (not a fixed daily/16-day cadence) | **Cut from the 1-month scope.** If time allows in a future phase, adds LST samples at times of day Landsat cannot see, which would strengthen the day/night modelling. Coverage over Mumbai on any given date must be checked in AppEEARS before assuming it is usable. | GeoTIFF (request this format in AppEEARS to avoid native HDF5/NetCDF) | https://appeears.earthdatacloud.nasa.gov/ (requires a free NASA Earthdata login) |
| 18 | **GHSL Built-up Surface / Height / Population** | EC Joint Research Centre | Built-up surface fraction, building height, and population grids. | 100 m (typical) | Fallback for building height/density (row 11) and for population (row 10) if those prove insufficient. | GeoTIFF | **UNVERIFIED exact asset ID — search "GHSL Earth Engine" and confirm the current catalog ID (e.g. under `JRC/GHSL/...`) before use**; general catalog: https://developers.google.com/earth-engine/datasets/catalog |

---

## 7. Cost

Earth Engine, all listed satellite/reanalysis datasets, OpenStreetMap, QGIS, Python, PyTorch, XGBoost, SHAP, and Pymoo/DEAP (for NSGA-II) are free. Earth Engine specifically:
- Free for **noncommercial** use (register the required Google Cloud project as noncommercial; no payment details needed for this route).
- Comes with a monthly compute quota under a **Community Tier** (**verify the exact current quota figure at https://earthengine.google.com before quoting a number in your report**, as these limits are administratively set and can change). Exceeding it slows requests; it does not incur charges or stop your project.
- A single 30 m raster export over Mumbai plus a pixel-sample CSV is a small workload relative to typical Earth Engine usage.

Real costs, if any, are: (a) compute hardware for training the PINN — a free-tier Google Colab or Kaggle GPU notebook is enough at this project's scale; (b) an Earthdata login for ECOSTRESS/AppEEARS (free registration), only relevant if that dataset is reinstated in a later phase.

---

## 8. Feature Engineering — Exact Definitions

All indices are computed from Sentinel-2 surface reflectance bands (row 3) after scaling by 0.0001:

| Feature | Formula | Purpose |
|---|---|---|
| NDVI (vegetation) | (NIR − Red) / (NIR + Red) = (B8 − B4) / (B8 + B4) | Vegetation cover; also the greening-scenario lever |
| NDBI (built-up) | (SWIR − NIR) / (SWIR + NIR) = (B11 − B8) / (B11 + B8) | Built-up surface indicator |
| NDWI (McFeeters, water) | (Green − NIR) / (Green + NIR) = (B3 − B8) / (B3 + B8) | Open-water detection |
| MNDWI (modified NDWI) | (Green − SWIR) / (Green + SWIR) = (B3 − B11) / (B3 + B11) | Water detection, more robust in built-up areas than NDWI |
| Distance to sea | Euclidean distance transform from the nearest sea-classified pixel (row 8, with the coastline-clipping caveat noted) | Scales the sea-breeze term |
| Building density | `building_fractional_count`, row 11 | Urban form input |
| Building height | `building_height` (m), row 11 | Urban form input; proxy for heat storage/roughness |
| Elevation, slope | From SRTM (row 9); slope computed from elevation via a standard terrain function | Terrain input |
| Land-cover class | ESA WorldCover class code (row 7) | Categorical input, and feasibility mask |

---

## 9. Modelling Specification

### 9.1 Baseline: Random Forest and XGBoost
- **Input:** the feature table (Section 8) plus emissivity, ERA5 variables, elevation, distance to sea — one row per training pixel (or per pixel-scene, if using multiple dates).
- **Output:** predicted LST (°C).
- **Train/test split:** **spatial**, not random — hold out entire contiguous areas of the city, not scattered random pixels. Random-pixel splits overstate accuracy because neighbouring pixels are highly correlated.
- **Purpose:** an accuracy benchmark for the PINN, and a source of a first SHAP attribution.

### 9.2 Physics-Informed Neural Network (PINN)
- **Input:** (x, y, t, feature vector) per point.
- **Output:** predicted LST (°C or K).
- **Loss function:** `L = L_data + λ_physics · L_physics` where:
  - `L_data` = mean squared error between the network's prediction and the observed Landsat/MODIS LST at the corresponding point.
  - `L_physics` = mean squared residual of the governing PDE, evaluated using automatic differentiation on the network's own output. An illustrative (not final — you must justify your exact form in the report) governing equation, written as an **advection-diffusion equation** rather than a pure energy-balance constraint, is:

    `∂T/∂t = α·∇²T − v(x,y,t)·∇T + S(x, y, t) − λ_exch·(T − T_air)`

    where `α` is a thermal diffusivity constant, `v(x,y,t) = (u, w)` is the local near-surface wind vector taken from ERA5 (row 4), scaled down from open-sea strength as a function of distance to coast `d` (row 8/Section 8) so the advective cooling is strongest near the shoreline and decays inland — **this is the sea-breeze term, expressed as directional advection of the temperature field by wind, not as a distance-weighted subtraction of sea temperature**. `S(x,y,t)` is the heat source term (solar-driven, switches to a much weaker or negative value at night — this is the "day/night switch"), `T_air` comes from ERA5, and `λ_exch` is a convective-exchange coefficient. **This exact functional form is a design decision your team must make and justify — do not present it as an established equation from the literature without citing where each term came from or stating that you derived it.**
  - **Learned vs. fixed physical parameters (recommended upgrade, cheap to add):** rather than hand-setting `α` and the wind-scaling function's coefficients as constants, make them **trainable parameters** optimized jointly with the network's weights (an "inverse PINN" / PDE-parameter-discovery setup — see Section 17, row 8, for where this idea comes from). This is a small code change (a few extra `nn.Parameter`s) but is a genuine methodological step up from fixing every physical constant by hand, and gives you an extra, defensible result to report: the data-fitted value of Mumbai's effective thermal diffusivity and sea-breeze strength, compared against literature ranges if any are available.
- **Staged training** (do this in order; stop at whichever stage is stable given the time budget):
  1. Stage 1 — plain feed-forward network on (x, y, t, features) → LST, data loss only (`λ_physics = 0`). This confirms the network can fit the data at all.
  2. Stage 2 — add the plain diffusion residual (`α·∇²T` term only) to the loss.
  3. Stage 3 — add the sea-breeze term `B(d,t)·(T − T_sea)` and the day/night source switch.
- **Ablation studies required for the report:** PINN without physics loss vs. with; PINN without the sea-breeze term vs. with — report both accuracy (RMSE) and physical-residual metrics for each variant, so the value of each added term is demonstrated, not assumed.

### 9.3 SHAP Explainability
Run SHAP (e.g., `TreeExplainer` for the RF/XGBoost baseline; a suitable gradient- or kernel-based explainer for the PINN) on both models. Produce: a mean-|SHAP value| ranking of features, a beeswarm/summary plot, and dependence plots for the top 2–3 features (e.g., NDVI vs. SHAP value, coloured by building density).

### 9.4 Scenario Simulation
For a chosen set of pixels (e.g., all built-up pixels in a ward), programmatically raise NDVI (simulating tree planting) or albedo (simulating cool roofs) by a stated amount, re-run the trained model (PINN or baseline) unchanged, and record ΔLST = LST_after − LST_before. This produces the cooling-gain input the optimizer needs.

### 9.5 NSGA-II Multi-Objective Optimization
- **Decision variables:** which grid-blocks/wards receive which intervention type (e.g., binary or fractional allocation of greening/cool-roof budget per unit).
- **Objectives** (typically formulated as: maximize objectives 1 and 3, minimize objective 2):
  1. Total cooling gain (sum of ΔLST × affected area, from Section 9.4).
  2. Total cost (a per-unit-area cost you define for each intervention type, stated explicitly and referenced, not invented without basis).
  3. Vulnerability-weighted benefit = cooling gain × exposed population (WorldPop, row 10) × a vulnerability multiplier (from Census slum/ward data, row 14, or a stated proxy if that data cannot be obtained).
- **Constraints:** a total budget cap; the feasibility mask from Section 6 rows 7/12 (cannot intervene on water, roads, etc.).
- **Output:** a Pareto front of non-dominated solutions — present it as a scatter of cooling gain vs. cost, with point size or colour showing the equity weight, plus a short table of 3–5 representative solutions along the front (e.g., "lowest-cost," "highest-equity," "balanced").
- **Tooling:** the Python `pymoo` or `DEAP` libraries implement NSGA-II directly; this does not need to be written from scratch.

### 9.6 Validation
- **CPCB comparison** (row 13): compare predicted near-surface temperature at each usable station's coordinates against its recorded values, for stations confirmed to report temperature. State plainly that this compares surface (satellite) and air (station) temperature, which correlate but are not identical — do not present this as a direct ground-truth match.
- **Physics-residual check:** evaluate the PDE residual (Section 9.2) across the study area for the trained PINN; a spatial map of residual magnitude shows where the model is and is not physically consistent.
- **Spatial hold-out check:** compare RMSE on the training area vs. a geographically separate held-out area, for both the baseline and the PINN — this is the generalization comparison referenced in Section 3.

---

## 10. Metrics — Exact List per Stage

| Stage | Metric | Definition / formula | What it tells you |
|---|---|---|---|
| Baseline (RF/XGBoost) | RMSE | √(mean((pred − obs)²)) | Average prediction error, in °C, penalizes large errors more |
| Baseline | MAE | mean(\|pred − obs\|) | Average absolute error, in °C |
| Baseline | R² | 1 − (SS_residual / SS_total) | Fraction of variance in LST explained by the model |
| Baseline | Feature importance | Gini importance (RF) / gain importance (XGBoost) | Quick, model-native ranking of feature relevance (SHAP is the more rigorous version, Section 9.3) |
| PINN | Data loss (train/val curves) | MSE(pred, obs) over training epochs | Confirms the network is fitting the data at all (Stage 1 check) |
| PINN | Physics residual | mean/RMS of the PDE residual evaluated at network output | Physical consistency (independent of accuracy against data) |
| PINN | RMSE, MAE, R² | same formulas as baseline, computed against held-out LST | Comparability with the baseline |
| PINN | Held-out area RMSE vs. training-area RMSE | same RMSE formula, two different spatial subsets | Generalization (Section 3 success criterion) |
| PINN ablations | RMSE and physics residual, with vs. without physics loss; with vs. without sea-breeze term | same formulas, computed per variant | Demonstrates each added term's contribution |
| Explainability | Mean \|SHAP value\| per feature | average magnitude of each feature's SHAP contribution | Driver ranking |
| Scenario simulation | ΔLST | LST_after − LST_before, per pixel/ward | Cooling gain from a simulated intervention |
| Optimization | Hypervolume (optional, if reporting convergence) | volume of objective space dominated by the current Pareto front, relative to a reference point | Whether NSGA-II is converging across generations |
| Optimization | Pareto front | set of non-dominated (cooling gain, cost, equity) triples | The actual recommendation output |
| Validation | Bias vs. CPCB stations | mean(pred − station observed), per station | Systematic over/under-prediction, per station |
| Validation | Residual by land-cover class | RMSE or bias, grouped by WorldPop class | Whether error is concentrated in a particular land-cover type |

---

## 11. Required Figures

| # | Figure | X-axis | Y-axis / content | Produced at stage |
|---|---|---|---|---|
| 1 | Mumbai LST map (median, dry season) | longitude | latitude, colour = °C | After data acquisition |
| 2 | Feature maps (NDVI, NDBI, distance-to-sea, building height) | longitude | latitude, colour = feature value | After feature engineering |
| 3 | Baseline predicted vs. observed LST scatter | observed LST (°C) | predicted LST (°C), with 1:1 line | After baseline training |
| 4 | SHAP summary/beeswarm plot (baseline) | mean \|SHAP value\| | feature name, coloured by feature value | After SHAP (Section 9.3) |
| 5 | SHAP dependence plot, top feature (e.g. NDVI) | feature value | SHAP value, coloured by a second feature | After SHAP |
| 6 | PINN training curves | training epoch | data loss and physics loss (two lines) | During PINN training |
| 7 | PINN physics-residual spatial map | longitude | latitude, colour = residual magnitude | After PINN training |
| 8 | Ablation bar chart | model variant (no-physics / physics / +sea-breeze) | RMSE (°C) and physics residual (grouped bars or two panels) | After ablation runs |
| 9 | ΔLST scenario map (e.g. +20% NDVI on built-up pixels) | longitude | latitude, colour = ΔLST (°C) | After scenario simulation |
| 10 | NSGA-II Pareto front | total cost | cooling gain, point size/colour = equity weight | After optimization |
| 11 | Predicted vs. CPCB station scatter | station observed temperature | predicted temperature, with 1:1 line | After validation |
| 12 | Ward-level recommendation map | longitude | latitude, colour = recommended intervention / priority rank | Final report figure |

---

## 12. Repository / File Structure (suggested)

```
mumbai-uhi-pinn/
├── data/
│   ├── raw/              # exported GeoTIFFs / CSVs from GEE, untouched
│   ├── processed/        # cleaned, gridded, joined tables (Parquet/CSV)
│   └── vector/           # ward boundaries, OSM extracts (GeoPackage/GeoJSON)
├── scripts/
│   ├── gee_export.py     # the Earth Engine pipeline (this project's existing script)
│   ├── features.py       # NDVI/NDBI/NDWI/MNDWI, distance-to-sea computation
│   ├── train_baseline.py
│   ├── train_pinn.py
│   ├── shap_analysis.py
│   ├── scenario_sim.py
│   └── optimize_nsga2.py
├── notebooks/            # exploration, figure generation
├── results/
│   ├── figures/          # numbered per Section 11
│   ├── metrics/          # CSV/JSON logs per Section 10
│   └── models/           # saved model weights
└── report/               # written report, this specification
```

---

## 13. Risks and Honest Limitations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Few usable Landsat scenes due to monsoon cloud | Directly limits PINN training data volume | Restrict to Oct–May, count usable scenes on day 1, add Landsat 9 |
| MODIS night LST is 1 km vs. 30 m target grid | Cannot be a per-pixel night label | Use as a coarse regime constraint only, not point-wise ground truth |
| CPCB station temperature coverage is historically unreliable in Mumbai (verified, Section 6 row 13) | Validation set may be very small or unusable | Check availability in week 1; treat as a secondary, not primary, validation source |
| No confirmed authoritative machine-readable Mumbai ward boundary file (Section 6 row 15) | Equity aggregation unit is uncertain | Verify the community GeoJSON against a current source, or fall back to grid-block aggregation |
| Census 2011 vulnerability data is over a decade old | Equity weighting reflects outdated demographics | State this explicitly as a limitation; do not present current-day equity claims |
| UT-GLOBUS availability unverified (Section 6 row 16) | Was a named building-height source in the original proposal | Use Open Buildings Temporal or GHSL instead; note the substitution in the report |
| PINN physics loss can be numerically unstable | Common failure mode for PINNs, independent of this project's code quality | Train in the three stages of Section 9.2 in order; do not attempt all loss terms simultaneously on day one |
| Sea-breeze term has no direct observational validation dataset | There is no ground-truth "sea-breeze cooling" measurement to check against | Validate indirectly via physical plausibility (cooler near-coast at expected times) and the ablation study, not a direct accuracy number |
| One-month timeline | Cannot support the full original 28-week scope | Section 4 defines exactly what is cut and why; state the cuts explicitly in the submitted report rather than silently under-delivering against the original proposal |

---

## 14. Four-Week Execution Plan

Team split: **Person A** — data/geospatial; **Person B** — modelling (baseline + PINN); **Person C** — explainability, optimization, validation, and the write-up/demo.

| Week | Person A | Person B | Person C | Week deliverable |
|---|---|---|---|---|
| 1 | Set up Earth Engine; run scene-count check (Section 6, row 1/2); export the 30 m feature stack (Section 5, 6) and a pixel-sample CSV; verify Open Buildings Temporal coverage for Mumbai | Set up the training environment (Colab/Kaggle); write the data-loading and spatial train/test split code | Attempt to obtain ward boundaries and Census/WorldPop vulnerability data (Section 6, rows 14–15); check CPCB station temperature availability | Clean, gridded dataset; confirmed data availability facts (no more assumptions) |
| 2 | Finalize distance-to-sea (with coastline caveat, Section 6 row 8) and building features | Train RF and XGBoost baseline (Section 9.1); compute metrics (Section 10) | Run SHAP on the baseline (Section 9.3); produce Figures 3–5 | Baseline metrics + SHAP figures. **Decision point:** if baseline RMSE is poor, fix features now before starting the PINN |
| 3 | Build the scenario-editing code (Section 9.4), independent of which model it will call | Train the PINN in the three stages of Section 9.2; run the ablations | Build the NSGA-II optimization scaffold using the baseline as a temporary surrogate, so this doesn't wait on the PINN | Working PINN (at least Stage 2, ideally Stage 3) with ablation table; optimization scaffold ready |
| 4 | Support the CPCB/physics-residual validation checks (Section 9.6) | Finish ablations; help debug optimization surrogate hand-off | Run NSGA-II with the real trained model; produce the Pareto front (Figure 10) and ward-level recommendation map (Figure 12); assemble the report and all figures | Final results, report, and demo |

---

## 15. Glossary

- **LST** — Land Surface Temperature: the temperature of the ground/roof/canopy surface as measured by a thermal satellite sensor, distinct from air temperature.
- **PINN** — Physics-Informed Neural Network: a neural network trained with a loss function that includes both a data-fitting term and a penalty for violating a known physical equation.
- **NDVI / NDBI / NDWI / MNDWI** — spectral indices computed from satellite reflectance bands (formulas in Section 8), indicating vegetation, built-up surfaces, and water respectively.
- **SHAP** — SHapley Additive exPlanations: a method from game theory used to attribute a model's prediction to its individual input features.
- **NSGA-II** — Non-dominated Sorting Genetic Algorithm II: a multi-objective evolutionary optimization algorithm that produces a Pareto front rather than a single answer.
- **Pareto front** — the set of solutions in a multi-objective problem where no objective can be improved without worsening another.
- **UHI** — Urban Heat Island: the phenomenon where urban areas are measurably hotter than surrounding non-urban areas.
- **GEE** — Google Earth Engine: a cloud platform for planetary-scale geospatial analysis, used here to acquire and pre-process all satellite/reanalysis data.

---

## 16. Sources Consulted for This Document

- Google Earth Engine dataset catalog pages (Landsat 8/9 C2 L2, Sentinel-2 SR Harmonized, ERA5-Land Hourly, MODIS MYD11A1, MODIS MCD43A3, ESA WorldCover v200, JRC Global Surface Water, USGS SRTM, WorldPop, Open Buildings Temporal V1) — https://developers.google.com/earth-engine/datasets/catalog
- CPCB continuous air-quality data portal and reporting on Mumbai/MMR station data availability (2019 news report; 2026 BMC CAAQMS expansion report)
- Community discussion and repository regarding Mumbai ward-level geographic data (DataMeet mailing list; mickeykedia/Mumbai-Population-Map GitHub repository)
- NASA AppEEARS (ECOSTRESS access point)

**Reminder:** re-verify every dataset ID, band name, and scale factor against its live source before submitting or building on this document — catalogs are updated over time and this document reflects a check performed at one point in time, not a guarantee of permanent accuracy.

---

## 17. Related Work and How This Project Differs

Eight public repositories were reviewed (READMEs and repo structure; not their full source code) at the person's request, to check this project is not a copy of, or too similar to, existing work. Assessed honestly, one of them (row 5 below) is a serious, closely related project and must be taken seriously rather than dismissed — the differentiation argument below is built around it specifically, not around the weaker projects.

| # | Repository | What it actually does | How it compares to this project |
|---|---|---|---|
| 1 | `Xmanish8/UrbanHeatAi` | Pune UHI prediction, ISRO Hackathon PS-1. Random Forest only (R² 0.998, claimed), Landsat/Sentinel-2/ERA5/SRTM/WorldPop features, SHAP, a cooling-scenario simulator, Streamlit dashboard. | Same data sources and general flow (RF baseline → SHAP → scenario simulation) as this project's baseline stage, but **no PINN, no physics term, no optimizer.** Different city. Not a close overlap once the PINN and NSGA-II stages are counted. |
| 2 | `Rashid075/Urban_Heat_Mitigation_System` | "Urban Heat Intelligence System," a hackathon dashboard using **live OpenWeather API data** (not satellite LST) with XGBoost/RF/KMeans, SHAP, a hand-written linear "heat risk formula," and a mitigation-strategy recommender. | No satellite remote sensing, no spatial grid, no PDE/physics term. Point-based live weather modelling, not geospatial LST prediction. Low overlap. |
| 3 | `Bit-Bard/Urban_Heat_island` | A tabular classifier (RF, auto-sklearn-selected) predicting whether a district/state will "become a UHI," from India-wide climate/AQI/urbanization CSVs, with a simple GUI. | Non-spatial, non-satellite, classification not regression, no physics. Low overlap. |
| 4 | `ashwin8332/UrbanYantra` | Could not be reviewed — GitHub blocked automated access to this repository in this session. **Check it yourself before assuming it doesn't overlap.** | Unknown — flagged, not cleared. |
| 5 | `AsthaChandel123/bah2026-ps1` (`urbanheat`) | **The most advanced and closely related repository found.** Built for the same competition family (ISRO Bharatiya Antariksh Hackathon 2026, PS-1) as repo 1. Explicitly physics-informed: a surface-energy-balance backbone, monotone-constrained gradient boosting, MGWR, and an **optional PINN with a surface-energy-balance closure loss**; hotspot detection with Getis-Ord Gi*/local Moran's I significance testing; SHAP + ALE + variance-partitioning attribution; intervention simulation cross-checked against the InVEST Urban Cooling model and a SOLWEIG hook; a lazy-greedy/ILP/**NSGA-II** optimizer with equity weighting; roughly 30 datasets across 5 LST sensors, 4 LULC products, and building-footprint sources including UT-GLOBUS and GHSL; **and it ships a Mumbai city preset by name.** Its headline result (R² 0.987, RMSE 0.751°C) is stated in its own README to come from a **synthetic, physically-plausible demo run**, not from a calibrated run on real satellite observations — the README says so explicitly. | **This is the project to differentiate against, not the others.** Three genuine, defensible differences: **(a) the governing equation.** This repo's physics is a surface-energy-balance closure (monotonic sign constraints on drivers, `L↑ = εσTs⁴`), not a directional advection-diffusion PDE. This project's sea-breeze term is wind-driven spatial advection (`−v·∇T`, Section 9.2) — a different, and for a coastal city, more physically direct mechanism than an energy-balance sign check. **(b) real vs. synthetic validation.** This project trains and validates on **actual Landsat/Sentinel-2/ERA5/CPCB observations for Mumbai** (Sections 6, 9.6); the comparable repo's headline metric is admittedly a synthetic-data demonstration of the pipeline, not a real-data result. Say this explicitly in your report — it is a genuine, checkable point of difference, not a rhetorical one. **(c) depth vs. breadth.** That repository is a broad, many-method framework (≥30 datasets, ~19 analytical methods) generalized across five Indian cities. This project is deliberately narrow: one city, one novel physics term, rigorously ablated (Section 9.2's required ablation table) within a one-month timeline. State this as a design choice, not a limitation — a small, well-validated result beats a large, thinly-validated one in a one-month student project. |
| 6 | `isatyamks/UHI-Detector` | "GeoSmartPredict" — UHI detection plus **LSTM time-series forecasting** of future temperature trends, GEE + OpenWeatherMap data, Streamlit/Leaflet dashboard. | Forecasting-focused (predicting UHI growth over time), not physics-informed, no PDE, no PINN. Different core method. Low overlap. |
| 7 | `Zephyrex21/urban-heat-mitigation` | A deployed full-stack app (React + FastAPI + Deck.gl) with a **single shared XGBoost model across 20 Indian cities**, SHAP attribution, a cooling-scenario builder, and — notably — a **leave-cities-out validation split** (train on some cities, test on a city never seen). Live demo deployed. | No physics, no PINN, no optimizer. Its leave-cities-out validation is a good idea worth borrowing in spirit: this project's equivalent is the spatial hold-out check in Section 9.6 (train on part of Mumbai, test on an unseen part) — same principle of testing generalization to unseen geography, applied within one city since this project is single-city. |
| 8 | `nkusla/heat-transfer-pinn` | Not an urban-heat project at all — a general tutorial/demo comparing a PINN against finite-difference for the plain 1D/2D heat equation on a plate, including **learning the thermal diffusivity α as an unknown parameter from data** ("PDE discovery"), citing the original Raissi et al. PINN papers. | No urban application, no satellite data, no sea-breeze term. Its useful idea — **treating a physical constant as learnable rather than fixed** — is exactly what Section 9.2's "learned vs. fixed physical parameters" recommendation borrows. Zero competitive overlap; purely a technique source. |

### What was changed in this document as a result

1. **Section 2 and 9.2** now state the sea-breeze term explicitly as **wind-driven advection** (`−v·∇T`) rather than a generic "cooling function of distance and time," to make the mechanism concretely different from repo 5's energy-balance approach — check this is still an accurate description of what your team actually builds, and adjust the equation if your implementation differs.
2. **Section 9.2** adds the option to **learn the diffusivity and sea-breeze coefficients as trainable parameters** instead of fixing them, borrowed from repo 8. This is optional but recommended: it is a small code change with a genuine methodological payoff, and gives you a novel, reportable number (Mumbai's data-fitted effective thermal diffusivity) that none of the eight reviewed repos report.
3. **This section (17) itself** should go into your submitted report as a short "Related Work" subsection — examiners who have seen hackathon repositories in this space (several of the above are recent, similarly-themed hackathon projects) will read a project more favorably if it demonstrates awareness of existing work and states its difference precisely, rather than presenting the idea as if no one has attempted anything similar.
4. **No change was made to cut scope further.** Repo 5's breadth is not a reason to try to match it feature-for-feature in one month — Section 4's reduced scope stands. The differentiation argument is about the *nature* of the physics term and the *reality* of the validation data, not about matching every feature of a much larger project.

**Caveat on this comparison:** it is based on each repository's README and file listing, not a full code review, because that is what was accessible in this session. A README can overstate what code actually does or runs reliably — this is true of the comparison repos and is a reason for realistic caution, not a license to assume they are weaker than they claim. If you want a firmer comparison before finalizing your report's related-work section, open the actual training/physics code in repo 5 (`urbanheat/physics/pinn.py`, `urbanheat/physics/energy_balance.py` per its stated repo layout) rather than relying on this summary alone.
