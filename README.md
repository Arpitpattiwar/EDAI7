# Mumbai UHI PINN

Physics-informed urban heat prediction and equity-aware cooling recommendations for Mumbai.

## Current Status

- Earth Engine authentication: configured through the local OAuth flow.
- Earth Engine Cloud project: `proud-lamp-453406-n1`.
- Verified Python API access with a server-side expression (`1 + 2 = 3`).
- Verified real catalog access over the Mumbai bounds: 4 Landsat 8 scenes and 26 Sentinel-2 scenes for January 2024.
- Implemented and smoke-tested the real Earth Engine feature export: 443 complete January 2024 pixels were written to `data/processed/mumbai_features.parquet` and `.csv`.
- Verified Open Buildings coverage over Mumbai and integrated the latest available building fractional count and height layers into the export.
- Added quality-masked MODIS nighttime LST as the auxiliary `lst_night_c` field for future day/night PINN evaluation; the daytime Landsat `lst_c` target remains unchanged.
- The downstream data loader now consumes the real export successfully, including the contiguous spatial holdout split.

## Environment

Use the project virtual environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Earth Engine authentication can be refreshed with:

```powershell
.venv\Scripts\earthengine.exe authenticate --auth_mode=localhost
.venv\Scripts\earthengine.exe set_project proud-lamp-453406-n1
```

The project ID must have Earth Engine enabled and the authenticated account must have permission to use the project.

## Real Data Export

After authentication, export a model-ready Mumbai sample with:

```powershell
.venv\Scripts\python.exe scripts\gee_export.py --num-pixels 10000
```

The exporter combines cloud-masked Landsat 8/9 L2 surface temperature, Sentinel-2 spectral indices, MODIS albedo, ERA5-Land meteorology, SRTM terrain, WorldCover land cover, JRC surface water distance, and WorldPop population. It writes the Parquet file consumed by `scripts/data_loader.py` and also writes a CSV copy for inspection.

The export uses the latest Open Buildings Temporal image for `building_density` and `building_height`, plus a January MODIS nighttime LST composite. The sample is limited to complete pixels across all layers, so the returned row count can be lower than `--num-pixels`.

## Real-Data Training

Train and evaluate the tree baselines:

```powershell
.venv\Scripts\python.exe scripts\train_baseline.py
```

Train the staged PINN:

```powershell
.venv\Scripts\python.exe scripts\train_pinn.py
```

The current January 2024 real-data run produced these spatial-holdout RMSE values: Random Forest `1.512` °C, XGBoost `1.451` °C, PINN stage 1 `2.472` °C, stage 2 `1.421` °C, and stage 3 `2.656` °C. Stage 2 is the current best checkpoint; stage 3's regression indicates that the sea-breeze physics weighting needs calibration before it is treated as an improvement. These values replace the earlier synthetic/demo metrics in `results/metrics/`.

## Scenarios And Recommendations

Generate NDVI greening and cool-roof impacts with:

```powershell
.venv\Scripts\python.exe scripts\scenario_sim.py
```

This writes `results/metrics/scenario_impacts.csv`, including signed per-pixel temperature deltas and population/vulnerability-weighted gains. The current small real sample predicts cooling for greening on average, while cool-roof edits are mixed; retain the signed values when interpreting recommendations.

Generate the zone-level budget-constrained NSGA-II recommendations with:

```powershell
.venv\Scripts\python.exe scripts\optimize_nsga2.py
```

Generate the real-data XGBoost SHAP ranking with:

```powershell
.venv\Scripts\python.exe scripts\shap_analysis.py
```

The optimizer currently operates over five spatial zones because verified ward boundaries are not present. The resulting Pareto file is `results/metrics/nsga2_recommendations.csv`; SHAP output is `results/metrics/shap_ranking.json`.

## Project References

- [Project specification](Mumbai_UHI_PINN_Project_Specification.md)
- [Interactive demonstration notebook](notebooks/mumbai_pinn_demo.ipynb)
- [Configuration](configs/config.py)
