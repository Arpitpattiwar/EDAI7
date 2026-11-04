"""Simulate NDVI and cool-roof interventions with a trained baseline model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import FEATURE_COLUMNS, MODELS_DIR, RESULTS_DIR, SCENARIO_CONFIG  # noqa: E402
from scripts.data_loader import load_mumbai_data  # noqa: E402


def simulate_scenarios(
	model_name: str = "xgboost",
	ndvi_boost: float = SCENARIO_CONFIG["greening_ndvi_boost"],
	albedo_boost: float = SCENARIO_CONFIG["cool_roof_albedo_boost"],
) -> pd.DataFrame:
	"""Return baseline and intervention predictions for every land pixel."""
	data = load_mumbai_data()
	frame = data["df_all"].copy()
	model = joblib.load(MODELS_DIR / f"{model_name}_baseline.joblib")

	baseline_features = frame[FEATURE_COLUMNS].copy()
	green_features = baseline_features.copy()
	green_features["ndvi"] = (green_features["ndvi"] + ndvi_boost).clip(-1.0, 1.0)
	roof_features = baseline_features.copy()
	roof_features["albedo"] = (roof_features["albedo"] + albedo_boost).clip(0.0, 1.0)
	combined_features = green_features.copy()
	combined_features["albedo"] = roof_features["albedo"]

	baseline = model.predict(baseline_features)
	green = model.predict(green_features)
	roof = model.predict(roof_features)
	combined = model.predict(combined_features)
	result = frame[["lon", "lat", "spatial_zone", "population_100m", "vulnerability_index"]].copy()
	result["baseline_lst_c"] = baseline
	result["greening_lst_c"] = green
	result["cool_roof_lst_c"] = roof
	result["combined_lst_c"] = combined
	result["greening_delta_lst_c"] = green - baseline
	result["cool_roof_delta_lst_c"] = roof - baseline
	result["combined_delta_lst_c"] = combined - baseline
	result["equity_weighted_greening_gain"] = -result["greening_delta_lst_c"] * result["population_100m"] * result["vulnerability_index"]
	result["equity_weighted_roof_gain"] = -result["cool_roof_delta_lst_c"] * result["population_100m"] * result["vulnerability_index"]
	return result


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model", choices=["random_forest", "xgboost"], default="xgboost")
	parser.add_argument("--ndvi-boost", type=float, default=SCENARIO_CONFIG["greening_ndvi_boost"])
	parser.add_argument("--albedo-boost", type=float, default=SCENARIO_CONFIG["cool_roof_albedo_boost"])
	args = parser.parse_args()
	result = simulate_scenarios(args.model, args.ndvi_boost, args.albedo_boost)
	output_path = RESULTS_DIR / "metrics" / "scenario_impacts.csv"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(output_path, index=False)
	print(f"Wrote {len(result):,} scenario rows to {output_path}")
	print(result[["greening_delta_lst_c", "cool_roof_delta_lst_c"]].describe().to_string())


if __name__ == "__main__":
	main()
