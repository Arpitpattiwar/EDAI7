"""Train Random Forest and XGBoost baselines on the real Mumbai dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import METRICS_DIR, MODELS_DIR, FEATURE_COLUMNS  # noqa: E402
from scripts.data_loader import load_mumbai_data  # noqa: E402


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
	return {
		"rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
		"mae": float(mean_absolute_error(targets, predictions)),
		"r2": float(r2_score(targets, predictions)),
	}


def train_baselines(random_seed: int = 42) -> dict[str, dict]:
	data = load_mumbai_data(random_seed=random_seed)
	train_frame = data["df_train"]
	validation_frame = data["df_val"]
	holdout_frame = data["df_test"]

	x_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
	y_train = train_frame["lst_c"].to_numpy(dtype=np.float32)
	x_validation = validation_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
	y_validation = validation_frame["lst_c"].to_numpy(dtype=np.float32)
	x_holdout = holdout_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
	y_holdout = holdout_frame["lst_c"].to_numpy(dtype=np.float32)

	models = {
		"random_forest": RandomForestRegressor(
			n_estimators=300,
			min_samples_leaf=2,
			n_jobs=-1,
			random_state=random_seed,
		),
		"xgboost": XGBRegressor(
			n_estimators=400,
			max_depth=5,
			learning_rate=0.04,
			subsample=0.85,
			colsample_bytree=0.9,
			objective="reg:squarederror",
			n_jobs=-1,
			random_state=random_seed,
		),
	}

	metrics = {}
	for name, model in models.items():
		model.fit(x_train, y_train)
		validation_predictions = model.predict(x_validation)
		holdout_predictions = model.predict(x_holdout)
		metrics[name] = {
			"validation": regression_metrics(y_validation, validation_predictions),
			"spatial_holdout": regression_metrics(y_holdout, holdout_predictions),
			"feature_importances": {
				feature: float(importance)
				for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_)
			},
		}
		joblib.dump(model, MODELS_DIR / f"{name}_baseline.joblib")

	METRICS_DIR.mkdir(parents=True, exist_ok=True)
	metrics_path = METRICS_DIR / "baseline_metrics.json"
	metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
	return metrics


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	metrics = train_baselines(random_seed=args.seed)
	for name, model_metrics in metrics.items():
		validation = model_metrics["validation"]
		holdout = model_metrics["spatial_holdout"]
		print(
			f"{name}: validation RMSE={validation['rmse']:.3f}, "
			f"holdout RMSE={holdout['rmse']:.3f}"
		)


if __name__ == "__main__":
	main()
