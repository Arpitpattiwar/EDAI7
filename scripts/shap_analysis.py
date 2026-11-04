"""Compute SHAP feature rankings for the trained tree baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import FEATURE_COLUMNS, METRICS_DIR, MODELS_DIR  # noqa: E402
from scripts.data_loader import load_mumbai_data  # noqa: E402


def compute_shap_rankings(model_name: str = "xgboost", max_samples: int = 500) -> dict[str, float]:
	data = load_mumbai_data()
	model = joblib.load(MODELS_DIR / f"{model_name}_baseline.joblib")
	frame = data["df_all"][FEATURE_COLUMNS].head(max_samples)
	try:
		explainer = shap.TreeExplainer(model)
		values = explainer.shap_values(frame)
	except (ValueError, TypeError):
		background = frame.head(min(50, len(frame)))
		explainer = shap.Explainer(model.predict, background, algorithm="permutation")
		values = explainer(frame).values
	absolute_mean = np.abs(np.asarray(values)).mean(axis=0)
	return {
		feature: float(score)
		for feature, score in sorted(
			zip(FEATURE_COLUMNS, absolute_mean), key=lambda item: item[1], reverse=True
		)
	}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model", choices=["random_forest", "xgboost"], default="xgboost")
	parser.add_argument("--max-samples", type=int, default=500)
	args = parser.parse_args()
	ranking = compute_shap_rankings(args.model, args.max_samples)
	output_path = METRICS_DIR / "shap_ranking.json"
	output_path.write_text(json.dumps({args.model: ranking}, indent=4), encoding="utf-8")
	print(f"Wrote SHAP ranking to {output_path}")
	print(json.dumps(dict(list(ranking.items())[:5]), indent=2))


if __name__ == "__main__":
	main()
