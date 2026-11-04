"""Train the staged Mumbai PINN on the exported real dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import (  # noqa: E402
	FEATURE_COLUMNS,
	METRICS_DIR,
	MODELS_DIR,
	PHYSICS_CONFIG,
)
from models.physics_loss import AdvectionDiffusionPhysicsLoss  # noqa: E402
from models.pinn import MumbaiPINN  # noqa: E402
from scripts.data_loader import load_mumbai_data  # noqa: E402


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def to_tensors(data: dict, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	return (
		torch.from_numpy(np.array(data[f"coords_{split}"], copy=True)).float(),
		torch.from_numpy(np.array(data[f"X_{split}"], copy=True)).float(),
		torch.from_numpy(np.array(data[f"y_{split}"], copy=True)).float().unsqueeze(1),
	)


def raw_physics_features(frame) -> dict[str, torch.Tensor]:
	return {
		name: torch.from_numpy(frame[name].to_numpy(dtype=np.float32, copy=True))
		for name in [
			"dist_to_sea_km",
			"u_wind",
			"v_wind",
			"air_temp_c",
			"solar_radiation",
			"albedo",
			"ndbi",
		]
	}


def predict(model: MumbaiPINN, coords: torch.Tensor, features: torch.Tensor) -> np.ndarray:
	model.eval()
	with torch.no_grad():
		return model(coords, features).squeeze(1).cpu().numpy()


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
	return {
		"rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
		"mae": float(mean_absolute_error(targets, predictions)),
		"r2": float(r2_score(targets, predictions)),
	}


def train_stage(
	model: MumbaiPINN,
	physics: AdvectionDiffusionPhysicsLoss,
	coords: torch.Tensor,
	features: torch.Tensor,
	targets: torch.Tensor,
	raw_features: dict[str, torch.Tensor],
	epochs: int,
	learning_rate: float,
	physics_weight: float,
) -> None:
	optimizer = torch.optim.Adam(
		list(model.parameters()) + list(physics.parameters()),
		lr=learning_rate,
		weight_decay=PHYSICS_CONFIG["weight_decay"],
	)
	for _ in range(epochs):
		model.train()
		optimizer.zero_grad(set_to_none=True)
		prediction = model(coords, features)
		data_loss = F.mse_loss(prediction, targets)
		if physics.stage == "stage1":
			physics_loss = torch.zeros((), dtype=data_loss.dtype)
		else:
			residual, _ = physics.compute_residual(model, coords, features, raw_features)
			physics_loss = residual.square().mean()
		total_loss = data_loss + physics_weight * physics_loss
		total_loss.backward()
		optimizer.step()


def train_pinn(random_seed: int = 42) -> dict[str, dict[str, float]]:
	set_seed(random_seed)
	data = load_mumbai_data(random_seed=random_seed)
	coords_train, features_train, targets_train = to_tensors(data, "train")
	coords_val, features_val, targets_val = to_tensors(data, "val")
	coords_test, features_test, targets_test = to_tensors(data, "test")
	raw_features = raw_physics_features(data["df_train"])

	model = MumbaiPINN(n_features=len(FEATURE_COLUMNS))
	stage_results = {}
	stages = [("stage1", "stage1"), ("stage2", "stage2"), ("stage3", "stage3")]
	for stage_name, physics_stage in stages:
		physics = AdvectionDiffusionPhysicsLoss(
			alpha_init=PHYSICS_CONFIG["alpha_init"],
			kd_init=PHYSICS_CONFIG["kd_wind_init"],
			lambda_exch=PHYSICS_CONFIG["lambda_exch"],
			stage=physics_stage,
		)
		train_stage(
			model,
			physics,
			coords_train,
			features_train,
			targets_train,
			raw_features,
			PHYSICS_CONFIG["staged_epochs"][stage_name],
			PHYSICS_CONFIG["learning_rate"],
			PHYSICS_CONFIG["lambda_physics"],
		)
		validation_metrics = metrics(
			targets_val.squeeze(1).numpy(), predict(model, coords_val, features_val)
		)
		holdout_metrics = metrics(
			targets_test.squeeze(1).numpy(), predict(model, coords_test, features_test)
		)
		stage_results[stage_name] = {
			"val_rmse": validation_metrics["rmse"],
			"val_mae": validation_metrics["mae"],
			"val_r2": validation_metrics["r2"],
			"test_rmse": holdout_metrics["rmse"],
			"test_mae": holdout_metrics["mae"],
			"test_r2": holdout_metrics["r2"],
			"alpha": float(physics.alpha.detach()),
			"kd": float(physics.kd.detach()),
		}
		torch.save(
			{
				"model_state_dict": model.state_dict(),
				"physics_state_dict": physics.state_dict(),
				"feature_names": FEATURE_COLUMNS,
				"stage": stage_name,
			},
			MODELS_DIR / f"pinn_{stage_name}.pt",
		)

	(METRICS_DIR / "pinn_ablation_metrics.json").write_text(
		json.dumps(stage_results, indent=4), encoding="utf-8"
	)
	return stage_results


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	results = train_pinn(random_seed=args.seed)
	for stage, values in results.items():
		print(f"{stage}: validation RMSE={values['val_rmse']:.3f}, holdout RMSE={values['test_rmse']:.3f}")


if __name__ == "__main__":
	main()
