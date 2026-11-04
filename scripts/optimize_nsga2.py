"""Find equity-aware intervention portfolios with NSGA-II."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import OPTIMIZATION_CONFIG, RESULTS_DIR, SCENARIO_CONFIG  # noqa: E402


INTERVENTIONS = {
	0: ("none", 0.0, 0.0, 0.0),
	1: ("greening", SCENARIO_CONFIG["cost_green_per_sqm"], "greening_delta_lst_c", "equity_weighted_greening_gain"),
	2: ("cool_roof", SCENARIO_CONFIG["cost_cool_roof_per_sqm"], "cool_roof_delta_lst_c", "equity_weighted_roof_gain"),
	3: ("combined", SCENARIO_CONFIG["cost_green_per_sqm"] + SCENARIO_CONFIG["cost_cool_roof_per_sqm"], "combined_delta_lst_c", "equity_weighted_greening_gain"),
}


def load_zone_options() -> tuple[pd.DataFrame, list[int]]:
	impacts_path = RESULTS_DIR / "metrics" / "scenario_impacts.csv"
	if not impacts_path.exists():
		raise FileNotFoundError("Run scripts/scenario_sim.py before the optimizer.")
	impacts = pd.read_csv(impacts_path)
	pixel_area_sqm = OPTIMIZATION_CONFIG["grid_res_m"] ** 2
	rows = []
	for zone, zone_frame in impacts.groupby("spatial_zone"):
		rows.append(
			{
				"spatial_zone": int(zone),
				"pixel_count": len(zone_frame),
				"population": zone_frame["population_100m"].sum(),
				"green_gain": -zone_frame["greening_delta_lst_c"].sum() * pixel_area_sqm,
				"roof_gain": -zone_frame["cool_roof_delta_lst_c"].sum() * pixel_area_sqm,
				"combined_gain": -zone_frame["combined_delta_lst_c"].sum() * pixel_area_sqm,
				"green_equity": zone_frame["equity_weighted_greening_gain"].sum(),
				"roof_equity": zone_frame["equity_weighted_roof_gain"].sum(),
				"combined_equity": zone_frame["equity_weighted_greening_gain"].sum() + zone_frame["equity_weighted_roof_gain"].sum(),
			}
		)
	zone_frame = pd.DataFrame(rows).sort_values("spatial_zone").reset_index(drop=True)
	return zone_frame, zone_frame["spatial_zone"].tolist()


class InterventionProblem(ElementwiseProblem):
	def __init__(self, zones: pd.DataFrame):
		self.zones = zones
		budget_inr = OPTIMIZATION_CONFIG["budget_cap_inr_cr"] * 1e7
		super().__init__(
			n_var=len(zones),
			n_obj=3,
			n_ieq_constr=1,
			xl=0,
			xu=3,
			vtype=int,
		)
		self.budget_inr = budget_inr

	def _evaluate(self, decision, out, *args, **kwargs):
		total_gain = 0.0
		total_cost = 0.0
		total_equity = 0.0
		for index, option_value in enumerate(np.rint(decision).astype(int)):
			option = INTERVENTIONS[option_value]
			option_name, cost_per_sqm, _, _ = option
			if option_name == "none":
				continue
			zone = self.zones.iloc[index]
			column_prefix = {"greening": "green", "cool_roof": "roof", "combined": "combined"}[option_name]
			gain_column = f"{column_prefix}_gain"
			equity_column = f"{column_prefix}_equity"
			total_gain += float(zone[gain_column])
			total_equity += float(zone[equity_column])
			total_cost += cost_per_sqm * OPTIMIZATION_CONFIG["grid_res_m"] ** 2 * zone["pixel_count"]
		out["F"] = [-total_gain, total_cost, -total_equity]
		out["G"] = [total_cost - self.budget_inr]


def optimize() -> pd.DataFrame:
	zones, zone_ids = load_zone_options()
	result = minimize(
		InterventionProblem(zones),
		NSGA2(pop_size=OPTIMIZATION_CONFIG["pop_size"]),
		termination=("n_gen", OPTIMIZATION_CONFIG["n_gen"]),
		seed=42,
		verbose=False,
	)
	recommendations = []
	for decision, objectives in zip(result.X, result.F):
		choices = np.rint(decision).astype(int)
		recommendations.append(
			{
				**{f"zone_{zone}": INTERVENTIONS[choice][0] for zone, choice in zip(zone_ids, choices)},
				"cooling_gain_c_m2": float(-objectives[0]),
				"cost_inr": float(objectives[1]),
				"equity_weighted_benefit": float(-objectives[2]),
			}
		)
	return pd.DataFrame(recommendations).drop_duplicates().reset_index(drop=True)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.parse_args()
	recommendations = optimize()
	output_path = RESULTS_DIR / "metrics" / "nsga2_recommendations.csv"
	recommendations.to_csv(output_path, index=False)
	print(f"Wrote {len(recommendations):,} Pareto recommendations to {output_path}")


if __name__ == "__main__":
	main()
