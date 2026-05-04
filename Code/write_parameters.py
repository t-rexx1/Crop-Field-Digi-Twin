"""Write default parameters for student Project10 plant-growth GA assignment."""

from __future__ import annotations

import pickle
from pathlib import Path

from plantga.data_generation import DEFAULT_TRUE_PARAMS


def default_parameters() -> dict:
    bounds = {
        "G0": (0.02, 0.25),
        "b_w": (0.00, 1.20),
        "b_f": (0.00, 1.20),
        "gamma": (0.30, 1.60),
        "alpha": (0.50, 1.40),
        "lambda": (0.00, 0.30),
        "rho": (0.20, 2.00),
    }

    return {
        "dataset": {
            "nx": 6,
            "ny": 6,
            "spacing": 0.5,
            "season_days": 100.0,
            "dt": 1.0,
            "noise_sigma": 0.03,
            "train_fraction": 0.70,
            "seed": 4,
        },
        "ga": {
            "population_size": 36,
            "num_generations": 55,
            "parent_fraction": 0.40,
            "mutation_rate": 0.20,
            "mutation_scale": 0.08,
            "elitism": 2,
            "patience": 12,
            "seed": 9,
        },
        "bounds": bounds,
        "true_params": DEFAULT_TRUE_PARAMS.copy(),
    }


def write_parameters(file_name: str = "parameters.pkl") -> Path:
    params = default_parameters()
    output_path = Path(__file__).resolve().parent / file_name
    with output_path.open("wb") as f:
        pickle.dump(params, f)
    return output_path


if __name__ == "__main__":
    path = write_parameters()
    print(f"Wrote default parameters to {path}")
