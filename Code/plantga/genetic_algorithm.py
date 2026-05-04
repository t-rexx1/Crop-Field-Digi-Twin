"""Student assignment: simple GA estimator for plant-growth model parameters."""

from __future__ import annotations

import numpy as np

from .simulation import simulate_heights


def normalized_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return a scale-normalized objective value from y_true and y_pred."""
    # FILL IN THE BLANK: Implement the objective function used by the GA.
    # use the normalized MSE, i.e.
    #   mean((y_pred - y_true)^2) / (var(y_true) + eps)

    _ = (y_true, y_pred) # Delete me
  
    # Include non-finite handling to return a large penalty when needed.
    if not np.all(np.isfinite(y_pred)):
        return 1.0e6
    return 0 # Placeholder return


class GeneticAlgorithmEstimator:
    """Genetic algorithm for minimizing training error."""

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        population_size: int = 36,
        num_generations: int = 60,
        parent_fraction: float = 0.4,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.08,
        elitism: int = 2, # top performing parents kept
        patience: int = 12, # wait this many generations without improvement before terminating search
        seed: int = 11,
    ) -> None:
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.lower_bounds = np.array([bounds[p][0] for p in self.param_names], dtype=float)
        self.upper_bounds = np.array([bounds[p][1] for p in self.param_names], dtype=float)

        self.population_size = int(population_size)
        self.num_generations = int(num_generations)
        self.parent_fraction = float(parent_fraction)
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self.elitism = int(max(1, min(elitism, population_size - 1)))
        self.patience = int(max(1, patience))

        self.dim = len(self.param_names)
        self.rng = np.random.default_rng(seed)

    def _vector_to_params(self, vector: np.ndarray) -> dict[str, float]:
        return {name: float(vector[i]) for i, name in enumerate(self.param_names)}

    def _sample_population(self) -> np.ndarray:
        return self.rng.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(self.population_size, self.dim),
        )

    def _evaluate_individual(
        self,
        vector: np.ndarray,
        dataset: dict,
    ) -> float:
        params = self._vector_to_params(vector)

        # generate heights from candidate model
        pred_train = simulate_heights(
            params=params,
            initial_heights=dataset["initial_heights"],
            time=dataset["time"],
            W=dataset["W"],
            F=dataset["F"],
            S=dataset["S"],
            positions=dataset["positions"],
        )

        if not np.all(np.isfinite(pred_train)):
            return 1.0e6

        train_obs = dataset["h_obs"][dataset["train_idx"]]
        train_pred = pred_train[dataset["train_idx"]]

        train_mse = normalized_mse(train_obs, train_pred)

        return float(train_mse)

    def _breed_children(self, parents: np.ndarray) -> np.ndarray:
        target_children = self.population_size - self.elitism
        span = self.upper_bounds - self.lower_bounds
        children = np.zeros((target_children, self.dim), dtype=float)

        for i in range(target_children):
            p1, p2 = self.rng.integers(0, len(parents), size=2)
            blend = self.rng.random(self.dim)
            child = blend * parents[p1] + (1.0 - blend) * parents[p2]

            mutation_mask = self.rng.random(self.dim) < self.mutation_rate
            if np.any(mutation_mask):
                child = child + mutation_mask * self.rng.normal(0.0, self.mutation_scale, size=self.dim) * span

            children[i] = np.clip(child, self.lower_bounds, self.upper_bounds)

        return children

    def run(self, dataset: dict) -> dict:
        population = self._sample_population()

        best_train = float("inf")
        best_vector = population[0].copy()
        no_improve = 0

        history = {
            "best_train": [],
            "mean_train": [],
        }

        last_generation = 0
        for generation in range(self.num_generations):
            train_costs = np.zeros(self.population_size, dtype=float)

            for i in range(self.population_size):
                train_costs[i] = self._evaluate_individual(
                    population[i],
                    dataset,
                )

            rank = np.argsort(train_costs)
            population = population[rank]
            train_costs = train_costs[rank]

            if train_costs[0] < best_train:
                best_train = float(train_costs[0])
                best_vector = population[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            history["best_train"].append(float(np.min(train_costs)))
            history["mean_train"].append(float(np.mean(train_costs)))
            last_generation = generation + 1

            if no_improve >= self.patience:
                break

            n_parents = int(np.ceil(self.parent_fraction * self.population_size))
            n_parents = max(2, min(n_parents, self.population_size))
            parents = population[:n_parents]

            elites = population[: self.elitism]
            children = self._breed_children(parents)
            population = np.vstack([elites, children])

        return {
            "best_vector": best_vector,
            "best_params": self._vector_to_params(best_vector),
            "best_train_cost": best_train,
            "history": history,
            "generations_run": last_generation,
            "param_names": self.param_names,
        }
