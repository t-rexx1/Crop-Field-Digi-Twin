"""Student assignment: numerical simulation for interacting plant-height dynamics."""

from __future__ import annotations

import numpy as np

EPS = 1.0e-9


def build_grid_positions(nx: int, ny: int, spacing: float) -> np.ndarray:
    """Create a 2D grid of plant locations with shape (nx * ny, 2)."""
    x = np.arange(nx, dtype=float) * spacing
    y = np.arange(ny, dtype=float) * spacing
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distance matrix for plant locations."""
    # FILL IN THE BLANK: Given `positions` with shape (N, 2), return an (N, N)
    # matrix where entry (i, j) is the Euclidean distance between plant i and j.

    # Placeholder return
    return np.zeros((positions.shape[0], positions.shape[0]), dtype=float)


def interaction_kernel(distances: np.ndarray, rho: float) -> np.ndarray:
    """Compute interaction matrix exp(-r_ij / rho) * 1(r_ij <= rho), with zero diagonal."""
    # FILL IN THE BLANK: Build K_ij = exp(-r_ij/rho) * 1(r_ij <= rho) and ensure
    # the diagonal is zero (no self-interaction term).
    _ = rho # Delete me

    # Placeholder return
    return np.zeros_like(distances, dtype=float)


def simulate_heights(
    params: dict,
    initial_heights: np.ndarray,
    time: np.ndarray,
    W: np.ndarray,
    F: np.ndarray,
    S: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Integrate dh_i/dt = G*h_i^alpha - sum_{j!=i} I_ij with explicit Euler.

    Parameters expected in params:
    - G0, b_w, b_f, gamma, alpha, lambda, rho
    """
    num_steps = len(time)
    num_plants = len(initial_heights)

    if num_steps < 2:
        raise ValueError("time array must contain at least two steps")

    dt = float(time[1] - time[0])
    max_height = float(params.get("max_height", 5.0))
    heights = np.zeros((num_steps, num_plants), dtype=float)
    heights[0] = np.maximum(initial_heights, 0.0)

    distances = compute_pairwise_distances(positions)
    kernel = interaction_kernel(distances, params["rho"])

    for k in range(num_steps - 1):
        current = np.clip(heights[k], 0.0, max_height)

        # FILL IN THE BLANK: Implement the model update for one Euler step.
        # Use the equations in the project handout:
        #   G = G0 * (1 + b_w*W + b_f*F) * S^gamma
        #   dh_i/dt = G * h_i^alpha - lambda * sum_{j!=i}(h_j * exp(-r_ij/rho) * 1(r_ij <= rho))
        # and include numerical safety clipping similar to current/next height handling.

        _ = (current, dt, kernel, W[k], F[k], S[k], params) # Delete me

    return heights
