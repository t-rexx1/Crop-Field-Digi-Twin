"""Student assignment: synthetic data generation for plant-growth parameter estimation."""

from __future__ import annotations

import numpy as np

from .simulation import build_grid_positions, simulate_heights

DEFAULT_TRUE_PARAMS = {
    "G0": 0.10,
    "b_w": 0.55,
    "b_f": 0.35,
    "gamma": 0.90,
    "alpha": 0.95,
    "lambda": 0.08,
    "rho": 1.10,
}


def create_forcing_signals(
    time: np.ndarray,
    water_period_days: float = 7.0,
    water_pulse_days: float = 1.5,
    water_level: float = 1.0,
    fertilization_days: tuple[float, ...] = (20.0, 45.0, 70.0),
    fertilization_dose: float = 1.0,
    sun_base: float = 0.80,
    sun_amp: float = 0.18,
    sun_period_days: float = 30.0,
    sun_phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build W(t), F(t), and S(t) signals for one growing season."""
    wrapped = np.mod(time, water_period_days)
    W = water_level * (wrapped < water_pulse_days).astype(float)

    F = np.zeros_like(time, dtype=float)
    for day in fertilization_days:
        idx = int(np.argmin(np.abs(time - day)))
        F[idx] += fertilization_dose

    S = (
        sun_base
        + sun_amp * np.sin(2.0 * np.pi * time / sun_period_days + sun_phase)
        + 0.05 * np.sin(2.0 * np.pi * time / 7.0)
    )
    S = np.clip(S, 0.10, None)
    return W, F, S


def generate_dataset(
    true_params: dict | None = None,
    nx: int = 6,
    ny: int = 6,
    spacing: float = 0.5,
    season_days: float = 100.0,
    dt: float = 1.0,
    noise_sigma: float = 0.03,
    train_fraction: float = 0.7,
    seed: int = 3,
) -> dict:
    """Create synthetic observations from the nonlinear plant-growth model."""
    rng = np.random.default_rng(seed)
    params = DEFAULT_TRUE_PARAMS.copy() if true_params is None else dict(true_params)

    time = np.arange(0.0, season_days + dt, dt, dtype=float)
    positions = build_grid_positions(nx=nx, ny=ny, spacing=spacing)

    # Season 1 (training): baseline growing-season trajectory.
    initial_heights_train = rng.uniform(0.03, 0.08, size=positions.shape[0])
    W_train, F_train, S_train = create_forcing_signals(time)
    h_clean_train = simulate_heights(
        params=params,
        initial_heights=initial_heights_train,
        time=time,
        W=W_train,
        F=F_train,
        S=S_train,
        positions=positions,
    )
    h_obs_train = np.maximum(
        h_clean_train + rng.normal(0.0, noise_sigma, size=h_clean_train.shape),
        0.0,
    )

    # Season 2 (held-out test): independent season with shifted forcing and initial state.
    fert_shift_days = rng.integers(-4, 5, size=3)
    fertilization_days_test = tuple(
        np.clip(np.array((20.0, 45.0, 70.0), dtype=float) + fert_shift_days, 5.0, season_days - 5.0)
    )
    W_test, F_test, S_test = create_forcing_signals(
        time,
        water_period_days=float(np.clip(7.0 + rng.uniform(-0.7, 0.7), 5.5, 8.5)),
        water_pulse_days=float(np.clip(1.5 + rng.uniform(-0.4, 0.4), 0.6, 2.4)),
        fertilization_days=fertilization_days_test,
        fertilization_dose=float(np.clip(1.0 + rng.uniform(-0.3, 0.3), 0.5, 1.5)),
        sun_base=float(np.clip(0.80 + rng.uniform(-0.08, 0.08), 0.6, 1.0)),
        sun_amp=float(np.clip(0.18 + rng.uniform(-0.04, 0.04), 0.08, 0.28)),
        sun_phase=float(rng.uniform(-np.pi / 3.0, np.pi / 3.0)),
    )
    initial_heights_test = rng.uniform(0.03, 0.08, size=positions.shape[0])
    h_clean_test = simulate_heights(
        params=params,
        initial_heights=initial_heights_test,
        time=time,
        W=W_test,
        F=F_test,
        S=S_test,
        positions=positions,
    )
    h_obs_test = np.maximum(
        h_clean_test + rng.normal(0.0, noise_sigma, size=h_clean_test.shape),
        0.0,
    )

    train_idx = np.arange(len(time))
    test_idx = np.arange(len(time))

    return {
        "true_params": params,
        "positions": positions,
        "time": time,
        "W": W_train,
        "F": F_train,
        "S": S_train,
        "initial_heights": initial_heights_train,
        "h_clean": h_clean_train,
        "h_obs": h_obs_train,
        "W_test": W_test,
        "F_test": F_test,
        "S_test": S_test,
        "initial_heights_test": initial_heights_test,
        "h_clean_test": h_clean_test,
        "h_obs_test": h_obs_test,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_fraction": train_fraction,
        "noise_sigma": noise_sigma,
    }
