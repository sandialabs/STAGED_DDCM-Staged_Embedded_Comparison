from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# =============================================================================
# Configuration containers
# =============================================================================

@dataclass(frozen=True)
class SIQRScenario:
    beta: float
    gamma: float
    delta: float
    name: str = "cal"


@dataclass(frozen=True)
class ObservationConfig:
    N_total: int = 2_000_000

    I0: int = 500
    Q0: int = 10
    R0: int = 10

    N_days: float = 200.0
    step: float = 0.25
    step_obs: float = 0.25

    obs_var: float = 1.0e-6

    xi_threshold_count: float = 3.0e5
    xi_transition_width_fraction: float = 0.05

    method: str = "RK45"


# =============================================================================
# Scenario helpers
# =============================================================================

def calibration_scenario() -> SIQRScenario:
    """
    Calibration scenario used by the original notebook.
    """
    return SIQRScenario(
        beta=0.15,
        gamma=0.013,
        delta=0.01,
        name="cal",
    )


def validation_scenarios():
    """
    Validation/prediction scenarios from the original notebook.
    """
    return [
        SIQRScenario(beta=0.30, gamma=0.02, delta=0.01, name="val1"),
        SIQRScenario(beta=0.10, gamma=0.01, delta=0.01, name="val2"),
        SIQRScenario(beta=0.15, gamma=0.10, delta=0.01, name="val3"),
    ]


def default_observation_operator() -> np.ndarray:
    """
    Fully observed SIQR observation operator.
    """
    return np.eye(4)


# =============================================================================
# Initial condition, closure, and ODEs
# =============================================================================

def make_initial_condition(config: ObservationConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return dimensional and nondimensional SIQR initial conditions.

    Returns
    -------
    y0_counts:
        Shape (4,).
    y0_nondim:
        Shape (4,).
    """
    S0 = config.N_total - config.I0 - config.Q0 - config.R0

    y0_counts = np.array(
        [S0, config.I0, config.Q0, config.R0],
        dtype=float,
    )

    y0_nondim = y0_counts / config.N_total

    return y0_counts, y0_nondim


def xi_f_nondim(
    y,
    N_total: int = 2_000_000,
    threshold_count: float = 3.0e5,
    transition_width_fraction: float = 0.05,
):
    """
    True nondimensional closure function.

    xi(s, i) = s^{0.5} * 0.5 *
               [1 + tanh((i N - threshold_count)/(width_fraction N))]

    Supports y shape (4,) or (4, N).
    """
    y = np.asarray(y)

    S, I, Q, R = y

    width = transition_width_fraction * N_total

    return S**0.5 * 0.5 * (
        np.tanh((I * N_total - threshold_count) / width) + 1.0
    )


def SIQR_ODEs_nondim(
    t,
    y,
    scenario: SIQRScenario,
    config: ObservationConfig,
):
    """
    True nondimensional SIQR ODE with state-dependent closure.
    """
    beta = scenario.beta
    gamma = scenario.gamma
    delta = scenario.delta

    S, I, Q, R = y

    xi = xi_f_nondim(
        y,
        N_total=config.N_total,
        threshold_count=config.xi_threshold_count,
        transition_width_fraction=config.xi_transition_width_fraction,
    )

    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I - xi * I
    dQdt = xi * I - delta * Q
    dRdt = gamma * I + delta * Q

    return dSdt, dIdt, dQdt, dRdt


def SIQR_ODEs_const_xi(
    t,
    y,
    beta: float,
    gamma: float,
    xi: float,
    delta: float,
):
    """
    Nondimensional SIQR ODE with constant xi.
    """
    S, I, Q, R = y

    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I - xi * I
    dQdt = xi * I - delta * Q
    dRdt = gamma * I + delta * Q

    return dSdt, dIdt, dQdt, dRdt


# =============================================================================
# Time grids and simulation
# =============================================================================

def make_time_grid(config: ObservationConfig):
    """
    Construct full and observation time grids.
    """
    t_span = (0.0, float(config.N_days))

    t_full = np.arange(
        t_span[0],
        t_span[1]+config.step,
        step=config.step,
        dtype=float,
    )

    t_obs = np.arange(
        t_span[0],
        t_span[1]+config.step_obs,
        step=config.step_obs,
        dtype=float,
    )

    return t_span, t_full, t_obs


def simulate_true_siqr(
    scenario: SIQRScenario,
    config: ObservationConfig,
    t_eval: np.ndarray,
):
    """
    Simulate the true SIQR system.

    Returns
    -------
    y:
        Shape (4, len(t_eval)).
    """
    _, y0_nondim = make_initial_condition(config)

    soln = solve_ivp(
        fun=lambda t, y: SIQR_ODEs_nondim(t, y, scenario, config),
        t_span=(0.0, float(config.N_days)),
        y0=y0_nondim,
        method=config.method,
        t_eval=t_eval,
    )

    if not soln.success:
        raise RuntimeError(
            f"solve_ivp failed for scenario {scenario.name}: {soln.message}"
        )

    return soln.y


def simulate_constant_xi_siqr(
    beta: float,
    gamma: float,
    xi: float,
    delta: float,
    config: ObservationConfig,
    t_eval: np.ndarray,
):
    """
    Simulate the constant-xi SIQR model.
    """
    _, y0_nondim = make_initial_condition(config)

    soln = solve_ivp(
        fun=lambda t, y: SIQR_ODEs_const_xi(t, y, beta, gamma, xi, delta),
        t_span=(0.0, float(config.N_days)),
        y0=y0_nondim,
        method=config.method,
        t_eval=t_eval,
    )

    if not soln.success:
        raise RuntimeError(f"constant-xi solve_ivp failed: {soln.message}")

    return soln.y


# =============================================================================
# Observation generation
# =============================================================================

def add_observation_noise(
    y_obs: np.ndarray,
    H_true: np.ndarray,
    obs_var: float,
    rng: np.random.Generator,
):
    """
    Generate noisy observations

        d = H_true y_obs + eta,
        eta ~ N(0, obs_var I).

    Parameters
    ----------
    y_obs:
        Shape (4, N_obs).
    H_true:
        Shape (num_outputs, 4).
    obs_var:
        Scalar observation-noise variance.
    rng:
        NumPy random generator.

    Returns
    -------
    d:
        Shape (num_outputs, N_obs).
    """
    num_outputs = H_true.shape[0]
    num_obs = y_obs.shape[1]

    noise = np.sqrt(obs_var) * rng.normal(
        loc=0.0,
        scale=1.0,
        size=(num_outputs, num_obs),
    )

    return H_true @ y_obs + noise


def generate_observation_dataset(
    scenario: SIQRScenario,
    config: ObservationConfig,
    H_true: Optional[np.ndarray] = None,
    noise_seed: int = 0,
    include_full_trajectory: bool = True,
):
    """
    Generate one noisy SIQR observation dataset.

    Saved array conventions match your embedded training workflow:

        d.shape       == (num_outputs, N_obs)
        y_true.shape  == (4, N_obs)
        xi_true.shape == (N_obs,)
        t_obs.shape   == (N_obs,)

    The embedded trainer later converts observations to time-major form using:

        d_time_major = data["d"].T
    """
    if H_true is None:
        H_true = default_observation_operator()

    H_true = np.asarray(H_true, dtype=float)

    rng = np.random.default_rng(noise_seed)

    t_span, t_full, t_obs = make_time_grid(config)

    y_obs = simulate_true_siqr(
        scenario=scenario,
        config=config,
        t_eval=t_obs,
    )

    d = add_observation_noise(
        y_obs=y_obs,
        H_true=H_true,
        obs_var=config.obs_var,
        rng=rng,
    )

    xi_true = xi_f_nondim(
        y_obs,
        N_total=config.N_total,
        threshold_count=config.xi_threshold_count,
        transition_width_fraction=config.xi_transition_width_fraction,
    )

    y0_counts, y0_nondim = make_initial_condition(config)

    dataset = {
        "d": d,
        "xi_true": xi_true,
        "t_obs": t_obs,
        "y_true": y_obs,
        "H_true": H_true,
        "obs_var": np.array(config.obs_var),
        "noise_seed": np.array(noise_seed),
        "params": np.array(
            [scenario.beta, scenario.gamma, scenario.delta],
            dtype=float,
        ),
        "scenario_name": np.array(scenario.name),
        "y0_counts": y0_counts,
        "y0_nondim": y0_nondim,
        "N_total": np.array(config.N_total),
        "config_json": np.array(
            json.dumps(
                {
                    "scenario": asdict(scenario),
                    "config": asdict(config),
                }
            )
        ),
    }

    if include_full_trajectory:
        y_full = simulate_true_siqr(
            scenario=scenario,
            config=config,
            t_eval=t_full,
        )

        xi_full = xi_f_nondim(
            y_full,
            N_total=config.N_total,
            threshold_count=config.xi_threshold_count,
            transition_width_fraction=config.xi_transition_width_fraction,
        )

        dataset["t_full"] = t_full
        dataset["y_full"] = y_full
        dataset["xi_full"] = xi_full

    return dataset


# =============================================================================
# Optional constant-xi baseline from original notebook
# =============================================================================

def mse_objective_const_xi(
    xi_array,
    scenario: SIQRScenario,
    obs: np.ndarray,
    H_true: np.ndarray,
    config: ObservationConfig,
    t_eval: np.ndarray,
):
    """
    MSE objective for fitting a constant xi baseline in observation space.
    """
    xi = float(np.asarray(xi_array).item())

    y_model = simulate_constant_xi_siqr(
        beta=scenario.beta,
        gamma=scenario.gamma,
        xi=xi,
        delta=scenario.delta,
        config=config,
        t_eval=t_eval,
    )

    d_model = H_true @ y_model

    if d_model.shape != obs.shape:
        raise ValueError(
            f"d_model shape {d_model.shape} and obs shape {obs.shape} differ."
        )

    return np.mean((d_model - obs) ** 2)


def optimize_constant_xi(
    scenario: SIQRScenario,
    obs: np.ndarray,
    H_true: np.ndarray,
    config: ObservationConfig,
    t_eval: np.ndarray,
    xi_init: float = 0.05,
    bounds: Tuple[float, float] = (0.0, 1.0),
):
    """
    Fit a constant xi baseline to noisy observations.
    """
    result = minimize(
        fun=lambda xi: mse_objective_const_xi(
            xi_array=xi,
            scenario=scenario,
            obs=obs,
            H_true=H_true,
            config=config,
            t_eval=t_eval,
        ),
        x0=np.array([xi_init], dtype=float),
        bounds=[bounds],
        method="L-BFGS-B",
    )

    return float(result.x[0]), float(result.fun), result


def add_constant_xi_baseline(
    dataset: Dict[str, Any],
    scenario: SIQRScenario,
    config: ObservationConfig,
    xi_init: float = 0.05,
    bounds: Tuple[float, float] = (0.0, 1.0),
):
    """
    Add constant-xi baseline information to an existing dataset.
    """
    H_true = dataset["H_true"]
    t_obs = dataset["t_obs"]
    obs = dataset["d"]

    opt_xi, opt_xi_fun, _ = optimize_constant_xi(
        scenario=scenario,
        obs=obs,
        H_true=H_true,
        config=config,
        t_eval=t_obs,
        xi_init=xi_init,
        bounds=bounds,
    )

    const_states = simulate_constant_xi_siqr(
        beta=scenario.beta,
        gamma=scenario.gamma,
        xi=opt_xi,
        delta=scenario.delta,
        config=config,
        t_eval=t_obs,
    )

    dataset["opt_xi"] = np.array(opt_xi)
    dataset["opt_xi_fun"] = np.array(opt_xi_fun)
    dataset["const_states"] = const_states

    return dataset


# =============================================================================
# Saving/loading
# =============================================================================

def save_dataset(dataset: Dict[str, Any], filename):
    """
    Save observation dataset as npz.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez(filename, **dataset)


def load_dataset(filename):
    """
    Load observation dataset.
    """
    return np.load(filename, allow_pickle=True)


# =============================================================================
# Convenience command-line generation
# =============================================================================

def generate_replicate_files(
    output_dir: str | Path,
    num_replicates: int,
    base_noise_seed: int = 10_000,
    config: Optional[ObservationConfig] = None,
    scenario: Optional[SIQRScenario] = None,
    H_true: Optional[np.ndarray] = None,
    include_constant_baseline: bool = False,
):
    """
    Generate one observation file per replicate.

    Output layout:

        output_dir/
            rep_000/
                observations.npz
            rep_001/
                observations.npz
            ...
            metadata.json
    """
    if config is None:
        config = ObservationConfig()

    if scenario is None:
        scenario = calibration_scenario()

    if H_true is None:
        H_true = default_observation_operator()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "num_replicates": num_replicates,
        "base_noise_seed": base_noise_seed,
        "scenario": asdict(scenario),
        "config": asdict(config),
        "replicates": [],
    }

    for r in range(num_replicates):
        noise_seed = base_noise_seed + r
        rep_dir = output_dir / f"rep_{r:03d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        dataset = generate_observation_dataset(
            scenario=scenario,
            config=config,
            H_true=H_true,
            noise_seed=noise_seed,
            include_full_trajectory=True,
        )

        if include_constant_baseline:
            dataset = add_constant_xi_baseline(
                dataset=dataset,
                scenario=scenario,
                config=config,
            )

        filename = rep_dir / "observations.npz"
        save_dataset(dataset, filename)

        metadata["replicates"].append(
            {
                "replicate": r,
                "noise_seed": noise_seed,
                "file": str(filename),
            }
        )

        print(f"Saved replicate {r:03d} to {filename}")

    metadata_file = output_dir / "metadata.json"

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_file}")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SIQR synthetic observation replicate files."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/study_replicates",
    )

    parser.add_argument(
        "--num-replicates",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--base-noise-seed",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--obs-var",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--include-constant-baseline",
        action="store_true",
    )

    args = parser.parse_args()

    obs_config = ObservationConfig(
        obs_var=args.obs_var,
    )

    generate_replicate_files(
        output_dir=args.output_dir,
        num_replicates=args.num_replicates,
        base_noise_seed=args.base_noise_seed,
        config=obs_config,
        scenario=calibration_scenario(),
        H_true=default_observation_operator(),
        include_constant_baseline=args.include_constant_baseline,
    )