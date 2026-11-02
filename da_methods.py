from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any
import json

import numpy as np
from scipy.optimize import minimize

from Epidemiology_EnKF import Epidemiology_EnKF


@dataclass(frozen=True)
class EnKFConfig:
    """
    Configuration for augmented SIQR EnKF.

    Final staged-method protocol:

        The EnKF is run once using the union of the original training and
        validation observations. Test observations are never assimilated.

    The smoother is enabled by default. It can be disabled from the driver with:

        --no-smoother
    """

    ensemble_size: int = 100
    dt_int: float = 0.25

    xi0: float = 0.0
    caseid: int = 1

    use_smoother: bool = True

    closure_process_noise: float = 4.578125e-05

    estimate_process_noise: bool = False
    mle_stride: int = 10
    mle_seed: int = 1
    mle_initial: float = 2.0e-5
    mle_bounds: tuple[float, float] = (1.0e-8, 0.5)
    mle_method: str = "Nelder-Mead"
    mle_tol: float = 1.0e-5


def make_augmented_observation_operator(H_true: np.ndarray) -> np.ndarray:
    """
    Convert H_true, shape (num_outputs, 4), into augmented observation operator

        H_aug = [H_true, 0],

    shape (num_outputs, 5).
    """
    H_true = np.asarray(H_true, dtype=float)
    zero_col = np.zeros((H_true.shape[0], 1), dtype=float)

    return np.hstack([H_true, zero_col])


def make_gamma(obs_var: float, num_outputs: int) -> np.ndarray:
    """
    Existing EnKF code expects Gamma as a vector of diagonal variances.
    """
    return float(obs_var) * np.ones(num_outputs, dtype=float)


def make_sigma(
    closure_process_noise: float,
    state_dim_aug: int = 5,
) -> np.ndarray:
    """
    Existing notebook used a vector Sigma with nonzero entry only in closure state.
    """
    Sigma = np.zeros(state_dim_aug, dtype=float)
    Sigma[-1] = float(closure_process_noise)

    return Sigma


def make_augmented_initial_condition(
    y0_nondim: np.ndarray,
    xi0: float,
) -> np.ndarray:
    """
    Create augmented EnKF state:

        x0 = [S0, I0, Q0, R0, xi0].
    """
    y0_nondim = np.asarray(y0_nondim, dtype=float)

    return np.concatenate([y0_nondim, np.asarray([xi0], dtype=float)])


def get_calibration_indices(split) -> np.ndarray:
    """
    Return sorted train+validation indices for EnKF calibration.

    This is the hard-coded DA choice for the staged method:

        calibration_idx = train_idx union val_idx

    Test indices are intentionally excluded.
    """
    return np.sort(
        np.concatenate(
            [
                np.asarray(split.train_idx, dtype=int),
                np.asarray(split.val_idx, dtype=int),
            ]
        )
    )


def collapse_duplicate_times_keep_last(
    t: np.ndarray,
    x: np.ndarray,
    tol: float = 1.0e-10,
):
    """
    Collapse duplicate or nearly duplicate times, keeping the last occurrence.

    The EnKF stores both a forecast value and an analysis/smoothed value at
    assimilation times. The last occurrence is the post-analysis/post-smoothing
    value and is what we want downstream for staged NN training.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x)

    if len(t) == 0:
        return t, x

    keep_indices = []

    for i, ti in enumerate(t):
        if len(keep_indices) == 0:
            keep_indices.append(i)
            continue

        last_i = keep_indices[-1]

        if abs(ti - t[last_i]) <= tol:
            keep_indices[-1] = i
        else:
            keep_indices.append(i)

    keep_indices = np.asarray(keep_indices, dtype=int)

    return t[keep_indices], x[keep_indices]


def align_times(
    t_available: np.ndarray,
    x_available: np.ndarray,
    t_query: np.ndarray,
    tol: float = 1.0e-8,
):
    """
    Align output arrays to requested times by nearest time match.
    """
    t_available = np.asarray(t_available, dtype=float)
    t_query = np.asarray(t_query, dtype=float)

    idx = np.array(
        [np.argmin(np.abs(t_available - tq)) for tq in t_query],
        dtype=int,
    )

    max_err = np.max(np.abs(t_available[idx] - t_query))

    if max_err > tol:
        raise ValueError(
            f"Could not align EnKF output times to query times. "
            f"Maximum mismatch was {max_err}, tolerance is {tol}."
        )

    return x_available[idx], idx


def build_enkf_model(
    closure_process_noise: float,
    obs_var: float,
    H_true: np.ndarray,
    params: tuple[float, float, float],
    cfg: EnKFConfig,
) -> Epidemiology_EnKF:
    """
    Construct Epidemiology_EnKF model using loaded replicate metadata.
    """
    H_aug = make_augmented_observation_operator(H_true)
    Gamma = make_gamma(obs_var, H_true.shape[0])
    Sigma = make_sigma(closure_process_noise)

    model = Epidemiology_EnKF(
        Sigma=Sigma,
        Gamma=Gamma,
        H=H_aug,
        dt_int=cfg.dt_int,
        model_params=params,
        caseid=cfg.caseid,
    )

    return model


def estimate_closure_process_noise(
    d_col_major: np.ndarray,
    t_obs: np.ndarray,
    split,
    y0_aug: np.ndarray,
    obs_var: float,
    H_true: np.ndarray,
    params: tuple[float, float, float],
    cfg: EnKFConfig,
):
    """
    Estimate closure process-noise parameter using EnKF likelihood.

    This is hard-coded to use train+validation observations only.

    Test observations are never used.

    Returns
    -------
    sigma_hat:
        Estimated closure process-noise value.
    result:
        scipy.optimize.OptimizeResult.
    mle_model_evaluations:
        Total model-evaluation count accumulated across likelihood evaluations.

    Notes
    -----
    In the current EnKF implementation, one call to compute_log_likelihood(...)
    increments model.num_eval by N_ensembles. Therefore, this accumulated count
    corresponds to the previous manuscript-style staged training cost metric:

        MLE model evaluations = mle_nfev * ensemble_size

    except that this implementation accumulates the actual EnKF model counter
    after each likelihood call.
    """
    estimate_idx = get_calibration_indices(split)

    if cfg.mle_stride is not None and cfg.mle_stride > 1:
        estimate_idx = estimate_idx[:: cfg.mle_stride]

    d_like = d_col_major[:, estimate_idx]
    t_like = t_obs[estimate_idx]

    lower, upper = cfg.mle_bounds

    mle_model_evaluations = 0

    def objective(x):
        nonlocal mle_model_evaluations

        sigma_val = float(np.asarray(x).ravel()[0])

        if sigma_val <= lower or sigma_val >= upper:
            return np.inf

        Sigma = make_sigma(sigma_val)

        model = build_enkf_model(
            closure_process_noise=sigma_val,
            obs_var=obs_var,
            H_true=H_true,
            params=params,
            cfg=cfg,
        )

        try:
            model.rng = np.random.default_rng(cfg.mle_seed)
        except Exception:
            pass

        try:
            loglike = np.sum(
                model.compute_log_likelihood(
                    y0_aug,
                    float(t_obs[0]),
                    d_like,
                    t_like,
                    Sigma,
                    N_ensembles=cfg.ensemble_size,
                    sample=True,
                )
            )

            try:
                mle_model_evaluations += int(model.get_model_evaluations())
            except Exception:
                mle_model_evaluations += int(cfg.ensemble_size)

        except Exception:
            return np.inf

        if not np.isfinite(loglike):
            return np.inf

        return -1.0 * float(loglike)

    result = minimize(
        objective,
        x0=np.asarray([cfg.mle_initial], dtype=float),
        method=cfg.mle_method,
        bounds=[cfg.mle_bounds],
        tol=cfg.mle_tol,
    )

    sigma_hat = float(result.x[0])

    return sigma_hat, result, int(mle_model_evaluations)


def run_enkf_assimilation(
    d_col_major: np.ndarray,
    t_obs: np.ndarray,
    split,
    y0_aug: np.ndarray,
    obs_var: float,
    H_true: np.ndarray,
    params: tuple[float, float, float],
    cfg: EnKFConfig,
    closure_process_noise: float,
    rng_seed: int,
):
    """
    Run EnKF assimilation using train+validation observations.

    This is hard-coded for the final staged algorithm:

        assim_idx = train_idx union val_idx

    Test observations are intentionally excluded.
    """
    assim_idx = get_calibration_indices(split)
    t_assim = t_obs[assim_idx]
    d_assim = d_col_major[:, assim_idx]

    model = build_enkf_model(
        closure_process_noise=closure_process_noise,
        obs_var=obs_var,
        H_true=H_true,
        params=params,
        cfg=cfg,
    )

    try:
        model.rng = np.random.default_rng(rng_seed)
    except Exception:
        pass

    t_model = np.asarray(t_obs, dtype=float)

    t_mu_raw, x_mu_raw, x_sd_raw, x_ens_raw = model.data_assimilation(
        y0_aug,
        d_assim,
        t_model,
        t_assim,
        N_ensembles=cfg.ensemble_size,
        smoother=cfg.use_smoother,
    )

    # Collapse duplicate output times, keeping post-analysis/post-smoothing rows.
    t_mu, x_mu = collapse_duplicate_times_keep_last(t_mu_raw, x_mu_raw)
    _, x_sd = collapse_duplicate_times_keep_last(t_mu_raw, x_sd_raw)
    _, x_ens = collapse_duplicate_times_keep_last(t_mu_raw, x_ens_raw)

    try:
        model_evaluations = int(model.get_model_evaluations())
    except Exception:
        model_evaluations = -1

    x_mu_assim, align_idx = align_times(t_mu, x_mu, t_assim)
    x_sd_assim, _ = align_times(t_mu, x_sd, t_assim)
    x_ens_assim, _ = align_times(t_mu, x_ens, t_assim)

    return {
        "model": model,
        "assim_idx": assim_idx,
        "t_assim": t_assim,
        "t_mu": t_mu,
        "x_mu": x_mu,
        "x_sd": x_sd,
        "x_ens": x_ens,
        "x_mu_assim": x_mu_assim,
        "x_sd_assim": x_sd_assim,
        "x_ens_assim": x_ens_assim,
        "align_idx": align_idx,
        "model_evaluations": model_evaluations,
        "use_smoother": bool(cfg.use_smoother),
    }


def save_enkf_result(
    filename,
    *,
    enkf_output: Dict[str, Any],
    cfg: EnKFConfig,
    closure_process_noise: float,
    noise_seed: int,
    split_seed: int,
    enkf_seed: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    obs_metadata: Dict[str, Any],
    mle_result=None,
    mle_model_evaluations: int = 0,
    save_sd: bool = False,
    save_ensemble: bool = False,
):
    """
    Save compact EnKF output.

    The staged NN needs:

        full trajectory:
            t_mu, x_mu

        assimilation-step:
            t_assim, x_mu_assim

    Standard deviations and ensembles are optional.

    Model-evaluation accounting
    ---------------------------
    model_evaluations:
        Final EnKF assimilation model evaluations only.

    mle_model_evaluations:
        Process-noise MLE model evaluations only.

    total_model_evaluations:
        MLE model evaluations + final EnKF assimilation model evaluations.

    For the manuscript training-cost comparison of STAGED versus embedded, use:

        total_model_evaluations

    because staged supervised NN training does not invoke the ODE/simulation model.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    mle_success = False
    mle_fun = np.nan
    mle_nfev = -1

    if mle_result is not None:
        mle_success = bool(getattr(mle_result, "success", False))
        mle_fun = float(getattr(mle_result, "fun", np.nan))
        mle_nfev = int(getattr(mle_result, "nfev", -1))

    enkf_model_evaluations = int(enkf_output["model_evaluations"])
    mle_model_evaluations = int(mle_model_evaluations)
    total_model_evaluations = int(
        mle_model_evaluations + enkf_model_evaluations
    )

    arrays_to_save = {
        "assim_idx": np.asarray(enkf_output["assim_idx"]),
        "train_idx": np.asarray(train_idx),
        "val_idx": np.asarray(val_idx),
        "test_idx": np.asarray(test_idx),

        "t_assim": np.asarray(enkf_output["t_assim"]),
        "x_mu_assim": np.asarray(enkf_output["x_mu_assim"]),
        "t_mu": np.asarray(enkf_output["t_mu"]),
        "x_mu": np.asarray(enkf_output["x_mu"]),

        "closure_process_noise": np.array(closure_process_noise),

        # Final EnKF assimilation model evaluations only.
        "model_evaluations": np.array(enkf_model_evaluations),

        # Process-noise MLE model evaluations only.
        "mle_model_evaluations": np.array(mle_model_evaluations),

        # Total staged DA-stage model evaluations:
        # process-noise MLE + final EnKF assimilation.
        "total_model_evaluations": np.array(total_model_evaluations),

        "noise_seed": np.array(noise_seed),
        "split_seed": np.array(split_seed),
        "enkf_seed": np.array(enkf_seed),
        "obs_var": np.array(obs_metadata["obs_var"]),
        "params": np.asarray(obs_metadata["params"]),

        "use_smoother": np.array(bool(cfg.use_smoother)),
        "mle_success": np.array(mle_success),
        "mle_fun": np.array(mle_fun),
        "mle_nfev": np.array(mle_nfev),

        "assimilation_policy": np.array("trainval_hard_coded"),
        "process_noise_estimation_policy": np.array("trainval_hard_coded"),
    }

    if save_sd:
        arrays_to_save.update(
            {
                "x_sd": np.asarray(enkf_output["x_sd"]),
                "x_sd_assim": np.asarray(enkf_output["x_sd_assim"]),
            }
        )

    if save_ensemble:
        arrays_to_save.update(
            {
                "x_ens": np.asarray(enkf_output["x_ens"]),
                "x_ens_assim": np.asarray(enkf_output["x_ens_assim"]),
            }
        )

    np.savez(filename, **arrays_to_save)

    metadata = {
        "assimilation_policy": "trainval_hard_coded",
        "process_noise_estimation_policy": "trainval_hard_coded",
        "method_note": (
            "EnKF assimilated original train+validation observations only. "
            "Original test observations were not assimilated."
        ),
        "closure_process_noise": float(closure_process_noise),
        "use_smoother": bool(cfg.use_smoother),

        # Final EnKF assimilation only.
        "model_evaluations": int(enkf_model_evaluations),

        # MLE only.
        "mle_model_evaluations": int(mle_model_evaluations),

        # MLE + final EnKF assimilation.
        "total_model_evaluations": int(total_model_evaluations),

        "noise_seed": int(noise_seed),
        "split_seed": int(split_seed),
        "enkf_seed": int(enkf_seed),
        "obs_var": float(obs_metadata["obs_var"]),
        "params": np.asarray(obs_metadata["params"]).tolist(),
        "num_train_obs": int(len(train_idx)),
        "num_val_obs": int(len(val_idx)),
        "num_test_obs": int(len(test_idx)),
        "num_assim_obs": int(len(enkf_output["assim_idx"])),
        "save_sd": bool(save_sd),
        "save_ensemble": bool(save_ensemble),
        "enkf_config": asdict(cfg),
        "mle": {
            "success": bool(mle_success),
            "fun": float(mle_fun),
            "nfev": int(mle_nfev),
            "model_evaluations": int(mle_model_evaluations),
        },
    }

    metadata_file = filename.with_name(filename.stem + "_metadata.json")

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)