import os

# Set these before importing JAX or anything that imports JAX.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import argparse
import json
import pickle
import traceback

import numpy as np

import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr

import equinox as eqx

from embedded_NN_methods import SIQRParams

from staged_NN_methods import (
    StagedNNConfig,
    StagedClosureNet,
    StagedNeuralODE,
    evaluate_staged_closure,
)

from siqr_observation_data import (
    ObservationConfig,
    make_initial_condition,
    simulate_true_siqr,
    xi_f_nondim,
    validation_scenarios,
)

from study_run_utils import load_pickle_if_exists


# =============================================================================
# Small file helpers
# =============================================================================

def rep_dir(study_dir, r: int) -> Path:
    return Path(study_dir) / f"rep_{r:03d}"


def load_json_if_exists(path):
    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def load_observation_config_from_obs(obs_npz) -> ObservationConfig:
    """
    Reconstruct the ObservationConfig from observations.npz.

    Falls back to ObservationConfig(obs_var=obs_var) if config_json is missing.
    """
    if "config_json" not in obs_npz.files:
        return ObservationConfig(obs_var=float(np.asarray(obs_npz["obs_var"])))

    raw = np.asarray(obs_npz["config_json"]).item()
    parsed = json.loads(raw)
    config_dict = parsed["config"]

    return ObservationConfig(**config_dict)


def staged_generalization_is_successful(rep_path: Path) -> bool:
    """
    Completion check for the generalization stage.
    """
    result_file = rep_path / "staged_generalization.npz"
    summary_file = rep_path / "staged_generalization_summary.json"

    if not result_file.exists() or not summary_file.exists():
        return False

    summary = load_json_if_exists(summary_file)

    if summary is None:
        return False

    return bool(summary.get("success", False))


def staged_generalization_has_failed(rep_path: Path) -> bool:
    """
    Failure check for the generalization stage.
    """
    error_file = rep_path / "staged_generalization_error.json"
    summary_file = rep_path / "staged_generalization_summary.json"

    if error_file.exists():
        return True

    summary = load_json_if_exists(summary_file)

    if summary is not None and summary.get("success", True) is False:
        return True

    return False


def staged_generalization_prerequisites_exist(rep_path: Path) -> bool:
    """
    Required files for staged generalization.
    """
    required = [
        "observations.npz",
        "staged_model.eqx",
        "staged_model_config.json",
        "staged_closure.npz",
        "staged_summary.json",
    ]

    return all((rep_path / name).exists() for name in required)


def select_generalization_replicates_to_run(
    study_dir,
    num_replicates: int,
    *,
    overwrite: bool = False,
    rerun_failed: bool = True,
):
    """
    Select replicate IDs for staged generalization rollout.
    """
    study_dir = Path(study_dir)

    tasks = []
    skipped_success = []
    skipped_failed = []
    skipped_missing_prereq = []

    for r in range(num_replicates):
        rp = rep_dir(study_dir, r)

        if overwrite:
            if not staged_generalization_prerequisites_exist(rp):
                skipped_missing_prereq.append(r)
            else:
                tasks.append(r)
            continue

        if staged_generalization_is_successful(rp):
            skipped_success.append(r)
            continue

        if staged_generalization_has_failed(rp) and not rerun_failed:
            skipped_failed.append(r)
            continue

        if not staged_generalization_prerequisites_exist(rp):
            skipped_missing_prereq.append(r)
            continue

        tasks.append(r)

    return tasks, skipped_success, skipped_failed, skipped_missing_prereq


def print_selection_summary(
    *,
    num_replicates,
    tasks,
    skipped_success,
    skipped_failed,
    skipped_missing_prereq,
):
    print("\n" + "=" * 80)
    print("Stage: staged_generalization")
    print("=" * 80)
    print(f"Requested replicate count:          {num_replicates}")
    print(f"Replicates selected to run:         {len(tasks)}")
    print(f"Replicates skipped, successful:     {len(skipped_success)}")
    print(f"Replicates skipped, failed:         {len(skipped_failed)}")
    print(f"Replicates skipped, missing prereq: {len(skipped_missing_prereq)}")

    if len(tasks) > 0:
        print(f"First selected replicates: {tasks[:10]}")
        print(f"Last selected replicates:  {tasks[-10:]}")

    print("=" * 80 + "\n")


def detect_num_replicates(study_dir: Path) -> int:
    rep_dirs = sorted(study_dir.glob("rep_*"))
    return len(rep_dirs)


# =============================================================================
# Model reconstruction and rollout
# =============================================================================

def load_staged_net_for_replicate(rep_path: Path):
    """
    Reconstruct and load a saved staged closure network.
    """
    model_file = rep_path / "staged_model.eqx"
    config_file = rep_path / "staged_model_config.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Missing staged model file: {model_file}")

    if not config_file.exists():
        raise FileNotFoundError(f"Missing staged model config file: {config_file}")

    with open(config_file, "r") as f:
        model_config = json.load(f)

    input_dim = int(model_config.get("input_dim", 4))
    width_size = int(model_config["width_size"])
    depth = int(model_config["depth"])

    template = StagedClosureNet(
        state_dim=input_dim,
        width_size=width_size,
        depth=depth,
        key=jr.PRNGKey(0),
        activation=jnn.sigmoid,
    )

    staged_net = eqx.tree_deserialise_leaves(model_file, template)

    return staged_net, model_config


def make_rollout_config(model_config: dict, args_dict: dict) -> StagedNNConfig:
    """
    Construct a StagedNNConfig for autonomous ODE rollout.

    Command-line values override model-config values when provided.
    """
    rtol = (
        float(args_dict["rtol"])
        if args_dict.get("rtol") is not None
        else float(model_config.get("rtol", 1.0e-5))
    )

    atol = (
        float(args_dict["atol"])
        if args_dict.get("atol") is not None
        else float(model_config.get("atol", 1.0e-7))
    )

    max_steps = (
        int(args_dict["max_steps"])
        if args_dict.get("max_steps") is not None
        else int(model_config.get("max_steps", 16384))
    )

    return StagedNNConfig(
        width_size=int(model_config["width_size"]),
        depth=int(model_config["depth"]),
        activation=jnn.sigmoid,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )


def rollout_staged_net_for_scenario(
    staged_net,
    scenario,
    t_eval,
    y0,
    cfg: StagedNNConfig,
):
    """
    Roll out one staged NN under one validation scenario.

    Returns
    -------
    dict with xi_pred, y_pred, success, error.
    """
    params = SIQRParams(
        beta=float(scenario.beta),
        gamma=float(scenario.gamma),
        delta=float(scenario.delta),
    )

    ode_model = StagedNeuralODE(
        net=staged_net,
        params=params,
        t0=float(t_eval[0]),
        t1=float(t_eval[-1]),
        dt0=float(t_eval[1] - t_eval[0]),
        cfg=cfg,
    )

    try:
        xi_pred, y_pred = evaluate_staged_closure(
            ode_model,
            jnp.asarray(t_eval),
            jnp.asarray(y0),
        )

        xi_pred_np = np.asarray(xi_pred, dtype=float)
        y_pred_np = np.asarray(y_pred, dtype=float)

        if (
            not np.all(np.isfinite(xi_pred_np))
            or not np.all(np.isfinite(y_pred_np))
        ):
            raise RuntimeError("Staged rollout produced nonfinite values.")

        return {
            "xi_pred": xi_pred_np,
            "y_pred": y_pred_np,
            "success": True,
            "error": None,
        }

    except Exception as exc:
        return {
            "xi_pred": np.full(len(t_eval), np.nan, dtype=float),
            "y_pred": np.full((len(t_eval), len(y0)), np.nan, dtype=float),
            "success": False,
            "error": repr(exc),
        }


def state_rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if not np.all(np.isfinite(y_pred)):
        return float(np.inf)

    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def closure_rmse(xi_pred, xi_true):
    xi_pred = np.asarray(xi_pred, dtype=float)
    xi_true = np.asarray(xi_true, dtype=float)

    if not np.all(np.isfinite(xi_pred)):
        return float(np.inf)

    return float(np.sqrt(np.mean((xi_pred - xi_true) ** 2)))


# =============================================================================
# Per-replicate worker
# =============================================================================

def run_one_replicate_generalization(
    r: int,
    study_dir: str,
    args_dict: dict,
):
    """
    Run staged-model generalization rollouts for one replicate.
    """
    try:
        study_dir = Path(study_dir)
        rp = rep_dir(study_dir, r)

        obs_file = rp / "observations.npz"
        staged_summary_file = rp / "staged_summary.json"

        result_file = rp / "staged_generalization.npz"
        summary_file = rp / "staged_generalization_summary.json"

        if not obs_file.exists():
            raise FileNotFoundError(f"Missing observation file: {obs_file}")

        if not staged_summary_file.exists():
            raise FileNotFoundError(
                f"Missing staged summary file: {staged_summary_file}"
            )

        staged_summary = load_json_if_exists(staged_summary_file)

        if not bool(staged_summary.get("success", False)):
            raise RuntimeError(
                f"Staged replicate {r:03d} was not successful; "
                "not running generalization."
            )

        # ---------------------------------------------------------------------
        # Load replicate observation metadata.
        # ---------------------------------------------------------------------
        obs = np.load(obs_file, allow_pickle=True)
        obs_config = load_observation_config_from_obs(obs)

        # Use the same observation grid as the original calibration data.
        t_eval = np.asarray(obs["t_obs"], dtype=float)

        _, y0_nondim = make_initial_condition(obs_config)

        # ---------------------------------------------------------------------
        # Load staged NN.
        # ---------------------------------------------------------------------
        staged_net, model_config = load_staged_net_for_replicate(rp)
        rollout_cfg = make_rollout_config(model_config, args_dict)

        # ---------------------------------------------------------------------
        # Validation scenarios.
        # ---------------------------------------------------------------------
        scenarios = validation_scenarios()
        n_scenarios = len(scenarios)
        n_times = len(t_eval)

        scenario_names = []
        scenario_params = np.zeros((n_scenarios, 3), dtype=float)

        xi_true_all = np.zeros((n_scenarios, n_times), dtype=float)
        y_true_all = np.zeros((n_scenarios, n_times, 4), dtype=float)

        xi_pred_all = np.full((n_scenarios, n_times), np.nan, dtype=float)
        y_pred_all = np.full((n_scenarios, n_times, 4), np.nan, dtype=float)

        success_all = np.zeros(n_scenarios, dtype=bool)
        state_rmse_all = np.full(n_scenarios, np.inf, dtype=float)
        closure_rmse_all = np.full(n_scenarios, np.inf, dtype=float)
        error_messages = []

        for s_idx, scenario in enumerate(scenarios):
            scenario_names.append(str(scenario.name))
            scenario_params[s_idx, :] = np.array(
                [scenario.beta, scenario.gamma, scenario.delta],
                dtype=float,
            )

            # True validation trajectory.
            y_true = simulate_true_siqr(
                scenario=scenario,
                config=obs_config,
                t_eval=t_eval,
            )

            xi_true = xi_f_nondim(
                y_true,
                N_total=obs_config.N_total,
                threshold_count=obs_config.xi_threshold_count,
                transition_width_fraction=obs_config.xi_transition_width_fraction,
            )

            # Store true data time-major for states.
            y_true_all[s_idx, :, :] = y_true.T
            xi_true_all[s_idx, :] = xi_true

            # Learned staged rollout.
            rollout = rollout_staged_net_for_scenario(
                staged_net=staged_net,
                scenario=scenario,
                t_eval=t_eval,
                y0=y0_nondim,
                cfg=rollout_cfg,
            )

            xi_pred_all[s_idx, :] = rollout["xi_pred"]
            y_pred_all[s_idx, :, :] = rollout["y_pred"]
            success_all[s_idx] = bool(rollout["success"])

            if rollout["success"]:
                state_rmse_all[s_idx] = state_rmse(
                    y_pred=rollout["y_pred"],
                    y_true=y_true.T,
                )
                closure_rmse_all[s_idx] = closure_rmse(
                    xi_pred=rollout["xi_pred"],
                    xi_true=xi_true,
                )
                error_messages.append("")
            else:
                error_messages.append(str(rollout["error"]))

        overall_success = bool(np.all(success_all))

        # ---------------------------------------------------------------------
        # Save per-replicate arrays.
        # ---------------------------------------------------------------------
        np.savez(
            result_file,
            replicate=np.array(r),
            success=np.array(overall_success),
            scenario_success=np.asarray(success_all),
            scenario_names=np.asarray(scenario_names),
            scenario_params=np.asarray(scenario_params),
            t_eval=np.asarray(t_eval),
            y0_nondim=np.asarray(y0_nondim),
            y_true=np.asarray(y_true_all),
            xi_true=np.asarray(xi_true_all),
            y_pred=np.asarray(y_pred_all),
            xi_pred=np.asarray(xi_pred_all),
            state_rmse=np.asarray(state_rmse_all),
            closure_rmse=np.asarray(closure_rmse_all),
            error_messages=np.asarray(error_messages),
            rtol=np.array(rollout_cfg.rtol),
            atol=np.array(rollout_cfg.atol),
            max_steps=np.array(rollout_cfg.max_steps),
        )

        summary = {
            "replicate": int(r),
            "success": bool(overall_success),
            "result_file": str(result_file),
            "summary_file": str(summary_file),
            "scenario_names": scenario_names,
            "scenario_success": [bool(v) for v in success_all],
            "scenario_params": scenario_params.tolist(),
            "state_rmse": state_rmse_all.tolist(),
            "closure_rmse": closure_rmse_all.tolist(),
            "error_messages": error_messages,
            "rtol": float(rollout_cfg.rtol),
            "atol": float(rollout_cfg.atol),
            "max_steps": int(rollout_cfg.max_steps),
            "method_note": (
                "Generalization rollouts use each trained staged NN closure "
                "embedded in SIQR ODEs under validation scenario parameters."
            ),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    except Exception as exc:
        err_summary = {
            "replicate": int(r),
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

        study_dir = Path(study_dir)
        rp = rep_dir(study_dir, r)
        rp.mkdir(parents=True, exist_ok=True)

        with open(rp / "staged_generalization_error.json", "w") as f:
            json.dump(err_summary, f, indent=2)

        return err_summary


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel generalization rollouts for trained staged NN closures. "
            "Each trained staged_model.eqx is embedded into the SIQR ODE and "
            "evaluated on validation_scenarios()."
        )
    )

    parser.add_argument(
        "--study-dir",
        type=str,
        default="data/study_replicates",
    )

    parser.add_argument(
        "--num-replicates",
        type=int,
        default=None,
        help="If omitted, infer from rep_* directories.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Default: min(8, number of selected tasks).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, rerun all requested generalization replicates.",
    )

    parser.add_argument(
        "--no-rerun-failed",
        action="store_true",
        help="If set, do not rerun previously failed generalization replicates.",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help=(
            "Override rollout solver rtol. If omitted, use value saved in "
            "staged_model_config.json."
        ),
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help=(
            "Override rollout solver atol. If omitted, use value saved in "
            "staged_model_config.json."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Override rollout solver max_steps. If omitted, use value saved in "
            "staged_model_config.json."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    study_dir = Path(args.study_dir)

    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory does not exist: {study_dir}")

    if args.num_replicates is None:
        num_replicates = detect_num_replicates(study_dir)
    else:
        num_replicates = args.num_replicates

    args_dict = {
        "rtol": args.rtol,
        "atol": args.atol,
        "max_steps": args.max_steps,
    }

    tasks, skipped_success, skipped_failed, skipped_missing_prereq = (
        select_generalization_replicates_to_run(
            study_dir,
            num_replicates,
            overwrite=args.overwrite,
            rerun_failed=not args.no_rerun_failed,
        )
    )

    if args.max_workers is None:
        max_workers = min(8, len(tasks)) if len(tasks) > 0 else 0
    else:
        max_workers = args.max_workers

    print_selection_summary(
        num_replicates=num_replicates,
        tasks=tasks,
        skipped_success=skipped_success,
        skipped_failed=skipped_failed,
        skipped_missing_prereq=skipped_missing_prereq,
    )

    run_metadata = {
        "stage": "staged_generalization",
        "study_dir": str(study_dir),
        "num_replicates": int(num_replicates),
        "max_workers": int(max_workers),
        "overwrite": bool(args.overwrite),
        "rerun_failed": bool(not args.no_rerun_failed),
        "selected_replicates": tasks,
        "skipped_success": skipped_success,
        "skipped_failed": skipped_failed,
        "skipped_missing_prereq": skipped_missing_prereq,
        "rtol_override": None if args.rtol is None else float(args.rtol),
        "atol_override": None if args.atol is None else float(args.atol),
        "max_steps_override": None if args.max_steps is None else int(args.max_steps),
        "method_note": (
            "Generalization rollouts evaluate trained staged NN closures on "
            "validation_scenarios() from siqr_observation_data.py."
        ),
    }

    with open(study_dir / "staged_generalization_last_run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    summaries = load_pickle_if_exists(
        study_dir / "staged_generalization_summaries.pkl",
        default={},
    )

    if len(tasks) == 0:
        print("No staged generalization replicates to run.")
        return

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
    ) as executor:
        futures = [
            executor.submit(
                run_one_replicate_generalization,
                r,
                str(study_dir),
                args_dict,
            )
            for r in tasks
        ]

        for future in as_completed(futures):
            summary = future.result()
            r = summary["replicate"]
            summaries[r] = summary

            if summary.get("success", False):
                state_rmse_vals = np.asarray(summary["state_rmse"], dtype=float)
                closure_rmse_vals = np.asarray(summary["closure_rmse"], dtype=float)

                print(
                    f"Finished staged generalization replicate {r:03d}: "
                    f"state_rmse={state_rmse_vals}, "
                    f"closure_rmse={closure_rmse_vals}"
                )
            else:
                print(
                    f"Staged generalization replicate {r:03d} failed: "
                    f"{summary.get('error', 'unknown error')}"
                )

    with open(study_dir / "staged_generalization_summaries.pkl", "wb") as f:
        pickle.dump(summaries, f)

    with open(study_dir / "staged_generalization_summaries.json", "w") as f:
        json.dump({str(k): v for k, v in summaries.items()}, f, indent=2)

    print(
        "Saved staged generalization summaries to "
        f"{study_dir / 'staged_generalization_summaries.pkl'}"
    )
    print(
        "Saved staged generalization summaries to "
        f"{study_dir / 'staged_generalization_summaries.json'}"
    )


if __name__ == "__main__":
    main()