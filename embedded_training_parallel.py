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
import equinox as eqx

from study_run_utils import (
    select_replicates_to_run,
    print_replicate_selection_summary,
    load_pickle_if_exists,
)

from siqr_observation_data import (
    ObservationConfig,
    SIQRScenario,
    calibration_scenario,
    default_observation_operator,
    generate_observation_dataset,
    add_constant_xi_baseline,
    save_dataset,
)

from embedded_NN_methods import (
    SIQRParams,
    TrainingConfig,
    EmbeddedTrainer,
    random_time_split,
    evaluate_closure,
    save_split,
)


def make_training_config(args) -> TrainingConfig:
    return TrainingConfig(
        lr=args.lr,
        adam_steps=args.adam_steps,
        width_size=args.width_size,
        depth=args.depth,
        num_starts=args.num_starts,
        patience=args.patience,
        val_every=args.val_every,
        print_every=args.print_every,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_lbfgs=not args.no_lbfgs,
        lbfgs_maxiter=args.lbfgs_maxiter,

        # Match old SIQR_Embedded.ipynb:
        # eqx.nn.MLP(..., activation=jnn.sigmoid)
        #
        # Note: this is the hidden-layer activation. The final MLP output
        # remains unconstrained, as in the old notebook.
        activation=jnn.sigmoid,

        rtol=args.rtol,
        atol=args.atol,
        max_steps=args.max_steps,
    )


def run_one_replicate(
    r: int,
    output_dir: str,
    obs_var: float,
    base_noise_seed: int,
    base_split_seed: int,
    base_model_seed: int,
    train_frac: float,
    val_frac: float,
    include_constant_baseline: bool,
    training_args_dict: dict,
):
    """
    Run one complete synthetic experiment replicate.

    Replicate r includes:
        1. new noisy observation realization,
        2. new train/validation/test split,
        3. embedded NN multistart training,
        4. saved closure/state prediction.

    This function runs inside a separate process.
    """
    try:
        output_dir = Path(output_dir)
        rep_dir = output_dir / f"rep_{r:03d}"

        rep_dir.mkdir(parents=True, exist_ok=True)

        obs_file = rep_dir / "observations.npz"
        split_file = rep_dir / "split.npz"
        model_file = rep_dir / "embedded_model.eqx"
        model_config_file = rep_dir / "embedded_model_config.json"
        closure_file = rep_dir / "embedded_closure.npz"
        summary_file = rep_dir / "summary.json"

        # ---------------------------------------------------------------------
        # Seeds for this full replicate.
        # ---------------------------------------------------------------------
        noise_seed = base_noise_seed + r
        split_seed = base_split_seed + r
        model_seed = base_model_seed + 1000 * r

        # ---------------------------------------------------------------------
        # Generate noisy observation dataset.
        # ---------------------------------------------------------------------
        obs_config = ObservationConfig(
            obs_var=obs_var,
        )

        scenario = calibration_scenario()
        H_true = default_observation_operator()

        dataset = generate_observation_dataset(
            scenario=scenario,
            config=obs_config,
            H_true=H_true,
            noise_seed=noise_seed,
            include_full_trajectory=True,
        )

        if include_constant_baseline:
            dataset = add_constant_xi_baseline(
                dataset=dataset,
                scenario=scenario,
                config=obs_config,
            )

        save_dataset(dataset, obs_file)

        # ---------------------------------------------------------------------
        # Convert to embedded-training convention.
        #
        # dataset["d"] shape:       (num_outputs, N_obs)
        # split/training d shape:   (N_obs, num_outputs)
        # ---------------------------------------------------------------------
        d = jnp.asarray(dataset["d"].T)
        t_obs = jnp.asarray(dataset["t_obs"])
        H = jnp.asarray(dataset["H_true"])
        y0 = jnp.asarray(dataset["y0_nondim"])

        beta, gamma, delta = map(float, dataset["params"])

        siqr_params = SIQRParams(
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        # ---------------------------------------------------------------------
        # Create and save train/validation/test split.
        # ---------------------------------------------------------------------
        split = random_time_split(
            d=d,
            t_obs=t_obs,
            rand_seed=split_seed,
            train_frac=train_frac,
            val_frac=val_frac,
        )

        save_split(split, split_file)

        # ---------------------------------------------------------------------
        # Build trainer.
        # ---------------------------------------------------------------------
        args_proxy = argparse.Namespace(**training_args_dict)
        train_cfg = make_training_config(args_proxy)

        trainer = EmbeddedTrainer(
            t_obs=t_obs,
            y0=y0,
            H=H,
            obs_var=obs_var,
            siqr_params=siqr_params,
            config=train_cfg,
        )

        # ---------------------------------------------------------------------
        # Train embedded NN on this replicate.
        # ---------------------------------------------------------------------
        result = trainer.train_split(
            split,
            seed=model_seed,
        )

        if result.model is not None:
            eqx.tree_serialise_leaves(model_file, result.model)

            model_config = {
                "width_size": int(training_args_dict["width_size"]),
                "depth": int(training_args_dict["depth"]),
                "activation": "sigmoid",
                "obs_var": float(obs_var),
                "rtol": float(training_args_dict["rtol"]),
                "atol": float(training_args_dict["atol"]),
                "max_steps": int(training_args_dict["max_steps"]),
                "siqr_params": {
                    "beta": float(beta),
                    "gamma": float(gamma),
                    "delta": float(delta),
                },
                "y0": np.asarray(y0).tolist(),
                "H": np.asarray(H).tolist(),
            }

            with open(model_config_file, "w") as f:
                json.dump(model_config, f, indent=2)

        # ---------------------------------------------------------------------
        # Save closure/state predictions.
        # ---------------------------------------------------------------------
        if result.model is not None:
            xi_pred, y_pred = evaluate_closure(result.model, t_obs, y0)
            xi_pred = np.asarray(xi_pred)
            y_pred = np.asarray(y_pred)
            success = True
        else:
            xi_pred = np.full_like(np.asarray(t_obs), np.nan, dtype=float)
            y_pred = np.full(
                (len(t_obs), len(y0)),
                np.nan,
                dtype=float,
            )
            success = False

        np.savez(
            closure_file,
            success=np.array(success),
            xi=np.asarray(xi_pred),
            y=np.asarray(y_pred),
            t_obs=np.asarray(t_obs),
            xi_true=np.asarray(dataset["xi_true"]),
            y_true=np.asarray(dataset["y_true"]).T,
            H_true=np.asarray(H),
            train_idx=np.asarray(split.train_idx),
            val_idx=np.asarray(split.val_idx),
            test_idx=np.asarray(split.test_idx),
            train_t=np.asarray(split.train_t),
            val_t=np.asarray(split.val_t),
            test_t=np.asarray(split.test_t),
            noise_seed=np.array(noise_seed),
            split_seed=np.array(split_seed),
            model_seed=np.array(model_seed),
            best_val_loss=np.array(result.best_val_loss),
            test_loss=np.array(result.test_loss),
            best_start=np.array(-1 if result.best_start is None else result.best_start),
            num_eval=np.array(result.num_eval),
            bad_starts=np.array(result.bad_starts),
            params=np.asarray(dataset["params"]),
            obs_var=np.array(obs_var),
        )

        summary = {
            "replicate": int(r),
            "success": bool(success),
            "noise_seed": int(noise_seed),
            "split_seed": int(split_seed),
            "model_seed": int(model_seed),
            "obs_file": str(obs_file),
            "split_file": str(split_file),
            "model_file": None if result.model is None else str(model_file),
            "model_config_file": None if result.model is None else str(model_config_file),
            "closure_file": str(closure_file),
            "best_val_loss": float(result.best_val_loss),
            "test_loss": float(result.test_loss),
            "best_start": None if result.best_start is None else int(result.best_start),
            "num_eval": int(result.num_eval),
            "bad_starts": int(result.bad_starts),
            "params": {
                "beta": float(beta),
                "gamma": float(gamma),
                "delta": float(delta),
            },
            "obs_var": float(obs_var),
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

        rep_dir = Path(output_dir) / f"rep_{r:03d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        with open(rep_dir / "error.json", "w") as f:
            json.dump(err_summary, f, indent=2)

        return err_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Single-loop parallel embedded NN study. "
            "Each replicate generates new noisy data, creates a new split, "
            "and trains the embedded NN model."
        )
    )

    # -------------------------------------------------------------------------
    # Study design.
    # -------------------------------------------------------------------------
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
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes. Default: min(8, num_replicates).",
    )

    parser.add_argument(
        "--obs-var",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--base-noise-seed",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--base-split-seed",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--base-model-seed",
        type=int,
        default=30_000,
    )

    parser.add_argument(
        "--include-constant-baseline",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, rerun all requested embedded replicates even if outputs exist.",
    )

    parser.add_argument(
        "--no-rerun-failed",
        action="store_true",
        help="If set, do not rerun replicates that previously failed.",
    )

    # -------------------------------------------------------------------------
    # Embedded NN model/training settings.
    # -------------------------------------------------------------------------
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--adam-steps", type=int, default=2000)

    # Match old notebook defaults:
    # width_size=2, depth=2
    parser.add_argument("--width-size", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)

    parser.add_argument("--num-starts", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10,
        help=( "Early-stopping patience measured in validation checks. "
            "With --val-every 10 and --patience 20, this allows up to "
            "200 AdamW iterations without validation improvement." ))

    parser.add_argument("--val-every", type=int, default=25,
        help="Run validation every N completed AdamW iterations." )
    parser.add_argument("--print-every", type=int, default=200)

    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--no-lbfgs", action="store_true")
    parser.add_argument("--lbfgs-maxiter", type=int, default=500)

    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=1.0e-7)
    parser.add_argument("--max-steps", type=int, default=16384)

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_replicates = args.num_replicates

    # -------------------------------------------------------------------------
    # Keep only primitive arguments for subprocess construction.
    # -------------------------------------------------------------------------
    training_args_dict = {
        "lr": args.lr,
        "adam_steps": args.adam_steps,
        "width_size": args.width_size,
        "depth": args.depth,
        "num_starts": args.num_starts,
        "patience": args.patience,
        "val_every": args.val_every,
        "print_every": args.print_every,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "no_lbfgs": args.no_lbfgs,
        "lbfgs_maxiter": args.lbfgs_maxiter,
        "rtol": args.rtol,
        "atol": args.atol,
        "max_steps": args.max_steps,
    }

    # -------------------------------------------------------------------------
    # Select only replicates that need to run.
    # -------------------------------------------------------------------------
    tasks, skipped_success, skipped_failed, skipped_missing_prereq = (
        select_replicates_to_run(
            output_dir,
            num_replicates,
            "embedded",
            overwrite=args.overwrite,
            rerun_failed=not args.no_rerun_failed,
            require_prerequisites=False,
        )
    )

    # -------------------------------------------------------------------------
    # Determine actual worker count based on selected tasks.
    # -------------------------------------------------------------------------
    if args.max_workers is None:
        max_workers = min(8, len(tasks)) if len(tasks) > 0 else 0
    else:
        max_workers = args.max_workers

    print_replicate_selection_summary(
        stage="embedded",
        num_replicates=num_replicates,
        tasks=tasks,
        skipped_success=skipped_success,
        skipped_failed=skipped_failed,
        skipped_missing_prereq=skipped_missing_prereq,
    )

    # -------------------------------------------------------------------------
    # Write metadata for this run.
    # Put it here, after task selection and max_workers determination.
    # -------------------------------------------------------------------------
    study_metadata = {
        "stage": "embedded",
        "num_replicates": num_replicates,
        "max_workers": max_workers,
        "output_dir": str(output_dir),
        "obs_var": args.obs_var,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "base_noise_seed": args.base_noise_seed,
        "base_split_seed": args.base_split_seed,
        "base_model_seed": args.base_model_seed,
        "include_constant_baseline": args.include_constant_baseline,
        "overwrite": args.overwrite,
        "rerun_failed": not args.no_rerun_failed,
        "selected_replicates": tasks,
        "skipped_success": skipped_success,
        "skipped_failed": skipped_failed,
        "skipped_missing_prereq": skipped_missing_prereq,
        "training_config": {
            **training_args_dict,
            "activation": "sigmoid",
        },
    }

    with open(output_dir / "embedded_last_run_metadata.json", "w") as f:
        json.dump(study_metadata, f, indent=2)

    # -------------------------------------------------------------------------
    # Load existing summaries so this run appends/updates rather than replaces.
    # -------------------------------------------------------------------------
    summaries = load_pickle_if_exists(
        output_dir / "embedded_summaries.pkl",
        default={},
    )

    if len(tasks) == 0:
        print("No embedded replicates to run.")
        return

    # -------------------------------------------------------------------------
    # Launch selected tasks.
    # -------------------------------------------------------------------------
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
    ) as executor:
        futures = [
            executor.submit(
                run_one_replicate,
                r,
                str(output_dir),
                args.obs_var,
                args.base_noise_seed,
                args.base_split_seed,
                args.base_model_seed,
                args.train_frac,
                args.val_frac,
                args.include_constant_baseline,
                training_args_dict,
            )
            for r in tasks
        ]

        for future in as_completed(futures):
            summary = future.result()
            r = summary["replicate"]
            summaries[r] = summary

            if summary.get("success", False):
                print(
                    f"Finished replicate {r:03d}: "
                    f"test_loss={summary['test_loss']:.4e}, "
                    f"best_val_loss={summary['best_val_loss']:.4e}, "
                    f"bad_starts={summary['bad_starts']}"
                )
            else:
                print(
                    f"Replicate {r:03d} failed: "
                    f"{summary.get('error', 'unknown error')}"
                )

    # -------------------------------------------------------------------------
    # Save updated aggregate summaries.
    # -------------------------------------------------------------------------
    with open(output_dir / "embedded_summaries.pkl", "wb") as f:
        pickle.dump(summaries, f)

    with open(output_dir / "embedded_summaries.json", "w") as f:
        json.dump(
            {str(k): v for k, v in summaries.items()},
            f,
            indent=2,
        )

    print(f"Saved summaries to {output_dir / 'embedded_summaries.pkl'}")
    print(f"Saved summaries to {output_dir / 'embedded_summaries.json'}")


if __name__ == "__main__":
    main()