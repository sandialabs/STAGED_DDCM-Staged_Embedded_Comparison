import os

# Set these before importing NumPy/JAX-heavy modules.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import argparse
import json
import pickle
import traceback

import numpy as np

from embedded_NN_methods import load_split

from da_methods import (
    EnKFConfig,
    make_augmented_initial_condition,
    estimate_closure_process_noise,
    run_enkf_assimilation,
    save_enkf_result,
)

from study_run_utils import (
    select_replicates_to_run,
    print_replicate_selection_summary,
    load_pickle_if_exists,
)


def run_one_replicate_enkf(
    r: int,
    study_dir: str,
    cfg_dict: dict,
    base_enkf_seed: int,
    save_enkf_sd: bool,
    save_enkf_ensemble: bool,
):
    """
    Run EnKF for one replicate.

    Final staged-method protocol:

        EnKF assimilates original train+validation observations only.
        Original test observations are not assimilated.

    Model-evaluation accounting:

        model_evaluations:
            Final EnKF assimilation model evaluations only.

        mle_model_evaluations:
            Process-noise MLE model evaluations only.

        total_model_evaluations:
            MLE model evaluations + final EnKF assimilation model evaluations.

    For the manuscript STAGED training-cost comparison, use:

        total_model_evaluations
    """
    try:
        study_dir = Path(study_dir)
        rep_dir = study_dir / f"rep_{r:03d}"

        obs_file = rep_dir / "observations.npz"
        split_file = rep_dir / "split.npz"
        result_file = rep_dir / "enkf_result.npz"
        summary_file = rep_dir / "enkf_summary.json"

        if not obs_file.exists():
            raise FileNotFoundError(f"Missing observation file: {obs_file}")

        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        cfg = EnKFConfig(**cfg_dict)

        obs = np.load(obs_file, allow_pickle=True)
        split = load_split(split_file)

        d_col_major = np.asarray(obs["d"])
        t_obs = np.asarray(obs["t_obs"])
        H_true = np.asarray(obs["H_true"])
        obs_var = float(np.asarray(obs["obs_var"]))
        y0_nondim = np.asarray(obs["y0_nondim"])
        params = tuple(map(float, np.asarray(obs["params"])))

        y0_aug = make_augmented_initial_condition(
            y0_nondim=y0_nondim,
            xi0=cfg.xi0,
        )

        noise_seed = int(np.asarray(obs["noise_seed"]))

        # Existing split-seed convention from the data generation script.
        split_seed = int(20_000 + r)

        enkf_seed = int(base_enkf_seed + r)

        mle_result = None
        mle_model_evaluations = 0

        if cfg.estimate_process_noise:
            (
                closure_process_noise,
                mle_result,
                mle_model_evaluations,
            ) = estimate_closure_process_noise(
                d_col_major=d_col_major,
                t_obs=t_obs,
                split=split,
                y0_aug=y0_aug,
                obs_var=obs_var,
                H_true=H_true,
                params=params,
                cfg=cfg,
            )
        else:
            closure_process_noise = float(cfg.closure_process_noise)
            mle_model_evaluations = 0

        enkf_output = run_enkf_assimilation(
            d_col_major=d_col_major,
            t_obs=t_obs,
            split=split,
            y0_aug=y0_aug,
            obs_var=obs_var,
            H_true=H_true,
            params=params,
            cfg=cfg,
            closure_process_noise=closure_process_noise,
            rng_seed=enkf_seed,
        )

        enkf_model_evaluations = int(enkf_output["model_evaluations"])
        total_model_evaluations = int(
            mle_model_evaluations + enkf_model_evaluations
        )

        obs_metadata = {
            "obs_var": obs_var,
            "params": np.asarray(params),
        }

        save_enkf_result(
            result_file,
            enkf_output=enkf_output,
            cfg=cfg,
            closure_process_noise=closure_process_noise,
            noise_seed=noise_seed,
            split_seed=split_seed,
            enkf_seed=enkf_seed,
            train_idx=np.asarray(split.train_idx),
            val_idx=np.asarray(split.val_idx),
            test_idx=np.asarray(split.test_idx),
            obs_metadata=obs_metadata,
            mle_result=mle_result,
            mle_model_evaluations=mle_model_evaluations,
            save_sd=save_enkf_sd,
            save_ensemble=save_enkf_ensemble,
        )

        summary = {
            "replicate": int(r),
            "success": True,
            "obs_file": str(obs_file),
            "split_file": str(split_file),
            "result_file": str(result_file),
            "metadata_file": str(
                result_file.with_name(result_file.stem + "_metadata.json")
            ),
            "closure_process_noise": float(closure_process_noise),
            "estimate_process_noise": bool(cfg.estimate_process_noise),
            "assimilation_policy": "trainval_hard_coded",
            "process_noise_estimation_policy": "trainval_hard_coded",
            "use_smoother": bool(cfg.use_smoother),
            "ensemble_size": int(cfg.ensemble_size),

            # Final EnKF assimilation only.
            "model_evaluations": int(enkf_model_evaluations),

            # Process-noise MLE only.
            "mle_model_evaluations": int(mle_model_evaluations),

            # MLE + final EnKF assimilation.
            "total_model_evaluations": int(total_model_evaluations),

            "num_train_obs": int(len(split.train_idx)),
            "num_val_obs": int(len(split.val_idx)),
            "num_test_obs": int(len(split.test_idx)),
            "num_assim_obs": int(len(enkf_output["assim_idx"])),
            "noise_seed": int(noise_seed),
            "split_seed": int(split_seed),
            "enkf_seed": int(enkf_seed),
            "save_enkf_sd": bool(save_enkf_sd),
            "save_enkf_ensemble": bool(save_enkf_ensemble),
            "method_note": (
                "EnKF assimilated original train+validation observations only. "
                "Original test observations were not assimilated. "
                "total_model_evaluations is the staged training-procedure "
                "simulation cost: process-noise MLE model evaluations plus "
                "final EnKF assimilation model evaluations."
            ),
        }

        if mle_result is not None:
            summary["mle_success"] = bool(getattr(mle_result, "success", False))
            summary["mle_fun"] = float(getattr(mle_result, "fun", np.nan))
            summary["mle_nfev"] = int(getattr(mle_result, "nfev", -1))
            summary["mle_model_evaluations"] = int(mle_model_evaluations)

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

        rep_dir = Path(study_dir) / f"rep_{r:03d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        with open(rep_dir / "enkf_error.json", "w") as f:
            json.dump(err_summary, f, indent=2)

        return err_summary


def detect_num_replicates(study_dir: Path) -> int:
    rep_dirs = sorted(study_dir.glob("rep_*"))

    return len(rep_dirs)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel EnKF assimilation over study replicates. "
            "This version hard-codes EnKF assimilation to original "
            "train+validation observations only."
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
        help="Default: min(8, num_replicates).",
    )

    parser.add_argument(
        "--base-enkf-seed",
        type=int,
        default=40_000,
    )

    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--dt-int",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--xi0",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--caseid",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--no-smoother",
        action="store_true",
        help=(
            "Disable EnKF smoothing. By default, smoothing is enabled. "
            "With this flag, the DA output uses filtering analysis states only."
        ),
    )

    parser.add_argument(
        "--closure-process-noise",
        type=float,
        default=4.578125e-05,
    )

    parser.add_argument(
        "--estimate-process-noise",
        action="store_true",
        help=(
            "Estimate closure process noise per replicate. "
            "Estimation is hard-coded to train+validation observations."
        ),
    )

    parser.add_argument(
        "--mle-stride",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--mle-seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--mle-initial",
        type=float,
        default=2.0e-5,
    )

    parser.add_argument(
        "--mle-lower",
        type=float,
        default=1.0e-8,
    )

    parser.add_argument(
        "--mle-upper",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--mle-method",
        type=str,
        default="Nelder-Mead",
    )

    parser.add_argument(
        "--mle-tol",
        type=float,
        default=1.0e-5,
    )

    parser.add_argument(
        "--save-enkf-sd",
        action="store_true",
        help="If set, save EnKF posterior standard deviations in enkf_result.npz.",
    )

    parser.add_argument(
        "--save-enkf-ensemble",
        action="store_true",
        help="If set, save EnKF ensemble arrays in enkf_result.npz. Can be large.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, rerun all requested EnKF replicates even if outputs exist.",
    )

    parser.add_argument(
        "--no-rerun-failed",
        action="store_true",
        help="If set, do not rerun EnKF replicates that previously failed.",
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

    cfg = EnKFConfig(
        ensemble_size=args.ensemble_size,
        dt_int=args.dt_int,
        xi0=args.xi0,
        caseid=args.caseid,
        use_smoother=not args.no_smoother,
        closure_process_noise=args.closure_process_noise,
        estimate_process_noise=args.estimate_process_noise,
        mle_stride=args.mle_stride,
        mle_seed=args.mle_seed,
        mle_initial=args.mle_initial,
        mle_bounds=(args.mle_lower, args.mle_upper),
        mle_method=args.mle_method,
        mle_tol=args.mle_tol,
    )

    cfg_dict = {
        "ensemble_size": cfg.ensemble_size,
        "dt_int": cfg.dt_int,
        "xi0": cfg.xi0,
        "caseid": cfg.caseid,
        "use_smoother": cfg.use_smoother,
        "closure_process_noise": cfg.closure_process_noise,
        "estimate_process_noise": cfg.estimate_process_noise,
        "mle_stride": cfg.mle_stride,
        "mle_seed": cfg.mle_seed,
        "mle_initial": cfg.mle_initial,
        "mle_bounds": cfg.mle_bounds,
        "mle_method": cfg.mle_method,
        "mle_tol": cfg.mle_tol,
    }

    tasks, skipped_success, skipped_failed, skipped_missing_prereq = (
        select_replicates_to_run(
            study_dir,
            num_replicates,
            "enkf",
            overwrite=args.overwrite,
            rerun_failed=not args.no_rerun_failed,
            require_prerequisites=True,
        )
    )

    if args.max_workers is None:
        max_workers = min(8, len(tasks)) if len(tasks) > 0 else 0
    else:
        max_workers = args.max_workers

    print_replicate_selection_summary(
        stage="enkf",
        num_replicates=num_replicates,
        tasks=tasks,
        skipped_success=skipped_success,
        skipped_failed=skipped_failed,
        skipped_missing_prereq=skipped_missing_prereq,
    )

    study_metadata = {
        "stage": "enkf",
        "method_note": (
            "EnKF assimilation is hard-coded to use original train+validation "
            "observations only. Original test observations are not assimilated. "
            "For staged training-cost comparison, use total_model_evaluations, "
            "which equals MLE model evaluations plus final EnKF assimilation "
            "model evaluations."
        ),
        "study_dir": str(study_dir),
        "num_replicates": num_replicates,
        "max_workers": max_workers,
        "base_enkf_seed": args.base_enkf_seed,
        "overwrite": args.overwrite,
        "rerun_failed": not args.no_rerun_failed,
        "selected_replicates": tasks,
        "skipped_success": skipped_success,
        "skipped_failed": skipped_failed,
        "skipped_missing_prereq": skipped_missing_prereq,
        "assimilation_policy": "trainval_hard_coded",
        "process_noise_estimation_policy": "trainval_hard_coded",
        "use_smoother": bool(cfg.use_smoother),
        "save_enkf_sd": bool(args.save_enkf_sd),
        "save_enkf_ensemble": bool(args.save_enkf_ensemble),
        "enkf_config": cfg_dict,
        "model_evaluation_accounting": {
            "model_evaluations": (
                "Final EnKF assimilation model evaluations only."
            ),
            "mle_model_evaluations": (
                "Process-noise MLE model evaluations only."
            ),
            "total_model_evaluations": (
                "MLE model evaluations plus final EnKF assimilation model "
                "evaluations."
            ),
        },
    }

    with open(study_dir / "enkf_last_run_metadata.json", "w") as f:
        json.dump(study_metadata, f, indent=2)

    summaries = load_pickle_if_exists(
        study_dir / "enkf_summaries.pkl",
        default={},
    )

    if len(tasks) == 0:
        print("No EnKF replicates to run.")
        return

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
    ) as executor:
        futures = [
            executor.submit(
                run_one_replicate_enkf,
                r,
                str(study_dir),
                cfg_dict,
                args.base_enkf_seed,
                args.save_enkf_sd,
                args.save_enkf_ensemble,
            )
            for r in tasks
        ]

        for future in as_completed(futures):
            summary = future.result()
            r = summary["replicate"]
            summaries[r] = summary

            if summary.get("success", False):
                smoother_label = (
                    "smoother" if summary.get("use_smoother", False) else "filter"
                )
                print(
                    f"Finished EnKF replicate {r:03d}: "
                    f"sigma={summary['closure_process_noise']:.4e}, "
                    f"mode={smoother_label}, "
                    f"enkf_model_evals={summary['model_evaluations']}, "
                    f"mle_model_evals={summary['mle_model_evaluations']}, "
                    f"total_model_evals={summary['total_model_evaluations']}, "
                    f"num_assim_obs={summary['num_assim_obs']}"
                )
            else:
                print(
                    f"EnKF replicate {r:03d} failed: "
                    f"{summary.get('error', 'unknown error')}"
                )

    with open(study_dir / "enkf_summaries.pkl", "wb") as f:
        pickle.dump(summaries, f)

    with open(study_dir / "enkf_summaries.json", "w") as f:
        json.dump({str(k): v for k, v in summaries.items()}, f, indent=2)

    print(f"Saved EnKF summaries to {study_dir / 'enkf_summaries.pkl'}")
    print(f"Saved EnKF summaries to {study_dir / 'enkf_summaries.json'}")


if __name__ == "__main__":
    main()