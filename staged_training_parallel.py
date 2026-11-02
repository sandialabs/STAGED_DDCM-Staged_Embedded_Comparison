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
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

from embedded_NN_methods import (
    SIQRParams,
    load_split,
)

from staged_NN_methods import (
    StagedNNConfig,
    get_enkf_supervised_pairs_for_original_split,
    train_staged_multistart,
    evaluate_staged_model_on_replicate,
)

from study_run_utils import (
    select_replicates_to_run,
    print_replicate_selection_summary,
    load_pickle_if_exists,
)


def eval_staged_net_batch(staged_net, X):
    """
    Evaluate staged_net on a batch of inputs.

    Returns shape (N,) for scalar-output staged networks.
    """
    X = np.asarray(X)

    if staged_net is None:
        return np.full((X.shape[0],), np.nan, dtype=float)

    pred = jax.vmap(staged_net)(jnp.asarray(X))
    pred = np.asarray(pred)

    pred = pred.reshape((X.shape[0], -1))

    if pred.shape[1] == 1:
        return pred[:, 0]

    return pred


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.nanmean(np.abs(a - b)))


def max_abs_error(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.nanmax(np.abs(a - b)))


def save_staged_trainval_diagnostic(
    diagnostic_file,
    diagnostic_plot_file,
    staged_net,
    t_pseudo_train,
    X_pseudo_train,
    y_pseudo_train,
    t_pseudo_val,
    X_pseudo_val,
    y_pseudo_val,
    t_obs,
    xi_true,
):
    """
    Evaluate the staged NN directly on EnKF train+val pseudo-data and compare
    to the EnKF-derived xi labels.

    This is a supervised-fit diagnostic. It does not use autonomous ODE rollout.
    """
    if staged_net is None:
        diagnostic = {
            "diagnostic_success": False,
            "diagnostic_error": "staged_net is None",
            "diagnostic_file": str(diagnostic_file),
            "diagnostic_plot_file": str(diagnostic_plot_file),
            "diagnostic_plot_error": None,
            "train_rmse": np.inf,
            "val_rmse": np.inf,
            "trainval_rmse": np.inf,
            "trainval_mae": np.inf,
            "trainval_max_abs_error": np.inf,
        }
        return diagnostic

    t_pseudo_train = np.asarray(t_pseudo_train)
    X_pseudo_train = np.asarray(X_pseudo_train)
    y_pseudo_train = np.asarray(y_pseudo_train).reshape(-1)

    t_pseudo_val = np.asarray(t_pseudo_val)
    X_pseudo_val = np.asarray(X_pseudo_val)
    y_pseudo_val = np.asarray(y_pseudo_val).reshape(-1)

    t_trainval = np.concatenate(
        [t_pseudo_train, t_pseudo_val],
        axis=0,
    )
    X_trainval = np.concatenate(
        [X_pseudo_train, X_pseudo_val],
        axis=0,
    )
    xi_da_trainval = np.concatenate(
        [y_pseudo_train, y_pseudo_val],
        axis=0,
    )

    split_label = np.concatenate(
        [
            np.full(t_pseudo_train.shape[0], "train", dtype="<U5"),
            np.full(t_pseudo_val.shape[0], "val", dtype="<U5"),
        ],
        axis=0,
    )

    xi_nn_train = eval_staged_net_batch(staged_net, X_pseudo_train)
    xi_nn_val = eval_staged_net_batch(staged_net, X_pseudo_val)
    xi_nn_trainval = eval_staged_net_batch(staged_net, X_trainval)

    train_rmse = rmse(xi_nn_train, y_pseudo_train)
    val_rmse = rmse(xi_nn_val, y_pseudo_val)
    trainval_rmse = rmse(xi_nn_trainval, xi_da_trainval)
    trainval_mae = mae(xi_nn_trainval, xi_da_trainval)
    trainval_max_abs = max_abs_error(xi_nn_trainval, xi_da_trainval)

    np.savez(
        diagnostic_file,
        diagnostic_success=np.array(True),
        t_trainval=np.asarray(t_trainval),
        X_trainval=np.asarray(X_trainval),
        xi_da_trainval=np.asarray(xi_da_trainval),
        xi_nn_trainval=np.asarray(xi_nn_trainval),
        split_label=np.asarray(split_label),
        t_pseudo_train=np.asarray(t_pseudo_train),
        X_pseudo_train=np.asarray(X_pseudo_train),
        xi_da_train=np.asarray(y_pseudo_train),
        xi_nn_train=np.asarray(xi_nn_train),
        t_pseudo_val=np.asarray(t_pseudo_val),
        X_pseudo_val=np.asarray(X_pseudo_val),
        xi_da_val=np.asarray(y_pseudo_val),
        xi_nn_val=np.asarray(xi_nn_val),
        t_obs=np.asarray(t_obs),
        xi_true=np.asarray(xi_true),
        train_rmse=np.array(train_rmse),
        val_rmse=np.array(val_rmse),
        trainval_rmse=np.array(trainval_rmse),
        trainval_mae=np.array(trainval_mae),
        trainval_max_abs_error=np.array(trainval_max_abs),
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5.5))

        ax.plot(
            np.asarray(t_obs),
            np.asarray(xi_true),
            "k-",
            linewidth=2.5,
            label=r"true $\xi(t)$",
        )

        train_order = np.argsort(t_pseudo_train)
        val_order = np.argsort(t_pseudo_val)

        ax.scatter(
            t_pseudo_train[train_order],
            y_pseudo_train[train_order],
            s=45,
            c="C2",
            marker="o",
            edgecolor="k",
            linewidth=0.3,
            label=r"EnKF $\xi$ train",
        )

        ax.scatter(
            t_pseudo_train[train_order],
            xi_nn_train[train_order],
            s=75,
            facecolors="none",
            edgecolors="C2",
            marker="o",
            linewidth=1.8,
            label=r"NN on train EnKF states",
        )

        ax.scatter(
            t_pseudo_val[val_order],
            y_pseudo_val[val_order],
            s=50,
            c="C1",
            marker="s",
            edgecolor="k",
            linewidth=0.3,
            label=r"EnKF $\xi$ val",
        )

        ax.scatter(
            t_pseudo_val[val_order],
            xi_nn_val[val_order],
            s=85,
            facecolors="none",
            edgecolors="C1",
            marker="s",
            linewidth=1.8,
            label=r"NN on val EnKF states",
        )

        ax.set_xlabel("time")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(
            "Staged NN evaluated on train+val EnKF pseudo-data\n"
            f"train+val RMSE = {trainval_rmse:.4e}, "
            f"train RMSE = {train_rmse:.4e}, "
            f"val RMSE = {val_rmse:.4e}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(diagnostic_plot_file, dpi=200)
        plt.close(fig)

        plot_error = None

    except Exception as exc:
        plot_error = repr(exc)

    diagnostic = {
        "diagnostic_success": True,
        "diagnostic_error": None,
        "diagnostic_file": str(diagnostic_file),
        "diagnostic_plot_file": str(diagnostic_plot_file),
        "diagnostic_plot_error": plot_error,
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "trainval_rmse": float(trainval_rmse),
        "trainval_mae": float(trainval_mae),
        "trainval_max_abs_error": float(trainval_max_abs),
    }

    return diagnostic


def make_staged_config(args) -> StagedNNConfig:
    return StagedNNConfig(
        width_size=args.width_size,
        depth=args.depth,

        # Match updated embedded implementation and old notebook:
        # eqx.nn.MLP(..., activation=jnn.sigmoid)
        #
        # This is the hidden-layer activation. The final MLP output remains
        # unconstrained.
        activation=jnn.sigmoid,

        lr=args.lr,
        adam_steps=args.adam_steps,
        num_starts=args.num_starts,
        val_every=args.val_every,
        patience=args.patience,
        print_every=args.print_every,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_lbfgs=not args.no_lbfgs,
        lbfgs_maxiter=args.lbfgs_maxiter,
        train_source=args.train_source,
        pseudo_time_tol=args.pseudo_time_tol,
        rtol=args.rtol,
        atol=args.atol,
        max_steps=args.max_steps,
    )


def run_one_replicate_staged(
    r: int,
    study_dir: str,
    cfg_dict: dict,
    base_staged_seed: int,
):
    """
    Run staged-NN phase for one replicate.

    Inputs:
        rep_r/observations.npz
        rep_r/split.npz
        rep_r/enkf_result.npz

    Expected methodology:
        The upstream EnKF result should have been generated by assimilating the
        union of original train and validation observations, but not test
        observations.

    Staged NN training:
        pseudo-train pairs are EnKF-derived pairs at original train_t.
        pseudo-val pairs are EnKF-derived pairs at original val_t.

    Outputs:
        rep_r/staged_model.eqx
        rep_r/staged_model_config.json
        rep_r/staged_closure.npz
        rep_r/staged_summary.json

    If --diagnostic-mode is enabled:
        rep_r/staged_trainval_diagnostic.npz
        rep_r/staged_trainval_diagnostic.png
    """
    try:
        study_dir = Path(study_dir)
        rep_dir = study_dir / f"rep_{r:03d}"

        obs_file = rep_dir / "observations.npz"
        split_file = rep_dir / "split.npz"
        enkf_file = rep_dir / "enkf_result.npz"

        model_file = rep_dir / "staged_model.eqx"
        model_config_file = rep_dir / "staged_model_config.json"
        closure_file = rep_dir / "staged_closure.npz"
        summary_file = rep_dir / "staged_summary.json"

        diagnostic_file = rep_dir / "staged_trainval_diagnostic.npz"
        diagnostic_plot_file = rep_dir / "staged_trainval_diagnostic.png"

        if not obs_file.exists():
            raise FileNotFoundError(f"Missing observation file: {obs_file}")

        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        if not enkf_file.exists():
            raise FileNotFoundError(f"Missing EnKF result file: {enkf_file}")

        # ---------------------------------------------------------------------
        # Reconstruct config.
        # ---------------------------------------------------------------------
        args_proxy = argparse.Namespace(**cfg_dict)
        cfg = make_staged_config(args_proxy)

        diagnostic_mode = bool(cfg_dict.get("diagnostic_mode", False))

        staged_seed = int(base_staged_seed + 1000 * r)

        # ---------------------------------------------------------------------
        # Load replicate data.
        # ---------------------------------------------------------------------
        obs = np.load(obs_file, allow_pickle=True)
        split = load_split(split_file)
        enkf = np.load(enkf_file, allow_pickle=True)

        d_time_major = np.asarray(obs["d"]).T
        t_obs = np.asarray(obs["t_obs"])
        H_true = np.asarray(obs["H_true"])
        obs_var = float(np.asarray(obs["obs_var"]))
        y0_nondim = np.asarray(obs["y0_nondim"])
        params_arr = np.asarray(obs["params"], dtype=float)
        beta, gamma, delta = map(float, params_arr)

        siqr_params = SIQRParams(
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        y_true_time_major = np.asarray(obs["y_true"]).T
        xi_true = np.asarray(obs["xi_true"])

        noise_seed = int(np.asarray(obs["noise_seed"]))

        # ---------------------------------------------------------------------
        # Extract EnKF-derived pseudo-pairs using original train/validation split.
        #
        # This implements Option C:
        #
        #   EnKF was run once using original train+validation observations.
        #   Staged NN pseudo-train = EnKF pairs at original train_t.
        #   Staged NN pseudo-val   = EnKF pairs at original val_t.
        #
        # The pseudo-validation set is used for internal NN selection only.
        # Final reporting uses original test_idx only.
        # ---------------------------------------------------------------------
        pseudo = get_enkf_supervised_pairs_for_original_split(
            enkf_data=enkf,
            split=split,
            source=cfg.train_source,
            remove_duplicate_times=True,
            tol=cfg.pseudo_time_tol,
        )

        t_pseudo_train = pseudo["t_pseudo_train"]
        X_pseudo_train = pseudo["X_pseudo_train"]
        y_pseudo_train = pseudo["y_pseudo_train"]

        t_pseudo_val = pseudo["t_pseudo_val"]
        X_pseudo_val = pseudo["X_pseudo_val"]
        y_pseudo_val = pseudo["y_pseudo_val"]

        t_all_pseudo = pseudo["t_all"]
        X_all_pseudo = pseudo["X_all"]
        y_all_pseudo = pseudo["y_all"]

        if X_pseudo_train.shape[0] == 0:
            raise RuntimeError("No valid pseudo-training pairs were found.")

        if X_pseudo_val.shape[0] == 0:
            raise RuntimeError("No valid pseudo-validation pairs were found.")

        # ---------------------------------------------------------------------
        # Train staged closure NN.
        #
        # Adam/L-BFGS parameter updates use pseudo-training only.
        # Pseudo-validation is used for checkpointing/selection.
        # ---------------------------------------------------------------------
        staged_train_result = train_staged_multistart(
            X_train=X_pseudo_train,
            y_train=y_pseudo_train,
            X_val=X_pseudo_val,
            y_val=y_pseudo_val,
            cfg=cfg,
            seed=staged_seed,
        )

        staged_net = staged_train_result["model"]

        # ---------------------------------------------------------------------
        # Evaluate staged model in autonomous rollout mode.
        # ---------------------------------------------------------------------
        if staged_net is None:
            success = False
            ode_success = False

            test_loss = np.inf
            state_rmse_test = np.inf
            closure_rmse_test = np.inf

            xi_pred = np.full_like(t_obs, np.nan, dtype=float)
            y_pred = np.full((len(t_obs), 4), np.nan, dtype=float)

        else:
            eval_result = evaluate_staged_model_on_replicate(
                staged_net=staged_net,
                t_obs=t_obs,
                y0=y0_nondim,
                params=siqr_params,
                cfg=cfg,
                d_obs_time_major=d_time_major,
                H=H_true,
                obs_var=obs_var,
                test_idx=np.asarray(split.test_idx),
                y_true_time_major=y_true_time_major,
                xi_true=xi_true,
            )

            xi_pred = eval_result["xi_pred"]
            y_pred = eval_result["y_pred"]

            test_loss = eval_result["test_loss"]
            state_rmse_test = eval_result["state_rmse_test"]
            closure_rmse_test = eval_result["closure_rmse_test"]
            ode_success = bool(eval_result.get("ode_success", True))

            success = bool(
                ode_success
                and np.isfinite(test_loss)
                and np.isfinite(state_rmse_test)
                and np.isfinite(closure_rmse_test)
            )

        # ---------------------------------------------------------------------
        # Optional diagnostic mode:
        # Evaluate NN directly on train+val EnKF pseudo-data and compare against
        # EnKF-derived xi labels. This avoids autonomous rollout entirely.
        # ---------------------------------------------------------------------
        if diagnostic_mode:
            diagnostic_summary = save_staged_trainval_diagnostic(
                diagnostic_file=diagnostic_file,
                diagnostic_plot_file=diagnostic_plot_file,
                staged_net=staged_net,
                t_pseudo_train=t_pseudo_train,
                X_pseudo_train=X_pseudo_train,
                y_pseudo_train=y_pseudo_train,
                t_pseudo_val=t_pseudo_val,
                X_pseudo_val=X_pseudo_val,
                y_pseudo_val=y_pseudo_val,
                t_obs=t_obs,
                xi_true=xi_true,
            )
        else:
            diagnostic_summary = {
                "diagnostic_success": None,
                "diagnostic_error": None,
                "diagnostic_file": None,
                "diagnostic_plot_file": None,
                "diagnostic_plot_error": None,
                "train_rmse": None,
                "val_rmse": None,
                "trainval_rmse": None,
                "trainval_mae": None,
                "trainval_max_abs_error": None,
            }

        # ---------------------------------------------------------------------
        # Save trained staged model using Equinox, not pickle.
        #
        # Do not pickle staged_net or staged_train_result directly, because the
        # model/config may contain JAX function objects such as jnn.sigmoid.
        # ---------------------------------------------------------------------
        if staged_net is not None:
            eqx.tree_serialise_leaves(model_file, staged_net)

            model_config = {
                "model_type": "staged_nn",
                "activation": "sigmoid",

                # Architecture.
                "width_size": int(cfg.width_size),
                "depth": int(cfg.depth),
                "input_dim": int(X_pseudo_train.shape[1]),
                "output_dim": int(
                    np.asarray(y_pseudo_train).shape[1]
                    if np.asarray(y_pseudo_train).ndim > 1
                    else 1
                ),

                # Training settings.
                "lr": float(cfg.lr),
                "adam_steps": int(cfg.adam_steps),
                "num_starts": int(cfg.num_starts),
                "val_every": int(cfg.val_every),
                "patience": None if cfg.patience is None else int(cfg.patience),
                "print_every": int(cfg.print_every),
                "weight_decay": float(cfg.weight_decay),
                "grad_clip": float(cfg.grad_clip),
                "use_lbfgs": bool(cfg.use_lbfgs),
                "lbfgs_maxiter": int(cfg.lbfgs_maxiter),

                # Data/protocol settings.
                "train_source": str(cfg.train_source),
                "pseudo_time_tol": float(cfg.pseudo_time_tol),

                # ODE/evaluation settings.
                "rtol": float(cfg.rtol),
                "atol": float(cfg.atol),
                "max_steps": int(cfg.max_steps),

                # Replicate-specific quantities needed for later evaluation.
                "obs_var": float(obs_var),
                "siqr_params": {
                    "beta": float(beta),
                    "gamma": float(gamma),
                    "delta": float(delta),
                },
                "y0": np.asarray(y0_nondim).tolist(),
                "H": np.asarray(H_true).tolist(),

                # Reproducibility.
                "replicate": int(r),
                "noise_seed": int(noise_seed),
                "staged_seed": int(staged_seed),

                "method_note": (
                    "This file contains metadata needed to reconstruct the "
                    "staged NN architecture before loading staged_model.eqx "
                    "with eqx.tree_deserialise_leaves."
                ),
            }

            with open(model_config_file, "w") as f:
                json.dump(model_config, f, indent=2)

        # ---------------------------------------------------------------------
        # Save lightweight closure/state outputs.
        # ---------------------------------------------------------------------
        np.savez(
            closure_file,
            success=np.array(success),
            ode_success=np.array(ode_success),
            xi=np.asarray(xi_pred),
            y=np.asarray(y_pred),
            t_obs=np.asarray(t_obs),
            xi_true=np.asarray(xi_true),
            y_true=np.asarray(y_true_time_major),
            H_true=np.asarray(H_true),
            train_idx=np.asarray(split.train_idx),
            val_idx=np.asarray(split.val_idx),
            test_idx=np.asarray(split.test_idx),

            # Original-split-derived pseudo-pairs.
            t_pseudo_train=np.asarray(t_pseudo_train),
            X_pseudo_train=np.asarray(X_pseudo_train),
            y_pseudo_train=np.asarray(y_pseudo_train),
            t_pseudo_val=np.asarray(t_pseudo_val),
            X_pseudo_val=np.asarray(X_pseudo_val),
            y_pseudo_val=np.asarray(y_pseudo_val),

            # Full available EnKF pseudo-data from selected source.
            t_all_pseudo=np.asarray(t_all_pseudo),
            X_all_pseudo=np.asarray(X_all_pseudo),
            y_all_pseudo=np.asarray(y_all_pseudo),

            # Backward-compatible aliases for previous analysis code.
            t_train=np.asarray(t_pseudo_train),
            X_train=np.asarray(X_pseudo_train),
            y_train=np.asarray(y_pseudo_train),

            train_source=np.array(cfg.train_source),
            pseudo_time_tol=np.array(cfg.pseudo_time_tol),
            noise_seed=np.array(noise_seed),
            staged_seed=np.array(staged_seed),
            best_train_loss=np.array(staged_train_result["best_train_loss"]),
            best_val_loss=np.array(staged_train_result["best_val_loss"]),
            test_loss=np.array(test_loss),
            state_rmse_test=np.array(state_rmse_test),
            closure_rmse_test=np.array(closure_rmse_test),
            best_start=np.array(
                -1 if staged_train_result["best_start"] is None
                else staged_train_result["best_start"]
            ),
            num_eval=np.array(staged_train_result["num_eval"]),
            bad_starts=np.array(staged_train_result["bad_starts"]),
            params=np.asarray(params_arr),
            obs_var=np.array(obs_var),
        )

        summary = {
            "replicate": int(r),
            "success": bool(success),
            "ode_success": bool(ode_success),
            "obs_file": str(obs_file),
            "split_file": str(split_file),
            "enkf_file": str(enkf_file),

            # Replaces old staged_result.pkl.
            "model_file": None if staged_net is None else str(model_file),
            "model_config_file": None if staged_net is None else str(model_config_file),

            "closure_file": str(closure_file),
            "train_source": cfg.train_source,
            "pseudo_time_tol": float(cfg.pseudo_time_tol),
            "num_pseudo_train_pairs": int(X_pseudo_train.shape[0]),
            "num_pseudo_val_pairs": int(X_pseudo_val.shape[0]),
            "num_all_pseudo_pairs": int(X_all_pseudo.shape[0]),
            "noise_seed": int(noise_seed),
            "staged_seed": int(staged_seed),
            "best_train_loss": float(staged_train_result["best_train_loss"]),
            "best_val_loss": float(staged_train_result["best_val_loss"]),
            "test_loss": float(test_loss),
            "state_rmse_test": float(state_rmse_test),
            "closure_rmse_test": float(closure_rmse_test),

            # Diagnostic-mode outputs.
            "diagnostic_mode": bool(diagnostic_mode),
            "diagnostic_success": diagnostic_summary["diagnostic_success"],
            "diagnostic_error": diagnostic_summary["diagnostic_error"],
            "diagnostic_file": diagnostic_summary["diagnostic_file"],
            "diagnostic_plot_file": diagnostic_summary["diagnostic_plot_file"],
            "diagnostic_plot_error": diagnostic_summary["diagnostic_plot_error"],
            "diagnostic_train_rmse": diagnostic_summary["train_rmse"],
            "diagnostic_val_rmse": diagnostic_summary["val_rmse"],
            "diagnostic_trainval_rmse": diagnostic_summary["trainval_rmse"],
            "diagnostic_trainval_mae": diagnostic_summary["trainval_mae"],
            "diagnostic_trainval_max_abs_error": diagnostic_summary[
                "trainval_max_abs_error"
            ],

            "best_start": (
                None if staged_train_result["best_start"] is None
                else int(staged_train_result["best_start"])
            ),
            "num_eval": int(staged_train_result["num_eval"]),
            "bad_starts": int(staged_train_result["bad_starts"]),
            "used_pseudo_validation": bool(
                staged_train_result["used_pseudo_validation"]
            ),
            "params": {
                "beta": float(beta),
                "gamma": float(gamma),
                "delta": float(delta),
            },
            "obs_var": float(obs_var),
            "method_note": (
                "For the staged method, EnKF-derived pairs were split using the "
                "original training and validation time split. This pseudo-validation "
                "set is used for internal NN selection only. Final evaluation is "
                "on held-out raw test observations. The trained staged model is "
                "saved as staged_model.eqx rather than as a full Python pickle."
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

        rep_dir = Path(study_dir) / f"rep_{r:03d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        with open(rep_dir / "staged_error.json", "w") as f:
            json.dump(err_summary, f, indent=2)

        return err_summary


def detect_num_replicates(study_dir: Path) -> int:
    rep_dirs = sorted(study_dir.glob("rep_*"))
    return len(rep_dirs)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel staged NN training over study replicates. "
            "Each replicate trains xi_NN(S,I,Q,R) from EnKF output. "
            "The EnKF-derived pseudo-pairs are split using the original "
            "train/validation time split."
        )
    )

    # -------------------------------------------------------------------------
    # Study design.
    # -------------------------------------------------------------------------
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
        "--base-staged-seed",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--train-source",
        type=str,
        choices=[
            "full_enkf_trajectory",
            "assimilation_step_enkf",
        ],
        default="assimilation_step_enkf",
        help=(
            "'full_enkf_trajectory' uses enkf_result['x_mu'] and "
            "enkf_result['t_mu'], the full EnKF-estimated mean trajectory. "
            "'assimilation_step_enkf' uses enkf_result['x_mu_assim'] and "
            "enkf_result['t_assim'], the EnKF estimates at assimilation times. "
            "For the one-EnKF train+validation assimilation protocol, "
            "'assimilation_step_enkf' is usually the cleanest choice."
        ),
    )

    parser.add_argument(
        "--pseudo-time-tol",
        type=float,
        default=1.0e-8,
        help=(
            "Tolerance for matching original train_t/val_t times to EnKF "
            "pseudo-data times."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, rerun all requested staged replicates even if outputs exist.",
    )

    parser.add_argument(
        "--no-rerun-failed",
        action="store_true",
        help="If set, do not rerun staged replicates that previously failed.",
    )

    parser.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help=(
            "If set, evaluate the staged NN directly on train+validation "
            "EnKF pseudo-data and save a diagnostic comparison against "
            "the EnKF-derived xi labels."
        ),
    )

    # -------------------------------------------------------------------------
    # Staged NN settings.
    # -------------------------------------------------------------------------
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--adam-steps", type=int, default=2000)

    # Match updated embedded implementation and old notebook defaults.
    parser.add_argument("--width-size", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)

    parser.add_argument("--num-starts", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=25,
        help="Run pseudo-validation every N completed AdamW iterations.")

    parser.add_argument("--patience", type=int, default=10,
        help=("Early-stopping patience measured in pseudo-validation checks. "
            "With --val-every 10 and --patience 20, this allows up to "
            "200 AdamW iterations without pseudo-validation improvement."))
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

    study_dir = Path(args.study_dir)

    if args.num_replicates is None:
        num_replicates = detect_num_replicates(study_dir)
    else:
        num_replicates = args.num_replicates

    cfg_dict = {
        "width_size": args.width_size,
        "depth": args.depth,
        "activation": "sigmoid",
        "lr": args.lr,
        "adam_steps": args.adam_steps,
        "num_starts": args.num_starts,
        "val_every": args.val_every,
        "patience": args.patience,
        "print_every": args.print_every,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "no_lbfgs": args.no_lbfgs,
        "lbfgs_maxiter": args.lbfgs_maxiter,
        "train_source": args.train_source,
        "pseudo_time_tol": args.pseudo_time_tol,
        "rtol": args.rtol,
        "atol": args.atol,
        "max_steps": args.max_steps,
        "diagnostic_mode": args.diagnostic_mode,
    }

    tasks, skipped_success, skipped_failed, skipped_missing_prereq = (
        select_replicates_to_run(
            study_dir,
            num_replicates,
            "staged",
            overwrite=args.overwrite,
            rerun_failed=not args.no_rerun_failed,
            require_prerequisites=True,
        )
    )

    print_replicate_selection_summary(
        stage="staged",
        num_replicates=num_replicates,
        tasks=tasks,
        skipped_success=skipped_success,
        skipped_failed=skipped_failed,
        skipped_missing_prereq=skipped_missing_prereq,
    )

    summaries = load_pickle_if_exists(
        study_dir / "staged_summaries.pkl",
        default={},
    )

    if len(tasks) == 0:
        print("No staged replicates to run.")
        return

    if args.max_workers is None:
        max_workers = min(8, len(tasks))
    else:
        max_workers = args.max_workers

    study_metadata = {
        "stage": "staged",
        "method_note": (
            "Staged NN uses EnKF-derived pseudo-pairs split according to the "
            "original train/validation time split. Pseudo-validation is used "
            "for internal NN selection. Final evaluation remains on original "
            "held-out test observations."
        ),
        "expected_enkf_usage": (
            "The upstream EnKF result should have assimilated original "
            "train+validation observations, but not test observations."
        ),
        "study_dir": str(study_dir),
        "num_replicates": num_replicates,
        "max_workers": max_workers,
        "base_staged_seed": args.base_staged_seed,
        "overwrite": args.overwrite,
        "rerun_failed": not args.no_rerun_failed,
        "selected_replicates": tasks,
        "skipped_success": skipped_success,
        "skipped_failed": skipped_failed,
        "skipped_missing_prereq": skipped_missing_prereq,
        "staged_config": cfg_dict,
    }

    with open(study_dir / "staged_last_run_metadata.json", "w") as f:
        json.dump(study_metadata, f, indent=2)

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
    ) as executor:
        futures = [
            executor.submit(
                run_one_replicate_staged,
                r,
                str(study_dir),
                cfg_dict,
                args.base_staged_seed,
            )
            for r in tasks
        ]

        for future in as_completed(futures):
            summary = future.result()
            r = summary["replicate"]
            summaries[r] = summary

            if summary.get("success", False):
                msg = (
                    f"Finished staged replicate {r:03d}: "
                    f"test_loss={summary['test_loss']:.4e}, "
                    f"state_rmse={summary['state_rmse_test']:.4e}, "
                    f"closure_rmse={summary['closure_rmse_test']:.4e}, "
                    f"pseudo_val_loss={summary['best_val_loss']:.4e}, "
                    f"bad_starts={summary['bad_starts']}"
                )

                if summary.get("diagnostic_mode", False):
                    msg += (
                        f", diagnostic_trainval_rmse="
                        f"{summary['diagnostic_trainval_rmse']:.4e}"
                    )

                print(msg)

            else:
                print(
                    f"Staged replicate {r:03d} failed: "
                    f"{summary.get('error', 'unknown error')}"
                )

    with open(study_dir / "staged_summaries.pkl", "wb") as f:
        pickle.dump(summaries, f)

    with open(study_dir / "staged_summaries.json", "w") as f:
        json.dump({str(k): v for k, v in summaries.items()}, f, indent=2)

    print(f"Saved staged summaries to {study_dir / 'staged_summaries.pkl'}")
    print(f"Saved staged summaries to {study_dir / 'staged_summaries.json'}")


if __name__ == "__main__":
    main()