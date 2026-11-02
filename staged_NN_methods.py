from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Literal
import pickle

import numpy as np

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr

import equinox as eqx
import diffrax
import optax
import jaxopt

from embedded_NN_methods import (
    SIQRParams,
    DataSplit,
    load_split,
    xavier_uniform_init,
    zeros_init,
    init_weights_biases,
)


# =============================================================================
# Constants
# =============================================================================

STAGED_FAIL_LOSS = 1.0e30


# =============================================================================
# Configuration/result containers
# =============================================================================

@dataclass(frozen=True)
class StagedNNConfig:
    """
    Configuration for staged closure learning.

    Staged pipeline:

        1. EnKF estimates x_mu = [S, I, Q, R, xi].
        2. NN learns xi_NN(S, I, Q, R) from EnKF-derived pseudo-pairs.
        3. NN is embedded back into SIQR ODE for final held-out test evaluation.

    Under the one-EnKF protocol, the EnKF should be run once using the union of
    the original training and validation observation times. Then the EnKF-derived
    pseudo-pairs are split using the original train/validation time split:

        pseudo-train = EnKF-derived pairs at original train_t
        pseudo-val   = EnKF-derived pairs at original val_t

    The pseudo-validation set is an internal EnKF-derived validation set. It is
    not independent in the same sense as raw held-out observation validation,
    because the validation observations were assimilated by the EnKF.
    """

    # Match updated embedded implementation and old notebook defaults.
    width_size: int = 2
    depth: int = 2
    activation: Any = jnn.sigmoid

    lr: float = 1.0e-3
    adam_steps: int = 2000
    num_starts: int = 10

    # Validate/checkpoint every 10 completed AdamW iterations.
    val_every: int = 10

    # Patience is counted in validation checks, not AdamW iterations.
    # With val_every=10 and patience=20, early stopping occurs after
    # 20 validation checks without improvement.
    patience: Optional[int] = 20

    print_every: int = 200
    weight_decay: float = 1.0e-6
    grad_clip: float = 1.0

    use_lbfgs: bool = True
    lbfgs_maxiter: int = 500

    # "full_enkf_trajectory":
    #     Train on the full EnKF-estimated mean trajectory, enkf_result["x_mu"],
    #     evaluated at enkf_result["t_mu"]. This does not mean all raw
    #     observations were assimilated; the EnKF phase controls which
    #     observations are used.
    #
    # "assimilation_step_enkf":
    #     Train only on EnKF estimates at assimilation times,
    #     enkf_result["x_mu_assim"], evaluated at enkf_result["t_assim"].
    #
    # For the one-EnKF staged protocol, "assimilation_step_enkf" is usually the
    # clearest option, assuming the EnKF assimilated original train+val times.
    train_source: Literal[
        "full_enkf_trajectory",
        "assimilation_step_enkf",
    ] = "assimilation_step_enkf"

    # Tolerance for matching original split times to EnKF pseudo-data times.
    pseudo_time_tol: float = 1.0e-8

    # ODE solver for embedding learned staged NN closure.
    rtol: float = 1.0e-5
    atol: float = 1.0e-7
    max_steps: int = 16384


@dataclass
class StagedStartResult:
    model: Any
    train_loss: float
    val_loss: float
    num_eval: int
    bad: bool


@dataclass
class StagedResult:
    model: Any
    best_train_loss: float
    best_val_loss: float
    test_loss: float
    state_rmse_test: float
    closure_rmse_test: float
    best_start: Optional[int]
    num_eval: int
    bad_starts: int
    start_summaries: list
    train_source: str


# =============================================================================
# Data extraction from EnKF results
# =============================================================================

def _unique_time_rows(t, x, tol: float = 1.0e-12):
    """
    Sort by time and drop duplicate/nearly duplicate time entries.

    If duplicate/nearly duplicate times occur, this keeps the first row in
    sorted-time order.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    if len(t) == 0:
        return t, x

    order = np.argsort(t)
    t_sorted = t[order]
    x_sorted = x[order]

    keep = [0]

    for i in range(1, len(t_sorted)):
        if abs(t_sorted[i] - t_sorted[keep[-1]]) > tol:
            keep.append(i)

    keep = np.asarray(keep, dtype=int)

    return t_sorted[keep], x_sorted[keep]


def get_enkf_supervised_pairs(
    enkf_data,
    source: Literal[
        "full_enkf_trajectory",
        "assimilation_step_enkf",
    ] = "full_enkf_trajectory",
    remove_duplicate_times: bool = True,
):
    """
    Extract supervised staged-NN training pairs from EnKF result.

    Parameters
    ----------
    enkf_data:
        np.load object from enkf_result.npz.
    source:
        "full_enkf_trajectory":
            Use enkf["x_mu"] and enkf["t_mu"]. This is the full EnKF-estimated
            mean trajectory produced by the EnKF run.

        "assimilation_step_enkf":
            Use enkf["x_mu_assim"] and enkf["t_assim"]. These are the EnKF
            estimates at assimilation times.
    remove_duplicate_times:
        If True, sort by time and drop duplicate time entries.

    Returns
    -------
    t_train:
        Times associated with pseudo-pairs, shape (N,).
    X_train:
        EnKF-estimated states [S,I,Q,R], shape (N, 4).
    y_train:
        EnKF-estimated closure xi, shape (N,).
    """
    if source == "full_enkf_trajectory":
        t = np.asarray(enkf_data["t_mu"], dtype=float)
        x = np.asarray(enkf_data["x_mu"], dtype=float)

    elif source == "assimilation_step_enkf":
        t = np.asarray(enkf_data["t_assim"], dtype=float)
        x = np.asarray(enkf_data["x_mu_assim"], dtype=float)

    else:
        raise ValueError(f"Unknown staged training source: {source}")

    if x.ndim != 2 or x.shape[1] < 5:
        raise ValueError(
            f"Expected EnKF state array shape (N, >=5). Got shape {x.shape}."
        )

    finite_mask = np.all(np.isfinite(x), axis=1) & np.isfinite(t)
    t = t[finite_mask]
    x = x[finite_mask]

    if remove_duplicate_times:
        t, x = _unique_time_rows(t, x)

    X_train = x[:, :4]
    y_train = x[:, 4]

    return (
        jnp.asarray(t),
        jnp.asarray(X_train),
        jnp.asarray(y_train),
    )


def _match_times_to_rows(
    t_all,
    X_all,
    y_all,
    target_t,
    tol: float = 1.0e-8,
):
    """
    Select rows from EnKF-derived pseudo-data whose times match target_t.

    This is used to apply the original train/validation time split to the
    EnKF-derived pseudo-pairs.
    """
    t_all_np = np.asarray(t_all, dtype=float)
    X_all_np = np.asarray(X_all, dtype=float)
    y_all_np = np.asarray(y_all, dtype=float)
    target_t_np = np.asarray(target_t, dtype=float)

    if len(t_all_np) == 0:
        raise ValueError("Cannot match times because EnKF pseudo-data is empty.")

    matched_indices = []

    for tt in target_t_np:
        diffs = np.abs(t_all_np - tt)
        idx = int(np.argmin(diffs))

        if diffs[idx] > tol:
            raise ValueError(
                f"Could not match target time {tt} to EnKF pseudo-data. "
                f"Closest time was {t_all_np[idx]} with difference {diffs[idx]}. "
                f"Increase pseudo_time_tol if this is expected."
            )

        matched_indices.append(idx)

    matched_indices = np.asarray(matched_indices, dtype=int)

    return (
        jnp.asarray(t_all_np[matched_indices]),
        jnp.asarray(X_all_np[matched_indices, :]),
        jnp.asarray(y_all_np[matched_indices]),
    )


def get_enkf_supervised_pairs_for_original_split(
    enkf_data,
    split: DataSplit,
    source: Literal[
        "full_enkf_trajectory",
        "assimilation_step_enkf",
    ] = "assimilation_step_enkf",
    remove_duplicate_times: bool = True,
    tol: float = 1.0e-8,
):
    """
    Extract EnKF-derived pseudo-training and pseudo-validation pairs using the
    original observation-level train/validation time split.

    Intended one-EnKF staged protocol:

        1. EnKF assimilates original train + validation observations.
        2. EnKF output is converted into supervised closure pairs.
        3. Original train_t selects pseudo-training pairs.
        4. Original val_t selects pseudo-validation pairs.

    The pseudo-validation set is used for NN early stopping, random-start
    selection, and L-BFGS accept/reject. It is not an independent raw-observation
    validation set if those validation observations were assimilated by EnKF.
    """
    t_all, X_all, y_all = get_enkf_supervised_pairs(
        enkf_data=enkf_data,
        source=source,
        remove_duplicate_times=remove_duplicate_times,
    )

    t_pseudo_train, X_pseudo_train, y_pseudo_train = _match_times_to_rows(
        t_all=t_all,
        X_all=X_all,
        y_all=y_all,
        target_t=split.train_t,
        tol=tol,
    )

    t_pseudo_val, X_pseudo_val, y_pseudo_val = _match_times_to_rows(
        t_all=t_all,
        X_all=X_all,
        y_all=y_all,
        target_t=split.val_t,
        tol=tol,
    )

    return {
        "t_pseudo_train": t_pseudo_train,
        "X_pseudo_train": X_pseudo_train,
        "y_pseudo_train": y_pseudo_train,
        "t_pseudo_val": t_pseudo_val,
        "X_pseudo_val": X_pseudo_val,
        "y_pseudo_val": y_pseudo_val,
        "t_all": t_all,
        "X_all": X_all,
        "y_all": y_all,
    }


# =============================================================================
# Staged NN closure model
# =============================================================================

class StagedClosureNet(eqx.Module):
    """
    Feedforward NN closure:

        xi_NN = xi_NN(S,I,Q,R)

    Output is unconstrained, matching the old notebook behavior.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        state_dim: int,
        width_size: int,
        depth: int,
        *,
        key,
        activation=jnn.sigmoid,
    ):
        self.mlp = eqx.nn.MLP(
            in_size=state_dim,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )

        self.mlp = init_weights_biases(
            self.mlp,
            xavier_uniform_init,
            zeros_init,
            key,
        )

    def __call__(self, y):
        return self.mlp(y)[0]


@eqx.filter_jit
def staged_predict_batch(model, X):
    return jax.vmap(model)(X)


def _flatten_prediction_and_target(model, X, y):
    """
    Evaluate model on X and flatten both prediction and target to shape (N,).

    This avoids accidental broadcasting such as:
        pred.shape == (N,)
        y.shape    == (N, 1)
    which would otherwise create an erroneous residual of shape (N, N).
    """
    pred = staged_predict_batch(model, X)

    pred = jnp.reshape(pred, (-1,))
    y = jnp.reshape(y, (-1,))

    return pred, y


def supervised_loss(model, X, y):
    """
    Staged supervised xi loss.

    This is intentionally a sum-of-squares style loss, not a mean-squared error,
    to be closer to the embedded loss convention:

        embedded:
            0.5 / obs_var * sum ||d_i - H y_i||^2

        staged:
            0.5 * sum |xi_NN(X_i) - xi_i|^2

    If a meaningful xi variance is introduced later, this could become:

        0.5 / xi_var * sum |xi_NN(X_i) - xi_i|^2

    The predictions and targets are explicitly flattened before subtraction to
    avoid silent broadcasting bugs.
    """
    pred, y_flat = _flatten_prediction_and_target(model, X, y)

    finite_ok = jnp.all(jnp.isfinite(pred)) & jnp.all(jnp.isfinite(y_flat))

    pred_safe = jnp.nan_to_num(
        pred,
        nan=0.0,
        posinf=1.0e10,
        neginf=-1.0e10,
    )

    y_safe = jnp.nan_to_num(
        y_flat,
        nan=0.0,
        posinf=1.0e10,
        neginf=-1.0e10,
    )

    residual = pred_safe - y_safe

    # Consistent with embedded's 0.5 * sum-of-squares structure.
    loss = 0.5 * jnp.sum(residual ** 2)

    loss = jnp.nan_to_num(
        loss,
        nan=STAGED_FAIL_LOSS,
        posinf=STAGED_FAIL_LOSS,
        neginf=STAGED_FAIL_LOSS,
    )

    return jnp.where(finite_ok, loss, STAGED_FAIL_LOSS)


def supervised_mse(model, X, y):
    """
    Mean squared xi error. Used for interpretable diagnostics only.
    """
    pred, y_flat = _flatten_prediction_and_target(model, X, y)

    finite_ok = jnp.all(jnp.isfinite(pred)) & jnp.all(jnp.isfinite(y_flat))

    pred_safe = jnp.nan_to_num(
        pred,
        nan=0.0,
        posinf=1.0e10,
        neginf=-1.0e10,
    )

    y_safe = jnp.nan_to_num(
        y_flat,
        nan=0.0,
        posinf=1.0e10,
        neginf=-1.0e10,
    )

    residual = pred_safe - y_safe
    mse = jnp.mean(residual ** 2)

    mse = jnp.nan_to_num(
        mse,
        nan=STAGED_FAIL_LOSS,
        posinf=STAGED_FAIL_LOSS,
        neginf=STAGED_FAIL_LOSS,
    )

    return jnp.where(finite_ok, mse, STAGED_FAIL_LOSS)


def supervised_rmse(model, X, y):
    """
    Root-mean-squared xi error. Same physical units as xi.
    """
    mse = supervised_mse(model, X, y)
    return jnp.sqrt(mse)


@eqx.filter_value_and_grad
def supervised_loss_and_grad(model, X, y):
    return supervised_loss(model, X, y)


@eqx.filter_jit
def supervised_eval_loss(model, X, y):
    return supervised_loss(model, X, y)


@eqx.filter_jit
def supervised_eval_mse(model, X, y):
    return supervised_mse(model, X, y)


@eqx.filter_jit
def supervised_eval_rmse(model, X, y):
    return supervised_rmse(model, X, y)


@eqx.filter_jit
def make_staged_adam_step(model, opt_state, optim, X, y):
    loss, grads = supervised_loss_and_grad(model, X, y)

    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )

    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


# =============================================================================
# Train staged NN closure
# =============================================================================

def train_staged_single_start(
    model,
    X_train,
    y_train,
    cfg: StagedNNConfig,
    X_val=None,
    y_val=None,
):
    """
    Train one staged-NN random initialization.

    AdamW/L-BFGS parameter updates use pseudo-training pairs only.

    Validation/checkpointing is performed every cfg.val_every completed AdamW
    iterations. Patience is counted in validation checks, not raw AdamW steps.

    If X_val/y_val are supplied, then pseudo-validation loss is used for:
        - checkpointing,
        - early stopping,
        - L-BFGS acceptance/rejection.

    This mirrors the embedded NN training pattern while recognizing that the
    staged validation set is EnKF-derived pseudo-validation.

    Loss convention:
        train_loss / val_loss are 0.5 * sum of squared xi residuals.

    Diagnostic convention:
        train_rmse / val_rmse are root-mean-squared xi residuals in xi units.
    """
    has_val = X_val is not None and y_val is not None and X_val.shape[0] > 0

    optim = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay,
        ),
    )

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    best_selection_loss = np.inf
    best_train_loss = np.inf
    best_params = None
    best_static = None
    best_iter = 0

    evals_since_improvement = 0
    total_eval = 0

    for step in range(cfg.adam_steps):
        model, opt_state, train_loss = make_staged_adam_step(
            model,
            opt_state,
            optim,
            X_train,
            y_train,
        )
        total_eval += 1

        adam_iter = step + 1
        train_loss_float = float(train_loss)

        if (
            not np.isfinite(train_loss_float)
            or train_loss_float >= 0.1 * STAGED_FAIL_LOSS
        ):
            print(
                f"Bad staged initialization during AdamW at iter {adam_iter}: "
                f"train loss={train_loss_float:.4e}"
            )
            return StagedStartResult(
                model=None,
                train_loss=np.inf,
                val_loss=np.inf,
                num_eval=total_eval,
                bad=True,
            )

        # Match embedded behavior: validate/checkpoint every cfg.val_every
        # completed AdamW iterations and also at the final iteration.
        do_validation = (
            adam_iter % cfg.val_every == 0
            or adam_iter == cfg.adam_steps
        )

        if not do_validation:
            continue

        if has_val:
            selection_loss = supervised_eval_loss(model, X_val, y_val)
            total_eval += 1
        else:
            selection_loss = train_loss

        selection_loss_float = float(selection_loss)

        if (
            not np.isfinite(selection_loss_float)
            or selection_loss_float >= 0.1 * STAGED_FAIL_LOSS
        ):
            print(
                f"Bad staged initialization during pseudo-validation "
                f"at AdamW iter {adam_iter}: "
                f"train loss={train_loss_float:.4e}, "
                f"selection loss={selection_loss_float:.4e}"
            )
            return StagedStartResult(
                model=None,
                train_loss=np.inf,
                val_loss=np.inf,
                num_eval=total_eval,
                bad=True,
            )

        if selection_loss_float < best_selection_loss:
            best_selection_loss = selection_loss_float
            best_train_loss = train_loss_float
            best_iter = adam_iter
            best_params, best_static = eqx.partition(model, eqx.is_array)
            evals_since_improvement = 0
        else:
            evals_since_improvement += 1

        if adam_iter % cfg.print_every == 0 or adam_iter == cfg.adam_steps:
            train_rmse = float(supervised_eval_rmse(model, X_train, y_train))

            if has_val:
                val_rmse = float(supervised_eval_rmse(model, X_val, y_val))
                print(
                    f"Staged AdamW iter {adam_iter:5d}, "
                    f"train loss={train_loss_float:.4e}, "
                    f"train rmse={train_rmse:.4e}, "
                    f"pseudo-val loss={selection_loss_float:.4e}, "
                    f"pseudo-val rmse={val_rmse:.4e}, "
                    f"best pseudo-val={best_selection_loss:.4e}"
                )
            else:
                print(
                    f"Staged AdamW iter {adam_iter:5d}, "
                    f"train loss={train_loss_float:.4e}, "
                    f"train rmse={train_rmse:.4e}, "
                    f"best train={best_selection_loss:.4e}"
                )

        if cfg.patience is not None and evals_since_improvement >= cfg.patience:
            print(
                f"Staged AdamW early stopping at iter {adam_iter}. "
                f"Best iter was {best_iter}."
            )
            break

    if best_params is None:
        return StagedStartResult(
            model=None,
            train_loss=np.inf,
            val_loss=np.inf,
            num_eval=total_eval,
            bad=True,
        )

    best_adam_model = eqx.combine(best_params, best_static)

    if not cfg.use_lbfgs:
        return StagedStartResult(
            model=best_adam_model,
            train_loss=best_train_loss,
            val_loss=best_selection_loss,
            num_eval=total_eval,
            bad=False,
        )

    print("Starting staged L-BFGS fine-tuning on pseudo-training loss...")

    params, static = eqx.partition(best_adam_model, eqx.is_array)

    def lbfgs_loss(params):
        """
        L-BFGS calibrates against pseudo-training pairs only.

        The objective is the same staged supervised loss:
            0.5 * sum |xi_NN(X_i) - xi_i|^2

        Pseudo-validation is not used inside this objective.
        """
        model_ = eqx.combine(params, static)
        return supervised_eval_loss(model_, X_train, y_train)

    lbfgs = jaxopt.LBFGS(
        fun=lbfgs_loss,
        maxiter=cfg.lbfgs_maxiter,
        jit=True,
    )

    try:
        result = lbfgs.run(params)
    except Exception as exc:
        print(f"Staged L-BFGS failed with exception: {exc}")
        print("Keeping best staged AdamW model.")
        return StagedStartResult(
            model=best_adam_model,
            train_loss=best_train_loss,
            val_loss=best_selection_loss,
            num_eval=total_eval,
            bad=False,
        )

    try:
        total_eval += int(result.state.num_fun_eval)
    except Exception:
        pass

    lbfgs_model = eqx.combine(result.params, static)

    lbfgs_train_loss = supervised_eval_loss(lbfgs_model, X_train, y_train)
    lbfgs_train_rmse = supervised_eval_rmse(lbfgs_model, X_train, y_train)
    total_eval += 1

    lbfgs_train_loss_float = float(lbfgs_train_loss)
    lbfgs_train_rmse_float = float(lbfgs_train_rmse)

    if has_val:
        lbfgs_selection_loss = supervised_eval_loss(lbfgs_model, X_val, y_val)
        lbfgs_selection_rmse = supervised_eval_rmse(lbfgs_model, X_val, y_val)
        total_eval += 1
    else:
        lbfgs_selection_loss = lbfgs_train_loss
        lbfgs_selection_rmse = lbfgs_train_rmse

    lbfgs_selection_loss_float = float(lbfgs_selection_loss)
    lbfgs_selection_rmse_float = float(lbfgs_selection_rmse)

    if has_val:
        best_adam_val_rmse = float(
            supervised_eval_rmse(best_adam_model, X_val, y_val)
        )
        print(
            f"Staged L-BFGS train loss={lbfgs_train_loss_float:.4e}, "
            f"train rmse={lbfgs_train_rmse_float:.4e}, "
            f"pseudo-val loss={lbfgs_selection_loss_float:.4e}, "
            f"pseudo-val rmse={lbfgs_selection_rmse_float:.4e}, "
            f"best AdamW pseudo-val loss={best_selection_loss:.4e}, "
            f"best AdamW pseudo-val rmse={best_adam_val_rmse:.4e}"
        )
    else:
        best_adam_train_rmse = float(
            supervised_eval_rmse(best_adam_model, X_train, y_train)
        )
        print(
            f"Staged L-BFGS train loss={lbfgs_train_loss_float:.4e}, "
            f"train rmse={lbfgs_train_rmse_float:.4e}, "
            f"best AdamW train loss={best_selection_loss:.4e}, "
            f"best AdamW train rmse={best_adam_train_rmse:.4e}"
        )

    if (
        np.isfinite(lbfgs_train_loss_float)
        and np.isfinite(lbfgs_selection_loss_float)
        and lbfgs_train_loss_float < 0.1 * STAGED_FAIL_LOSS
        and lbfgs_selection_loss_float < 0.1 * STAGED_FAIL_LOSS
        and lbfgs_selection_loss_float <= best_selection_loss
    ):
        return StagedStartResult(
            model=lbfgs_model,
            train_loss=lbfgs_train_loss_float,
            val_loss=lbfgs_selection_loss_float,
            num_eval=total_eval,
            bad=False,
        )

    print("Keeping best staged AdamW model after L-BFGS selection check.")

    return StagedStartResult(
        model=best_adam_model,
        train_loss=best_train_loss,
        val_loss=best_selection_loss,
        num_eval=total_eval,
        bad=False,
    )

def train_staged_multistart(
    X_train,
    y_train,
    cfg: StagedNNConfig,
    seed: int = 50_000,
    X_val=None,
    y_val=None,
):
    """
    Train staged closure NN over multiple random starts.

    If pseudo-validation data are supplied, start selection is by
    pseudo-validation loss. Otherwise, start selection is by pseudo-training loss.

    Returned loss values are 0.5 * sum of squared xi residuals.
    RMSE values are also returned for interpretability.
    """
    has_val = X_val is not None and y_val is not None and X_val.shape[0] > 0

    best_model = None
    best_selection_loss = np.inf
    best_train_loss = np.inf
    best_start = None

    total_eval = 0
    bad_starts = 0
    start_summaries = []

    state_dim = X_train.shape[1]

    for start in range(cfg.num_starts):
        print(f"\n=== Staged start {start + 1} / {cfg.num_starts} ===")

        key = jr.PRNGKey(seed + start)

        model = StagedClosureNet(
            state_dim=state_dim,
            width_size=cfg.width_size,
            depth=cfg.depth,
            key=key,
            activation=cfg.activation,
        )

        start_result = train_staged_single_start(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cfg=cfg,
        )

        total_eval += start_result.num_eval

        if start_result.bad or start_result.model is None:
            bad_starts += 1
            start_summaries.append(
                {
                    "start": start,
                    "bad": True,
                    "train_loss": np.inf,
                    "val_loss": np.inf,
                    "train_rmse": np.inf,
                    "val_rmse": np.inf,
                    "num_eval": start_result.num_eval,
                }
            )
            continue

        train_rmse = float(
            supervised_eval_rmse(start_result.model, X_train, y_train)
        )

        if has_val:
            val_rmse = float(
                supervised_eval_rmse(start_result.model, X_val, y_val)
            )
        else:
            val_rmse = train_rmse

        start_summaries.append(
            {
                "start": start,
                "bad": False,
                "train_loss": float(start_result.train_loss),
                "val_loss": float(start_result.val_loss),
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "num_eval": start_result.num_eval,
            }
        )

        selection_loss = (
            float(start_result.val_loss)
            if has_val
            else float(start_result.train_loss)
        )

        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_train_loss = float(start_result.train_loss)
            best_model = start_result.model
            best_start = start

    if best_model is not None:
        best_train_rmse = float(
            supervised_eval_rmse(best_model, X_train, y_train)
        )
        if has_val:
            best_val_rmse = float(
                supervised_eval_rmse(best_model, X_val, y_val)
            )
        else:
            best_val_rmse = best_train_rmse
    else:
        best_train_rmse = np.inf
        best_val_rmse = np.inf

    return {
        "model": best_model,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_selection_loss,
        "best_train_rmse": best_train_rmse,
        "best_val_rmse": best_val_rmse,
        "best_start": best_start,
        "num_eval": total_eval,
        "bad_starts": bad_starts,
        "start_summaries": start_summaries,
        "used_pseudo_validation": bool(has_val),
    }


# =============================================================================
# Embed staged closure NN in SIQR ODE
# =============================================================================

class StagedSIQRFunc(eqx.Module):
    net: StagedClosureNet
    beta: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    delta: float = eqx.field(static=True)

    def __init__(self, net, params: SIQRParams):
        self.net = net
        self.beta = params.beta
        self.gamma = params.gamma
        self.delta = params.delta

    def closure(self, y):
        return self.net(y)

    def __call__(self, t, y, args):
        xi = self.closure(y)

        S, I, Q, R = y

        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I - xi * I
        dQdt = xi * I - self.delta * Q
        dRdt = self.gamma * I + self.delta * Q

        return jnp.stack([dSdt, dIdt, dQdt, dRdt])


class StagedNeuralODE(eqx.Module):
    func: StagedSIQRFunc
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    dt0: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        net,
        params: SIQRParams,
        t0,
        t1,
        dt0,
        cfg: StagedNNConfig,
    ):
        self.func = StagedSIQRFunc(net, params)

        self.t0 = float(t0)
        self.t1 = float(t1)
        self.dt0 = float(dt0)
        self.rtol = float(cfg.rtol)
        self.atol = float(cfg.atol)
        self.max_steps = int(cfg.max_steps)

    def solve(self, ts, y0):
        ts = jnp.asarray(ts)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Dopri5(),
            t0=self.t0,
            t1=self.t1,
            dt0=self.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol,
                atol=self.atol,
            ),
            max_steps=self.max_steps,
            throw=False,
        )

        return solution

    def __call__(self, ts, y0):
        solution = self.solve(ts, y0)
        return solution.ys


@eqx.filter_jit
def evaluate_staged_closure(ode_model, ts, y0):
    ys = ode_model(ts, y0)
    xi = jax.vmap(ode_model.func.closure)(ys)
    return xi, ys


def observation_loss_np(y_pred, d_obs_time_major, H, obs_var, idx):
    """
    Observation-space held-out loss:

        0.5 / obs_var * sum ||d_i - H y_i||^2

    over specified indices.
    """
    y_pred = np.asarray(y_pred)
    d_obs_time_major = np.asarray(d_obs_time_major)
    H = np.asarray(H)

    if not np.all(np.isfinite(y_pred)):
        return float(np.inf)

    d_pred = y_pred @ H.T

    if not np.all(np.isfinite(d_pred)):
        return float(np.inf)

    residual = d_obs_time_major[idx, :] - d_pred[idx, :]

    if not np.all(np.isfinite(residual)):
        return float(np.inf)

    return float(0.5 * np.sum(residual ** 2) / float(obs_var))


def state_rmse_np(y_pred, y_true_time_major, idx):
    y_pred = np.asarray(y_pred)
    y_true_time_major = np.asarray(y_true_time_major)

    if not np.all(np.isfinite(y_pred)):
        return float(np.inf)

    residual = y_true_time_major[idx, :] - y_pred[idx, :]

    if not np.all(np.isfinite(residual)):
        return float(np.inf)

    return float(np.sqrt(np.mean(residual ** 2)))


def closure_rmse_np(xi_pred, xi_true, idx):
    xi_pred = np.asarray(xi_pred)
    xi_true = np.asarray(xi_true)

    if not np.all(np.isfinite(xi_pred)):
        return float(np.inf)

    residual = xi_true[idx] - xi_pred[idx]

    if not np.all(np.isfinite(residual)):
        return float(np.inf)

    return float(np.sqrt(np.mean(residual ** 2)))


def evaluate_staged_model_on_replicate(
    staged_net,
    t_obs,
    y0,
    params: SIQRParams,
    cfg: StagedNNConfig,
    d_obs_time_major,
    H,
    obs_var,
    test_idx,
    y_true_time_major=None,
    xi_true=None,
):
    """
    Embed staged NN in SIQR ODE, integrate over t_obs, and evaluate on test_idx.

    This function does not use raw train/validation observations for scoring.
    The final reported loss is on the original held-out test indices.
    """
    t_obs = np.asarray(t_obs)
    y0 = np.asarray(y0)

    ode_model = StagedNeuralODE(
        net=staged_net,
        params=params,
        t0=t_obs[0],
        t1=t_obs[-1],
        dt0=t_obs[1] - t_obs[0],
        cfg=cfg,
    )

    try:
        xi_pred, y_pred = evaluate_staged_closure(
            ode_model,
            jnp.asarray(t_obs),
            jnp.asarray(y0),
        )

        xi_pred_np = np.asarray(xi_pred)
        y_pred_np = np.asarray(y_pred)

        if (
            not np.all(np.isfinite(xi_pred_np))
            or not np.all(np.isfinite(y_pred_np))
        ):
            raise RuntimeError("Staged embedded ODE produced nonfinite values.")

    except Exception as exc:
        print(f"Staged embedded ODE evaluation failed: {exc}")

        xi_pred_np = np.full_like(t_obs, np.nan, dtype=float)
        y_pred_np = np.full((len(t_obs), len(y0)), np.nan, dtype=float)

        return {
            "ode_model": ode_model,
            "xi_pred": xi_pred_np,
            "y_pred": y_pred_np,
            "test_loss": float(np.inf),
            "state_rmse_test": float(np.inf),
            "closure_rmse_test": float(np.inf),
            "ode_success": False,
        }

    test_loss = observation_loss_np(
        y_pred=y_pred_np,
        d_obs_time_major=d_obs_time_major,
        H=H,
        obs_var=obs_var,
        idx=test_idx,
    )

    if y_true_time_major is not None:
        state_rmse_test = state_rmse_np(
            y_pred=y_pred_np,
            y_true_time_major=y_true_time_major,
            idx=test_idx,
        )
    else:
        state_rmse_test = np.nan

    if xi_true is not None:
        closure_rmse_test = closure_rmse_np(
            xi_pred=xi_pred_np,
            xi_true=xi_true,
            idx=test_idx,
        )
    else:
        closure_rmse_test = np.nan

    ode_success = bool(
        np.isfinite(test_loss)
        and np.isfinite(state_rmse_test)
        and np.isfinite(closure_rmse_test)
    )

    return {
        "ode_model": ode_model,
        "xi_pred": xi_pred_np,
        "y_pred": y_pred_np,
        "test_loss": float(test_loss),
        "state_rmse_test": float(state_rmse_test),
        "closure_rmse_test": float(closure_rmse_test),
        "ode_success": ode_success,
    }


# =============================================================================
# Saving/loading helpers
# =============================================================================

def save_pickle(obj, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)