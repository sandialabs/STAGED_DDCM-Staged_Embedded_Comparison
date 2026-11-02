from __future__ import annotations

from pathlib import Path
import pickle

from dataclasses import dataclass
from typing import Optional, Any
import math
import numpy as np

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr

import equinox as eqx
import diffrax
import optax
import jaxopt


# =============================================================================
# Constants
# =============================================================================

FAIL_LOSS = 1.0e30


# =============================================================================
# Configuration/data containers
# =============================================================================

@dataclass(frozen=True)
class SIQRParams:
    beta: float = 0.15
    gamma: float = 0.013
    delta: float = 0.01

@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 1.0e-3
    adam_steps: int = 2000

    # Match the old notebook call:
    # conduct_embedded_network_training(..., width_size=2, depth=2)
    width_size: int = 2
    depth: int = 2

    num_starts: int = 10

    # Patience is counted in validation checks, not AdamW iterations.
    # With val_every=10 and patience=20, early stopping occurs after
    # 20 validation checks without improvement, i.e. up to 200 AdamW
    # iterations without validation improvement.
    patience: Optional[int] = 20

    # Validate every 10 completed AdamW iterations.
    val_every: int = 10

    print_every: int = 200

    weight_decay: float = 1.0e-6
    grad_clip: float = 1.0

    use_lbfgs: bool = True
    lbfgs_maxiter: int = 500

    rtol: float = 1.0e-5
    atol: float = 1.0e-7
    max_steps: int = 16384

    # Match the old notebook:
    # eqx.nn.MLP(..., activation=jnn.sigmoid)
    #
    # Note: this is the hidden-layer activation. The final MLP output remains
    # unconstrained/identity, as in the old notebook.
    activation: Any = jnn.sigmoid

@dataclass
class DataSplit:
    train_data: jnp.ndarray
    val_data: jnp.ndarray
    test_data: jnp.ndarray
    train_t: jnp.ndarray
    val_t: jnp.ndarray
    test_t: jnp.ndarray

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class StartResult:
    model: Any
    val_loss: float
    num_eval: int
    bad: bool


@dataclass
class SplitResult:
    model: Any
    best_val_loss: float
    test_loss: float
    best_start: Optional[int]
    num_eval: int
    bad_starts: int
    start_summaries: list


# =============================================================================
# Data splitting, saving, and loading
# =============================================================================

def random_time_split(
    d,
    t_obs,
    rand_seed: int = 0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> DataSplit:
    num_data = len(t_obs)

    n_train = int(train_frac * num_data)
    n_val = int(val_frac * num_data)

    rng = np.random.default_rng(rand_seed)
    rand_ind = rng.permutation(num_data)

    train_idx = rand_ind[:n_train]
    val_idx = rand_ind[n_train:n_train + n_val]
    test_idx = rand_ind[n_train + n_val:]

    def sort_subset(idx):
        idx = np.asarray(idx)
        ts_np = np.asarray(t_obs)[idx]
        order = np.argsort(ts_np)
        sorted_idx = idx[order]

        ds = d[sorted_idx, :]
        ts = t_obs[sorted_idx]

        return ds, ts, sorted_idx

    train_data, train_t, train_idx = sort_subset(train_idx)
    val_data, val_t, val_idx = sort_subset(val_idx)
    test_data, test_t, test_idx = sort_subset(test_idx)

    return DataSplit(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_t=train_t,
        val_t=val_t,
        test_t=test_t,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


def save_split(split: DataSplit, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        filename,
        train_idx=np.asarray(split.train_idx),
        val_idx=np.asarray(split.val_idx),
        test_idx=np.asarray(split.test_idx),
        train_t=np.asarray(split.train_t),
        val_t=np.asarray(split.val_t),
        test_t=np.asarray(split.test_t),
        train_data=np.asarray(split.train_data),
        val_data=np.asarray(split.val_data),
        test_data=np.asarray(split.test_data),
    )


def load_split(filename) -> DataSplit:
    z = np.load(filename)

    return DataSplit(
        train_data=jnp.asarray(z["train_data"]),
        val_data=jnp.asarray(z["val_data"]),
        test_data=jnp.asarray(z["test_data"]),
        train_t=jnp.asarray(z["train_t"]),
        val_t=jnp.asarray(z["val_t"]),
        test_t=jnp.asarray(z["test_t"]),
        train_idx=np.asarray(z["train_idx"]),
        val_idx=np.asarray(z["val_idx"]),
        test_idx=np.asarray(z["test_idx"]),
    )


def save_pickle(obj, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Model functionality
# =============================================================================

def xavier_uniform_init(weight, key):
    out, in_ = weight.shape
    limit = math.sqrt(6.0 / (in_ + out))
    return jax.random.uniform(
        key,
        shape=(out, in_),
        minval=-limit,
        maxval=limit,
    )


def zeros_init(bias, key):
    return jnp.zeros(bias.shape)


def init_weights_biases(model, init_fn_weight, init_fn_bias, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)

    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]

    get_biases = lambda m: [
        x.bias
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]

    weights = get_weights(model)
    biases = get_biases(model)

    weight_keys = jax.random.split(key, len(weights))
    bias_keys = jax.random.split(key, len(biases))

    new_weights = [
        init_fn_weight(weight, subkey)
        for weight, subkey in zip(weights, weight_keys)
    ]

    new_biases = [
        init_fn_bias(bias, subkey)
        for bias, subkey in zip(biases, bias_keys)
    ]

    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)

    return new_model


class Func(eqx.Module):
    mlp: eqx.nn.MLP
    beta: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    delta: float = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        width_size: int,
        depth: int,
        params: SIQRParams,
        *,
        key,
        activation,
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

        self.beta = params.beta
        self.gamma = params.gamma
        self.delta = params.delta

    def closure(self, y):
        # Matches old notebook behavior:
        # NN_out = self.mlp(y)[0]
        return self.mlp(y)[0]

    def __call__(self, t, y, args):
        xi = self.closure(y)
        S, I, Q, R = y

        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I - xi * I
        dQdt = xi * I - self.delta * Q
        dRdt = self.gamma * I + self.delta * Q

        return jnp.stack([dSdt, dIdt, dQdt, dRdt])


class NeuralODE(eqx.Module):
    func: Func
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    dt0: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        config: TrainingConfig,
        params: SIQRParams,
        t0,
        t1,
        dt0,
        *,
        key,
    ):
        self.func = Func(
            state_dim=state_dim,
            width_size=config.width_size,
            depth=config.depth,
            params=params,
            key=key,
            activation=config.activation,
        )

        self.t0 = float(t0)
        self.t1 = float(t1)
        self.dt0 = float(dt0)
        self.rtol = float(config.rtol)
        self.atol = float(config.atol)
        self.max_steps = int(config.max_steps)

    def solve(self, ts, y0):
        """
        Solve over the full observation interval, as in the old notebook:

            t0 = t_obs[0]
            t1 = t_obs[-1]

        and save only at the requested times.

        throw=False prevents failed solves from raising inside JAX/JAXopt.
        The loss function below converts failed solves into a large finite
        penalty.
        """
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


# =============================================================================
# Loss functions
# =============================================================================

def observation_loss(model, ts, d_obs, y0, H, obs_var):
    """
    Observation negative log-likelihood up to an additive constant.

    If the ODE solve fails or produces nonfinite states, return a large finite
    penalty. This lets Adam/L-BFGS reject bad parameter regions without crashing
    the whole replicate.
    """
    sol = model.solve(ts, y0)
    y_pred = sol.ys

    finite_y = jnp.all(jnp.isfinite(y_pred))

    # With throw=False, Diffrax records solver failure in sol.result.
    # This should catch max_steps failures even if sol.ys is finite-looking.
    successful_solve = sol.result == diffrax.RESULTS.successful

    bad_solve = jnp.logical_or(~finite_y, ~successful_solve)

    # Prevent NaN/Inf from contaminating arithmetic before applying the penalty.
    y_safe = jnp.nan_to_num(
        y_pred,
        nan=0.0,
        posinf=1.0e10,
        neginf=-1.0e10,
    )

    d_pred = y_safe @ H.T
    residual = d_obs - d_pred

    loss = 0.5 * jnp.sum(residual ** 2) / obs_var

    loss = jnp.nan_to_num(
        loss,
        nan=FAIL_LOSS,
        posinf=FAIL_LOSS,
        neginf=FAIL_LOSS,
    )

    return jnp.where(bad_solve, FAIL_LOSS, loss)


@eqx.filter_value_and_grad
def loss_and_grad(model, ts, d_obs, y0, H, obs_var):
    return observation_loss(model, ts, d_obs, y0, H, obs_var)


@eqx.filter_jit
def eval_loss(model, ts, d_obs, y0, H, obs_var):
    return observation_loss(model, ts, d_obs, y0, H, obs_var)


@eqx.filter_jit
def make_adam_step(model, opt_state, optim, ts, d_obs, y0, H, obs_var):
    loss, grads = loss_and_grad(model, ts, d_obs, y0, H, obs_var)

    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )

    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


# =============================================================================
# Trainer class
# =============================================================================

class EmbeddedTrainer:
    def __init__(
        self,
        t_obs,
        y0,
        H,
        obs_var: float,
        siqr_params: SIQRParams,
        config: TrainingConfig,
    ):
        self.t_obs = jnp.asarray(t_obs)
        self.y0 = jnp.asarray(y0)
        self.H = jnp.asarray(H)
        self.obs_var = float(obs_var)
        self.siqr_params = siqr_params
        self.config = config

        self.state_dim = len(self.y0)
        self.t0 = self.t_obs[0]
        self.t1 = self.t_obs[-1]
        self.dt0 = self.t_obs[1] - self.t_obs[0]

    def make_model(self, seed: int):
        key = jr.PRNGKey(seed)

        return NeuralODE(
            state_dim=self.state_dim,
            config=self.config,
            params=self.siqr_params,
            t0=self.t0,
            t1=self.t1,
            dt0=self.dt0,
            key=key,
        )

    def train_single_start(self, model) -> StartResult:
        raise NotImplementedError(
            "Use train_start_on_split(model, split) instead."
        )

    def train_start_on_split(self, model, split: DataSplit) -> StartResult:
        cfg = self.config

        optim = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adamw(
                learning_rate=cfg.lr,
                weight_decay=cfg.weight_decay,
            ),
        )

        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        best_val_loss = np.inf
        best_params = None
        best_static = None
        best_step = 0

        evals_since_improvement = 0
        total_eval = 0

        for step in range(cfg.adam_steps):
            model, opt_state, train_loss = make_adam_step(
                model,
                opt_state,
                optim,
                split.train_t,
                split.train_data,
                self.y0,
                self.H,
                self.obs_var,
            )
            total_eval += 1

            adam_iter = step + 1

            # Validate every cfg.val_every completed AdamW iterations.
            do_validation = (
                adam_iter % cfg.val_every == 0
                or adam_iter == cfg.adam_steps
            )

            if do_validation:
                val_loss = eval_loss(
                    model,
                    split.val_t,
                    split.val_data,
                    self.y0,
                    self.H,
                    self.obs_var,
                )
                total_eval += 1

                train_loss_float = float(train_loss)
                val_loss_float = float(val_loss)

                # Treat solver-penalty losses as bad starts rather than valid
                # candidates. This avoids accepting FAIL_LOSS as the first
                # finite "best" loss.
                if (
                    not np.isfinite(val_loss_float)
                    or val_loss_float >= 0.1 * FAIL_LOSS
                ):
                    print(
                        f"Bad initialization/solver failure at AdamW iter {adam_iter}: "
                        f"train loss={train_loss_float:.4e}, "
                        f"val loss={val_loss_float:.4e}"
                    )
                    return StartResult(
                        model=None,
                        val_loss=np.inf,
                        num_eval=total_eval,
                        bad=True,
                    )

                if val_loss_float < best_val_loss:
                    best_val_loss = val_loss_float
                    best_step = adam_iter
                    best_params, best_static = eqx.partition(model, eqx.is_array)
                    evals_since_improvement = 0
                else:
                    evals_since_improvement += 1

                if adam_iter % cfg.print_every == 0 or adam_iter == cfg.adam_steps:
                    print(
                        f"AdamW iter {adam_iter:5d}, "
                        f"train loss={train_loss_float:.4e}, "
                        f"val loss={val_loss_float:.4e}, "
                        f"best val={best_val_loss:.4e}"
                    )

                if (
                    cfg.patience is not None
                    and evals_since_improvement >= cfg.patience
                ):
                    print(
                        f"AdamW early stopping at iter {adam_iter}. "
                        f"Best iter was {best_step}."
                        )
                    break

        if best_params is None:
            return StartResult(
                model=None,
                val_loss=np.inf,
                num_eval=total_eval,
                bad=True,
            )

        best_adam_model = eqx.combine(best_params, best_static)

        if not cfg.use_lbfgs:
            return StartResult(
                model=best_adam_model,
                val_loss=best_val_loss,
                num_eval=total_eval,
                bad=False,
            )

        print("Starting L-BFGS fine-tuning on training data...")

        params, static = eqx.partition(best_adam_model, eqx.is_array)

        def lbfgs_train_loss(params):
            """
            Methodologically important:

            L-BFGS calibrates against TRAINING data only. Validation data is
            not used inside this objective.
            """
            model_ = eqx.combine(params, static)
            return eval_loss(
                model_,
                split.train_t,
                split.train_data,
                self.y0,
                self.H,
                self.obs_var,
            )

        lbfgs = jaxopt.LBFGS(
            fun=lbfgs_train_loss,
            maxiter=cfg.lbfgs_maxiter,
            jit=True,
        )

        try:
            result = lbfgs.run(params)
        except Exception as exc:
            print(f"L-BFGS failed with exception: {exc}")
            print("Keeping best Adam model.")
            return StartResult(
                model=best_adam_model,
                val_loss=best_val_loss,
                num_eval=total_eval,
                bad=False,
            )

        try:
            total_eval += int(result.state.num_fun_eval)
        except Exception:
            pass

        lbfgs_model = eqx.combine(result.params, static)

        # Validation is used only for model selection, not for calibration.
        lbfgs_val_loss = eval_loss(
            lbfgs_model,
            split.val_t,
            split.val_data,
            self.y0,
            self.H,
            self.obs_var,
        )
        total_eval += 1

        lbfgs_val_loss_float = float(lbfgs_val_loss)

        print(
            f"L-BFGS val loss={lbfgs_val_loss_float:.4e}, "
            f"best Adam val loss={best_val_loss:.4e}"
        )

        if (
            np.isfinite(lbfgs_val_loss_float)
            and lbfgs_val_loss_float < 0.1 * FAIL_LOSS
            and lbfgs_val_loss_float <= best_val_loss
        ):
            return StartResult(
                model=lbfgs_model,
                val_loss=lbfgs_val_loss_float,
                num_eval=total_eval,
                bad=False,
            )

        print("Keeping best Adam model after L-BFGS model-selection check.")

        return StartResult(
            model=best_adam_model,
            val_loss=best_val_loss,
            num_eval=total_eval,
            bad=False,
        )

    def train_split(self, split: DataSplit, seed: int = 100) -> SplitResult:
        cfg = self.config

        best_model = None
        best_val_loss = np.inf
        best_start = None

        total_eval = 0
        bad_starts = 0
        start_summaries = []

        for start in range(cfg.num_starts):
            print(f"\n=== Start {start + 1} / {cfg.num_starts} ===")

            model = self.make_model(seed + start)

            start_result = self.train_start_on_split(model, split)

            total_eval += start_result.num_eval

            if start_result.bad or start_result.model is None:
                bad_starts += 1
                start_summaries.append(
                    {
                        "start": start,
                        "bad": True,
                        "val_loss": np.inf,
                        "num_eval": start_result.num_eval,
                    }
                )
                continue

            start_summaries.append(
                {
                    "start": start,
                    "bad": False,
                    "val_loss": start_result.val_loss,
                    "num_eval": start_result.num_eval,
                }
            )

            if start_result.val_loss < best_val_loss:
                best_val_loss = start_result.val_loss
                best_model = start_result.model
                best_start = start

        if best_model is None:
            return SplitResult(
                model=None,
                best_val_loss=np.inf,
                test_loss=np.inf,
                best_start=None,
                num_eval=total_eval,
                bad_starts=bad_starts,
                start_summaries=start_summaries,
            )

        test_loss = eval_loss(
            best_model,
            split.test_t,
            split.test_data,
            self.y0,
            self.H,
            self.obs_var,
        )
        # total_eval += 1

        test_loss_float = float(test_loss)

        return SplitResult(
            model=best_model,
            best_val_loss=best_val_loss,
            test_loss=test_loss_float,
            best_start=best_start,
            num_eval=total_eval,
            bad_starts=bad_starts,
            start_summaries=start_summaries,
        )


# =============================================================================
# Evaluation helper
# =============================================================================

@eqx.filter_jit
def evaluate_closure(model, ts, y0):
    ys = model(ts, y0)
    xi = jax.vmap(model.func.closure)(ys)
    return xi, ys