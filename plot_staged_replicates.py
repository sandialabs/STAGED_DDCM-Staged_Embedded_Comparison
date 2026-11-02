# %%
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


# %%
# =============================================================================
# User settings
# =============================================================================

STUDY_DIR = Path("data/study_replicates")
FIG_DIR = Path("figures_staged")
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUM_REPLICATES = 50

STATE_NAMES = ["Susceptible", "Infectious", "Quarantined", "Removed"]
STATE_LABELS = [r"$s(t)$", r"$i(t)$", r"$q(t)$", r"$r(t)$"]

PLOT_NOISY_OBS = True
SAVE_DPI = 300

# Manuscript-friendly dimensions: no wider than 6 inches.
FIG_WIDTH = 6.0
STATE_GRID_HEIGHT = 4.6
XI_FIG_HEIGHT = 3.4

# Plot aesthetics for individual staged replicate traces.
STAGED_COLOR = "C3"
STAGED_ALPHA = 0.24
STAGED_LINEWIDTH = 0.7

TRUE_COLOR = "black"
TRUE_LINEWIDTH = 1.7


# %%
# =============================================================================
# Loading helpers
# =============================================================================

def rep_dir(r: int) -> Path:
    return STUDY_DIR / f"rep_{r:03d}"


def load_json_if_exists(path):
    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def staged_replicate_successful(r: int) -> bool:
    """
    Check staged success using staged_summary.json if available.

    If the summary is missing but staged_closure.npz exists, fall back to the
    success flag inside staged_closure.npz.
    """
    rd = rep_dir(r)
    closure_file = rd / "staged_closure.npz"
    summary_file = rd / "staged_summary.json"

    if not closure_file.exists():
        return False

    summary = load_json_if_exists(summary_file)

    if summary is not None:
        return bool(summary.get("success", False))

    try:
        z = np.load(closure_file, allow_pickle=True)
        return bool(np.asarray(z["success"]).item())
    except Exception:
        return False


def load_reference_truth():
    """
    Load true trajectory, noisy observations, true xi, and split indices from the
    first available replicate with observations.npz.

    observations.npz contains:
        t_obs    shape (N,)
        d        shape (num_outputs, N)
        y_true   shape (4, N)
        xi_true  shape (N,)
    """
    for r in range(NUM_REPLICATES):
        obs_file = rep_dir(r) / "observations.npz"

        if obs_file.exists():
            obs = np.load(obs_file, allow_pickle=True)

            t_obs = np.asarray(obs["t_obs"], dtype=float)
            d_obs = np.asarray(obs["d"], dtype=float)
            y_true = np.asarray(obs["y_true"], dtype=float)
            xi_true = np.asarray(obs["xi_true"], dtype=float)

            split_file = rep_dir(r) / "split.npz"

            if split_file.exists():
                split = np.load(split_file)
                train_idx = np.asarray(split["train_idx"], dtype=int)
                val_idx = np.asarray(split["val_idx"], dtype=int)
                test_idx = np.asarray(split["test_idx"], dtype=int)
            else:
                train_idx = None
                val_idx = None
                test_idx = None

            return {
                "replicate": r,
                "t_obs": t_obs,
                "d_obs": d_obs,
                "y_true": y_true,
                "xi_true": xi_true,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }

    raise FileNotFoundError(
        f"No observations.npz files found under {STUDY_DIR}"
    )


def load_staged_replicates(t_ref):
    """
    Load staged autonomous rollout trajectories from all successful staged replicates.

    staged_closure.npz contains:
        t_obs  shape (N,)
        y      shape (N, 4)
        xi     shape (N,)

    Returns
    -------
    y_stack:
        Shape (R, N, 4).
    xi_stack:
        Shape (R, N).
    rep_ids:
        Replicate IDs loaded.
    """
    y_list = []
    xi_list = []
    rep_ids = []

    for r in range(NUM_REPLICATES):
        if not staged_replicate_successful(r):
            print(f"Skipping replicate {r:03d}: no successful staged result")
            continue

        closure_file = rep_dir(r) / "staged_closure.npz"

        try:
            z = np.load(closure_file, allow_pickle=True)

            if "t_obs" not in z.files or "y" not in z.files or "xi" not in z.files:
                print(f"Skipping replicate {r:03d}: missing t_obs, y, or xi")
                continue

            t = np.asarray(z["t_obs"], dtype=float)
            y = np.asarray(z["y"], dtype=float)
            xi = np.asarray(z["xi"], dtype=float)

            if len(t) != len(t_ref) or np.max(np.abs(t - t_ref)) > 1.0e-10:
                print(f"Skipping replicate {r:03d}: time grid mismatch")
                continue

            if y.shape != (len(t_ref), 4):
                print(f"Skipping replicate {r:03d}: unexpected y shape {y.shape}")
                continue

            if xi.shape != (len(t_ref),):
                print(f"Skipping replicate {r:03d}: unexpected xi shape {xi.shape}")
                continue

            if not np.all(np.isfinite(y)) or not np.all(np.isfinite(xi)):
                print(f"Skipping replicate {r:03d}: nonfinite y or xi")
                continue

            y_list.append(y)
            xi_list.append(xi)
            rep_ids.append(r)

        except Exception as exc:
            print(f"Skipping replicate {r:03d}: failed to load because {exc}")

    if len(y_list) == 0:
        raise RuntimeError("No successful staged replicate outputs were loaded.")

    y_stack = np.stack(y_list, axis=0)
    xi_stack = np.stack(xi_list, axis=0)

    return y_stack, xi_stack, rep_ids


# %%
# =============================================================================
# Plot styling
# =============================================================================

def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": SAVE_DPI,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def save_fig(fig, stem):
    png_file = FIG_DIR / f"{stem}.png"
    pdf_file = FIG_DIR / f"{stem}.pdf"

    fig.savefig(png_file, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")

    print(f"Saved {png_file}")
    print(f"Saved {pdf_file}")


# %%
# =============================================================================
# Plotting functions
# =============================================================================

def plot_staged_state_grid_individual_replicates(
    t,
    y_true,
    d_obs,
    y_stack,
    ref_data,
    rep_ids,
):
    """
    Plot S, I, Q, R in a 2x2 manuscript-style grid.

    This plots individual staged autonomous rollout trajectories, not a summary
    statistic across replicates.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_WIDTH, STATE_GRID_HEIGHT),
        sharex=True,
    )

    axes = axes.ravel()

    for j, ax in enumerate(axes):
        # Individual staged replicate traces.
        for k in range(y_stack.shape[0]):
            label = "STAGED replicates" if (j == 0 and k == 0) else None

            ax.plot(
                t,
                y_stack[k, :, j],
                color=STAGED_COLOR,
                alpha=STAGED_ALPHA,
                linewidth=STAGED_LINEWIDTH,
                label=label,
            )

        # True state.
        ax.plot(
            t,
            y_true[j, :],
            color=TRUE_COLOR,
            linewidth=TRUE_LINEWIDTH,
            label="true" if j == 0 else None,
        )

        # Noisy observations from reference replicate.
        if PLOT_NOISY_OBS:
            ax.scatter(
                t,
                d_obs[j, :],
                s=3,
                color="gray",
                alpha=0.20,
                label="noisy obs" if j == 0 else None,
                rasterized=True,
            )

        ax.set_title(STATE_NAMES[j])
        ax.set_ylabel(STATE_LABELS[j])

        if j in [2, 3]:
            ax.set_xlabel("time [days]")

        if j == 0:
            ax.legend(loc="best", frameon=True)

    fig.suptitle(
        f"STAGED DDCM state trajectories, {len(rep_ids)} replicates",
        y=1.02,
        fontsize=8,
    )

    fig.tight_layout()

    save_fig(fig, "staged_state_grid_individual_replicates")

    return fig, axes


def plot_staged_xi_individual_replicates(
    t,
    xi_true,
    xi_stack,
    rep_ids,
):
    """
    Plot true closure xi(t) against individual staged learned xi(t) traces.

    This plots each replicate's autonomous staged rollout closure, not a summary
    statistic across replicates.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, XI_FIG_HEIGHT))

    for k in range(xi_stack.shape[0]):
        label = r"STAGED learned $\xi$ replicates" if k == 0 else None

        ax.plot(
            t,
            xi_stack[k, :],
            color=STAGED_COLOR,
            alpha=STAGED_ALPHA,
            linewidth=STAGED_LINEWIDTH,
            label=label,
        )

    ax.plot(
        t,
        xi_true,
        color=TRUE_COLOR,
        linewidth=TRUE_LINEWIDTH,
        label=r"true $\xi$",
    )

    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$\xi(t)$")
    ax.set_title(
        f"STAGED DDCM learned closure, {len(rep_ids)} replicates",
        fontsize=8,
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    save_fig(fig, "staged_xi_individual_replicates")

    return fig, ax


def plot_staged_trainval_diagnostic_if_available(r=0):
    """
    Optional helper: plot direct supervised NN fit to EnKF pseudo-data for one
    replicate if staged_trainval_diagnostic.npz exists.

    This is not autonomous rollout. It shows the staged NN evaluated directly on
    the EnKF-derived pseudo-pairs.
    """
    diag_file = rep_dir(r) / "staged_trainval_diagnostic.npz"

    if not diag_file.exists():
        print(f"No diagnostic file found for replicate {r:03d}: {diag_file}")
        return None, None

    z = np.load(diag_file, allow_pickle=True)

    if "diagnostic_success" in z.files:
        success = bool(np.asarray(z["diagnostic_success"]).item())
        if not success:
            print(f"Diagnostic file for replicate {r:03d} has success=False")
            return None, None

    t_obs = np.asarray(z["t_obs"], dtype=float)
    xi_true = np.asarray(z["xi_true"], dtype=float)

    t_pseudo_train = np.asarray(z["t_pseudo_train"], dtype=float)
    xi_da_train = np.asarray(z["xi_da_train"], dtype=float)
    xi_nn_train = np.asarray(z["xi_nn_train"], dtype=float)

    t_pseudo_val = np.asarray(z["t_pseudo_val"], dtype=float)
    xi_da_val = np.asarray(z["xi_da_val"], dtype=float)
    xi_nn_val = np.asarray(z["xi_nn_val"], dtype=float)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, XI_FIG_HEIGHT))

    ax.plot(
        t_obs,
        xi_true,
        color="black",
        linewidth=TRUE_LINEWIDTH,
        label=r"true $\xi$",
    )

    ax.scatter(
        t_pseudo_train,
        xi_da_train,
        s=10,
        color="C2",
        alpha=0.75,
        label=r"EnKF $\hat{\xi}$ train",
    )

    ax.scatter(
        t_pseudo_train,
        xi_nn_train,
        s=18,
        facecolors="none",
        edgecolors="C2",
        linewidth=0.8,
        label=r"NN on train pseudo-states",
    )

    ax.scatter(
        t_pseudo_val,
        xi_da_val,
        s=10,
        color="C1",
        alpha=0.75,
        label=r"EnKF $\hat{\xi}$ val",
    )

    ax.scatter(
        t_pseudo_val,
        xi_nn_val,
        s=18,
        facecolors="none",
        edgecolors="C1",
        linewidth=0.8,
        label=r"NN on val pseudo-states",
    )

    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$\xi$")
    ax.set_title(f"Staged supervised diagnostic, replicate {r:03d}", fontsize=8)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    save_fig(fig, f"staged_trainval_diagnostic_rep_{r:03d}")

    return fig, ax


# %%
# =============================================================================
# Load data
# =============================================================================

apply_style()

ref = load_reference_truth()
print(f"Loaded reference truth from replicate {ref['replicate']:03d}")

t = ref["t_obs"]
d_obs = ref["d_obs"]
y_true = ref["y_true"]
xi_true = ref["xi_true"]

y_stack, xi_stack, rep_ids = load_staged_replicates(t)

print(f"Loaded {len(rep_ids)} successful staged replicates:")
print(rep_ids)

print(f"t shape:        {t.shape}")
print(f"y_stack shape:  {y_stack.shape}")
print(f"xi_stack shape: {xi_stack.shape}")


# %%
# =============================================================================
# Plot S, I, Q, R in manuscript-style grid
# =============================================================================

fig_states, axes_states = plot_staged_state_grid_individual_replicates(
    t=t,
    y_true=y_true,
    d_obs=d_obs,
    y_stack=y_stack,
    ref_data=ref,
    rep_ids=rep_ids,
)

plt.show()


# %%
# =============================================================================
# Plot staged learned xi traces
# =============================================================================

fig_xi, ax_xi = plot_staged_xi_individual_replicates(
    t=t,
    xi_true=xi_true,
    xi_stack=xi_stack,
    rep_ids=rep_ids,
)

plt.show()


# %%
# =============================================================================
# Optional: plot direct staged pseudo-data diagnostic for one replicate
# =============================================================================

# This only works if staged_training_parallel.py was run with --diagnostic-mode.
# Change the replicate ID as desired.

DIAGNOSTIC_REP = rep_ids[0]

fig_diag, ax_diag = plot_staged_trainval_diagnostic_if_available(
    r=DIAGNOSTIC_REP,
)

if fig_diag is not None:
    plt.show()