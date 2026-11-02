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
FIG_DIR = Path("figures_enkf")
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUM_REPLICATES = 50

STATE_NAMES = ["Susceptible", "Infectious", "Quarantined", "Removed"]
STATE_LABELS = [r"$s(t)$", r"$i(t)$", r"$q(t)$", r"$r(t)$"]

# If True, show noisy observations from the reference replicate.
PLOT_NOISY_OBS = True

# If True, show only the observations that were assimilated by the EnKF
# for the reference replicate. If False, show all noisy observations.
PLOT_ASSIMILATED_OBS_ONLY = False

SAVE_DPI = 300

# Manuscript-friendly dimensions: no wider than 6 inches.
FIG_WIDTH = 6.0
STATE_GRID_HEIGHT = 4.6
XI_FIG_HEIGHT = 3.4

# Plot aesthetics for individual EnKF replicate means.
ENKF_COLOR = "C2"
ENKF_ALPHA = 0.24
ENKF_LINEWIDTH = 0.7

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


def enkf_replicate_successful(r: int) -> bool:
    """
    Check EnKF success using enkf_summary.json if available.

    If the summary is missing but enkf_result.npz exists, return True.
    """
    rd = rep_dir(r)
    result_file = rd / "enkf_result.npz"
    summary_file = rd / "enkf_summary.json"

    if not result_file.exists():
        return False

    summary = load_json_if_exists(summary_file)

    if summary is None:
        return True

    return bool(summary.get("success", False))


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
                assim_idx = np.sort(np.concatenate([train_idx, val_idx]))
            else:
                train_idx = None
                val_idx = None
                test_idx = None
                assim_idx = None

            return {
                "replicate": r,
                "t_obs": t_obs,
                "d_obs": d_obs,
                "y_true": y_true,
                "xi_true": xi_true,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "assim_idx": assim_idx,
            }

    raise FileNotFoundError(
        f"No observations.npz files found under {STUDY_DIR}"
    )


def interpolate_truth_to_times(t_source, y_source, xi_source, t_target):
    """
    Interpolate true states and xi from observation times to EnKF output times.

    y_source is expected shape (4, N_source).
    Returns:
        y_interp  shape (4, N_target)
        xi_interp shape (N_target,)
    """
    t_source = np.asarray(t_source, dtype=float)
    t_target = np.asarray(t_target, dtype=float)
    y_source = np.asarray(y_source, dtype=float)
    xi_source = np.asarray(xi_source, dtype=float)

    y_interp = np.vstack(
        [
            np.interp(t_target, t_source, y_source[j, :])
            for j in range(y_source.shape[0])
        ]
    )

    xi_interp = np.interp(t_target, t_source, xi_source)

    return y_interp, xi_interp


def load_enkf_mean_replicates():
    """
    Load EnKF ensemble-mean trajectories from all successful EnKF replicates.

    enkf_result.npz contains:
        t_mu  shape (N,)
        x_mu  shape (N, 5)

    x_mu columns:
        0: S
        1: I
        2: Q
        3: R
        4: xi

    Returns
    -------
    t_ref:
        Common EnKF time grid.
    x_stack:
        Shape (R, N, 5), one EnKF mean trajectory per successful replicate.
    rep_ids:
        Replicate IDs loaded.
    """
    t_ref = None
    x_list = []
    rep_ids = []

    for r in range(NUM_REPLICATES):
        if not enkf_replicate_successful(r):
            print(f"Skipping replicate {r:03d}: no successful EnKF result")
            continue

        result_file = rep_dir(r) / "enkf_result.npz"

        try:
            z = np.load(result_file, allow_pickle=True)

            if "t_mu" not in z.files or "x_mu" not in z.files:
                print(f"Skipping replicate {r:03d}: missing t_mu or x_mu")
                continue

            t_mu = np.asarray(z["t_mu"], dtype=float)
            x_mu = np.asarray(z["x_mu"], dtype=float)

            if x_mu.ndim != 2 or x_mu.shape[1] < 5:
                print(
                    f"Skipping replicate {r:03d}: unexpected x_mu shape {x_mu.shape}"
                )
                continue

            if not np.all(np.isfinite(t_mu)) or not np.all(np.isfinite(x_mu)):
                print(f"Skipping replicate {r:03d}: nonfinite t_mu or x_mu")
                continue

            if t_ref is None:
                t_ref = t_mu
            else:
                if len(t_mu) != len(t_ref) or np.max(np.abs(t_mu - t_ref)) > 1.0e-10:
                    print(f"Skipping replicate {r:03d}: EnKF time grid mismatch")
                    continue

            x_list.append(x_mu[:, :5])
            rep_ids.append(r)

        except Exception as exc:
            print(f"Skipping replicate {r:03d}: failed to load because {exc}")

    if len(x_list) == 0:
        raise RuntimeError("No successful EnKF replicate outputs were loaded.")

    x_stack = np.stack(x_list, axis=0)

    return t_ref, x_stack, rep_ids


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

def plot_enkf_state_grid_individual_means(
    t_enkf,
    x_stack,
    y_true_interp,
    ref_data,
    rep_ids,
):
    """
    Plot S, I, Q, R in a 2x2 manuscript-style grid.

    This plots the individual EnKF ensemble-mean trajectories from each replicate,
    not a summary statistic across replicates.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_WIDTH, STATE_GRID_HEIGHT),
        sharex=True,
    )

    axes = axes.ravel()

    t_obs = ref_data["t_obs"]
    d_obs = ref_data["d_obs"]
    assim_idx = ref_data["assim_idx"]

    for j, ax in enumerate(axes):
        # Individual EnKF replicate ensemble means.
        for k in range(x_stack.shape[0]):
            label = "EnKF replicate means" if (j == 0 and k == 0) else None

            ax.plot(
                t_enkf,
                x_stack[k, :, j],
                color=ENKF_COLOR,
                alpha=ENKF_ALPHA,
                linewidth=ENKF_LINEWIDTH,
                label=label,
            )

        # True state.
        ax.plot(
            t_enkf,
            y_true_interp[j, :],
            color=TRUE_COLOR,
            linewidth=TRUE_LINEWIDTH,
            label="true" if j == 0 else None,
        )

        # Noisy observations from reference replicate.
        if PLOT_NOISY_OBS:
            if PLOT_ASSIMILATED_OBS_ONLY and assim_idx is not None:
                obs_idx = assim_idx
                obs_label = "assimilated obs"
            else:
                obs_idx = np.arange(len(t_obs))
                obs_label = "noisy obs"

            ax.scatter(
                t_obs[obs_idx],
                d_obs[j, obs_idx],
                s=3,
                color="gray",
                alpha=0.20,
                label=obs_label if j == 0 else None,
                rasterized=True,
            )

        ax.set_title(STATE_NAMES[j])
        ax.set_ylabel(STATE_LABELS[j])

        if j in [2, 3]:
            ax.set_xlabel("time [days]")

        if j == 0:
            ax.legend(loc="best", frameon=True)

    fig.suptitle(
        f"EnKF ensemble-mean state trajectories, {len(rep_ids)} replicates",
        y=1.02,
        fontsize=8,
    )

    fig.tight_layout()

    save_fig(fig, "enkf_state_grid_individual_replicate_means")

    return fig, axes


def plot_enkf_xi_individual_means(
    t_enkf,
    x_stack,
    xi_true_interp,
    rep_ids,
):
    """
    Plot true closure xi(t) against individual EnKF mean xi(t) traces.

    This plots each replicate's EnKF posterior mean xi trajectory, not a summary
    statistic across replicates.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, XI_FIG_HEIGHT))

    for k in range(x_stack.shape[0]):
        label = r"EnKF mean $\hat{\xi}$ replicates" if k == 0 else None

        ax.plot(
            t_enkf,
            x_stack[k, :, 4],
            color=ENKF_COLOR,
            alpha=ENKF_ALPHA,
            linewidth=ENKF_LINEWIDTH,
            label=label,
        )

    ax.plot(
        t_enkf,
        xi_true_interp,
        color=TRUE_COLOR,
        linewidth=TRUE_LINEWIDTH,
        label=r"true $\xi$",
    )

    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$\xi(t)$")
    ax.set_title(
        f"EnKF inferred closure means, {len(rep_ids)} replicates",
        fontsize=8,
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    save_fig(fig, "enkf_xi_individual_replicate_means")

    return fig, ax


# %%
# =============================================================================
# Load data
# =============================================================================

apply_style()

ref = load_reference_truth()
print(f"Loaded reference truth from replicate {ref['replicate']:03d}")

t_enkf, x_stack, rep_ids = load_enkf_mean_replicates()

print(f"Loaded {len(rep_ids)} successful EnKF replicates:")
print(rep_ids)

print(f"t_enkf shape:   {t_enkf.shape}")
print(f"x_stack shape: {x_stack.shape}")

y_true_interp, xi_true_interp = interpolate_truth_to_times(
    t_source=ref["t_obs"],
    y_source=ref["y_true"],
    xi_source=ref["xi_true"],
    t_target=t_enkf,
)

print(f"Interpolated y_true shape:  {y_true_interp.shape}")
print(f"Interpolated xi_true shape: {xi_true_interp.shape}")


# %%
# =============================================================================
# Plot S, I, Q, R in manuscript-style grid
# =============================================================================

fig_states, axes_states = plot_enkf_state_grid_individual_means(
    t_enkf=t_enkf,
    x_stack=x_stack,
    y_true_interp=y_true_interp,
    ref_data=ref,
    rep_ids=rep_ids,
)

plt.show()


# %%
# =============================================================================
# Plot EnKF inferred xi traces
# =============================================================================

fig_xi, ax_xi = plot_enkf_xi_individual_means(
    t_enkf=t_enkf,
    x_stack=x_stack,
    xi_true_interp=xi_true_interp,
    rep_ids=rep_ids,
)

plt.show()