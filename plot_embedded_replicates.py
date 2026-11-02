# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# %%
# =============================================================================
# User settings
# =============================================================================

STUDY_DIR = Path("data/study_replicates")
FIG_DIR = Path("figures_embedded")
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


# %%
# =============================================================================
# Loading helpers
# =============================================================================

def rep_dir(r: int) -> Path:
    return STUDY_DIR / f"rep_{r:03d}"


def load_reference_truth():
    """
    Load true trajectory, noisy observations, and true xi from the first available
    observations.npz file.
    """
    for r in range(NUM_REPLICATES):
        obs_file = rep_dir(r) / "observations.npz"

        if obs_file.exists():
            obs = np.load(obs_file, allow_pickle=True)

            t_obs = np.asarray(obs["t_obs"])
            d_obs = np.asarray(obs["d"])          # shape (4, N)
            y_true = np.asarray(obs["y_true"])    # shape (4, N)
            xi_true = np.asarray(obs["xi_true"])  # shape (N,)

            return {
                "replicate": r,
                "t_obs": t_obs,
                "d_obs": d_obs,
                "y_true": y_true,
                "xi_true": xi_true,
            }

    raise FileNotFoundError(
        f"No observations.npz files found under {STUDY_DIR}"
    )


def load_embedded_replicates(t_ref):
    """
    Load all successful embedded replicate trajectories.

    embedded_closure.npz contains:
        y  shape (N, 4)
        xi shape (N,)
    """
    y_list = []
    xi_list = []
    rep_ids = []

    for r in range(NUM_REPLICATES):
        closure_file = rep_dir(r) / "embedded_closure.npz"

        if not closure_file.exists():
            print(f"Skipping replicate {r:03d}: missing embedded_closure.npz")
            continue

        try:
            z = np.load(closure_file, allow_pickle=True)

            success = bool(np.asarray(z["success"]).item())

            if not success:
                print(f"Skipping replicate {r:03d}: success=False")
                continue

            t = np.asarray(z["t_obs"])
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
                print(f"Skipping replicate {r:03d}: nonfinite values")
                continue

            y_list.append(y)
            xi_list.append(xi)
            rep_ids.append(r)

        except Exception as exc:
            print(f"Skipping replicate {r:03d}: failed to load because {exc}")

    if len(y_list) == 0:
        raise RuntimeError("No successful embedded replicate outputs were loaded.")

    y_stack = np.stack(y_list, axis=0)      # shape (R, N, 4)
    xi_stack = np.stack(xi_list, axis=0)    # shape (R, N)

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

def plot_state_grid_individual_replicates(
    t,
    y_true,
    d_obs,
    y_stack,
    rep_ids,
):
    """
    Plot S, I, Q, R in a 2x2 manuscript-style grid.

    This plots individual embedded replicate trajectories only, not mean/std
    summaries.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_WIDTH, STATE_GRID_HEIGHT),
        sharex=True,
    )

    axes = axes.ravel()

    for j, ax in enumerate(axes):
        # Individual embedded replicate traces.
        for k in range(y_stack.shape[0]):
            label = "embedded replicates" if (j == 0 and k == 0) else None
            ax.plot(
                t,
                y_stack[k, :, j],
                color="C0",
                alpha=0.22,
                linewidth=0.6,
                label=label,
            )

        # True state.
        ax.plot(
            t,
            y_true[j, :],
            color="black",
            linewidth=1.6,
            label="true" if j == 0 else None,
        )

        # Noisy observations.
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
        f"Embedded state trajectories, {len(rep_ids)} replicates",
        y=1.02,
        fontsize=8,
    )

    fig.tight_layout()

    save_fig(fig, "embedded_state_grid_individual_replicates")

    return fig, axes


def plot_xi_individual_replicates(
    t,
    xi_true,
    xi_stack,
    rep_ids,
):
    """
    Plot true closure xi(t) against individual learned embedded xi(t) traces.
    No mean/std summary is plotted.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, XI_FIG_HEIGHT))

    # Individual embedded replicate traces.
    for k in range(xi_stack.shape[0]):
        label = r"embedded learned $\xi$ replicates" if k == 0 else None
        ax.plot(
            t,
            xi_stack[k, :],
            color="C3",
            alpha=0.24,
            linewidth=0.7,
            label=label,
        )

    # True xi.
    ax.plot(
        t,
        xi_true,
        color="black",
        linewidth=1.8,
        label=r"true $\xi$",
    )

    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$\xi(t)$")
    ax.set_title(
        f"Learned embedded closure, {len(rep_ids)} replicates",
        fontsize=8,
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    save_fig(fig, "embedded_xi_individual_replicates")

    return fig, ax


# %%
# =============================================================================
# Load data
# =============================================================================

apply_style()

ref = load_reference_truth()

print(f"Loaded reference truth from replicate {ref['replicate']:03d}")

t = ref["t_obs"]
y_true = ref["y_true"]
d_obs = ref["d_obs"]
xi_true = ref["xi_true"]

y_stack, xi_stack, rep_ids = load_embedded_replicates(t)

print(f"Loaded {len(rep_ids)} successful embedded replicates:")
print(rep_ids)

print(f"y_stack shape:  {y_stack.shape}")
print(f"xi_stack shape: {xi_stack.shape}")


# %%
# =============================================================================
# Plot S, I, Q, R in manuscript-style grid
# =============================================================================

fig_states, axes_states = plot_state_grid_individual_replicates(
    t=t,
    y_true=y_true,
    d_obs=d_obs,
    y_stack=y_stack,
    rep_ids=rep_ids,
)

plt.show()


# %%
# =============================================================================
# Plot learned xi traces
# =============================================================================

fig_xi, ax_xi = plot_xi_individual_replicates(
    t=t,
    xi_true=xi_true,
    xi_stack=xi_stack,
    rep_ids=rep_ids,
)

plt.show()
# %%
