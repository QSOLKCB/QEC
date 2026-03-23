"""v86.2.0 — Deterministic spectral trajectory visualization.

Renders the output of ``run_phase_trajectory_analysis`` as a 3-panel
static figure: dominant eigenvalue, spectral drift with transition
markers, and rank/degeneracy evolution.

Same input always produces identical output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive, deterministic backend
import matplotlib.pyplot as plt  # noqa: E402

# ── styling constants ────────────────────────────────────────────────

TRAJECTORY_COLORS: Dict[str, str] = {
    "convergent": "green",
    "oscillatory": "purple",
    "divergent": "red",
    "undetermined": "gray",
}

_DEFAULT_TRAJ_COLOR = "gray"


# ── main plot ────────────────────────────────────────────────────────


def plot_spectral_trajectory(
    traj: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    *,
    mode: str = "debug",
) -> Dict[str, Any]:
    """Render a spectral trajectory as a deterministic 3-panel figure.

    Parameters
    ----------
    traj : dict
        Output of ``run_phase_trajectory_analysis``.  Must contain
        ``lambda_max``, ``drift``, ``rank_evolution``, and
        ``trajectory_type``.
    output_path : str or Path, optional
        When given the figure is saved as PNG.
    mode : str
        ``"paper"`` for clean output (300 dpi, no labels),
        ``"debug"`` (default) for full detail (120 dpi, labels).

    Returns
    -------
    dict
        ``{"output_path": str | None, "n_steps": int}``
    """
    if mode == "paper":
        dpi = 300
        show_labels = False
    elif mode == "debug":
        dpi = 120
        show_labels = True
    else:
        raise ValueError("mode must be 'paper' or 'debug'")

    lambda_max: List[float] = traj.get("lambda_max", [])
    drift: List[float] = traj.get("drift", [])
    rank_evolution: List[int] = traj.get("rank_evolution", [])
    degeneracy_evolution: List[int] = traj.get("degeneracy_evolution", [])
    transitions: List[Dict[str, Any]] = traj.get("temporal_transitions", [])
    trajectory_type: str = traj.get("trajectory_type", "undetermined")
    n_steps: int = traj.get("n_steps", len(lambda_max))

    color = TRAJECTORY_COLORS.get(trajectory_type, _DEFAULT_TRAJ_COLOR)
    steps = list(range(n_steps))

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    # ── panel 1: λ_max ───────────────────────────────────────────────
    ax0 = axes[0]
    if lambda_max:
        ax0.plot(steps, lambda_max, color=color, linewidth=1.5)
    ax0.set_ylabel("λ_max")
    if show_labels:
        ax0.set_title(
            f"Dominant eigenvalue  ({trajectory_type})", fontsize=9,
        )

    # ── panel 2: spectral drift + transitions ────────────────────────
    ax1 = axes[1]
    drift_steps = list(range(len(drift)))
    if drift:
        ax1.plot(drift_steps, drift, color="steelblue", linewidth=1.2)
    for tr in transitions:
        t_idx = tr.get("time_index", 0)
        ax1.axvline(t_idx, linestyle="--", color="black", linewidth=0.8)
    ax1.set_ylabel("drift")
    if show_labels:
        ax1.set_title("Spectral drift", fontsize=9)

    # ── panel 3: rank evolution ──────────────────────────────────────
    ax2 = axes[2]
    if rank_evolution:
        ax2.step(steps, rank_evolution, where="mid", color="teal",
                 linewidth=1.2, label="rank")
    if degeneracy_evolution:
        ax2.step(steps, degeneracy_evolution, where="mid", color="teal",
                 linewidth=1.0, linestyle="--", label="degeneracy")
    ax2.set_ylabel("rank / degeneracy")
    ax2.set_xlabel("time step")
    if show_labels:
        ax2.set_title("Rank evolution", fontsize=9)
        if rank_evolution or degeneracy_evolution:
            ax2.legend(fontsize=7, loc="best")

    plt.tight_layout()

    saved_path: Optional[str] = None
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        saved_path = str(out)

    plt.close(fig)

    return {
        "output_path": saved_path,
        "n_steps": n_steps,
    }
