"""v87.3.0 — Resonance phase diagram visualization.

Renders resonance locks, attractor field scores, and trajectory
as a deterministic static figure.

Layer 6 — Visualization.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive, deterministic backend
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402


# ── styling constants ────────────────────────────────────────────────

_FIELD_COLORS: Dict[str, str] = {
    "single_attractor": "darkgreen",
    "multi_attractor": "darkorange",
    "resonant": "steelblue",
    "transient": "gray",
    "dispersed": "lightcoral",
}

_DEFAULT_FIELD_COLOR = "gray"


# ── main plot ────────────────────────────────────────────────────────


def plot_resonance_phase_diagram(
    series: List[Tuple[int, ...]],
    drift: List[float],
    attractor_field: Dict[str, Any],
    locks: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    *,
    mode: str = "debug",
    field_type: str = "undetermined",
) -> Dict[str, Any]:
    """Render a resonance phase diagram.

    Parameters
    ----------
    series:
        Ternary-encoded trajectory.
    drift:
        Spectral drift values.
    attractor_field:
        Dict with ``nodes`` list (each having ``state``, ``score``,
        ``is_attractor``).
    locks:
        Dict with ``locks`` list (each having ``start``, ``end``, ``length``).
    output_path:
        When given the figure is saved as PNG.
    mode:
        ``"paper"`` for clean output (300 dpi), ``"debug"`` (default)
        for full detail (120 dpi, labels + legend).
    field_type:
        Resonance field classification label for the title.

    Returns
    -------
    dict with ``output_path`` and ``n_steps``.
    """
    if mode == "paper":
        dpi = 300
        show_labels = False
    elif mode == "debug":
        dpi = 120
        show_labels = True
    else:
        raise ValueError("mode must be 'paper' or 'debug'")

    n_steps = len(series)
    steps = list(range(n_steps))

    # Project each ternary tuple to a scalar for plotting (L2 norm).
    trajectory_values = [
        float(np.linalg.norm(np.array(s, dtype=np.float64))) for s in series
    ]

    color = _FIELD_COLORS.get(field_type, _DEFAULT_FIELD_COLOR)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    # ── panel 1: trajectory + lock spans ─────────────────────────────
    ax0 = axes[0]
    ax0.plot(steps, trajectory_values, color=color, linewidth=1.5,
             label="trajectory")

    # Shade lock regions.
    lock_list = locks.get("locks", [])
    for lk in lock_list:
        ax0.axvspan(lk["start"], lk["end"], alpha=0.15, color="gold",
                     label=None)
    # Single legend entry for locks (avoid duplicates).
    if lock_list:
        ax0.axvspan(0, 0, alpha=0.15, color="gold", label="lock region")

    # Overlay attractor nodes.
    attractor_nodes = attractor_field.get("nodes", [])
    # Map state → first occurrence index in series for x-position.
    state_to_first: Dict[Tuple[int, ...], int] = {}
    for idx, s in enumerate(series):
        if s not in state_to_first:
            state_to_first[s] = idx

    for anode in attractor_nodes:
        state = anode["state"]
        score = anode["score"]
        if state in state_to_first:
            x = state_to_first[state]
            y = trajectory_values[x]
            size = 20 + score * 80  # size proportional to score
            marker = "D" if anode["is_attractor"] else "o"
            ax0.scatter([x], [y], s=size, color="red", marker=marker,
                        zorder=5)

    ax0.set_ylabel("||state||")
    if show_labels:
        ax0.set_title(f"Resonance Phase Diagram — {field_type}", fontsize=9)
        ax0.legend(fontsize=7, loc="best")

    # ── panel 2: drift overlay ───────────────────────────────────────
    ax1 = axes[1]
    drift_steps = list(range(len(drift)))
    if drift:
        ax1.plot(drift_steps, drift, color="steelblue", linewidth=1.2,
                 label="drift")
    ax1.set_ylabel("drift")
    ax1.set_xlabel("time step")
    if show_labels:
        ax1.set_title("Spectral drift", fontsize=9)
        if drift:
            ax1.legend(fontsize=7, loc="best")

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
