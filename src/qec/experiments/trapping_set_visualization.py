"""Visualization utilities for NB trapping-set predictions."""

from __future__ import annotations

from typing import Any

import numpy as np


_ROUND = 12


def render_trapping_sets_ascii(
    H: np.ndarray,
    prediction: dict[str, Any],
) -> str:
    """Render trapping-set prediction summary as deterministic ASCII."""
    _ = np.asarray(H, dtype=np.float64)

    lines: list[str] = ["Predicted Trapping Regions", ""]

    candidate_sets = prediction.get("candidate_sets", [])
    if not candidate_sets:
        lines.append("Nodes: (none)")
        lines.append("Cluster size: 0")
    else:
        first = candidate_sets[0]
        lines.append(f"Nodes: {', '.join(str(v) for v in first)}")
        lines.append(f"Cluster size: {len(first)}")

    lines.append("")
    lines.append(f"Spectral radius: {round(float(prediction.get('spectral_radius', 0.0)), _ROUND)}")
    lines.append(f"IPR: {round(float(prediction.get('ipr', 0.0)), _ROUND)}")
    lines.append(f"Risk score: {round(float(prediction.get('risk_score', 0.0)), _ROUND)}")

    if len(candidate_sets) > 1:
        lines.append("")
        lines.append("Additional clusters:")
        for idx, comp in enumerate(candidate_sets[1:], start=2):
            lines.append(f"  {idx}. [{', '.join(str(v) for v in comp)}]")

    return "\n".join(lines)


def plot_trapping_sets_matplotlib(
    H: np.ndarray,
    prediction: dict[str, Any],
) -> Any:
    """Plot trapping-set node instability if matplotlib is available.

    If matplotlib is unavailable, returns ASCII rendering.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return render_trapping_sets_ascii(H, prediction)

    H_arr = np.asarray(H, dtype=np.float64)
    n = H_arr.shape[1] if H_arr.ndim == 2 else 0

    node_scores = prediction.get("node_scores", {})
    xs = list(range(n))
    ys = [float(node_scores.get(i, 0.0)) for i in xs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(xs, ys, color="#4c72b0")

    candidate_nodes = {v for comp in prediction.get("candidate_sets", []) for v in comp}
    for vi in candidate_nodes:
        if 0 <= vi < len(bars):
            bars[vi].set_color("#c44e52")

    ax.set_xlabel("Variable node index")
    ax.set_ylabel("Instability score")
    ax.set_title("NB Predicted Trapping-Set Instability")

    subtitle = (
        f"spectral_radius={round(float(prediction.get('spectral_radius', 0.0)), _ROUND)}, "
        f"ipr={round(float(prediction.get('ipr', 0.0)), _ROUND)}, "
        f"risk={round(float(prediction.get('risk_score', 0.0)), _ROUND)}"
    )
    ax.text(0.01, 1.02, subtitle, transform=ax.transAxes, fontsize=9)

    fig.tight_layout()
    return fig
