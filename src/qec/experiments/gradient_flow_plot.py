"""
v12.6.0 — Instability gradient field visualization.

Provides deterministic ASCII summaries for NB instability gradients and
optional matplotlib rendering when matplotlib is available.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import numpy as np
import scipy.sparse

from src.qec.analysis.nb_instability_gradient import NBInstabilityGradientAnalyzer


_ROUND = 12


def render_gradient_ascii(
    H: np.ndarray | scipy.sparse.spmatrix,
    *,
    top_k: int = 20,
) -> str:
    """Render deterministic ASCII view of strongest instability edges."""
    analyzer = NBInstabilityGradientAnalyzer()
    gradient = analyzer.compute_gradient(H)
    edge_scores = gradient["edge_scores"]
    node_instability = gradient["node_instability"]

    if scipy.sparse.issparse(H):
        H_arr = np.asarray(H.todense(), dtype=np.float64)
    else:
        H_arr = np.asarray(H, dtype=np.float64)

    m, n = H_arr.shape

    ranked_edges = sorted(
        edge_scores,
        key=lambda e: (-edge_scores[e], e[0], e[1]),
    )

    lines = ["Edge Instability Field", ""]
    for ci, vi in ranked_edges[: max(0, top_k)]:
        best_vj = None
        best_grad = None
        for vj in range(n):
            if vj == vi or H_arr[ci, vj] != 0:
                continue
            target_grad = round(
                float(node_instability[ci] - node_instability[m + vj]),
                _ROUND,
            )
            if best_grad is None or target_grad < best_grad:
                best_grad = target_grad
                best_vj = vj

        target_text = f"v{best_vj}" if best_vj is not None else "-"
        lines.append(
            f"(c{ci},v{vi}) {edge_scores[(ci, vi)]:.{_ROUND}f} -> {target_text}",
        )

    return "\n".join(lines)


def plot_gradient_matplotlib(
    H: np.ndarray | scipy.sparse.spmatrix,
    *,
    title: str = "NB Instability Gradient Field",
) -> Any | None:
    """Optionally plot the gradient field if matplotlib is installed."""
    mpl_spec = importlib.util.find_spec("matplotlib.pyplot")
    if mpl_spec is None:
        return None

    plt = importlib.import_module("matplotlib.pyplot")

    analyzer = NBInstabilityGradientAnalyzer()
    gradient = analyzer.compute_gradient(H)
    edge_scores = gradient["edge_scores"]

    if scipy.sparse.issparse(H):
        H_arr = np.asarray(H.todense(), dtype=np.float64)
    else:
        H_arr = np.asarray(H, dtype=np.float64)

    m, n = H_arr.shape
    if not edge_scores:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title(title)
        ax.set_axis_off()
        return fig

    max_score = max(edge_scores.values())
    denom = max(max_score, 1e-12)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(4, m * 0.8)))
    cmap = plt.get_cmap("viridis")

    for (ci, vi), score in sorted(edge_scores.items(), key=lambda x: (x[0][0], x[0][1])):
        norm = score / denom
        color = cmap(norm)
        ax.plot([vi, vi], [m - 1 - ci, m + 1], color=color, linewidth=1.5)
        ax.annotate(
            "",
            xy=(vi, m + 0.9),
            xytext=(vi, m + 1.1),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
        )

    for ci in range(m):
        ax.scatter([-1], [m - 1 - ci], color="black", s=30)
        ax.text(-1.2, m - 1 - ci, f"c{ci}", ha="right", va="center", fontsize=8)

    for vi in range(n):
        ax.scatter([vi], [m + 1.2], color="black", s=30)
        ax.text(vi, m + 1.4, f"v{vi}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title)
    ax.set_xlim(-2.0, max(1.0, n - 0.2))
    ax.set_ylim(-1.0, m + 2.0)
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(np.array(list(edge_scores.values()), dtype=np.float64))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Edge instability")

    fig.tight_layout()
    return fig
