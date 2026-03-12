"""
v12.4.0 — NB Flow Heatmap Visualization.

Visualizes non-backtracking flow magnitude on Tanner graph edges.
Produces ASCII heatmaps and optional matplotlib plots.

Layer 5 — Experiments.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer


_ROUND = 12
_BAR_WIDTH = 8


def compute_edge_flow_scores(
    H: np.ndarray,
    flow_vector: np.ndarray,
    directed_edges: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Compute NB flow magnitude score per undirected Tanner graph edge.

    For each undirected edge (ci, vi), the score is:
        score(ci, vi) = |v_(vi->ci)| + |v_(ci->vi)|

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    flow_vector : np.ndarray
        Directed edge flow magnitudes from NonBacktrackingFlowAnalyzer.
    directed_edges : list[tuple[int, int]]
        Sorted directed edge list matching flow_vector ordering.

    Returns
    -------
    list[dict]
        Sorted (descending score) list of dicts with keys:
        ``check``, ``variable``, ``score``, ``normalized_score``.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if m == 0 or n == 0 or len(flow_vector) == 0:
        return []

    # Build directed edge index.
    de_index: dict[tuple[int, int], int] = {
        e: i for i, e in enumerate(directed_edges)
    }

    # Collect undirected edges deterministically.
    edges: list[tuple[int, int]] = []
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                edges.append((ci, vi))
    edges.sort()

    # Compute scores.
    scores: list[dict[str, Any]] = []
    for ci, vi in edges:
        j1 = de_index.get((vi, n + ci))
        j2 = de_index.get((n + ci, vi))
        s = 0.0
        if j1 is not None:
            s += abs(float(flow_vector[j1]))
        if j2 is not None:
            s += abs(float(flow_vector[j2]))
        scores.append({
            "check": ci,
            "variable": vi,
            "score": round(s, _ROUND),
        })

    # Normalize to [0, 1].
    max_score = max((d["score"] for d in scores), default=0.0)
    for d in scores:
        if max_score > 1e-15:
            d["normalized_score"] = round(d["score"] / max_score, _ROUND)
        else:
            d["normalized_score"] = 0.0

    # Sort descending by score, then by (check, variable) for determinism.
    scores.sort(key=lambda d: (-d["score"], d["check"], d["variable"]))
    return scores


def compute_edge_flow_scores_from_H(
    H: np.ndarray,
) -> list[dict[str, Any]]:
    """Convenience: compute NB flow and edge scores in one call.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    list[dict]
        Sorted edge flow scores (see ``compute_edge_flow_scores``).
    """
    H_arr = np.asarray(H, dtype=np.float64)
    analyzer = NonBacktrackingFlowAnalyzer()
    flow = analyzer.compute_flow(H_arr)
    return compute_edge_flow_scores(
        H_arr,
        flow["directed_edge_flow"],
        flow["directed_edges"],
    )


def format_ascii_heatmap(
    scores: list[dict[str, Any]],
    *,
    top_k: int = 10,
) -> str:
    """Format edge flow scores as an ASCII bar chart.

    Parameters
    ----------
    scores : list[dict]
        Output of ``compute_edge_flow_scores``.
    top_k : int
        Number of top edges to display.

    Returns
    -------
    str
        ASCII heatmap string.
    """
    if not scores:
        return "Edge Flow Heatmap\n(no edges)"

    lines = ["Edge Flow Heatmap", ""]
    for entry in scores[:top_k]:
        ci = entry["check"]
        vi = entry["variable"]
        ns = entry["normalized_score"]
        bar_len = max(1, int(round(ns * _BAR_WIDTH)))
        bar = "\u2588" * bar_len
        label = f"(c{ci},v{vi})"
        lines.append(f"{label:<10} {bar:<{_BAR_WIDTH}} {ns:.2f}")

    return "\n".join(lines)


def plot_flow_heatmap(
    H: np.ndarray,
    scores: list[dict[str, Any]],
    *,
    output_path: str | None = None,
) -> Any:
    """Plot bipartite Tanner graph with edges colored by NB flow magnitude.

    Requires matplotlib. Returns the figure object if matplotlib is
    available. Saves to ``output_path`` if specified.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    scores : list[dict]
        Output of ``compute_edge_flow_scores``.
    output_path : str or None
        If set, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The plotted figure.

    Raises
    ------
    RuntimeError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        raise RuntimeError(
            "matplotlib is required for plot_flow_heatmap but is not installed"
        )

    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    # Build score lookup.
    score_map: dict[tuple[int, int], float] = {}
    for entry in scores:
        score_map[(entry["check"], entry["variable"])] = entry["normalized_score"]

    fig, ax = plt.subplots(1, 1, figsize=(max(6, n * 0.5), max(4, m * 0.5)))

    # Position variable nodes on top row, check nodes on bottom row.
    var_y = 1.0
    chk_y = 0.0

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Draw edges.
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                ns = score_map.get((ci, vi), 0.0)
                color = cmap(norm(ns))
                ax.plot(
                    [vi, ci * (n / max(m, 1))],
                    [var_y, chk_y],
                    color=color,
                    linewidth=1.5 + ns * 2.0,
                    alpha=0.6 + ns * 0.4,
                )

    # Draw nodes.
    for vi in range(n):
        ax.plot(vi, var_y, "o", color="steelblue", markersize=8)
        ax.text(vi, var_y + 0.05, f"v{vi}", ha="center", fontsize=7)
    for ci in range(m):
        x = ci * (n / max(m, 1))
        ax.plot(x, chk_y, "s", color="firebrick", markersize=8)
        ax.text(x, chk_y - 0.08, f"c{ci}", ha="center", fontsize=7)

    ax.set_xlim(-0.5, max(n, m) - 0.5)
    ax.set_ylim(-0.3, 1.3)
    ax.set_title("NB Flow Heatmap on Tanner Graph")
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Normalized NB Flow")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    return fig
