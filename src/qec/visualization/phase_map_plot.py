"""v85.0.0 — Deterministic phase map visualization.

Converts a phase_map (nodes + edges) into a publication-quality static
figure using only matplotlib.  Same input always produces identical layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive, deterministic backend
import matplotlib.pyplot as plt  # noqa: E402


# ── node styling ──────────────────────────────────────────────────────

CLASS_COLORS: Dict[str, str] = {
    "stable": "green",
    "fragile": "orange",
    "chaotic": "red",
    "boundary_rider": "blue",
}

PHASE_MARKERS: Dict[str, str] = {
    "aligned": "o",
    "misaligned": "s",
    "unknown": "x",
}

_DEFAULT_COLOR = "gray"
_DEFAULT_MARKER = "o"

# ── edge styling ──────────────────────────────────────────────────────

EDGE_COLORS: Dict[str, str] = {
    "strong_boundary": "black",
    "phase_boundary": "purple",
    "class_boundary": "orange",
    "structural_boundary": "blue",
    "degenerate_interface": "gray",
}

_DEFAULT_EDGE_COLOR = "gray"


# ── layout ────────────────────────────────────────────────────────────

def compute_linear_layout(
    nodes: List[Dict[str, Any]],
) -> Dict[int, Tuple[float, float]]:
    """Place nodes along the x-axis in order of *id*, y = 0."""
    sorted_nodes = sorted(nodes, key=lambda n: n["id"])
    return {n["id"]: (float(i), 0.0) for i, n in enumerate(sorted_nodes)}


# ── weight scaling ────────────────────────────────────────────────────

def _scale_linewidth(weight: float, max_weight: float) -> float:
    if max_weight <= 0.0:
        return 1.0
    return 1.0 + 3.0 * (weight / max_weight)


# ── main plot ─────────────────────────────────────────────────────────

def plot_phase_map(
    phase_map: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Render *phase_map* as a deterministic static figure.

    Parameters
    ----------
    phase_map : dict
        Must contain ``"nodes"`` and ``"edges"`` lists.
    output_path : str or Path, optional
        When given the figure is saved as PNG.

    Returns
    -------
    dict
        ``{"n_nodes": int, "n_edges": int, "output_path": str | None}``
    """
    nodes: List[Dict[str, Any]] = phase_map.get("nodes", [])
    edges: List[Dict[str, Any]] = phase_map.get("edges", [])

    layout = compute_linear_layout(nodes)

    fig, ax = plt.subplots(figsize=(max(6, len(nodes) * 1.5), 4))

    # ── edges (drawn first so nodes overlay) ──────────────────────────
    max_weight = max((e.get("weight", 1.0) for e in edges), default=0.0)

    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in layout or tgt not in layout:
            continue
        x0, y0 = layout[src]
        x1, y1 = layout[tgt]
        etype = edge.get("type", "")
        color = EDGE_COLORS.get(etype, _DEFAULT_EDGE_COLOR)
        lw = _scale_linewidth(edge.get("weight", 1.0), max_weight)
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, zorder=1)

    # ── nodes ─────────────────────────────────────────────────────────
    for node in nodes:
        nid = node["id"]
        if nid not in layout:
            continue
        x, y = layout[nid]
        cls = node.get("dominant_class", "")
        phase = node.get("dominant_phase", "")
        color = CLASS_COLORS.get(cls, _DEFAULT_COLOR)
        marker = PHASE_MARKERS.get(phase, _DEFAULT_MARKER)
        # unfilled markers (e.g. "x") ignore edgecolors — skip for them
        filled = marker not in ("x", "+", "1", "2", "3", "4")
        scatter_kw: Dict[str, Any] = dict(
            c=color, marker=marker, s=120, zorder=2,
        )
        if filled:
            scatter_kw.update(edgecolors="black", linewidths=0.5)
        ax.scatter(x, y, **scatter_kw)

        label = str(nid)
        node_range = node.get("range")
        if node_range is not None:
            label = f"{nid} [{node_range[0]}-{node_range[1]}]"
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7)

    ax.set_title("Phase Map", fontsize=10)
    ax.set_xlabel("Regime index")
    ax.set_yticks([])
    ax.margins(x=0.15, y=0.4)

    plt.tight_layout()

    saved_path: Optional[str] = None
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        saved_path = str(out)

    plt.close(fig)

    return {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "output_path": saved_path,
    }
