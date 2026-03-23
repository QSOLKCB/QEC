"""v85.1.0 — Deterministic phase map visualization.

Converts a phase_map (nodes + edges) into a publication-quality static
figure using only matplotlib.  Same input always produces identical layout.

v85.1.0 adds annotated overlays: dominant-boundary emphasis, node labels,
transition summary box, and a deterministic legend.
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

def _strongest_edge_key(
    interface_ranking: Optional[Dict[str, Any]],
) -> Optional[Tuple[int, int]]:
    """Return ``(source, target)`` of the strongest interface, or *None*."""
    if interface_ranking is None:
        return None
    strongest = interface_ranking.get("strongest_interface")
    if strongest is None:
        return None
    return (strongest.get("from_index"), strongest.get("to_index"))


def _render_summary_overlay(
    ax: Any,
    summary: Dict[str, Any],
) -> None:
    """Draw a small transition-summary text box in the top-left corner."""
    lines = [
        f"Transitions: {summary.get('n_transitions', 0)}",
        f"Max \u0394score: {summary.get('max_delta_score', 0.0):.2f}",
        f"Class changes: {summary.get('class_change_count', 0)}",
        f"Phase changes: {summary.get('phase_change_count', 0)}",
    ]
    text = "\n".join(lines)
    ax.text(
        0.01, 0.95, text,
        transform=ax.transAxes,
        fontsize=6, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.8),
    )


def _add_legend(ax: Any) -> None:
    """Add a deterministic legend for node classes and edge types."""
    from matplotlib.lines import Line2D

    handles: List[Any] = []
    # node class legend
    for cls in sorted(CLASS_COLORS):
        handles.append(Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=CLASS_COLORS[cls], markersize=6,
            label=cls,
        ))
    # edge type legend
    for etype in sorted(EDGE_COLORS):
        handles.append(Line2D(
            [0], [0], color=EDGE_COLORS[etype], linewidth=1.5,
            label=etype,
        ))
    ax.legend(handles=handles, fontsize=5, loc="lower right",
              framealpha=0.8, borderpad=0.4)


def plot_phase_map(
    phase_map: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    *,
    interface_ranking: Optional[Dict[str, Any]] = None,
    transition_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render *phase_map* as a deterministic static figure.

    Parameters
    ----------
    phase_map : dict
        Must contain ``"nodes"`` and ``"edges"`` lists.
    output_path : str or Path, optional
        When given the figure is saved as PNG.
    interface_ranking : dict, optional
        Output of ``rank_regime_interfaces``.  When provided the strongest
        interface edge is visually emphasised.
    transition_summary : dict, optional
        Output of ``summarize_transitions``.  When provided a small summary
        text box is overlaid on the figure.

    Returns
    -------
    dict
        ``{"n_nodes": int, "n_edges": int, "output_path": str | None}``
    """
    nodes: List[Dict[str, Any]] = phase_map.get("nodes", [])
    edges: List[Dict[str, Any]] = phase_map.get("edges", [])

    layout = compute_linear_layout(nodes)
    strongest_key = _strongest_edge_key(interface_ranking)

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
        # emphasise strongest interface
        if strongest_key == (src, tgt) or strongest_key == (tgt, src):
            lw += 2.0
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

    # ── summary overlay ───────────────────────────────────────────────
    if transition_summary is not None:
        _render_summary_overlay(ax, transition_summary)

    # ── legend ────────────────────────────────────────────────────────
    _add_legend(ax)

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
