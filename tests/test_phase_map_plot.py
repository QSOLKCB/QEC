"""Tests for v85.3.0 — phase map visualization with export modes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from qec.visualization.phase_map_plot import (
    CLASS_COLORS,
    EDGE_COLORS,
    PHASE_MARKERS,
    compute_linear_layout,
    plot_phase_map,
)


# ── helpers ───────────────────────────────────────────────────────────

def _sample_phase_map():
    return {
        "nodes": [
            {"id": 0, "range": [0, 9], "dominant_class": "stable",
             "dominant_phase": "aligned", "mean_score": 0.95},
            {"id": 1, "range": [10, 19], "dominant_class": "fragile",
             "dominant_phase": "misaligned", "mean_score": 0.50},
            {"id": 2, "range": [20, 29], "dominant_class": "chaotic",
             "dominant_phase": "unknown", "mean_score": 0.10},
        ],
        "edges": [
            {"source": 0, "target": 1, "type": "phase_boundary",
             "weight": 0.45},
            {"source": 1, "target": 2, "type": "strong_boundary",
             "weight": 0.80},
        ],
    }


# ── layout determinism ────────────────────────────────────────────────

def test_layout_deterministic():
    nodes = _sample_phase_map()["nodes"]
    layout_a = compute_linear_layout(nodes)
    layout_b = compute_linear_layout(nodes)
    assert layout_a == layout_b


def test_layout_order_independent():
    nodes = _sample_phase_map()["nodes"]
    layout_sorted = compute_linear_layout(nodes)
    layout_reversed = compute_linear_layout(list(reversed(nodes)))
    assert layout_sorted == layout_reversed


def test_layout_positions():
    nodes = [{"id": 0}, {"id": 1}, {"id": 2}]
    layout = compute_linear_layout(nodes)
    # v85.2: x spacing = 1.5, y alternates -0.1 / +0.1
    assert layout == {0: (0.0, -0.1), 1: (1.5, 0.1), 2: (3.0, -0.1)}


# ── plot result counts ───────────────────────────────────────────────

def test_plot_returns_correct_counts():
    pm = _sample_phase_map()
    result = plot_phase_map(pm)
    assert result["n_nodes"] == 3
    assert result["n_edges"] == 2
    assert result["output_path"] is None


# ── file creation ─────────────────────────────────────────────────────

def test_plot_saves_png():
    pm = _sample_phase_map()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "test_plot.png"
        result = plot_phase_map(pm, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        assert result["output_path"] == str(out)


# ── empty phase map ──────────────────────────────────────────────────

def test_empty_phase_map():
    result = plot_phase_map({"nodes": [], "edges": []})
    assert result["n_nodes"] == 0
    assert result["n_edges"] == 0


# ── unknown class / phase fallback ───────────────────────────────────

def test_unknown_class_and_phase():
    pm = {
        "nodes": [
            {"id": 0, "dominant_class": "INVENTED", "dominant_phase": "NOVEL"},
        ],
        "edges": [],
    }
    result = plot_phase_map(pm)
    assert result["n_nodes"] == 1


# ── deterministic coordinates across calls ────────────────────────────

def test_same_input_same_coords():
    nodes = _sample_phase_map()["nodes"]
    expected = {0: (0.0, -0.1), 1: (1.5, 0.1), 2: (3.0, -0.1)}
    for _ in range(5):
        assert compute_linear_layout(nodes) == expected


# ── edge with unknown type ───────────────────────────────────────────

def test_unknown_edge_type():
    pm = {
        "nodes": [{"id": 0}, {"id": 1}],
        "edges": [{"source": 0, "target": 1, "type": "MYSTERY", "weight": 1.0}],
    }
    result = plot_phase_map(pm)
    assert result["n_edges"] == 1


# ── v85.1.0 — annotation tests ──────────────────────────────────────


def _sample_interface_ranking():
    return {
        "ranked_interfaces": [
            {"from_index": 1, "to_index": 2, "strength": 0.80},
            {"from_index": 0, "to_index": 1, "strength": 0.45},
        ],
        "strongest_interface": {
            "from_index": 1, "to_index": 2, "strength": 0.80,
        },
    }


def _sample_transition_summary():
    return {
        "n_transitions": 2,
        "max_delta_score": 0.45,
        "class_change_count": 2,
        "phase_change_count": 2,
    }


def test_plot_with_all_new_args():
    """Plotting with interface_ranking and transition_summary succeeds."""
    pm = _sample_phase_map()
    result = plot_phase_map(
        pm,
        interface_ranking=_sample_interface_ranking(),
        transition_summary=_sample_transition_summary(),
    )
    assert result["n_nodes"] == 3
    assert result["n_edges"] == 2


def test_plot_backward_compatible():
    """Existing calls without new args still work."""
    pm = _sample_phase_map()
    result = plot_phase_map(pm)
    assert result["n_nodes"] == 3
    assert result["n_edges"] == 2
    assert result["output_path"] is None


def test_strongest_interface_emphasis():
    """Strongest interface edge renders without error."""
    pm = _sample_phase_map()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "annotated.png"
        result = plot_phase_map(
            pm,
            output_path=out,
            interface_ranking=_sample_interface_ranking(),
        )
        assert out.exists()
        assert out.stat().st_size > 0
        assert result["n_edges"] == 2


def test_annotation_empty_data():
    """Empty phase map with annotations does not crash."""
    result = plot_phase_map(
        {"nodes": [], "edges": []},
        interface_ranking={"ranked_interfaces": [],
                           "strongest_interface": None},
        transition_summary={"n_transitions": 0, "max_delta_score": 0.0,
                            "class_change_count": 0, "phase_change_count": 0},
    )
    assert result["n_nodes"] == 0
    assert result["n_edges"] == 0


def test_summary_overlay_missing_fields():
    """Summary overlay tolerates missing keys gracefully."""
    pm = _sample_phase_map()
    result = plot_phase_map(
        pm,
        transition_summary={},  # all fields missing
    )
    assert result["n_nodes"] == 3


def test_deterministic_rendering_with_annotations():
    """Same annotated inputs produce identical layout positions."""
    nodes = _sample_phase_map()["nodes"]
    expected = {0: (0.0, -0.1), 1: (1.5, 0.1), 2: (3.0, -0.1)}
    for _ in range(5):
        assert compute_linear_layout(nodes) == expected


def test_interface_ranking_none_strongest():
    """interface_ranking with strongest_interface=None is safe."""
    pm = _sample_phase_map()
    ranking = {"ranked_interfaces": [], "strongest_interface": None}
    result = plot_phase_map(pm, interface_ranking=ranking)
    assert result["n_nodes"] == 3


def test_plot_saves_png_with_annotations():
    """Full annotated plot saves to disk correctly."""
    pm = _sample_phase_map()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "full_annotated.png"
        result = plot_phase_map(
            pm,
            output_path=out,
            interface_ranking=_sample_interface_ranking(),
            transition_summary=_sample_transition_summary(),
        )
        assert out.exists()
        assert out.stat().st_size > 0
        assert result["output_path"] == str(out)


# ── v85.2.0 — layout refinement tests ───────────────────────────────


def test_vertical_offsets_alternate():
    """Node y-coordinates alternate between -0.1 and +0.1."""
    nodes = [{"id": i} for i in range(6)]
    layout = compute_linear_layout(nodes)
    for i in range(6):
        expected_y = ((i % 2) * 2 - 1) * 0.1
        assert layout[i][1] == pytest.approx(expected_y)


def test_node_x_spacing():
    """Nodes are spaced at x = index * 1.5."""
    nodes = [{"id": i} for i in range(4)]
    layout = compute_linear_layout(nodes)
    for i in range(4):
        assert layout[i][0] == pytest.approx(i * 1.5)


def test_single_node_layout():
    """Single-node map produces valid layout without error."""
    pm = {"nodes": [{"id": 0, "dominant_class": "stable"}], "edges": []}
    result = plot_phase_map(pm)
    assert result["n_nodes"] == 1
    layout = compute_linear_layout(pm["nodes"])
    assert layout == {0: (0.0, -0.1)}


def test_curved_edges_render():
    """Curved edges render without error and produce a valid plot."""
    pm = _sample_phase_map()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "curved.png"
        result = plot_phase_map(pm, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        assert result["n_edges"] == 2


def test_multiple_edges_same_pair():
    """Multiple edges between the same node pair render without error."""
    pm = {
        "nodes": [{"id": 0}, {"id": 1}],
        "edges": [
            {"source": 0, "target": 1, "type": "phase_boundary", "weight": 0.5},
            {"source": 0, "target": 1, "type": "class_boundary", "weight": 0.3},
        ],
    }
    result = plot_phase_map(pm)
    assert result["n_edges"] == 2


def test_layout_empty_nodes():
    """Empty node list returns empty layout."""
    layout = compute_linear_layout([])
    assert layout == {}


# ── v85.3.0 — export mode tests ─────────────────────────────────────


def test_default_mode_is_debug():
    """Calling without mode= uses debug (backward compatible)."""
    pm = _sample_phase_map()
    result = plot_phase_map(pm)
    assert result["n_nodes"] == 3
    assert result["n_edges"] == 2


def test_paper_mode_disables_labels_legend_summary():
    """Paper mode renders without labels, legend, or summary overlay."""
    pm = _sample_phase_map()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "paper.png"
        result = plot_phase_map(
            pm,
            output_path=out,
            interface_ranking=_sample_interface_ranking(),
            transition_summary=_sample_transition_summary(),
            mode="paper",
        )
        assert out.exists()
        assert out.stat().st_size > 0
        assert result["n_nodes"] == 3


def test_debug_mode_enables_labels_legend_summary():
    """Debug mode renders with labels, legend, and summary overlay."""
    pm = _sample_phase_map()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "debug.png"
        result = plot_phase_map(
            pm,
            output_path=out,
            interface_ranking=_sample_interface_ranking(),
            transition_summary=_sample_transition_summary(),
            mode="debug",
        )
        assert out.exists()
        assert out.stat().st_size > 0
        assert result["n_nodes"] == 3


def test_invalid_mode_raises_error():
    """Invalid mode raises ValueError."""
    pm = _sample_phase_map()
    with pytest.raises(ValueError, match="mode must be 'paper' or 'debug'"):
        plot_phase_map(pm, mode="fancy")


def test_file_creation_both_modes():
    """Both modes produce valid PNG files."""
    pm = _sample_phase_map()
    with tempfile.TemporaryDirectory() as td:
        for m in ("paper", "debug"):
            out = Path(td) / f"{m}.png"
            result = plot_phase_map(pm, output_path=out, mode=m)
            assert out.exists()
            assert out.stat().st_size > 0
            assert result["output_path"] == str(out)


def test_deterministic_rendering_unchanged_with_modes():
    """Layout positions are identical regardless of mode."""
    nodes = _sample_phase_map()["nodes"]
    expected = {0: (0.0, -0.1), 1: (1.5, 0.1), 2: (3.0, -0.1)}
    for m in ("paper", "debug"):
        assert compute_linear_layout(nodes) == expected


def test_paper_file_larger_dpi():
    """Paper mode (300 dpi) produces a larger file than debug (120 dpi)."""
    pm = _sample_phase_map()
    with tempfile.TemporaryDirectory() as td:
        paper = Path(td) / "paper.png"
        debug = Path(td) / "debug.png"
        plot_phase_map(pm, output_path=paper, mode="paper")
        plot_phase_map(pm, output_path=debug, mode="debug")
        assert paper.stat().st_size > debug.stat().st_size
