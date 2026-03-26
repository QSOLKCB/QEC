"""Tests for flow geometry embedding (v102.7.0).

Verifies:
- deterministic embedding (identical inputs -> identical outputs)
- consistent coordinates across runs
- normalization bounds [-1, 1]
- distance metrics correctness
- empty graph handling
- adjacency matrix construction
- row normalization
- ASCII map rendering
- integration via strategy adapter
"""

from __future__ import annotations

import copy

from qec.analysis.flow_geometry import (
    ASCII_GRID_SIZE,
    ROUND_PRECISION,
    build_adjacency_matrix,
    compute_flow_geometry,
    compute_geometric_metrics,
    embed_types,
    normalize_coordinates,
    normalize_matrix,
    render_ascii_map,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(strategies):
    """Build a run dict from a list of (name, design_score) tuples."""
    return {
        "strategies": [
            {
                "name": name,
                "metrics": {
                    "design_score": score,
                    "confidence_efficiency": 0.5,
                    "consistency_gap": 0.1,
                    "revival_strength": 0.0,
                },
            }
            for name, score in strategies
        ],
    }


def _make_runs_with_scores(name, scores):
    """Build a list of runs where a single strategy has varying scores."""
    return [_make_run([(name, s)]) for s in scores]


# ---------------------------------------------------------------------------
# build_adjacency_matrix tests
# ---------------------------------------------------------------------------


class TestBuildAdjacencyMatrix:
    """Tests for build_adjacency_matrix."""

    def test_empty_graph(self):
        nodes, matrix = build_adjacency_matrix({})
        assert nodes == []
        assert matrix == []

    def test_single_edge(self):
        graph = {("a", "b"): 3}
        nodes, matrix = build_adjacency_matrix(graph)
        assert nodes == ["a", "b"]
        assert matrix == [[0.0, 3.0], [0.0, 0.0]]

    def test_self_loop(self):
        graph = {("a", "a"): 5}
        nodes, matrix = build_adjacency_matrix(graph)
        assert nodes == ["a"]
        assert matrix == [[5.0]]

    def test_multiple_edges(self):
        graph = {("a", "b"): 2, ("b", "a"): 3, ("a", "a"): 1}
        nodes, matrix = build_adjacency_matrix(graph)
        assert nodes == ["a", "b"]
        # a->a=1, a->b=2, b->a=3, b->b=0
        assert matrix[0][0] == 1.0
        assert matrix[0][1] == 2.0
        assert matrix[1][0] == 3.0
        assert matrix[1][1] == 0.0

    def test_deterministic_ordering(self):
        graph = {("c", "a"): 1, ("a", "b"): 2}
        nodes, _ = build_adjacency_matrix(graph)
        assert nodes == ["a", "b", "c"]

    def test_float_conversion(self):
        graph = {("a", "b"): 7}
        _, matrix = build_adjacency_matrix(graph)
        assert isinstance(matrix[0][1], float)


# ---------------------------------------------------------------------------
# normalize_matrix tests
# ---------------------------------------------------------------------------


class TestNormalizeMatrix:
    """Tests for normalize_matrix."""

    def test_empty(self):
        result = normalize_matrix([])
        assert result == []

    def test_single_row(self):
        result = normalize_matrix([[4.0]])
        assert abs(result[0][0] - 1.0) < 1e-10

    def test_row_sums_to_one(self):
        A = [[1.0, 3.0], [2.0, 2.0]]
        result = normalize_matrix(A)
        for row in result:
            assert abs(sum(row) - 1.0) < 1e-10

    def test_zero_row(self):
        A = [[0.0, 0.0], [1.0, 1.0]]
        result = normalize_matrix(A)
        # Zero row stays near zero (due to epsilon).
        assert abs(sum(result[0])) < 1e-10

    def test_no_mutation(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        original = copy.deepcopy(A)
        normalize_matrix(A)
        assert A == original


# ---------------------------------------------------------------------------
# embed_types tests
# ---------------------------------------------------------------------------


class TestEmbedTypes:
    """Tests for embed_types."""

    def test_empty(self):
        result = embed_types([])
        assert result == []

    def test_single_node(self):
        A = [[1.0]]
        result = embed_types(A)
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    def test_deterministic(self):
        A = [[0.5, 0.5], [0.3, 0.7]]
        r1 = embed_types(A)
        r2 = embed_types(A)
        assert r1 == r2

    def test_output_shape(self):
        A = [[0.25, 0.25, 0.25, 0.25]] * 4
        result = embed_types(A)
        assert len(result) == 4
        for x, y in result:
            assert isinstance(x, float)
            assert isinstance(y, float)


# ---------------------------------------------------------------------------
# normalize_coordinates tests
# ---------------------------------------------------------------------------


class TestNormalizeCoordinates:
    """Tests for normalize_coordinates."""

    def test_empty(self):
        result = normalize_coordinates({})
        assert result == {}

    def test_single_point(self):
        result = normalize_coordinates({"a": (0.5, 0.3)})
        assert result == {"a": (0.0, 0.0)}

    def test_bounds(self):
        coords = {"a": (1.0, 0.0), "b": (-1.0, 0.0), "c": (0.0, 0.5)}
        result = normalize_coordinates(coords)
        for name in result:
            x, y = result[name]
            assert -1.0 <= x <= 1.0
            assert -1.0 <= y <= 1.0

    def test_max_at_boundary(self):
        coords = {"a": (1.0, 0.0), "b": (-1.0, 0.0)}
        result = normalize_coordinates(coords)
        # After centering: a=(1,0), b=(-1,0), max_abs=1
        assert result["a"] == (1.0, 0.0)
        assert result["b"] == (-1.0, 0.0)

    def test_identical_points(self):
        coords = {"a": (0.5, 0.5), "b": (0.5, 0.5)}
        result = normalize_coordinates(coords)
        assert result["a"] == (0.0, 0.0)
        assert result["b"] == (0.0, 0.0)

    def test_deterministic(self):
        coords = {"a": (0.1, 0.2), "b": (0.3, -0.1), "c": (-0.2, 0.4)}
        r1 = normalize_coordinates(coords)
        r2 = normalize_coordinates(dict(coords))
        assert r1 == r2


# ---------------------------------------------------------------------------
# compute_geometric_metrics tests
# ---------------------------------------------------------------------------


class TestComputeGeometricMetrics:
    """Tests for compute_geometric_metrics."""

    def test_empty(self):
        result = compute_geometric_metrics({})
        assert result == {}

    def test_single_at_origin(self):
        result = compute_geometric_metrics({"a": (0.0, 0.0)})
        assert result["a"]["distance_from_center"] == 0.0
        assert result["a"]["density"] == 0.0
        assert result["a"]["cluster_score"] == 1.0

    def test_distance_from_center(self):
        result = compute_geometric_metrics({"a": (0.6, 0.8)})
        # sqrt(0.36 + 0.64) = 1.0
        assert abs(result["a"]["distance_from_center"] - 1.0) < 1e-10

    def test_cluster_score_bounded(self):
        coords = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (-1.0, 0.0)}
        result = compute_geometric_metrics(coords)
        for name in result:
            assert 0.0 <= result[name]["cluster_score"] <= 1.0

    def test_density_correctness(self):
        coords = {"a": (0.0, 0.0), "b": (1.0, 0.0)}
        result = compute_geometric_metrics(coords)
        # Distance between a and b is 1.0.
        assert abs(result["a"]["density"] - 1.0) < 1e-10
        assert abs(result["b"]["density"] - 1.0) < 1e-10

    def test_cluster_score_inverse_density(self):
        coords = {"a": (0.0, 0.0), "b": (1.0, 0.0)}
        result = compute_geometric_metrics(coords)
        # density = 1.0, cluster_score = 1/(1+1) = 0.5
        assert abs(result["a"]["cluster_score"] - 0.5) < 1e-10

    def test_sorted_keys(self):
        coords = {"c": (0.0, 0.0), "a": (1.0, 0.0), "b": (0.5, 0.5)}
        result = compute_geometric_metrics(coords)
        assert list(result.keys()) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# compute_flow_geometry (full pipeline) tests
# ---------------------------------------------------------------------------


class TestComputeFlowGeometry:
    """Tests for compute_flow_geometry."""

    def test_empty_graph(self):
        result = compute_flow_geometry({})
        assert result["coordinates"] == {}
        assert result["metrics"] == {}
        assert result["nodes"] == []

    def test_single_self_loop(self):
        graph = {("a", "a"): 5}
        result = compute_flow_geometry(graph)
        assert "a" in result["coordinates"]
        assert "a" in result["metrics"]
        assert result["nodes"] == ["a"]

    def test_deterministic(self):
        graph = {("a", "b"): 3, ("b", "a"): 2, ("a", "a"): 1}
        r1 = compute_flow_geometry(graph)
        r2 = compute_flow_geometry(dict(graph))
        assert r1["coordinates"] == r2["coordinates"]
        assert r1["metrics"] == r2["metrics"]
        assert r1["nodes"] == r2["nodes"]

    def test_coordinates_bounded(self):
        graph = {
            ("a", "b"): 3,
            ("b", "c"): 2,
            ("c", "a"): 1,
            ("a", "a"): 4,
        }
        result = compute_flow_geometry(graph)
        for name, (x, y) in result["coordinates"].items():
            assert -1.0 <= x <= 1.0, f"{name}: x={x}"
            assert -1.0 <= y <= 1.0, f"{name}: y={y}"

    def test_metrics_present(self):
        graph = {("a", "b"): 1, ("b", "a"): 1}
        result = compute_flow_geometry(graph)
        for name in result["nodes"]:
            m = result["metrics"][name]
            assert "distance_from_center" in m
            assert "density" in m
            assert "cluster_score" in m

    def test_no_input_mutation(self):
        graph = {("a", "b"): 3, ("b", "a"): 2}
        original = dict(graph)
        compute_flow_geometry(graph)
        assert graph == original


# ---------------------------------------------------------------------------
# render_ascii_map tests
# ---------------------------------------------------------------------------


class TestRenderAsciiMap:
    """Tests for render_ascii_map."""

    def test_empty(self):
        result = render_ascii_map({})
        assert "empty" in result

    def test_single_type(self):
        result = render_ascii_map({"stable_core": (0.0, 0.0)})
        assert "=== Flow Geometry Map ===" in result
        assert "S" in result
        assert "Legend:" in result
        assert "stable_core" in result

    def test_deterministic(self):
        coords = {"a": (0.5, 0.5), "b": (-0.5, -0.5)}
        r1 = render_ascii_map(coords)
        r2 = render_ascii_map(dict(coords))
        assert r1 == r2

    def test_contains_legend(self):
        coords = {"alpha": (0.1, 0.2), "beta": (-0.3, 0.4)}
        result = render_ascii_map(coords)
        assert "A = alpha" in result
        assert "B = beta" in result


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFlowGeometryIntegration:
    """Integration tests via strategy_adapter.run_flow_geometry_analysis."""

    def test_basic_integration(self):
        from qec.analysis.strategy_adapter import run_flow_geometry_analysis

        runs = _make_runs_with_scores("alpha", [0.5, 0.5, 0.5])
        result = run_flow_geometry_analysis(runs)

        assert "flow_geometry" in result
        geo = result["flow_geometry"]
        assert "coordinates" in geo
        assert "metrics" in geo
        assert "nodes" in geo

    def test_format_summary(self):
        from qec.analysis.strategy_adapter import (
            format_flow_geometry_summary,
            run_flow_geometry_analysis,
        )

        runs = _make_runs_with_scores("beta", [0.5, 0.6, 0.5])
        result = run_flow_geometry_analysis(runs)
        summary = format_flow_geometry_summary(result)

        assert "=== Flow Geometry ===" in summary

    def test_determinism_integration(self):
        from qec.analysis.strategy_adapter import run_flow_geometry_analysis

        runs = _make_runs_with_scores("delta", [0.5, 0.6, 0.4, 0.5])
        r1 = run_flow_geometry_analysis(runs)
        r2 = run_flow_geometry_analysis(copy.deepcopy(runs))

        assert r1["flow_geometry"] == r2["flow_geometry"]

    def test_empty_runs(self):
        from qec.analysis.strategy_adapter import run_flow_geometry_analysis

        result = run_flow_geometry_analysis([])
        geo = result["flow_geometry"]
        assert geo["coordinates"] == {}
        assert geo["metrics"] == {}
        assert geo["nodes"] == []

    def test_no_mutation(self):
        from qec.analysis.strategy_adapter import run_flow_geometry_analysis

        runs = _make_runs_with_scores("epsilon", [0.5, 0.6])
        original = copy.deepcopy(runs)
        run_flow_geometry_analysis(runs)
        assert runs == original

    def test_coordinates_bounded_integration(self):
        from qec.analysis.strategy_adapter import run_flow_geometry_analysis

        runs = _make_runs_with_scores("zeta", [0.5, 0.6, 0.4, 0.5, 0.7])
        result = run_flow_geometry_analysis(runs)

        for name, (x, y) in result["flow_geometry"]["coordinates"].items():
            assert -1.0 <= x <= 1.0, f"{name}: x={x}"
            assert -1.0 <= y <= 1.0, f"{name}: y={y}"

    def test_format_with_geometry_flag(self):
        from qec.analysis.strategy_adapter import (
            format_flow_geometry_summary,
            run_flow_geometry_analysis,
        )

        runs = [
            _make_run([("A", 0.5), ("B", 0.6)]),
            _make_run([("A", 0.5), ("B", 0.7)]),
            _make_run([("A", 0.5), ("B", 0.8)]),
        ]
        result = run_flow_geometry_analysis(runs)
        summary = format_flow_geometry_summary(result, show_map=True)
        assert "=== Flow Geometry ===" in summary
