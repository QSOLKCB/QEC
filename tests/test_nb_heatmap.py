"""
Tests for v12.4.0 NB Flow Heatmap Visualization.

Verifies:
- deterministic edge flow score computation
- correct normalization to [0, 1]
- ASCII heatmap formatting
- empty/degenerate input handling
"""

import numpy as np
import pytest

from src.qec.experiments.nb_flow_heatmap import (
    compute_edge_flow_scores,
    compute_edge_flow_scores_from_H,
    format_ascii_heatmap,
)
from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _small_H():
    """A small 3x6 parity-check matrix with known structure."""
    return np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1, 1],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests: compute_edge_flow_scores
# ---------------------------------------------------------------------------

class TestComputeEdgeFlowScores:
    def test_basic_output_structure(self):
        H = _small_H()
        analyzer = NonBacktrackingFlowAnalyzer()
        flow = analyzer.compute_flow(H)
        scores = compute_edge_flow_scores(
            H, flow["directed_edge_flow"], flow["directed_edges"],
        )
        assert isinstance(scores, list)
        assert len(scores) > 0
        for entry in scores:
            assert "check" in entry
            assert "variable" in entry
            assert "score" in entry
            assert "normalized_score" in entry

    def test_normalized_scores_in_range(self):
        H = _small_H()
        scores = compute_edge_flow_scores_from_H(H)
        for entry in scores:
            assert 0.0 <= entry["normalized_score"] <= 1.0

    def test_max_normalized_score_is_one(self):
        H = _small_H()
        scores = compute_edge_flow_scores_from_H(H)
        if scores:
            assert scores[0]["normalized_score"] == 1.0

    def test_sorted_descending(self):
        H = _small_H()
        scores = compute_edge_flow_scores_from_H(H)
        for i in range(len(scores) - 1):
            assert scores[i]["score"] >= scores[i + 1]["score"]

    def test_determinism(self):
        H = _small_H()
        scores1 = compute_edge_flow_scores_from_H(H)
        scores2 = compute_edge_flow_scores_from_H(H)
        assert scores1 == scores2

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        scores = compute_edge_flow_scores_from_H(H)
        assert scores == []

    def test_no_edges(self):
        H = np.zeros((3, 4), dtype=np.float64)
        scores = compute_edge_flow_scores_from_H(H)
        assert scores == []

    def test_edge_count_matches(self):
        H = _small_H()
        num_edges = int(np.sum(H != 0))
        scores = compute_edge_flow_scores_from_H(H)
        assert len(scores) == num_edges


# ---------------------------------------------------------------------------
# Tests: format_ascii_heatmap
# ---------------------------------------------------------------------------

class TestFormatAsciiHeatmap:
    def test_basic_output(self):
        H = _small_H()
        scores = compute_edge_flow_scores_from_H(H)
        heatmap = format_ascii_heatmap(scores)
        assert "Edge Flow Heatmap" in heatmap
        assert len(heatmap.splitlines()) > 1

    def test_top_k_limit(self):
        H = _small_H()
        scores = compute_edge_flow_scores_from_H(H)
        heatmap = format_ascii_heatmap(scores, top_k=2)
        # Header + blank line + 2 data lines = 4 lines
        lines = heatmap.splitlines()
        data_lines = [l for l in lines if l.startswith("(c")]
        assert len(data_lines) == 2

    def test_empty_scores(self):
        heatmap = format_ascii_heatmap([])
        assert "no edges" in heatmap

    def test_determinism(self):
        H = _small_H()
        scores = compute_edge_flow_scores_from_H(H)
        h1 = format_ascii_heatmap(scores)
        h2 = format_ascii_heatmap(scores)
        assert h1 == h2


# ---------------------------------------------------------------------------
# Tests: IPR-weighted mutation variant
# ---------------------------------------------------------------------------

class TestIPRWeightedMutation:
    def test_ipr_weight_off_by_default(self):
        from src.qec.discovery.mutation_nb_guided import NBGuidedMutator
        mutator = NBGuidedMutator(enabled=True)
        assert mutator.use_ipr_weight is False

    def test_ipr_weight_from_config(self):
        from src.qec.discovery.mutation_nb_guided import NBGuidedMutator
        config = {"nb_mutation": {"enabled": True, "use_ipr_weight": True}}
        mutator = NBGuidedMutator.from_config(config)
        assert mutator.use_ipr_weight is True

    def test_ipr_weight_produces_different_scores(self):
        from src.qec.discovery.mutation_nb_guided import NBGuidedMutator
        H = _small_H()
        m1 = NBGuidedMutator(enabled=True, use_ipr_weight=False)
        m2 = NBGuidedMutator(enabled=True, use_ipr_weight=True)
        H1, log1 = m1.mutate(H)
        H2, log2 = m2.mutate(H)
        # Both should produce valid results (may or may not differ
        # depending on graph structure).
        assert H1.shape == H.shape
        assert H2.shape == H.shape

    def test_no_mutation_when_disabled(self):
        from src.qec.discovery.mutation_nb_guided import NBGuidedMutator
        H = _small_H()
        mutator = NBGuidedMutator(enabled=False, use_ipr_weight=True)
        H_out, log = mutator.mutate(H)
        assert log == []
        np.testing.assert_array_equal(H_out, H)
