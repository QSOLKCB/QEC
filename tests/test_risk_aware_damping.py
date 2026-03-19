"""
Tests for risk-aware damping experiment (v6.5).

Verifies:
  - Deterministic high-risk node selection
  - Experiment runs complete with baseline and experimental results
  - Baseline vs experiment results both produced
  - Decoder core remains unchanged (no decoder import in module)
  - JSON output stability
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.experiments.risk_aware_damping import (
    run_risk_aware_damping_experiment,
    _identify_high_risk_nodes,
)


# ── Toy inputs ────────────────────────────────────────────────────────

def _make_risk_result(
    node_risk_scores: list[list],
    cluster_risk_scores: list[float] | None = None,
    top_risk_clusters: list[int] | None = None,
) -> dict[str, object]:
    """Build a minimal v6.4-compatible risk result."""
    return {
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": cluster_risk_scores or [],
        "top_risk_clusters": top_risk_clusters or [],
        "cluster_risk_ranking": [],
        "max_cluster_risk": max((s for _, s in node_risk_scores), default=0.0),
        "mean_cluster_risk": 0.0,
        "num_high_risk_clusters": 0,
    }


def _make_simple_code() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple parity-check matrix with LLR and syndrome."""
    H = np.array([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
    ], dtype=np.float64)
    llr = np.array([2.0, -1.5, 0.5, 1.0, -0.8], dtype=np.float64)
    syndrome_vec = np.array([0, 0, 0], dtype=np.uint8)
    return H, llr, syndrome_vec


class TestHighRiskNodeSelection:
    """Tests for deterministic high-risk node identification."""

    def test_basic_selection(self):
        """Nodes with risk >= 0.5 * max are selected."""
        scores = [[0, 1.0], [1, 0.6], [2, 0.2], [3, 0.8]]
        result = _identify_high_risk_nodes(scores)
        # max = 1.0, threshold = 0.5
        # nodes 0 (1.0), 1 (0.6), 3 (0.8) pass threshold
        assert result == [0, 1, 3]

    def test_empty_scores(self):
        """Empty scores → empty result."""
        assert _identify_high_risk_nodes([]) == []

    def test_all_zero_risk(self):
        """All zero risk → empty result."""
        scores = [[0, 0.0], [1, 0.0]]
        assert _identify_high_risk_nodes(scores) == []

    def test_custom_threshold(self):
        """Custom threshold fraction works correctly."""
        scores = [[0, 1.0], [1, 0.3], [2, 0.8]]
        result = _identify_high_risk_nodes(scores, threshold_fraction=0.9)
        # threshold = 0.9 * 1.0 = 0.9; only node 0 passes
        assert result == [0]

    def test_sorted_output(self):
        """Output is sorted by node index."""
        scores = [[5, 1.0], [2, 0.8], [7, 0.9]]
        result = _identify_high_risk_nodes(scores)
        assert result == sorted(result)

    def test_deterministic(self):
        """Repeated calls produce identical results."""
        scores = [[0, 1.0], [1, 0.6], [2, 0.2]]
        r1 = _identify_high_risk_nodes(scores)
        r2 = _identify_high_risk_nodes(scores)
        assert r1 == r2


class TestExperimentExecution:
    """Tests for complete experiment execution."""

    def test_experiment_runs(self):
        """Experiment completes and returns expected keys."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [1, 0.5], [2, 0.2]])

        result = run_risk_aware_damping_experiment(
            H, llr, s, risk, max_iters=10,
        )

        assert "high_risk_nodes" in result
        assert "baseline_metrics" in result
        assert "experiment_metrics" in result
        assert "delta_iterations" in result
        assert "delta_success" in result
        assert "node_risk_scores" in result
        assert "cluster_risk_scores" in result
        assert "top_risk_clusters" in result

    def test_baseline_metrics_structure(self):
        """Baseline metrics contain expected fields."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0]])

        result = run_risk_aware_damping_experiment(
            H, llr, s, risk, max_iters=10,
        )

        bm = result["baseline_metrics"]
        assert "iterations" in bm
        assert "success" in bm
        assert "residual_norms" in bm
        assert "final_residual_norm" in bm
        assert isinstance(bm["iterations"], int)
        assert isinstance(bm["success"], bool)

    def test_experiment_metrics_structure(self):
        """Experiment metrics contain expected fields."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0]])

        result = run_risk_aware_damping_experiment(
            H, llr, s, risk, max_iters=10,
        )

        em = result["experiment_metrics"]
        assert "iterations" in em
        assert "success" in em
        assert "residual_norms" in em
        assert "final_residual_norm" in em

    def test_delta_consistency(self):
        """Delta values are consistent with metrics."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [1, 0.8]])

        result = run_risk_aware_damping_experiment(
            H, llr, s, risk, max_iters=20,
        )

        assert result["delta_iterations"] == (
            result["experiment_metrics"]["iterations"]
            - result["baseline_metrics"]["iterations"]
        )
        assert result["delta_success"] == (
            int(result["experiment_metrics"]["success"])
            - int(result["baseline_metrics"]["success"])
        )

    def test_no_risk_nodes(self):
        """Empty risk scores → no high-risk nodes, experiment still runs."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([])

        result = run_risk_aware_damping_experiment(
            H, llr, s, risk, max_iters=10,
        )

        assert result["high_risk_nodes"] == []
        assert "baseline_metrics" in result
        assert "experiment_metrics" in result

    def test_risk_passthrough(self):
        """Risk result fields are passed through to output."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.5]],
            cluster_risk_scores=[0.8, 0.3],
            top_risk_clusters=[0],
        )

        result = run_risk_aware_damping_experiment(
            H, llr, s, risk, max_iters=10,
        )

        assert result["node_risk_scores"] == [[0, 1.0], [1, 0.5]]
        assert result["cluster_risk_scores"] == [0.8, 0.3]
        assert result["top_risk_clusters"] == [0]


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_calls_identical(self):
        """Repeated calls produce identical JSON results."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [1, 0.5], [2, 0.2]])

        r1 = run_risk_aware_damping_experiment(
            H, llr, s, risk, max_iters=15,
        )
        r2 = run_risk_aware_damping_experiment(
            H, llr, s, risk, max_iters=15,
        )

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is fully JSON-serializable."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [1, 0.5]])

        result = run_risk_aware_damping_experiment(
            H, llr, s, risk, max_iters=10,
        )

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON serialization roundtrip."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [2, 0.8]])

        result = run_risk_aware_damping_experiment(
            H, llr, s, risk, max_iters=10,
        )

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)


class TestNoDecoderImport:
    """Tests that this module does not import decoder code."""

    def test_no_decoder_import(self):
        """Module does not import any decoder code."""
        import qec.experiments.risk_aware_damping as mod
        source = open(mod.__file__).read()
        assert "import bp_decode" not in source
        assert "from qec.decoder" not in source
        assert "from qec_qldpc_codes" not in source


class TestEndToEndWithV64:
    """Tests end-to-end pipeline from v6.4 risk scoring to experiment."""

    def test_full_pipeline(self):
        """Full pipeline: risk scoring → damping experiment."""
        from qec.diagnostics.nb_localization import (
            compute_nb_localization_metrics,
        )
        from qec.diagnostics.nb_trapping_candidates import (
            compute_nb_trapping_candidates,
        )
        from qec.diagnostics.spectral_bp_alignment import (
            compute_spectral_bp_alignment,
        )
        from qec.diagnostics.spectral_failure_risk import (
            compute_spectral_failure_risk,
        )

        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)

        loc = compute_nb_localization_metrics(H)
        trapping = compute_nb_trapping_candidates(H, loc)
        bp_scores = {0: 3.0, 1: 7.0, 2: 1.0, 3: 5.0}
        alignment = compute_spectral_bp_alignment(trapping, bp_scores)
        risk = compute_spectral_failure_risk(loc, trapping, alignment)

        llr = np.array([2.0, -1.5, 0.5, 1.0], dtype=np.float64)
        syndrome_vec = np.array([0, 0, 0], dtype=np.uint8)

        result = run_risk_aware_damping_experiment(
            H, llr, syndrome_vec, risk, max_iters=20,
        )

        # All required keys present.
        assert "high_risk_nodes" in result
        assert "baseline_metrics" in result
        assert "experiment_metrics" in result
        assert "delta_iterations" in result
        assert "delta_success" in result

        # JSON-serializable.
        json.dumps(result)
