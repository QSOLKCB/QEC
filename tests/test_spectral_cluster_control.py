"""
Tests for cluster-aware decoder control (v7.1).

Verifies:
  - Determinism: repeated runs with cluster control produce identical outputs.
  - Cluster ordering: cluster nodes are deterministically sorted.
  - Controller mode activation: high-risk cluster activates cluster scheduling;
    no clusters falls back to standard mode.
  - JSON artifact stability: controller outputs serialize without NumPy types.
  - Decoder safety: enabling cluster control does not change baseline behavior
    when controller is disabled.
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

from qec.experiments.spectral_decoder_controller import (
    run_spectral_decoder_control_experiment,
    _extract_cluster_nodes,
    _experimental_bp_cluster_scheduled,
    _experimental_bp_flooding,
    _compute_syndrome,
    CONTROL_MODE_STANDARD,
    CONTROL_MODE_RISK_CLUSTER_SCHEDULE,
)


# ── Toy inputs ────────────────────────────────────────────────────────

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


def _make_risk_result_with_clusters(
    node_risk_scores: list[list],
    cluster_risk_scores: list[float] | None = None,
    top_risk_clusters: list[int] | None = None,
    candidate_clusters: list[dict] | None = None,
) -> dict[str, object]:
    """Build a v6.4-compatible risk result enriched with candidate_clusters."""
    return {
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": cluster_risk_scores or [],
        "top_risk_clusters": top_risk_clusters or [],
        "cluster_risk_ranking": [],
        "max_cluster_risk": max((s for _, s in node_risk_scores), default=0.0),
        "mean_cluster_risk": 0.0,
        "num_high_risk_clusters": 0,
        "candidate_clusters": candidate_clusters or [],
    }


def _make_prediction(
    bp_failure_risk: float = 0.0,
    predicted_instability: bool = False,
    spectral_instability_ratio: float = 0.0,
) -> dict[str, object]:
    """Build a minimal v6.8-compatible prediction result."""
    return {
        "bp_stability_score": bp_failure_risk * 1.5,
        "bp_failure_risk": bp_failure_risk,
        "predicted_instability": predicted_instability,
        "spectral_radius": 1.0,
        "nb_instability_threshold": 1.0,
        "spectral_instability_ratio": spectral_instability_ratio,
        "localization_strength": 0.0,
        "cluster_risk_signal": 0.0,
        "cycle_density": 0.0,
    }


# ── Cluster node extraction tests ────────────────────────────────────

class TestExtractClusterNodes:
    """Tests for _extract_cluster_nodes helper."""

    def test_extracts_sorted_nodes(self):
        """Cluster nodes are extracted and sorted."""
        risk = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.5]],
            cluster_risk_scores=[0.9, 0.3],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [3, 1, 0], "check_nodes": [0]},
                {"variable_nodes": [2, 4], "check_nodes": [1]},
            ],
        )
        nodes = _extract_cluster_nodes(risk, 5)
        assert nodes == [0, 1, 3]

    def test_empty_top_clusters(self):
        """No top clusters → empty result."""
        risk = _make_risk_result_with_clusters(
            [[0, 0.1]],
            top_risk_clusters=[],
            candidate_clusters=[
                {"variable_nodes": [0, 1], "check_nodes": []},
            ],
        )
        assert _extract_cluster_nodes(risk, 5) == []

    def test_empty_candidate_clusters(self):
        """No candidate clusters → empty result."""
        risk = _make_risk_result_with_clusters(
            [[0, 0.1]],
            top_risk_clusters=[0],
            candidate_clusters=[],
        )
        assert _extract_cluster_nodes(risk, 5) == []

    def test_out_of_bounds_filtered(self):
        """Nodes beyond n are filtered out."""
        risk = _make_risk_result_with_clusters(
            [[0, 1.0]],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [0, 2, 10], "check_nodes": []},
            ],
        )
        nodes = _extract_cluster_nodes(risk, 5)
        assert nodes == [0, 2]
        assert 10 not in nodes

    def test_deterministic_ordering(self):
        """Repeated calls produce identical sorted output."""
        risk = _make_risk_result_with_clusters(
            [[0, 1.0]],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [4, 2, 0, 3], "check_nodes": []},
            ],
        )
        n1 = _extract_cluster_nodes(risk, 5)
        n2 = _extract_cluster_nodes(risk, 5)
        assert n1 == n2
        assert n1 == sorted(n1)


# ── Cluster-scheduled BP tests ───────────────────────────────────────

class TestClusterScheduledBP:
    """Tests for _experimental_bp_cluster_scheduled."""

    def test_deterministic(self):
        """Repeated calls produce identical results."""
        H, llr, s = _make_simple_code()
        damping = np.full(5, 0.5, dtype=np.float64)
        cluster = [0, 1]

        c1, i1, r1 = _experimental_bp_cluster_scheduled(
            H, llr, s, 10, damping, cluster,
        )
        c2, i2, r2 = _experimental_bp_cluster_scheduled(
            H, llr, s, 10, damping, cluster,
        )

        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2
        assert r1 == r2

    def test_empty_cluster_matches_flooding(self):
        """Empty cluster list produces same result as flooding."""
        H, llr, s = _make_simple_code()
        damping = np.full(5, 0.5, dtype=np.float64)

        c_flood, i_flood, r_flood = _experimental_bp_flooding(
            H, llr, s, 10, damping,
        )
        c_clust, i_clust, r_clust = _experimental_bp_cluster_scheduled(
            H, llr, s, 10, damping, [],
        )

        np.testing.assert_array_equal(c_flood, c_clust)
        assert i_flood == i_clust
        assert r_flood == r_clust

    def test_returns_valid_correction(self):
        """Cluster-scheduled BP returns a valid binary correction."""
        H, llr, s = _make_simple_code()
        damping = np.full(5, 0.5, dtype=np.float64)
        cluster = [0, 3]

        correction, iters, residuals = _experimental_bp_cluster_scheduled(
            H, llr, s, 20, damping, cluster,
        )

        assert correction.shape == (5,)
        assert set(np.unique(correction)).issubset({0, 1})
        assert isinstance(iters, int)
        assert all(isinstance(r, float) for r in residuals)


# ── Controller mode activation tests ─────────────────────────────────

class TestClusterControlActivation:
    """Tests for cluster control mode activation."""

    def test_cluster_mode_activated_high_risk(self):
        """High-risk cluster + enable_cluster_control → risk_cluster_schedule."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.8], [3, 0.6]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [0, 1, 3], "check_nodes": [0]},
            ],
        )
        pred = _make_prediction(
            bp_failure_risk=0.8, predicted_instability=True,
        )

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=10,
            enable_cluster_control=True,
        )

        assert result["control_mode"] == CONTROL_MODE_RISK_CLUSTER_SCHEDULE
        assert result["cluster_control_enabled"] is True
        assert result["cluster_nodes"] == [0, 1, 3]
        assert result["cluster_size"] == 3

    def test_fallback_no_clusters(self):
        """No clusters available → fallback, no cluster scheduling."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.8]],
            cluster_risk_scores=[],
            top_risk_clusters=[],
            candidate_clusters=[],
        )
        pred = _make_prediction(
            bp_failure_risk=0.8, predicted_instability=True,
        )

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=10,
            enable_cluster_control=True,
        )

        assert result["control_mode"] != CONTROL_MODE_RISK_CLUSTER_SCHEDULE
        assert result["cluster_control_enabled"] is False
        assert result["cluster_nodes"] == []
        assert result["cluster_size"] == 0

    def test_no_activation_standard_mode(self):
        """Low risk (standard mode) → cluster control not activated."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result_with_clusters(
            [[0, 0.1]],
            cluster_risk_scores=[0.5],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [0, 1], "check_nodes": []},
            ],
        )
        pred = _make_prediction(
            bp_failure_risk=0.2, predicted_instability=False,
        )

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=10,
            enable_cluster_control=True,
        )

        assert result["control_mode"] == CONTROL_MODE_STANDARD
        assert result["cluster_control_enabled"] is False

    def test_disabled_by_default(self):
        """Without enable_cluster_control, cluster mode is not activated."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.8]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [0, 1], "check_nodes": [0]},
            ],
        )
        pred = _make_prediction(
            bp_failure_risk=0.8, predicted_instability=True,
        )

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=10,
        )

        assert result["control_mode"] != CONTROL_MODE_RISK_CLUSTER_SCHEDULE
        assert result["cluster_control_enabled"] is False


# ── Determinism tests ────────────────────────────────────────────────

class TestClusterControlDeterminism:
    """Tests for deterministic cluster-control behavior."""

    def test_repeated_runs_identical(self):
        """Two identical cluster-control runs produce identical JSON."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.8], [2, 0.3]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [0, 1], "check_nodes": [0]},
            ],
        )
        pred = _make_prediction(
            bp_failure_risk=0.8, predicted_instability=True,
        )

        r1 = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=15, enable_cluster_control=True,
        )
        r2 = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=15, enable_cluster_control=True,
        )

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_cluster_nodes_always_sorted(self):
        """Cluster nodes in output are always deterministically sorted."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.5]],
            cluster_risk_scores=[0.8],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [4, 2, 0], "check_nodes": []},
            ],
        )
        pred = _make_prediction(
            bp_failure_risk=0.8, predicted_instability=True,
        )

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=10, enable_cluster_control=True,
        )

        assert result["cluster_nodes"] == sorted(result["cluster_nodes"])


# ── JSON artifact stability tests ────────────────────────────────────

class TestClusterControlJSONStability:
    """Tests for JSON serialization stability of cluster control outputs."""

    def test_json_serializable(self):
        """All cluster control output fields serialize to JSON."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.5]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [0, 1], "check_nodes": [0]},
            ],
        )
        pred = _make_prediction(
            bp_failure_risk=0.8, predicted_instability=True,
        )

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=10, enable_cluster_control=True,
        )

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_cluster_fields_present(self):
        """All cluster-specific artifact fields are present."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result_with_clusters(
            [[0, 1.0]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [0, 1], "check_nodes": []},
            ],
        )
        pred = _make_prediction(
            bp_failure_risk=0.8, predicted_instability=True,
        )

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=10, enable_cluster_control=True,
        )

        assert "cluster_control_enabled" in result
        assert "cluster_nodes" in result
        assert "cluster_size" in result
        assert "cluster_risk_score" in result
        assert "cluster_priority_fraction" in result

        assert isinstance(result["cluster_control_enabled"], bool)
        assert isinstance(result["cluster_nodes"], list)
        assert isinstance(result["cluster_size"], int)
        assert isinstance(result["cluster_risk_score"], float)
        assert isinstance(result["cluster_priority_fraction"], float)

    def test_float_precision(self):
        """Float outputs are rounded to 12 decimals."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.5]],
            cluster_risk_scores=[0.123456789012345],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [0, 1, 2], "check_nodes": []},
            ],
        )
        pred = _make_prediction(
            bp_failure_risk=0.8, predicted_instability=True,
        )

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=10, enable_cluster_control=True,
        )

        # cluster_risk_score should be rounded to 12 decimals.
        score_str = f"{result['cluster_risk_score']:.15f}"
        # Check that we don't have more than 12 significant decimal digits.
        assert result["cluster_risk_score"] == round(
            result["cluster_risk_score"], 12,
        )

        # cluster_priority_fraction should also be rounded.
        assert result["cluster_priority_fraction"] == round(
            result["cluster_priority_fraction"], 12,
        )


# ── Decoder safety tests ─────────────────────────────────────────────

class TestClusterControlDecoderSafety:
    """Tests that cluster control does not modify decoder core."""

    def test_no_decoder_import(self):
        """Module does not import any decoder code."""
        import qec.experiments.spectral_decoder_controller as mod
        source = open(mod.__file__).read()
        assert "import bp_decode" not in source
        assert "from qec.decoder" not in source

    def test_baseline_unchanged_when_disabled(self):
        """Baseline decode is identical with and without cluster control."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.5]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [0, 1], "check_nodes": [0]},
            ],
        )
        pred = _make_prediction(
            bp_failure_risk=0.2, predicted_instability=False,
        )

        # With cluster control disabled (standard mode due to low risk).
        r_without = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=10, enable_cluster_control=False,
        )

        # With cluster control enabled but standard mode (low risk).
        r_with = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred,
            max_iters=10, enable_cluster_control=True,
        )

        # Baseline metrics must be identical.
        assert r_without["baseline_metrics"] == r_with["baseline_metrics"]
        # Both should be standard mode — no cluster scheduling.
        assert r_without["control_mode"] == CONTROL_MODE_STANDARD
        assert r_with["control_mode"] == CONTROL_MODE_STANDARD

    def test_baseline_decode_not_affected_by_cluster_mode(self):
        """Baseline decode metrics are the same regardless of cluster mode."""
        H, llr, s = _make_simple_code()
        risk_no_cluster = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.8]],
            cluster_risk_scores=[],
            top_risk_clusters=[],
            candidate_clusters=[],
        )
        risk_with_cluster = _make_risk_result_with_clusters(
            [[0, 1.0], [1, 0.8]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
            candidate_clusters=[
                {"variable_nodes": [0, 1], "check_nodes": [0]},
            ],
        )
        pred = _make_prediction(
            bp_failure_risk=0.8, predicted_instability=True,
        )

        r1 = run_spectral_decoder_control_experiment(
            H, llr, s, risk_no_cluster, pred,
            max_iters=10, enable_cluster_control=True,
        )
        r2 = run_spectral_decoder_control_experiment(
            H, llr, s, risk_with_cluster, pred,
            max_iters=10, enable_cluster_control=True,
        )

        # Baseline is computed identically regardless of cluster info.
        assert r1["baseline_metrics"] == r2["baseline_metrics"]


# ── Pipeline integration test ────────────────────────────────────────

class TestClusterControlPipeline:
    """Tests end-to-end pipeline with cluster control."""

    def test_full_pipeline_with_cluster_control(self):
        """Full pipeline: spectral diagnostics → predictor → cluster controller."""
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
        from qec.diagnostics.bp_stability_predictor import (
            compute_bp_stability_prediction,
        )

        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)

        # Step 1: Spectral diagnostics.
        loc = compute_nb_localization_metrics(H)
        trapping = compute_nb_trapping_candidates(H, loc)
        bp_scores = {0: 3.0, 1: 7.0, 2: 1.0, 3: 5.0}
        alignment = compute_spectral_bp_alignment(trapping, bp_scores)
        risk = compute_spectral_failure_risk(loc, trapping, alignment)

        # Enrich risk with candidate_clusters from trapping.
        risk["candidate_clusters"] = trapping.get("candidate_clusters", [])

        # Step 2: BP stability predictor.
        m, n = H.shape
        num_edges = int(np.count_nonzero(H))
        avg_degree = (2.0 * num_edges) / (n + m) if (n + m) > 0 else 0.0
        graph_info = {
            "num_variable_nodes": n,
            "num_check_nodes": m,
            "num_edges": num_edges,
            "avg_degree": round(avg_degree, 12),
        }
        nb_max_ipr = loc.get("max_ipr", 0.0)
        if nb_max_ipr is None:
            ipr_scores = loc.get("ipr_scores", [])
            nb_max_ipr = max(ipr_scores) if ipr_scores else 0.0
        nb_num_localized = loc.get("localized_modes", [])
        diag = {
            "spectral_radius": risk.get("max_cluster_risk", 0.0),
            "nb_max_ipr": nb_max_ipr,
            "nb_num_localized_modes": (
                len(nb_num_localized)
                if isinstance(nb_num_localized, list)
                else nb_num_localized
            ),
            "max_cluster_risk": risk.get("max_cluster_risk", 0.0),
        }
        prediction = compute_bp_stability_prediction(graph_info, diag)

        # Step 3: Cluster-aware controller.
        llr = np.array([2.0, -1.5, 0.5, 1.0], dtype=np.float64)
        syndrome_vec = np.array([0, 0, 0], dtype=np.uint8)

        result = run_spectral_decoder_control_experiment(
            H, llr, syndrome_vec, risk, prediction,
            max_iters=20, enable_cluster_control=True,
        )

        # All required keys present.
        assert "control_mode" in result
        assert "cluster_control_enabled" in result
        assert "cluster_nodes" in result
        assert "cluster_size" in result
        assert "cluster_risk_score" in result
        assert "cluster_priority_fraction" in result

        # JSON-serializable.
        json.dumps(result)

        # Cluster nodes are sorted.
        assert result["cluster_nodes"] == sorted(result["cluster_nodes"])
