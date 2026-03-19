"""
Tests for spectral decoder controller experiment (v7.0).

Verifies:
  - Determinism: repeated runs produce identical outputs.
  - Mode selection: high bp_failure_risk triggers controller mode;
    low risk leaves decoding in standard mode.
  - Damping bounds: per-node damping remains within [0.5, 0.9].
  - JSON serialization: controller artifacts serialize correctly.
  - Pipeline integration: full stack executes from spectral diagnostics
    through BP stability predictor to decoder controller.
  - Decoder safety: controller does not import decoder core.
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
    _identify_high_risk_nodes,
    _compute_adaptive_damping,
    _select_control_mode,
    CONTROL_MODE_STANDARD,
    CONTROL_MODE_RISK_GUIDED_DAMPING,
    CONTROL_MODE_RISK_GUIDED_SCHEDULE,
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


# ── Mode selection tests ──────────────────────────────────────────────

class TestModeSelection:
    """Tests for control mode selection logic."""

    def test_standard_mode_low_risk(self):
        """Low risk, no instability → standard mode."""
        pred = _make_prediction(bp_failure_risk=0.2, predicted_instability=False)
        assert _select_control_mode(pred) == CONTROL_MODE_STANDARD

    def test_risk_guided_damping_high_risk(self):
        """High risk with instability → risk_guided_damping."""
        pred = _make_prediction(bp_failure_risk=0.8, predicted_instability=True)
        assert _select_control_mode(pred) == CONTROL_MODE_RISK_GUIDED_DAMPING

    def test_risk_guided_schedule_moderate_risk(self):
        """Moderate risk with instability → risk_guided_schedule."""
        pred = _make_prediction(bp_failure_risk=0.5, predicted_instability=True)
        assert _select_control_mode(pred) == CONTROL_MODE_RISK_GUIDED_SCHEDULE

    def test_standard_mode_no_instability(self):
        """Even high risk without instability flag → standard."""
        pred = _make_prediction(bp_failure_risk=0.9, predicted_instability=False)
        assert _select_control_mode(pred) == CONTROL_MODE_STANDARD

    def test_custom_threshold(self):
        """Custom threshold changes mode boundary."""
        pred = _make_prediction(bp_failure_risk=0.5, predicted_instability=True)
        # With threshold 0.4, bp_failure_risk 0.5 > 0.4 → damping mode.
        assert _select_control_mode(pred, risk_threshold=0.4) == CONTROL_MODE_RISK_GUIDED_DAMPING


# ── Adaptive damping tests ────────────────────────────────────────────

class TestAdaptiveDamping:
    """Tests for adaptive per-node damping computation."""

    def test_damping_bounds(self):
        """All damping values stay within [0.5, 0.9]."""
        scores = [[0, 1.0], [1, 0.5], [2, 0.8], [3, 0.1], [4, 0.0]]
        damping = _compute_adaptive_damping(5, scores)
        assert damping.shape == (5,)
        assert np.all(damping >= 0.5)
        assert np.all(damping <= 0.9)

    def test_high_risk_gets_higher_damping(self):
        """High-risk nodes get higher damping than low-risk nodes."""
        scores = [[0, 1.0], [1, 0.0]]
        damping = _compute_adaptive_damping(2, scores)
        assert damping[0] > damping[1]

    def test_empty_scores(self):
        """Empty scores → base damping for all nodes."""
        damping = _compute_adaptive_damping(3, [])
        assert damping.shape == (3,)
        assert np.all(damping == 0.5)

    def test_all_zero_scores(self):
        """All-zero scores → base damping for all nodes."""
        scores = [[0, 0.0], [1, 0.0]]
        damping = _compute_adaptive_damping(2, scores)
        assert np.all(damping == 0.5)

    def test_custom_parameters(self):
        """Custom base_damping and alpha work correctly."""
        scores = [[0, 1.0]]  # max_risk = 1.0, normalized = 1.0
        damping = _compute_adaptive_damping(
            1, scores, base_damping=0.6, alpha=0.2,
            damping_min=0.5, damping_max=0.9,
        )
        # damping = 0.6 + 0.2 * 1.0 = 0.8
        assert abs(damping[0] - 0.8) < 1e-12

    def test_clamping_upper(self):
        """Damping is clamped to damping_max."""
        scores = [[0, 1.0]]
        damping = _compute_adaptive_damping(
            1, scores, base_damping=0.8, alpha=0.25,
            damping_min=0.5, damping_max=0.9,
        )
        # damping = 0.8 + 0.25 = 1.05 → clamped to 0.9
        assert abs(damping[0] - 0.9) < 1e-12

    def test_deterministic(self):
        """Repeated calls produce identical results."""
        scores = [[0, 1.0], [1, 0.5], [2, 0.3]]
        d1 = _compute_adaptive_damping(3, scores)
        d2 = _compute_adaptive_damping(3, scores)
        np.testing.assert_array_equal(d1, d2)


# ── High-risk node selection tests ────────────────────────────────────

class TestHighRiskNodes:
    """Tests for high-risk node identification."""

    def test_basic_selection(self):
        """Nodes with risk >= 0.5 * max are selected."""
        scores = [[0, 1.0], [1, 0.6], [2, 0.2], [3, 0.8]]
        result = _identify_high_risk_nodes(scores)
        assert result == [0, 1, 3]

    def test_empty_scores(self):
        """Empty scores → empty result."""
        assert _identify_high_risk_nodes([]) == []

    def test_sorted_output(self):
        """Output is sorted by node index."""
        scores = [[5, 1.0], [2, 0.8], [7, 0.9]]
        result = _identify_high_risk_nodes(scores)
        assert result == sorted(result)


# ── Experiment execution tests ────────────────────────────────────────

class TestExperimentExecution:
    """Tests for complete experiment execution."""

    def test_experiment_runs_standard_mode(self):
        """Experiment completes in standard mode (low risk)."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 0.1], [1, 0.05]])
        pred = _make_prediction(bp_failure_risk=0.2, predicted_instability=False)

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        assert result["control_mode"] == CONTROL_MODE_STANDARD
        assert result["adaptive_damping_enabled"] is False
        assert "baseline_metrics" in result
        assert "controlled_metrics" in result
        assert "delta_iterations" in result
        assert "delta_success" in result
        assert "controller_metadata" in result

    def test_experiment_runs_risk_guided_mode(self):
        """Experiment completes in risk_guided_damping mode (high risk)."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [1, 0.8], [2, 0.3]])
        pred = _make_prediction(bp_failure_risk=0.8, predicted_instability=True)

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        assert result["control_mode"] == CONTROL_MODE_RISK_GUIDED_DAMPING
        assert result["adaptive_damping_enabled"] is True
        assert result["predicted_instability"] is True

    def test_baseline_metrics_structure(self):
        """Baseline metrics contain expected fields."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0]])
        pred = _make_prediction(bp_failure_risk=0.8, predicted_instability=True)

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        bm = result["baseline_metrics"]
        assert "iterations" in bm
        assert "success" in bm
        assert "residual_norms" in bm
        assert "final_residual_norm" in bm
        assert isinstance(bm["iterations"], int)
        assert isinstance(bm["success"], bool)

    def test_controlled_metrics_structure(self):
        """Controlled metrics contain expected fields."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0]])
        pred = _make_prediction(bp_failure_risk=0.8, predicted_instability=True)

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        cm = result["controlled_metrics"]
        assert "iterations" in cm
        assert "success" in cm
        assert "residual_norms" in cm
        assert "final_residual_norm" in cm

    def test_delta_consistency(self):
        """Delta values are consistent with metrics."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [1, 0.8]])
        pred = _make_prediction(bp_failure_risk=0.8, predicted_instability=True)

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=20,
        )

        assert result["delta_iterations"] == (
            result["controlled_metrics"]["iterations"]
            - result["baseline_metrics"]["iterations"]
        )
        assert result["delta_success"] == (
            int(result["controlled_metrics"]["success"])
            - int(result["baseline_metrics"]["success"])
        )

    def test_risk_passthrough(self):
        """Risk result fields are passed through to output."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.5]],
            cluster_risk_scores=[0.8, 0.3],
            top_risk_clusters=[0],
        )
        pred = _make_prediction(bp_failure_risk=0.5, predicted_instability=False)

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        assert result["node_risk_scores"] == [[0, 1.0], [1, 0.5]]
        assert result["cluster_risk_scores"] == [0.8, 0.3]
        assert result["top_risk_clusters"] == [0]

    def test_no_risk_nodes(self):
        """Empty risk scores → experiment still runs."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([])
        pred = _make_prediction(bp_failure_risk=0.1, predicted_instability=False)

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        assert result["scheduled_high_risk_nodes"] == []
        assert "baseline_metrics" in result
        assert "controlled_metrics" in result

    def test_controller_metadata(self):
        """Controller metadata contains parameter info."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0]])
        pred = _make_prediction(
            bp_failure_risk=0.8,
            predicted_instability=True,
            spectral_instability_ratio=1.5,
        )

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        meta = result["controller_metadata"]
        assert "base_damping" in meta
        assert "alpha" in meta
        assert "damping_min" in meta
        assert "damping_max" in meta
        assert "risk_threshold" in meta
        assert "spectral_instability_ratio" in meta
        assert "num_high_risk_nodes" in meta

    def test_output_keys_complete(self):
        """Output contains all required keys."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [1, 0.5]])
        pred = _make_prediction(bp_failure_risk=0.8, predicted_instability=True)

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        required_keys = {
            "control_mode", "predicted_instability", "bp_failure_risk",
            "adaptive_damping_enabled", "scheduled_high_risk_nodes",
            "baseline_metrics", "controlled_metrics",
            "delta_iterations", "delta_success",
            "controller_metadata",
            "node_risk_scores", "cluster_risk_scores", "top_risk_clusters",
        }
        assert required_keys.issubset(set(result.keys()))


# ── Determinism tests ─────────────────────────────────────────────────

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_calls_identical(self):
        """Repeated calls produce identical JSON results."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [1, 0.5], [2, 0.2]])
        pred = _make_prediction(bp_failure_risk=0.8, predicted_instability=True)

        r1 = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=15,
        )
        r2 = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=15,
        )

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_standard_mode_determinism(self):
        """Standard mode is deterministic."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 0.1]])
        pred = _make_prediction(bp_failure_risk=0.2, predicted_instability=False)

        r1 = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )
        r2 = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is fully JSON-serializable."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [1, 0.5]])
        pred = _make_prediction(bp_failure_risk=0.8, predicted_instability=True)

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON serialization roundtrip."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([[0, 1.0], [2, 0.8]])
        pred = _make_prediction(bp_failure_risk=0.7, predicted_instability=True)

        result = run_spectral_decoder_control_experiment(
            H, llr, s, risk, pred, max_iters=10,
        )

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)


# ── Decoder safety tests ─────────────────────────────────────────────

class TestDecoderSafety:
    """Tests that this module does not import decoder core."""

    def test_no_decoder_import(self):
        """Module does not import any decoder code."""
        import qec.experiments.spectral_decoder_controller as mod
        source = open(mod.__file__).read()
        assert "import bp_decode" not in source
        assert "from qec.decoder" not in source
        assert "from qec_qldpc_codes" not in source


# ── Pipeline integration test ─────────────────────────────────────────

class TestPipelineIntegration:
    """Tests end-to-end pipeline from spectral diagnostics to controller."""

    def test_full_pipeline(self):
        """Full pipeline: spectral diagnostics → predictor → controller."""
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
            "nb_num_localized_modes": len(nb_num_localized) if isinstance(nb_num_localized, list) else nb_num_localized,
            "max_cluster_risk": risk.get("max_cluster_risk", 0.0),
        }
        prediction = compute_bp_stability_prediction(graph_info, diag)

        # Step 3: Decoder controller.
        llr = np.array([2.0, -1.5, 0.5, 1.0], dtype=np.float64)
        syndrome_vec = np.array([0, 0, 0], dtype=np.uint8)

        result = run_spectral_decoder_control_experiment(
            H, llr, syndrome_vec, risk, prediction, max_iters=20,
        )

        # All required keys present.
        assert "control_mode" in result
        assert "predicted_instability" in result
        assert "bp_failure_risk" in result
        assert "adaptive_damping_enabled" in result
        assert "scheduled_high_risk_nodes" in result
        assert "baseline_metrics" in result
        assert "controlled_metrics" in result
        assert "delta_iterations" in result
        assert "delta_success" in result
        assert "controller_metadata" in result

        # JSON-serializable.
        json.dumps(result)

        # Control mode is a valid value.
        assert result["control_mode"] in {
            CONTROL_MODE_STANDARD,
            CONTROL_MODE_RISK_GUIDED_DAMPING,
            CONTROL_MODE_RISK_GUIDED_SCHEDULE,
        }
