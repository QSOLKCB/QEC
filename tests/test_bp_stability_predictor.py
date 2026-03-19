"""
Tests for BP stability predictor diagnostics (v6.8).

Verifies:
  - Deterministic prediction output
  - Risk monotonicity (higher spectral radius → higher failure risk)
  - Threshold behavior (spectral_radius > threshold → instability flag)
  - JSON serialization stability
  - Pipeline integration with v6.1–v6.4 outputs
  - No decoder behavior changes (no decoder import)
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

from qec.diagnostics.bp_stability_predictor import (
    compute_bp_stability_prediction,
)


# ── Toy inputs ────────────────────────────────────────────────────────


def _make_graph(
    num_variable_nodes: int = 10,
    num_check_nodes: int = 5,
    num_edges: int = 20,
    avg_degree: float = 2.67,
    num_short_cycles: int | None = None,
) -> dict[str, object]:
    """Build a minimal graph info dict."""
    d: dict[str, object] = {
        "num_variable_nodes": num_variable_nodes,
        "num_check_nodes": num_check_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
    }
    if num_short_cycles is not None:
        d["num_short_cycles"] = num_short_cycles
    return d


def _make_diagnostics(
    spectral_radius: float = 2.0,
    nb_max_ipr: float = 0.3,
    nb_num_localized_modes: int = 2,
    max_cluster_risk: float = 0.5,
) -> dict[str, object]:
    """Build a minimal diagnostics dict."""
    return {
        "spectral_radius": spectral_radius,
        "nb_max_ipr": nb_max_ipr,
        "nb_num_localized_modes": nb_num_localized_modes,
        "max_cluster_risk": max_cluster_risk,
    }


class TestBasicComputation:
    """Tests for basic stability prediction computation."""

    def test_output_keys(self):
        """Output contains all required keys."""
        graph = _make_graph()
        diag = _make_diagnostics()

        result = compute_bp_stability_prediction(graph, diag)

        expected_keys = {
            "bp_stability_score",
            "bp_failure_risk",
            "predicted_instability",
            "spectral_radius",
            "nb_instability_threshold",
            "spectral_instability_ratio",
            "localization_strength",
            "cluster_risk_signal",
            "cycle_density",
        }
        assert set(result.keys()) == expected_keys

    def test_output_types(self):
        """Output values have correct types."""
        graph = _make_graph()
        diag = _make_diagnostics()

        result = compute_bp_stability_prediction(graph, diag)

        assert isinstance(result["bp_stability_score"], float)
        assert isinstance(result["bp_failure_risk"], float)
        assert isinstance(result["predicted_instability"], bool)
        assert isinstance(result["spectral_radius"], float)
        assert isinstance(result["nb_instability_threshold"], float)
        assert isinstance(result["spectral_instability_ratio"], float)
        assert isinstance(result["localization_strength"], float)
        assert isinstance(result["cluster_risk_signal"], float)
        assert isinstance(result["cycle_density"], float)

    def test_nb_instability_threshold(self):
        """NB instability threshold = sqrt(avg_degree)."""
        graph = _make_graph(avg_degree=4.0)
        diag = _make_diagnostics()

        result = compute_bp_stability_prediction(graph, diag)

        assert abs(result["nb_instability_threshold"] - 2.0) < 1e-10

    def test_spectral_instability_ratio(self):
        """Spectral instability ratio = spectral_radius / threshold."""
        graph = _make_graph(avg_degree=4.0)
        diag = _make_diagnostics(spectral_radius=3.0)

        result = compute_bp_stability_prediction(graph, diag)

        # threshold = sqrt(4) = 2.0, ratio = 3.0 / 2.0 = 1.5
        assert abs(result["spectral_instability_ratio"] - 1.5) < 1e-10

    def test_localization_strength(self):
        """Localization strength = nb_max_ipr * nb_num_localized_modes."""
        graph = _make_graph()
        diag = _make_diagnostics(
            nb_max_ipr=0.4,
            nb_num_localized_modes=3,
        )

        result = compute_bp_stability_prediction(graph, diag)

        assert abs(result["localization_strength"] - 1.2) < 1e-10

    def test_cluster_risk_signal(self):
        """Cluster risk signal = max_cluster_risk."""
        graph = _make_graph()
        diag = _make_diagnostics(max_cluster_risk=0.75)

        result = compute_bp_stability_prediction(graph, diag)

        assert abs(result["cluster_risk_signal"] - 0.75) < 1e-10

    def test_cycle_density_with_explicit_cycles(self):
        """Cycle density = num_short_cycles / num_variable_nodes."""
        graph = _make_graph(
            num_variable_nodes=10,
            num_short_cycles=5,
        )
        diag = _make_diagnostics()

        result = compute_bp_stability_prediction(graph, diag)

        assert abs(result["cycle_density"] - 0.5) < 1e-10

    def test_cycle_density_proxy_without_cycles(self):
        """Cycle density uses proxy when num_short_cycles is absent."""
        graph = _make_graph(
            num_variable_nodes=10,
            num_check_nodes=5,
            num_edges=20,
        )
        diag = _make_diagnostics()

        result = compute_bp_stability_prediction(graph, diag)

        # Proxy: max(0, edges - total_nodes + 1) / variable_nodes
        # = max(0, 20 - 15 + 1) / 10 = 6 / 10 = 0.6
        assert abs(result["cycle_density"] - 0.6) < 1e-10

    def test_failure_risk_clamped_to_one(self):
        """bp_failure_risk is clamped to [0, 1]."""
        graph = _make_graph(avg_degree=1.0)
        # Very high spectral radius → high score → risk would exceed 1
        diag = _make_diagnostics(
            spectral_radius=100.0,
            nb_max_ipr=1.0,
            nb_num_localized_modes=10,
            max_cluster_risk=10.0,
        )

        result = compute_bp_stability_prediction(graph, diag)

        assert result["bp_failure_risk"] <= 1.0

    def test_zero_inputs_yield_zero(self):
        """All-zero inputs yield zero scores."""
        graph = _make_graph(
            num_variable_nodes=0,
            num_check_nodes=0,
            num_edges=0,
            avg_degree=0.0,
        )
        diag = _make_diagnostics(
            spectral_radius=0.0,
            nb_max_ipr=0.0,
            nb_num_localized_modes=0,
            max_cluster_risk=0.0,
        )

        result = compute_bp_stability_prediction(graph, diag)

        assert result["bp_stability_score"] == 0.0
        assert result["bp_failure_risk"] == 0.0
        assert result["predicted_instability"] is False


class TestRiskMonotonicity:
    """Tests for risk monotonicity — higher spectral radius → higher risk."""

    def test_higher_spectral_radius_higher_risk(self):
        """Higher spectral radius produces higher failure risk."""
        graph = _make_graph(avg_degree=4.0)
        diag_low = _make_diagnostics(spectral_radius=1.5)
        diag_high = _make_diagnostics(spectral_radius=3.0)

        result_low = compute_bp_stability_prediction(graph, diag_low)
        result_high = compute_bp_stability_prediction(graph, diag_high)

        assert result_high["bp_failure_risk"] > result_low["bp_failure_risk"]
        assert result_high["bp_stability_score"] > result_low["bp_stability_score"]

    def test_higher_localization_higher_risk(self):
        """Higher localization strength produces higher failure risk."""
        graph = _make_graph()
        diag_low = _make_diagnostics(nb_max_ipr=0.1, nb_num_localized_modes=1)
        diag_high = _make_diagnostics(nb_max_ipr=0.5, nb_num_localized_modes=4)

        result_low = compute_bp_stability_prediction(graph, diag_low)
        result_high = compute_bp_stability_prediction(graph, diag_high)

        assert result_high["bp_failure_risk"] > result_low["bp_failure_risk"]

    def test_higher_cluster_risk_higher_failure_risk(self):
        """Higher cluster risk produces higher failure risk."""
        graph = _make_graph()
        diag_low = _make_diagnostics(max_cluster_risk=0.1)
        diag_high = _make_diagnostics(max_cluster_risk=2.0)

        result_low = compute_bp_stability_prediction(graph, diag_low)
        result_high = compute_bp_stability_prediction(graph, diag_high)

        assert result_high["bp_failure_risk"] > result_low["bp_failure_risk"]


class TestThresholdBehavior:
    """Tests for instability threshold behavior."""

    def test_above_threshold_triggers_instability(self):
        """spectral_radius > nb_instability_threshold → instability flag."""
        graph = _make_graph(avg_degree=4.0)
        # threshold = sqrt(4) = 2.0; spectral_radius = 3.0 → ratio = 1.5
        # With enough other signals, bp_failure_risk should exceed 0.6.
        diag = _make_diagnostics(
            spectral_radius=3.0,
            nb_max_ipr=0.5,
            nb_num_localized_modes=3,
            max_cluster_risk=1.0,
        )

        result = compute_bp_stability_prediction(graph, diag)

        assert result["spectral_instability_ratio"] > 1.0
        assert result["predicted_instability"] is True

    def test_below_threshold_stable(self):
        """spectral_radius < nb_instability_threshold → stable regime."""
        graph = _make_graph(avg_degree=16.0)
        # threshold = sqrt(16) = 4.0; spectral_radius = 1.0 → ratio = 0.25
        diag = _make_diagnostics(
            spectral_radius=1.0,
            nb_max_ipr=0.01,
            nb_num_localized_modes=0,
            max_cluster_risk=0.0,
        )

        result = compute_bp_stability_prediction(graph, diag)

        assert result["spectral_instability_ratio"] < 1.0
        assert result["predicted_instability"] is False

    def test_critical_regime(self):
        """spectral_radius ≈ threshold → ratio ≈ 1.0."""
        graph = _make_graph(avg_degree=4.0)
        # threshold = 2.0; spectral_radius = 2.0 → ratio = 1.0
        diag = _make_diagnostics(spectral_radius=2.0)

        result = compute_bp_stability_prediction(graph, diag)

        assert abs(result["spectral_instability_ratio"] - 1.0) < 1e-10


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_calls_identical(self):
        """Repeated calls produce identical JSON results."""
        graph = _make_graph(
            num_variable_nodes=10,
            num_check_nodes=5,
            num_edges=20,
            avg_degree=2.67,
            num_short_cycles=8,
        )
        diag = _make_diagnostics(
            spectral_radius=2.5,
            nb_max_ipr=0.35,
            nb_num_localized_modes=3,
            max_cluster_risk=0.7,
        )

        r1 = compute_bp_stability_prediction(graph, diag)
        r2 = compute_bp_stability_prediction(graph, diag)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        graph = _make_graph()
        diag = _make_diagnostics()

        result = compute_bp_stability_prediction(graph, diag)

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON serialization roundtrip."""
        graph = _make_graph(num_short_cycles=12)
        diag = _make_diagnostics(
            spectral_radius=3.1,
            nb_max_ipr=0.42,
            nb_num_localized_modes=4,
            max_cluster_risk=1.2,
        )

        result = compute_bp_stability_prediction(graph, diag)

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)


class TestPipelineIntegration:
    """Tests for end-to-end pipeline compatibility."""

    def test_full_pipeline(self):
        """Full pipeline: NB localization → trapping → alignment → risk → predictor."""
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
        from qec.diagnostics.non_backtracking_spectrum import (
            compute_non_backtracking_spectrum,
        )

        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)

        m, n = H.shape
        num_edges = int(np.count_nonzero(H))
        avg_degree = (2.0 * num_edges) / (n + m) if (n + m) > 0 else 0.0

        # Run v6.0 NB spectrum.
        nb_spectrum = compute_non_backtracking_spectrum(H)

        # Run v6.1 localization.
        loc = compute_nb_localization_metrics(H)

        # Run v6.2 trapping candidates.
        trapping = compute_nb_trapping_candidates(H, loc)

        # Simulate BP activity scores.
        bp_scores = {0: 3.0, 1: 7.0, 2: 1.0, 3: 5.0}

        # Run v6.3 alignment.
        alignment = compute_spectral_bp_alignment(trapping, bp_scores)

        # Run v6.4 failure risk.
        risk = compute_spectral_failure_risk(loc, trapping, alignment)

        # Build graph + diagnostics for v6.8 predictor.
        graph = {
            "num_variable_nodes": n,
            "num_check_nodes": m,
            "num_edges": num_edges,
            "avg_degree": round(avg_degree, 12),
        }
        diagnostics = {
            "spectral_radius": nb_spectrum["spectral_radius"],
            "nb_max_ipr": max(loc["ipr_scores"]) if loc["ipr_scores"] else 0.0,
            "nb_num_localized_modes": len(loc["localized_modes"]),
            "max_cluster_risk": risk["max_cluster_risk"],
        }

        # Run v6.8 predictor.
        result = compute_bp_stability_prediction(graph, diagnostics)

        # All required keys present.
        assert "bp_stability_score" in result
        assert "bp_failure_risk" in result
        assert "predicted_instability" in result
        assert "spectral_radius" in result
        assert "nb_instability_threshold" in result
        assert "spectral_instability_ratio" in result
        assert "localization_strength" in result
        assert "cluster_risk_signal" in result
        assert "cycle_density" in result

        # Types are correct.
        assert isinstance(result["bp_stability_score"], float)
        assert isinstance(result["bp_failure_risk"], float)
        assert isinstance(result["predicted_instability"], bool)

        # Non-negative risk.
        assert result["bp_stability_score"] >= 0.0
        assert result["bp_failure_risk"] >= 0.0
        assert result["bp_failure_risk"] <= 1.0

        # JSON-serializable.
        json.dumps(result)


class TestCustomWeights:
    """Tests for custom weight parameters."""

    def test_custom_weights_change_score(self):
        """Custom weights produce different scores."""
        graph = _make_graph(avg_degree=4.0, num_short_cycles=5)
        diag = _make_diagnostics(spectral_radius=3.0)

        result_default = compute_bp_stability_prediction(graph, diag)
        result_custom = compute_bp_stability_prediction(
            graph, diag,
            w1=1.0, w2=0.0, w3=0.0, w4=0.0,
        )

        # With only w1, score = spectral_instability_ratio
        assert result_custom["bp_stability_score"] != result_default["bp_stability_score"]

    def test_custom_threshold(self):
        """Custom failure risk threshold changes risk normalization."""
        graph = _make_graph(avg_degree=4.0)
        diag = _make_diagnostics(spectral_radius=2.0)

        result_default = compute_bp_stability_prediction(graph, diag)
        result_custom = compute_bp_stability_prediction(
            graph, diag,
            failure_risk_threshold=10.0,
        )

        assert result_custom["bp_failure_risk"] < result_default["bp_failure_risk"]


class TestNoDecoderImport:
    """Tests that this module does not import decoder code."""

    def test_no_decoder_import(self):
        """This module does not import any decoder code."""
        import qec.diagnostics.bp_stability_predictor as mod
        source = open(mod.__file__).read()
        assert "bp_decode" not in source
        assert "from qec.decoder" not in source
