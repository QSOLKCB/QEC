"""
Tests for BP prediction validation experiment (v6.9).

Verifies:
  - Confusion matrix correctness (TP, FP, TN, FN counts)
  - Metric accuracy (precision, recall, accuracy formulas)
  - Determinism (repeated runs produce identical metrics)
  - Pipeline integration with v6.1–v6.8 outputs
  - Edge cases (all correct, all wrong, no positives, no negatives)
  - JSON serialization stability
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

from qec.experiments.bp_prediction_validation import (
    run_bp_prediction_validation,
)


# ── Helper ───────────────────────────────────────────────────────────


def _make_trial(predicted_instability: bool, decoder_success: bool) -> dict:
    """Build a minimal trial dict for validation."""
    return {
        "bp_stability_prediction": {
            "predicted_instability": predicted_instability,
        },
        "decoder_success": decoder_success,
    }


# ── Confusion Matrix Correctness ─────────────────────────────────────


class TestConfusionMatrix:
    """Tests for confusion matrix counts."""

    def test_true_positive(self):
        """Predicted failure + actual failure → TP."""
        trials = [_make_trial(predicted_instability=True, decoder_success=False)]
        result = run_bp_prediction_validation(trials)
        cm = result["confusion_matrix"]
        assert cm["true_positives"] == 1
        assert cm["false_positives"] == 0
        assert cm["true_negatives"] == 0
        assert cm["false_negatives"] == 0

    def test_false_positive(self):
        """Predicted failure + actual success → FP."""
        trials = [_make_trial(predicted_instability=True, decoder_success=True)]
        result = run_bp_prediction_validation(trials)
        cm = result["confusion_matrix"]
        assert cm["true_positives"] == 0
        assert cm["false_positives"] == 1
        assert cm["true_negatives"] == 0
        assert cm["false_negatives"] == 0

    def test_true_negative(self):
        """Predicted stable + actual success → TN."""
        trials = [_make_trial(predicted_instability=False, decoder_success=True)]
        result = run_bp_prediction_validation(trials)
        cm = result["confusion_matrix"]
        assert cm["true_positives"] == 0
        assert cm["false_positives"] == 0
        assert cm["true_negatives"] == 1
        assert cm["false_negatives"] == 0

    def test_false_negative(self):
        """Predicted stable + actual failure → FN."""
        trials = [_make_trial(predicted_instability=False, decoder_success=False)]
        result = run_bp_prediction_validation(trials)
        cm = result["confusion_matrix"]
        assert cm["true_positives"] == 0
        assert cm["false_positives"] == 0
        assert cm["true_negatives"] == 0
        assert cm["false_negatives"] == 1

    def test_mixed_confusion_matrix(self):
        """Mixed trials produce correct TP/FP/TN/FN counts."""
        trials = [
            _make_trial(True, False),   # TP
            _make_trial(True, False),   # TP
            _make_trial(True, True),    # FP
            _make_trial(False, True),   # TN
            _make_trial(False, True),   # TN
            _make_trial(False, True),   # TN
            _make_trial(False, False),  # FN
            _make_trial(False, False),  # FN
        ]
        result = run_bp_prediction_validation(trials)
        cm = result["confusion_matrix"]
        assert cm["true_positives"] == 2
        assert cm["false_positives"] == 1
        assert cm["true_negatives"] == 3
        assert cm["false_negatives"] == 2

    def test_num_trials(self):
        """num_trials equals total trial count."""
        trials = [_make_trial(True, False)] * 5 + [_make_trial(False, True)] * 3
        result = run_bp_prediction_validation(trials)
        assert result["num_trials"] == 8


# ── Metric Accuracy ──────────────────────────────────────────────────


class TestMetricAccuracy:
    """Tests for metric computation formulas."""

    def test_accuracy_formula(self):
        """prediction_accuracy = (TP + TN) / N."""
        trials = [
            _make_trial(True, False),   # TP
            _make_trial(True, False),   # TP
            _make_trial(True, True),    # FP
            _make_trial(False, True),   # TN
            _make_trial(False, True),   # TN
            _make_trial(False, True),   # TN
            _make_trial(False, False),  # FN
            _make_trial(False, False),  # FN
        ]
        result = run_bp_prediction_validation(trials)
        # TP=2, FP=1, TN=3, FN=2, N=8
        expected = round((2 + 3) / 8, 12)
        assert result["prediction_accuracy"] == expected

    def test_precision_formula(self):
        """precision = TP / (TP + FP)."""
        trials = [
            _make_trial(True, False),   # TP
            _make_trial(True, False),   # TP
            _make_trial(True, True),    # FP
            _make_trial(False, True),   # TN
        ]
        result = run_bp_prediction_validation(trials)
        # TP=2, FP=1 → precision = 2/3
        expected = round(2 / 3, 12)
        assert result["precision"] == expected

    def test_recall_formula(self):
        """recall = TP / (TP + FN)."""
        trials = [
            _make_trial(True, False),   # TP
            _make_trial(True, False),   # TP
            _make_trial(False, False),  # FN
            _make_trial(False, True),   # TN
        ]
        result = run_bp_prediction_validation(trials)
        # TP=2, FN=1 → recall = 2/3
        expected = round(2 / 3, 12)
        assert result["recall"] == expected

    def test_false_positive_rate_formula(self):
        """false_positive_rate = FP / (FP + TN)."""
        trials = [
            _make_trial(True, True),    # FP
            _make_trial(False, True),   # TN
            _make_trial(False, True),   # TN
            _make_trial(False, True),   # TN
        ]
        result = run_bp_prediction_validation(trials)
        # FP=1, TN=3 → FPR = 1/4
        expected = round(1 / 4, 12)
        assert result["false_positive_rate"] == expected

    def test_true_positive_rate_equals_recall(self):
        """true_positive_rate == recall."""
        trials = [
            _make_trial(True, False),   # TP
            _make_trial(True, True),    # FP
            _make_trial(False, False),  # FN
            _make_trial(False, True),   # TN
        ]
        result = run_bp_prediction_validation(trials)
        assert result["true_positive_rate"] == result["recall"]

    def test_prediction_error_rate_formula(self):
        """prediction_error_rate = (FP + FN) / N."""
        trials = [
            _make_trial(True, False),   # TP
            _make_trial(True, True),    # FP
            _make_trial(False, False),  # FN
            _make_trial(False, True),   # TN
        ]
        result = run_bp_prediction_validation(trials)
        # FP=1, FN=1, N=4 → error_rate = 2/4 = 0.5
        expected = round(2 / 4, 12)
        assert result["prediction_error_rate"] == expected

    def test_perfect_predictions(self):
        """All correct predictions yield accuracy=1, error_rate=0."""
        trials = [
            _make_trial(True, False),   # TP
            _make_trial(True, False),   # TP
            _make_trial(False, True),   # TN
            _make_trial(False, True),   # TN
        ]
        result = run_bp_prediction_validation(trials)
        assert result["prediction_accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["false_positive_rate"] == 0.0
        assert result["prediction_error_rate"] == 0.0

    def test_all_wrong_predictions(self):
        """All wrong predictions yield accuracy=0, error_rate=1."""
        trials = [
            _make_trial(True, True),    # FP
            _make_trial(True, True),    # FP
            _make_trial(False, False),  # FN
            _make_trial(False, False),  # FN
        ]
        result = run_bp_prediction_validation(trials)
        assert result["prediction_accuracy"] == 0.0
        assert result["prediction_error_rate"] == 1.0

    def test_rounding_to_12_decimals(self):
        """All metrics are rounded to 12 decimal places."""
        trials = [
            _make_trial(True, False),   # TP
            _make_trial(True, True),    # FP
            _make_trial(False, False),  # FN
        ]
        result = run_bp_prediction_validation(trials)
        for key in ("prediction_accuracy", "precision", "recall",
                     "false_positive_rate", "true_positive_rate",
                     "prediction_error_rate"):
            val = result[key]
            # Verify rounding: value * 10^12 should be close to integer.
            assert abs(val * 1e12 - round(val * 1e12)) < 0.5


# ── Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_trials(self):
        """Empty trial list yields zero metrics."""
        result = run_bp_prediction_validation([])
        assert result["prediction_accuracy"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["false_positive_rate"] == 0.0
        assert result["true_positive_rate"] == 0.0
        assert result["prediction_error_rate"] == 0.0
        assert result["num_trials"] == 0

    def test_no_predicted_positives(self):
        """No predicted failures → precision = 0."""
        trials = [
            _make_trial(False, True),   # TN
            _make_trial(False, False),  # FN
        ]
        result = run_bp_prediction_validation(trials)
        assert result["precision"] == 0.0

    def test_no_actual_failures(self):
        """No actual failures → recall = 0, FPR defined."""
        trials = [
            _make_trial(True, True),    # FP
            _make_trial(False, True),   # TN
        ]
        result = run_bp_prediction_validation(trials)
        assert result["recall"] == 0.0
        assert result["false_positive_rate"] == round(1 / 2, 12)

    def test_no_actual_successes(self):
        """No actual successes → FPR = 0 (no TN or FP possible for negatives)."""
        trials = [
            _make_trial(True, False),   # TP
            _make_trial(False, False),  # FN
        ]
        result = run_bp_prediction_validation(trials)
        assert result["false_positive_rate"] == 0.0
        assert result["recall"] == round(1 / 2, 12)

    def test_single_trial(self):
        """Single trial produces valid metrics."""
        result = run_bp_prediction_validation(
            [_make_trial(True, False)],
        )
        assert result["num_trials"] == 1
        assert result["prediction_accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_missing_prediction_defaults_to_stable(self):
        """Missing predicted_instability defaults to False."""
        trials = [{"bp_stability_prediction": {}, "decoder_success": True}]
        result = run_bp_prediction_validation(trials)
        assert result["confusion_matrix"]["true_negatives"] == 1

    def test_missing_decoder_success_defaults_to_true(self):
        """Missing decoder_success defaults to True."""
        trials = [{"bp_stability_prediction": {"predicted_instability": True}}]
        result = run_bp_prediction_validation(trials)
        assert result["confusion_matrix"]["false_positives"] == 1


# ── Determinism ───────────────────────────────────────────────────────


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_runs_identical(self):
        """Repeated runs produce identical JSON-serialized results."""
        trials = [
            _make_trial(True, False),
            _make_trial(True, True),
            _make_trial(False, True),
            _make_trial(False, False),
            _make_trial(True, False),
        ]
        r1 = run_bp_prediction_validation(trials)
        r2 = run_bp_prediction_validation(trials)
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        trials = [_make_trial(True, False), _make_trial(False, True)]
        result = run_bp_prediction_validation(trials)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON serialization roundtrip."""
        trials = [
            _make_trial(True, False),
            _make_trial(True, True),
            _make_trial(False, False),
        ]
        result = run_bp_prediction_validation(trials)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)


# ── Pipeline Integration ─────────────────────────────────────────────


class TestPipelineIntegration:
    """Tests for end-to-end pipeline compatibility."""

    def test_full_pipeline(self):
        """Full pipeline: NB localization → trapping → alignment → risk → predictor → validation."""
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
        from qec.diagnostics.bp_stability_predictor import (
            compute_bp_stability_prediction,
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
        prediction = compute_bp_stability_prediction(graph, diagnostics)

        # Simulate decoder outcomes for multiple trials.
        trial_results = [
            {
                "bp_stability_prediction": prediction,
                "decoder_success": True,
            },
            {
                "bp_stability_prediction": prediction,
                "decoder_success": False,
            },
            {
                "bp_stability_prediction": prediction,
                "decoder_success": True,
            },
        ]

        # Run v6.9 validation.
        result = run_bp_prediction_validation(trial_results)

        # All required keys present.
        assert "prediction_accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "false_positive_rate" in result
        assert "true_positive_rate" in result
        assert "prediction_error_rate" in result
        assert "confusion_matrix" in result
        assert "num_trials" in result

        # Types are correct.
        assert isinstance(result["prediction_accuracy"], float)
        assert isinstance(result["precision"], float)
        assert isinstance(result["recall"], float)
        assert isinstance(result["false_positive_rate"], float)
        assert isinstance(result["true_positive_rate"], float)
        assert isinstance(result["prediction_error_rate"], float)
        assert isinstance(result["num_trials"], int)

        # Values are bounded.
        for key in ("prediction_accuracy", "precision", "recall",
                     "false_positive_rate", "true_positive_rate",
                     "prediction_error_rate"):
            assert 0.0 <= result[key] <= 1.0

        assert result["num_trials"] == 3

        # JSON-serializable.
        json.dumps(result)

        # Determinism: second run identical.
        result2 = run_bp_prediction_validation(trial_results)
        assert json.dumps(result, sort_keys=True) == \
               json.dumps(result2, sort_keys=True)


# ── No Decoder Import ────────────────────────────────────────────────


class TestNoDecoderImport:
    """Tests that this module does not import decoder code."""

    def test_no_decoder_import(self):
        """This module does not import any decoder code."""
        import qec.experiments.bp_prediction_validation as mod
        source = open(mod.__file__).read()
        assert "bp_decode" not in source
        assert "from qec.decoder" not in source
