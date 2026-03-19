"""
Tests for v5.7.0 — Ternary Decoder Topology Classifier.

Verifies correct classification into stable success (+1),
boundary/metastable (0), and failure basin (-1) states,
deterministic repeated execution, JSON serialization stability,
and graceful handling of absent optional evidence.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qec.diagnostics.bp_phase_space import compute_bp_phase_space
from qec.diagnostics.ternary_decoder_topology import (
    compute_ternary_decoder_topology,
)


# ── Test helpers ──────────────────────────────────────────────────────

def _success_phase_space() -> dict:
    """Phase-space result for a converged, successful decode.

    Uses exponential decay to ensure residual norms fall well below
    the default residual_epsilon=1e-6.
    """
    fixed_point = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    states = []
    for t in range(10):
        perturbation = np.array([1.0, 0.5, 0.3], dtype=np.float64) * (0.001 ** t)
        states.append(fixed_point + perturbation)
    return compute_bp_phase_space(states)


def _failure_phase_space() -> dict:
    """Phase-space result for a converged but failed decode.

    Converges to a fixed point (residuals → 0) but syndrome remains
    unsatisfied.
    """
    fixed_point = np.array([1.5, 0.8, 0.4], dtype=np.float64)
    states = []
    for t in range(10):
        perturbation = np.array([2.0, 1.0, 0.6], dtype=np.float64) * (0.001 ** t)
        states.append(fixed_point + perturbation)
    return compute_bp_phase_space(states)


def _oscillating_phase_space() -> dict:
    """Phase-space result for an oscillating, non-converging decode."""
    states = []
    for t in range(20):
        if t % 2 == 0:
            states.append(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        else:
            states.append(np.array([3.0, 2.0, 1.0], dtype=np.float64))
    return compute_bp_phase_space(states)


def _minimal_phase_space() -> dict:
    """Minimal phase-space result for edge-case testing."""
    return {
        "trajectory_length": 1,
        "state_dimension": 3,
        "residual_norms": [],
        "phase_coordinates": [[1.0, 2.0, 3.0]],
        "final_phase_coordinate": [1.0, 2.0, 3.0],
        "oscillation_score": 0.0,
    }


# ── Classification tests ────────────────────────────────────────────

class TestStableSuccess:
    """Verify +1 (stable success) classification."""

    def test_syndrome_zero_converged(self):
        ps = _success_phase_space()
        syndrome_residuals = [0] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        assert out["final_ternary_state"] == 1

    def test_classification_reason_mentions_success(self):
        ps = _success_phase_space()
        syndrome_residuals = [0] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        assert "success" in out["classification_reason"].lower()

    def test_ternary_trace_contains_positive(self):
        ps = _success_phase_space()
        syndrome_residuals = [0] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        assert 1 in out["ternary_trace"]


class TestFailureBasin:
    """Verify -1 (failure basin) classification."""

    def test_syndrome_nonzero_converged(self):
        ps = _failure_phase_space()
        syndrome_residuals = [3] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        assert out["final_ternary_state"] == -1

    def test_classification_reason_mentions_failure(self):
        ps = _failure_phase_space()
        syndrome_residuals = [3] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        assert "failure" in out["classification_reason"].lower()

    def test_ternary_trace_contains_negative(self):
        ps = _failure_phase_space()
        syndrome_residuals = [3] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        assert -1 in out["ternary_trace"]


class TestBoundaryMetastable:
    """Verify 0 (boundary/metastable) classification."""

    def test_oscillating_is_boundary(self):
        ps = _oscillating_phase_space()
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
        )
        assert out["final_ternary_state"] == 0

    def test_no_syndrome_info_boundary(self):
        ps = _success_phase_space()
        # No syndrome residuals → cannot assign basin.
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
        )
        assert out["final_ternary_state"] == 0

    def test_classification_reason_mentions_boundary(self):
        ps = _oscillating_phase_space()
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
        )
        assert "boundary" in out["classification_reason"].lower() or \
               "metastable" in out["classification_reason"].lower()


# ── State duration tests ─────────────────────────────────────────────

class TestStateDurations:
    """Verify correct state duration counting."""

    def test_durations_sum_to_length(self):
        ps = _success_phase_space()
        syndrome_residuals = [0] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        total = (
            out["state_durations"]["positive"]
            + out["state_durations"]["zero"]
            + out["state_durations"]["negative"]
        )
        assert total == ps["trajectory_length"]

    def test_all_success_positive_duration(self):
        ps = _success_phase_space()
        syndrome_residuals = [0] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        assert out["state_durations"]["positive"] > 0


# ── Final state tests ────────────────────────────────────────────────

class TestFinalState:
    """Verify correct final state assignment."""

    def test_final_state_is_int(self):
        ps = _success_phase_space()
        out = compute_ternary_decoder_topology(phase_space_result=ps)
        assert isinstance(out["final_ternary_state"], int)

    def test_final_state_in_valid_range(self):
        ps = _success_phase_space()
        out = compute_ternary_decoder_topology(phase_space_result=ps)
        assert out["final_ternary_state"] in (-1, 0, 1)

    def test_ternary_trace_values_valid(self):
        ps = _oscillating_phase_space()
        syndrome_residuals = [2] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        for s in out["ternary_trace"]:
            assert s in (-1, 0, 1)


# ── Transition iteration tests ──────────────────────────────────────

class TestTransitionIteration:
    """Verify transition iteration detection."""

    def test_constant_success_transition_at_zero(self):
        ps = _success_phase_space()
        syndrome_residuals = [0] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        if out["transition_iteration"] is not None:
            assert out["transition_iteration"] >= 0
            assert out["transition_iteration"] < ps["trajectory_length"]

    def test_transition_iteration_type(self):
        ps = _success_phase_space()
        out = compute_ternary_decoder_topology(phase_space_result=ps)
        assert out["transition_iteration"] is None or isinstance(
            out["transition_iteration"], int
        )


# ── Evidence tests ───────────────────────────────────────────────────

class TestEvidence:
    """Verify evidence values are correctly populated."""

    def test_evidence_keys_present(self):
        ps = _success_phase_space()
        out = compute_ternary_decoder_topology(phase_space_result=ps)
        evidence = out["evidence"]
        assert "residual_norm_final" in evidence
        assert "oscillation_score" in evidence
        assert "alignment_max_final" in evidence
        assert "boundary_eps_final" in evidence
        assert "barrier_eps_final" in evidence
        assert "syndrome_residual_final" in evidence

    def test_evidence_none_when_absent(self):
        ps = _success_phase_space()
        out = compute_ternary_decoder_topology(phase_space_result=ps)
        assert out["evidence"]["alignment_max_final"] is None
        assert out["evidence"]["boundary_eps_final"] is None
        assert out["evidence"]["barrier_eps_final"] is None
        assert out["evidence"]["syndrome_residual_final"] is None

    def test_evidence_populated_with_optional(self):
        ps = _success_phase_space()
        alignment = {"max_alignment": 0.85, "mean_alignment": 0.5}
        boundary = {"boundary_eps": 0.01}
        barrier = {"barrier_eps": 0.05}
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            alignment_result=alignment,
            boundary_result=boundary,
            barrier_result=barrier,
        )
        assert out["evidence"]["alignment_max_final"] == 0.85
        assert out["evidence"]["boundary_eps_final"] == 0.01
        assert out["evidence"]["barrier_eps_final"] == 0.05

    def test_syndrome_residual_populated(self):
        ps = _success_phase_space()
        syndrome_residuals = [3, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        assert out["evidence"]["syndrome_residual_final"] == 0


# ── Input validation tests ──────────────────────────────────────────

class TestInputValidation:
    """Verify error handling for invalid inputs."""

    def test_zero_trajectory_raises(self):
        ps = {"trajectory_length": 0, "residual_norms": [], "oscillation_score": 0.0}
        with pytest.raises(ValueError, match="zero trajectory_length"):
            compute_ternary_decoder_topology(phase_space_result=ps)


# ── Determinism tests ────────────────────────────────────────────────

class TestDeterminism:
    """Verify byte-identical outputs across repeated calls."""

    def test_success_determinism(self):
        ps = _success_phase_space()
        syndrome_residuals = [0] * ps["trajectory_length"]
        out1 = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        out2 = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_failure_determinism(self):
        ps = _failure_phase_space()
        syndrome_residuals = [3] * ps["trajectory_length"]
        out1 = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        out2 = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_boundary_determinism(self):
        ps = _oscillating_phase_space()
        out1 = compute_ternary_decoder_topology(phase_space_result=ps)
        out2 = compute_ternary_decoder_topology(phase_space_result=ps)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2


# ── JSON serialization tests ────────────────────────────────────────

class TestJsonSerialization:
    """Verify JSON roundtrip stability."""

    def test_json_roundtrip_success(self):
        ps = _success_phase_space()
        syndrome_residuals = [0] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        s = json.dumps(out, sort_keys=True)
        roundtrip = json.loads(s)
        assert roundtrip == out

    def test_json_roundtrip_failure(self):
        ps = _failure_phase_space()
        syndrome_residuals = [3] * ps["trajectory_length"]
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            syndrome_residuals=syndrome_residuals,
        )
        s = json.dumps(out, sort_keys=True)
        roundtrip = json.loads(s)
        assert roundtrip == out

    def test_json_roundtrip_with_optional(self):
        ps = _success_phase_space()
        alignment = {"max_alignment": 0.85}
        out = compute_ternary_decoder_topology(
            phase_space_result=ps,
            alignment_result=alignment,
        )
        s = json.dumps(out, sort_keys=True)
        roundtrip = json.loads(s)
        assert roundtrip == out

    def test_all_values_json_types(self):
        ps = _success_phase_space()
        out = compute_ternary_decoder_topology(phase_space_result=ps)
        s = json.dumps(out)
        assert isinstance(s, str)


# ── Output structure tests ──────────────────────────────────────────

class TestOutputStructure:
    """Verify all required keys are present."""

    REQUIRED_KEYS = [
        "ternary_trace",
        "final_ternary_state",
        "state_durations",
        "transition_iteration",
        "classification_reason",
        "evidence",
    ]

    def test_all_keys_present(self):
        ps = _success_phase_space()
        out = compute_ternary_decoder_topology(phase_space_result=ps)
        for key in self.REQUIRED_KEYS:
            assert key in out, f"Missing key: {key}"

    def test_state_durations_keys(self):
        ps = _success_phase_space()
        out = compute_ternary_decoder_topology(phase_space_result=ps)
        assert "positive" in out["state_durations"]
        assert "zero" in out["state_durations"]
        assert "negative" in out["state_durations"]

    def test_evidence_keys(self):
        ps = _success_phase_space()
        out = compute_ternary_decoder_topology(phase_space_result=ps)
        expected = [
            "residual_norm_final",
            "oscillation_score",
            "alignment_max_final",
            "boundary_eps_final",
            "barrier_eps_final",
            "syndrome_residual_final",
        ]
        for key in expected:
            assert key in out["evidence"], f"Missing evidence key: {key}"

    def test_ternary_trace_length(self):
        ps = _success_phase_space()
        out = compute_ternary_decoder_topology(phase_space_result=ps)
        assert len(out["ternary_trace"]) == ps["trajectory_length"]
