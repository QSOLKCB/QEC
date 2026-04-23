from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.state_conditioned_filter_mesh import (
    CLASSIFICATION_CONFLICTED,
    CLASSIFICATION_STABLE_PREFERENCE,
    FilterMeshReceipt,
    FilterMeshState,
    FilterOrdering,
    score_filter_mesh,
)


def _state(**overrides: float | str) -> FilterMeshState:
    base: dict[str, float | str] = {
        "invariant_class": "parity",
        "geometry_class": "surface",
        "spectral_regime": "high_frequency",
        "hardware_class": "superconducting",
        "recurrence_class": "oscillatory",
        "thermal_pressure": 0.5,
        "latency_drift": 0.3,
        "timing_skew": 0.3,
        "power_pressure": 0.2,
        "consensus_instability": 0.4,
    }
    base.update(overrides)
    return FilterMeshState(**base)


def _candidate(input_filters: tuple[str, ...], control_filters: tuple[str, ...]) -> FilterOrdering:
    return FilterOrdering.build(input_filters, control_filters)


def test_deterministic_replay() -> None:
    state = _state()
    candidates = (
        _candidate(("thermal_stabilize", "parity_gate"), ("boundary_control",)),
        _candidate(("latency_buffer", "spectral_phase"), ("surface_sync",)),
    )
    first = score_filter_mesh(state, candidates)
    second = score_filter_mesh(state, candidates)
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_canonical_candidate_tie_break() -> None:
    state = _state(recurrence_class="stable", consensus_instability=0.1, thermal_pressure=0.2, latency_drift=0.2, timing_skew=0.2, power_pressure=0.2)
    a = _candidate(("generic_a",), ("generic_ctrl",))
    b = _candidate(("generic_b",), ("generic_ctrl",))
    receipt = score_filter_mesh(state, (b, a))
    assert receipt.ordered_scores[0].total_score == receipt.ordered_scores[1].total_score
    assert tuple(item.ordering_signature for item in receipt.ordered_scores) == tuple(sorted((a.ordering_signature, b.ordering_signature)))


def test_dominant_thermal_alignment() -> None:
    state = _state(thermal_pressure=0.95, latency_drift=0.1, timing_skew=0.1, power_pressure=0.1, consensus_instability=0.8)
    thermal = _candidate(("thermal_stabilize", "parity_gate"), ("boundary_control",))
    latency = _candidate(("latency_buffer", "parity_gate"), ("boundary_control",))
    receipt = score_filter_mesh(state, (latency, thermal))
    assert receipt.ordered_scores[0].ordering_signature == thermal.ordering_signature


def test_dominant_latency_alignment() -> None:
    state = _state(thermal_pressure=0.1, latency_drift=0.95, timing_skew=0.1, power_pressure=0.1)
    thermal = _candidate(("thermal_stabilize",), ("boundary_control",))
    latency = _candidate(("latency_buffer",), ("boundary_control",))
    receipt = score_filter_mesh(state, (thermal, latency))
    assert receipt.ordered_scores[0].ordering_signature == latency.ordering_signature


def test_recurrence_penalty() -> None:
    state = _state(recurrence_class="oscillatory")
    amplifying = _candidate(("resonance_amplify",), ("feedback_loop",))
    damping = _candidate(("damp_filter",), ("consensus_stabilize",))
    receipt = score_filter_mesh(state, (amplifying, damping))
    assert receipt.ordered_scores[0].ordering_signature == damping.ordering_signature


def test_classification_stable_preference() -> None:
    state = _state(thermal_pressure=0.95, latency_drift=0.05, timing_skew=0.05, power_pressure=0.05, consensus_instability=0.9)
    strong = _candidate(("thermal_stabilize", "stabilize_core", "parity_sync"), ("boundary_control",))
    weak = _candidate(("generic_filter",), ("generic_control",))
    receipt = score_filter_mesh(state, (weak, strong))
    assert receipt.classification == CLASSIFICATION_STABLE_PREFERENCE


def test_classification_conflicted() -> None:
    state = _state(recurrence_class="stable", consensus_instability=0.2, thermal_pressure=0.25, latency_drift=0.25, timing_skew=0.25, power_pressure=0.25)
    a = _candidate(("generic_a",), ("generic_c",))
    b = _candidate(("generic_b",), ("generic_c",))
    receipt = score_filter_mesh(state, (a, b))
    assert receipt.classification == CLASSIFICATION_CONFLICTED


def test_duplicate_ordering_signature_rejection() -> None:
    state = _state()
    a = _candidate(("thermal",), ("ctrl",))
    dup = replace(a, stable_hash=a.stable_hash)
    with pytest.raises(ValueError, match="duplicate ordering_signature"):
        score_filter_mesh(state, (a, dup))


def test_invalid_metric_rejection() -> None:
    with pytest.raises(ValueError, match=r"within \[0,1\]"):
        _state(thermal_pressure=1.1)


def test_immutable_artifact_enforcement() -> None:
    receipt = score_filter_mesh(_state(), (_candidate(("thermal",), ("ctrl",)),))
    with pytest.raises(FrozenInstanceError):
        receipt.classification = "x"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        receipt.ordered_scores[0].rank = 2  # type: ignore[misc]


def test_stable_hash_self_validation() -> None:
    candidate = _candidate(("thermal",), ("ctrl",))
    with pytest.raises(ValueError, match="stable_hash must match"):
        replace(candidate, stable_hash="0" * 64)

    receipt = score_filter_mesh(_state(), (candidate,))
    payload = receipt.to_dict()
    payload["classification"] = CLASSIFICATION_CONFLICTED
    with pytest.raises(ValueError, match="stable_hash must match"):
        FilterMeshReceipt(
            state=receipt.state,
            candidate_count=receipt.candidate_count,
            ordered_scores=receipt.ordered_scores,
            dominant_ordering_signature=receipt.dominant_ordering_signature,
            dominant_score=receipt.dominant_score,
            classification=CLASSIFICATION_CONFLICTED,
            stable_hash=receipt.stable_hash,
        )


def test_score_boundedness() -> None:
    receipt = score_filter_mesh(
        _state(thermal_pressure=1.0, latency_drift=0.0, timing_skew=1.0, power_pressure=0.0, consensus_instability=1.0),
        (
            _candidate(("thermal_stabilize", "timing_sync"), ("boundary_control",)),
            _candidate(("resonance_amplify", "feedback_loop"), ("generic",)),
        ),
    )
    for score in receipt.ordered_scores:
        assert 0.0 <= score.invariant_alignment <= 1.0
        assert 0.0 <= score.hardware_alignment <= 1.0
        assert 0.0 <= score.recurrence_avoidance <= 1.0
        assert 0.0 <= score.stability_alignment <= 1.0
        assert 0.0 <= score.total_score <= 1.0


def test_point_free_no_external_state() -> None:
    state = _state()
    candidates = (
        _candidate(("thermal_stabilize",), ("boundary_control",)),
        _candidate(("latency_buffer",), ("surface_sync",)),
    )
    first = score_filter_mesh(state, candidates)
    second = score_filter_mesh(state, candidates)
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
