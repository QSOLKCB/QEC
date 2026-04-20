from __future__ import annotations

import math

import pytest

from qec.analysis.resonance_lock_diagnostic_kernel import (
    ResonanceDiagnosticPolicy,
    run_resonance_lock_diagnostic,
)


def test_determinism_same_input_same_bytes_and_hash() -> None:
    kwargs = {
        "state_sequence": ("A", "A", "A", "B", "B", "A", "A"),
        "drift_sequence": (0.01, 0.01, 0.02, 0.4, 0.4, 0.01, 0.01),
    }
    a = run_resonance_lock_diagnostic(**kwargs)
    b = run_resonance_lock_diagnostic(**kwargs)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_material_input_change_changes_receipt_or_hash() -> None:
    a = run_resonance_lock_diagnostic(
        state_sequence=(1, 1, 1, 2, 2, 3),
        drift_sequence=(0.01, 0.01, 0.2, 0.2, 0.2, 0.2),
    )
    b = run_resonance_lock_diagnostic(
        state_sequence=(1, 2, 3, 4, 5, 6),
        drift_sequence=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    )
    assert a.to_canonical_bytes() != b.to_canonical_bytes()
    assert a.stable_hash() != b.stable_hash()


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="state_sequence must be non-empty"):
        run_resonance_lock_diagnostic(state_sequence=())

    with pytest.raises(ValueError, match="drift_sequence length"):
        run_resonance_lock_diagnostic(state_sequence=("x", "y"), drift_sequence=(0.1, 0.2, 0.3, 0.4))

    with pytest.raises(ValueError, match="must be finite"):
        run_resonance_lock_diagnostic(state_sequence=("x", "y"), drift_sequence=(0.1, math.inf))

    with pytest.raises(ValueError, match="policy min_lock_span_length"):
        run_resonance_lock_diagnostic(
            state_sequence=("x", "x"),
            policy=ResonanceDiagnosticPolicy(min_lock_span_length=1),
        )


def test_lock_detection_contiguous_and_ordered() -> None:
    receipt = run_resonance_lock_diagnostic(
        state_sequence=("a", "a", "a", "b", "c", "c", "c", "d"),
        drift_sequence=(0.01, 0.01, 0.01, 0.7, 0.01, 0.01, 0.01, 0.7),
        policy=ResonanceDiagnosticPolicy(min_lock_span_length=2, drift_lock_threshold=0.05),
    )
    assert len(receipt.lock_spans) >= 2
    assert tuple((s.start_index, s.end_index) for s in receipt.lock_spans) == tuple(
        sorted((s.start_index, s.end_index) for s in receipt.lock_spans)
    )


def test_low_drift_increases_lock_strength() -> None:
    low = run_resonance_lock_diagnostic(
        state_sequence=("x", "x", "x", "y", "y", "y"),
        drift_sequence=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
    )
    high = run_resonance_lock_diagnostic(
        state_sequence=("x", "x", "x", "y", "y", "y"),
        drift_sequence=(0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
    )
    assert low.bounded_metrics["lock_strength_score"] > high.bounded_metrics["lock_strength_score"]


def test_dominant_state_tie_break_is_deterministic() -> None:
    receipt = run_resonance_lock_diagnostic(
        state_sequence=("b", "a", "b", "a", "b", "a"),
        drift_sequence=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
    )
    assert receipt.lock_spans
    assert receipt.lock_spans[0].dominant_state == "a"


def test_attractor_classification_single_and_multi_and_dispersed() -> None:
    single = run_resonance_lock_diagnostic(
        state_sequence=("z", "z", "z", "z", "z", "a"),
        drift_sequence=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert single.resonance_classification == "single_attractor_lock"

    multi = run_resonance_lock_diagnostic(
        state_sequence=("a", "a", "a", "b", "b", "b", "a", "b"),
        drift_sequence=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        policy=ResonanceDiagnosticPolicy(single_attractor_threshold=0.8),
    )
    assert multi.resonance_classification in {"multi_attractor_lock", "weak_lock_field"}

    dispersed = run_resonance_lock_diagnostic(
        state_sequence=(1, 2, 3, 4, 5, 6, 7),
        drift_sequence=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert dispersed.resonance_classification in {"dispersed_field", "weak_lock_field"}


def test_bounded_metrics_and_invariants() -> None:
    receipt = run_resonance_lock_diagnostic(
        state_sequence=(1, 1, 2, 2, 1, 1),
        drift_sequence=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    )
    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0
    assert receipt.advisory_only is True
    assert receipt.decoder_core_modified is False


def test_canonical_json_and_hash_stability() -> None:
    receipt = run_resonance_lock_diagnostic(
        state_sequence=("a", "a", "b", "b", "a"),
        drift_sequence=(0.1, 0.1, 0.1, 0.1, 0.1),
    )
    assert receipt.to_canonical_json() == receipt.to_canonical_json()
    ref = receipt.stable_hash()
    for _ in range(5):
        assert run_resonance_lock_diagnostic(
            state_sequence=("a", "a", "b", "b", "a"),
            drift_sequence=(0.1, 0.1, 0.1, 0.1, 0.1),
        ).stable_hash() == ref


def test_mapping_like_fields_are_immutable() -> None:
    receipt = run_resonance_lock_diagnostic(
        state_sequence=("a", "a", "b", "b", "a"),
        drift_sequence=(0.1, 0.1, 0.1, 0.1, 0.1),
    )
    with pytest.raises(TypeError):
        receipt.bounded_metrics["lock_strength_score"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        receipt.attractor_profile.occupancy_counts["s:a"] = 999  # type: ignore[index]


def test_single_state_empty_drift_no_error() -> None:
    """Regression: transition-aligned empty drift for a single state must not raise IndexError."""
    receipt = run_resonance_lock_diagnostic(
        state_sequence=("x",),
        drift_sequence=(),
    )
    assert receipt.trajectory_length == 1
    assert receipt.lock_spans == ()
