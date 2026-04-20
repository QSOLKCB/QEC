from __future__ import annotations

import math

import pytest

from qec.analysis.phase_coherence_audit_layer import (
    PhaseCoherenceAuditPolicy,
    run_phase_coherence_audit,
)
from qec.analysis.resonance_lock_diagnostic_kernel import run_resonance_lock_diagnostic


def test_determinism_same_input_same_bytes_and_hash() -> None:
    kwargs = {
        "state_sequence": ("A", "A", "A", "B", "B", "A"),
        "phase_sequence": (0.01, 0.02, 0.02, 0.5, 0.52, 0.02),
    }
    a = run_phase_coherence_audit(**kwargs)
    b = run_phase_coherence_audit(**kwargs)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_material_input_change_changes_receipt_or_hash() -> None:
    a = run_phase_coherence_audit(
        state_sequence=(1, 1, 1, 2, 2),
        phase_sequence=(0.0, 0.0, 0.01, 0.01, 0.0),
    )
    b = run_phase_coherence_audit(
        state_sequence=(1, 2, 3, 4, 5),
        phase_sequence=(0.0, 0.9, -0.9, 0.9, -0.9),
    )
    assert a.to_canonical_bytes() != b.to_canonical_bytes()
    assert a.stable_hash() != b.stable_hash()


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="state_sequence must be non-empty"):
        run_phase_coherence_audit(state_sequence=(), phase_sequence=(0.1,))

    with pytest.raises(ValueError, match="phase_sequence must be non-empty"):
        run_phase_coherence_audit(state_sequence=("x",), phase_sequence=())

    with pytest.raises(ValueError, match="phase_sequence length"):
        run_phase_coherence_audit(state_sequence=("x", "y"), phase_sequence=(0.1,))

    with pytest.raises(ValueError, match="must be finite"):
        run_phase_coherence_audit(state_sequence=("x", "y"), phase_sequence=(0.1, math.inf))

    with pytest.raises(ValueError, match="policy min_window_length"):
        run_phase_coherence_audit(
            state_sequence=("x", "x"),
            phase_sequence=(0.1, 0.1),
            policy=PhaseCoherenceAuditPolicy(min_window_length=1),
        )


def test_malformed_and_wrong_version_resonance_rejected() -> None:
    with pytest.raises(ValueError, match="missing field 'trajectory_length'"):
        run_phase_coherence_audit(
            state_sequence=("a", "a"),
            phase_sequence=("p", "p"),
            resonance_receipt={"release_version": "v138.5.0", "diagnostic_kind": "resonance_lock_diagnostic_kernel"},
        )

    valid = run_resonance_lock_diagnostic(state_sequence=("a", "a"), drift_sequence=(0.0, 0.0)).to_dict()
    valid["release_version"] = "v138.4.9"
    with pytest.raises(ValueError, match="release_version"):
        run_phase_coherence_audit(
            state_sequence=("a", "a"),
            phase_sequence=("p", "p"),
            resonance_receipt=valid,
        )


def test_stable_numeric_region_yields_coherence_window() -> None:
    receipt = run_phase_coherence_audit(
        state_sequence=(1, 1, 1, 1, 1),
        phase_sequence=(0.0, 0.02, 0.01, 0.03, 0.02),
    )
    assert receipt.coherence_windows


def test_symbolic_repetition_yields_coherence_window() -> None:
    receipt = run_phase_coherence_audit(
        state_sequence=("s1", "s2", "s3", "s4"),
        phase_sequence=("alpha", "alpha", "beta", "beta"),
    )
    assert receipt.coherence_windows


def test_sharp_phase_jump_yields_break_span() -> None:
    receipt = run_phase_coherence_audit(
        state_sequence=(1, 2, 3, 4),
        phase_sequence=(0.0, 0.01, 2.0, 2.01),
    )
    assert receipt.phase_break_spans


def test_multiple_windows_and_breaks_are_deterministic_ordered() -> None:
    receipt = run_phase_coherence_audit(
        state_sequence=(0, 1, 2, 3, 4, 5),
        phase_sequence=(0.0, 0.0, 1.0, 1.0, 2.0, 2.0),
    )
    assert len(receipt.coherence_windows) >= 2
    assert tuple((w.start_index, w.end_index) for w in receipt.coherence_windows) == tuple(
        sorted((w.start_index, w.end_index) for w in receipt.coherence_windows)
    )
    assert tuple((b.start_index, b.end_index) for b in receipt.phase_break_spans) == tuple(
        sorted((b.start_index, b.end_index) for b in receipt.phase_break_spans)
    )


def test_dominant_phase_tie_break_is_deterministic() -> None:
    receipt = run_phase_coherence_audit(
        state_sequence=(0, 1, 2, 3),
        phase_sequence=("b", "a", "b", "a"),
        policy=PhaseCoherenceAuditPolicy(min_window_length=2),
    )
    assert receipt.coherence_windows
    assert receipt.coherence_windows[0].dominant_phase == "a"


def test_alignment_score_higher_when_overlapping_lock_spans() -> None:
    states = ("a", "a", "a", "b", "b", "b")
    resonance = run_resonance_lock_diagnostic(
        state_sequence=states,
        drift_sequence=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    overlapping = run_phase_coherence_audit(
        state_sequence=states,
        phase_sequence=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05),
        resonance_receipt=resonance,
    )
    non_overlapping = run_phase_coherence_audit(
        state_sequence=states,
        phase_sequence=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        resonance_receipt=resonance,
    )
    assert (
        overlapping.bounded_metrics["phase_lock_alignment_score"]
        > non_overlapping.bounded_metrics["phase_lock_alignment_score"]
    )


def test_no_resonance_receipt_still_valid() -> None:
    receipt = run_phase_coherence_audit(
        state_sequence=(1, 2, 3),
        phase_sequence=("x", "x", "y"),
    )
    assert receipt.resonance_source_identity is None


def test_classification_behavior() -> None:
    strong = run_phase_coherence_audit(
        state_sequence=(0, 1, 2, 3, 4),
        phase_sequence=(0.0, 0.01, 0.01, 0.02, 0.02),
    )
    assert strong.coherence_classification == "strong_phase_coherence"

    partial = run_phase_coherence_audit(
        state_sequence=(0, 1, 2, 3, 4, 5),
        phase_sequence=(0.0, 0.0, 1.0, 1.01, 2.0, 2.01),
        policy=PhaseCoherenceAuditPolicy(localized_coherence_threshold=0.4),
    )
    assert partial.coherence_classification in {"localized_phase_coherence", "weak_phase_structure"}

    fragmented = run_phase_coherence_audit(
        state_sequence=(0, 1, 2, 3, 4, 5),
        phase_sequence=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    )
    assert fragmented.coherence_classification in {"fragmented_phase_field", "phase_incoherent"}


def test_bounded_metrics_and_invariants() -> None:
    receipt = run_phase_coherence_audit(
        state_sequence=(1, 1, 2, 2, 1, 1),
        phase_sequence=(0.0, 0.1, 0.0, 0.1, 0.0, 0.1),
    )
    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0
    assert receipt.advisory_only is True
    assert receipt.decoder_core_modified is False


def test_canonical_json_and_hash_stability() -> None:
    receipt = run_phase_coherence_audit(
        state_sequence=("a", "a", "b", "b", "a"),
        phase_sequence=("p", "p", "q", "q", "p"),
    )
    assert receipt.to_canonical_json() == receipt.to_canonical_json()
    ref = receipt.stable_hash()
    for _ in range(5):
        assert (
            run_phase_coherence_audit(
                state_sequence=("a", "a", "b", "b", "a"),
                phase_sequence=("p", "p", "q", "q", "p"),
            ).stable_hash()
            == ref
        )


def test_mapping_like_fields_are_immutable() -> None:
    receipt = run_phase_coherence_audit(
        state_sequence=("a", "a", "b"),
        phase_sequence=("x", "x", "y"),
    )
    with pytest.raises(TypeError):
        receipt.bounded_metrics["phase_coherence_score"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        receipt.input_summary["input_hash"] = "oops"  # type: ignore[index]


def test_tampered_resonance_source_rejected_by_hash_mismatch() -> None:
    source = run_resonance_lock_diagnostic(
        state_sequence=("a", "a", "a", "b"),
        drift_sequence=(0.0, 0.0, 0.0, 0.0),
    ).to_dict()
    source["resonance_classification"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        run_phase_coherence_audit(
            state_sequence=("a", "a", "a", "b"),
            phase_sequence=(0.0, 0.0, 0.1, 0.1),
            resonance_receipt=source,
        )


def test_resonance_source_binding_changes_receipt_hash() -> None:
    states = ("a", "a", "b", "b")
    source_a = run_resonance_lock_diagnostic(state_sequence=states, drift_sequence=(0.0, 0.0, 0.0, 0.0))
    source_b = run_resonance_lock_diagnostic(state_sequence=states, drift_sequence=(0.8, 0.8, 0.8, 0.8))

    receipt_a = run_phase_coherence_audit(
        state_sequence=states,
        phase_sequence=(0.0, 0.0, 0.1, 0.1),
        resonance_receipt=source_a,
    )
    receipt_b = run_phase_coherence_audit(
        state_sequence=states,
        phase_sequence=(0.0, 0.0, 0.1, 0.1),
        resonance_receipt=source_b,
    )
    assert receipt_a.stable_hash() != receipt_b.stable_hash()
