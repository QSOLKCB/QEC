from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.phase_lock_state_transition_kernel import (
    PhaseTransition,
    compute_lock_strength,
    export_phase_transition_bytes,
    generate_phase_transition_receipt,
    synthesize_phase_transition,
)


def _hex(char: str) -> str:
    return char * 64


def _build_transition() -> PhaseTransition:
    return synthesize_phase_transition(
        source_field_hash=_hex("1"),
        target_field_hash=_hex("2"),
        parent_coherence_field_hash=_hex("a"),
    )


def test_repeated_run_determinism() -> None:
    t1 = _build_transition()
    t2 = _build_transition()
    assert t1 == t2


def test_identical_inputs_produce_identical_bytes() -> None:
    t1 = _build_transition()
    t2 = _build_transition()
    assert export_phase_transition_bytes(t1) == export_phase_transition_bytes(t2)


def test_bounded_lock_strength_scores() -> None:
    lock_strength, stability = compute_lock_strength(_hex("f"), _hex("0"))

    assert 0.0 <= lock_strength <= 1.0
    assert 0.0 <= stability <= 1.0


def test_stable_transition_hash() -> None:
    t1 = _build_transition()
    t2 = _build_transition()
    assert t1.stable_transition_hash == t2.stable_transition_hash


def test_receipt_stability() -> None:
    r1 = generate_phase_transition_receipt(_build_transition())
    r2 = generate_phase_transition_receipt(_build_transition())
    assert r1 == r2


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError, match="source_field_hash must be 64 hex characters"):
        synthesize_phase_transition(
            source_field_hash="123",
            target_field_hash=_hex("2"),
            parent_coherence_field_hash=_hex("a"),
        )

    with pytest.raises(ValueError, match="target_field_hash must be hexadecimal"):
        synthesize_phase_transition(
            source_field_hash=_hex("1"),
            target_field_hash="z" * 64,
            parent_coherence_field_hash=_hex("a"),
        )

    with pytest.raises(ValueError, match="parent_coherence_field_hash must be 64 hex characters"):
        synthesize_phase_transition(
            source_field_hash=_hex("1"),
            target_field_hash=_hex("2"),
            parent_coherence_field_hash="abc",
        )

    transition = _build_transition()

    tampered_hash = replace(transition, stable_transition_hash=_hex("b"))
    with pytest.raises(ValueError, match="stable_transition_hash mismatch"):
        export_phase_transition_bytes(tampered_hash)

    tampered_chain = replace(
        transition,
        transition_identity_chain=(_hex("b"), transition.transition_identity_chain[1]),
    )
    with pytest.raises(ValueError, match="transition_identity_chain must start with parent_coherence_field_hash"):
        export_phase_transition_bytes(tampered_chain)

    tampered_score = replace(transition, phase_lock_strength=1.1)
    with pytest.raises(ValueError, match=r"phase_lock_strength must be in \[0, 1\]"):
        export_phase_transition_bytes(tampered_score)
