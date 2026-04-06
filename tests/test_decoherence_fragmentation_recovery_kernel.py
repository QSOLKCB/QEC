from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.decoherence_fragmentation_recovery_kernel import (
    DecoherenceState,
    compute_fragmentation_score,
    detect_fragmentation,
    export_recovery_bytes,
    generate_recovery_receipt,
    synthesize_recovery_state,
    validate_recovery_artifact,
)


def _hex(char: str) -> str:
    return char * 64


def _state() -> DecoherenceState:
    return DecoherenceState(
        state_id="state-alpha",
        field_amplitudes=(0.9, 0.4, 0.1, 0.95, 0.98),
        coherence_profile=(0.95, 0.88, 0.21, 0.22, 0.93),
    )


def test_repeated_run_determinism() -> None:
    state = _state()
    artifact_a = synthesize_recovery_state(state, parent_transition_hash=_hex("a"))
    artifact_b = synthesize_recovery_state(state, parent_transition_hash=_hex("a"))

    assert artifact_a == artifact_b
    assert validate_recovery_artifact(artifact_a)
    assert validate_recovery_artifact(artifact_b)


def test_identical_inputs_identical_bytes() -> None:
    state = _state()
    artifact_a = synthesize_recovery_state(state, parent_transition_hash=_hex("b"))
    artifact_b = synthesize_recovery_state(state, parent_transition_hash=_hex("b"))

    assert export_recovery_bytes(artifact_a) == export_recovery_bytes(artifact_b)


def test_bounded_fragmentation_scores() -> None:
    fragmented = _state()
    boundaries = detect_fragmentation(fragmented)
    score = compute_fragmentation_score(fragmented, boundaries)
    assert 0.0 <= score <= 1.0

    coherent = DecoherenceState(
        state_id="coherent",
        field_amplitudes=(0.1, 0.2, 0.3),
        coherence_profile=(0.4, 0.45, 0.5),
    )
    coherent_boundaries = detect_fragmentation(coherent)
    coherent_score = compute_fragmentation_score(coherent, coherent_boundaries)

    assert coherent_boundaries == ()
    assert coherent_score == 0.0


def test_stable_recovery_hash_and_receipt_stability() -> None:
    state = _state()
    artifact_a = synthesize_recovery_state(state, parent_transition_hash=_hex("c"))
    artifact_b = synthesize_recovery_state(state, parent_transition_hash=_hex("c"))

    assert artifact_a.stable_recovery_hash == artifact_b.stable_recovery_hash

    receipt_a = generate_recovery_receipt(artifact_a)
    receipt_b = generate_recovery_receipt(artifact_b)
    assert receipt_a == receipt_b


def test_tamper_detection() -> None:
    artifact = synthesize_recovery_state(_state(), parent_transition_hash=_hex("d"))
    tampered = replace(artifact, fragmentation_score=min(1.0, artifact.fragmentation_score + 0.1))

    with pytest.raises(ValueError, match="stable_recovery_hash mismatch"):
        validate_recovery_artifact(tampered)


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError, match="field_amplitudes and coherence_profile must have equal lengths"):
        DecoherenceState(
            state_id="bad",
            field_amplitudes=(1.0, 2.0),
            coherence_profile=(1.0,),
        )

    with pytest.raises(ValueError, match="coherence_profile entries must be finite"):
        DecoherenceState(
            state_id="bad",
            field_amplitudes=(1.0,),
            coherence_profile=(float("nan"),),
        )

    with pytest.raises(ValueError, match="parent_transition_hash must be 64 hex characters"):
        synthesize_recovery_state(_state(), parent_transition_hash="bad")

    with pytest.raises(ValueError, match="gap_threshold must be finite and > 0"):
        detect_fragmentation(_state(), gap_threshold=0.0)

    with pytest.raises(ValueError, match="fragmentation_boundaries entries out of range"):
        compute_fragmentation_score(_state(), (100,))
