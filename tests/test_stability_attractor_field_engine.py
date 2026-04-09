"""Deterministic tests for v137.5.3 stability attractor field engine."""

from __future__ import annotations

import pytest

from qec.analysis.stability_attractor_field_engine import (
    AttractorField,
    AttractorReceipt,
    compute_basin_strength,
    detect_attractor_basin,
    export_attractor_bytes,
    generate_attractor_receipt,
    synthesize_attractor_field,
)


_SOURCE_HASH = "1" * 64
_PARENT_HASH = "2" * 64


def _state_trace() -> tuple[tuple[float, ...], ...]:
    return (
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (0.5, 0.5),
        (1.0, 1.0),
    )


def test_detect_attractor_basin_is_deterministic() -> None:
    trace = _state_trace()
    b1 = detect_attractor_basin(trace)
    b2 = detect_attractor_basin(trace)
    assert b1 == b2
    assert b1 == ((0.5, 0.5), (1.0, 1.0))


def test_compute_basin_strength_bounded() -> None:
    trace = _state_trace()
    basin = detect_attractor_basin(trace)
    score = compute_basin_strength(trace, basin)
    assert 0.0 <= score <= 1.0


def test_identical_inputs_produce_identical_bytes() -> None:
    trace = _state_trace()
    f1 = synthesize_attractor_field(
        source_recovery_hash=_SOURCE_HASH,
        parent_recovery_hash=_PARENT_HASH,
        state_trace=trace,
        enable_attractor_engine=True,
    )
    f2 = synthesize_attractor_field(
        source_recovery_hash=_SOURCE_HASH,
        parent_recovery_hash=_PARENT_HASH,
        state_trace=trace,
        enable_attractor_engine=True,
    )

    b1 = export_attractor_bytes(f1)
    b2 = export_attractor_bytes(f2)
    assert isinstance(f1, AttractorField)
    assert b1 == b2
    assert f1.stable_attractor_hash == f2.stable_attractor_hash


def test_receipt_stability() -> None:
    field = synthesize_attractor_field(
        source_recovery_hash=_SOURCE_HASH,
        parent_recovery_hash=_PARENT_HASH,
        state_trace=_state_trace(),
        enable_attractor_engine=True,
    )
    exported = export_attractor_bytes(field)

    r1 = generate_attractor_receipt(field, exported)
    r2 = generate_attractor_receipt(field, exported)
    assert isinstance(r1, AttractorReceipt)
    assert r1 == r2
    assert r1.tamper_detected is False


def test_tamper_detection() -> None:
    field = synthesize_attractor_field(
        source_recovery_hash=_SOURCE_HASH,
        parent_recovery_hash=_PARENT_HASH,
        state_trace=_state_trace(),
        enable_attractor_engine=True,
    )
    exported = bytearray(export_attractor_bytes(field))
    exported[-1] = ord("0") if exported[-1] != ord("0") else ord("1")

    receipt = generate_attractor_receipt(field, bytes(exported))
    assert receipt.tamper_detected is True


def test_fail_fast_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="enable_attractor_engine"):
        synthesize_attractor_field(
            source_recovery_hash=_SOURCE_HASH,
            parent_recovery_hash=_PARENT_HASH,
            state_trace=_state_trace(),
        )

    with pytest.raises(ValueError, match="source_recovery_hash"):
        synthesize_attractor_field(
            source_recovery_hash="bad",
            parent_recovery_hash=_PARENT_HASH,
            state_trace=_state_trace(),
            enable_attractor_engine=True,
        )

    with pytest.raises(ValueError, match="state_trace"):
        detect_attractor_basin(())


def test_repeated_run_determinism() -> None:
    trace = _state_trace()
    reference = synthesize_attractor_field(
        source_recovery_hash=_SOURCE_HASH,
        parent_recovery_hash=_PARENT_HASH,
        state_trace=trace,
        enable_attractor_engine=True,
    )
    reference_bytes = export_attractor_bytes(reference)

    for _ in range(100):
        field = synthesize_attractor_field(
            source_recovery_hash=_SOURCE_HASH,
            parent_recovery_hash=_PARENT_HASH,
            state_trace=trace,
            enable_attractor_engine=True,
        )
        assert field == reference
        assert export_attractor_bytes(field) == reference_bytes


def test_convergence_stability_score_phase_invariant() -> None:
    """Phase-shifted cyclic traversal must score as fully converged."""
    # Basin is (A, B, C); trace ends with a phase-shifted traversal (C, A, B).
    a = (0.0, 1.0)
    b = (1.0, 0.0)
    c = (0.5, 0.5)
    # Pre-period states establish the basin, then one complete phase-shifted cycle.
    trace = (a, b, c, a, b, c, a, b)
    basin = detect_attractor_basin(trace)
    assert set(basin) == {a, b, c}

    field = synthesize_attractor_field(
        source_recovery_hash=_SOURCE_HASH,
        parent_recovery_hash=_PARENT_HASH,
        state_trace=trace,
        enable_attractor_engine=True,
    )
    # Convergence should be perfect (1.0) because the tail is a cyclic shift of the basin.
    assert field.convergence_stability_score == 1.0


def test_convergence_stability_score_regression() -> None:
    """Convergence score for known trace must remain stable across code changes."""
    trace = _state_trace()
    field = synthesize_attractor_field(
        source_recovery_hash=_SOURCE_HASH,
        parent_recovery_hash=_PARENT_HASH,
        state_trace=trace,
        enable_attractor_engine=True,
    )
    # Basin is ((0.5, 0.5), (1.0, 1.0)); tail matches the basin exactly → score must be 1.0.
    assert field.convergence_stability_score == 1.0


def test_fail_fast_string_state_trace() -> None:
    with pytest.raises(ValueError, match="state_trace"):
        detect_attractor_basin("invalid")  # type: ignore[arg-type]


def test_fail_fast_bool_elements_rejected() -> None:
    with pytest.raises(ValueError, match="bool"):
        detect_attractor_basin([[True, False]])
