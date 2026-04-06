from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.coherence_field_kernel import (
    CoherenceField,
    compute_coherence_score,
    export_coherence_bytes,
    generate_coherence_receipt,
    synthesize_coherence_field,
)


def _hex(char: str) -> str:
    return char * 64


def _build_field() -> CoherenceField:
    return synthesize_coherence_field(
        {
            "alpha": 4.0,
            "beta": 2.0,
            "gamma": 1.0,
            "delta": 1.0,
        },
        parent_certification_root=_hex("a"),
    )


def test_repeated_run_determinism() -> None:
    f1 = _build_field()
    f2 = _build_field()
    assert f1 == f2


def test_identical_inputs_produce_identical_bytes() -> None:
    f1 = _build_field()
    f2 = _build_field()

    assert export_coherence_bytes(f1) == export_coherence_bytes(f2)


def test_bounded_coherence_scores() -> None:
    coherence, stability, entropy = compute_coherence_score(
        (
            ("alpha", 5.0),
            ("beta", -5.0),
            ("gamma", 0.0),
        )
    )

    assert 0.0 <= coherence <= 1.0
    assert 0.0 <= stability <= 1.0
    assert 0.0 <= entropy <= 1.0


def test_stable_field_hash() -> None:
    f1 = _build_field()
    f2 = _build_field()
    assert f1.stable_field_hash == f2.stable_field_hash


def test_receipt_stability() -> None:
    r1 = generate_coherence_receipt(_build_field())
    r2 = generate_coherence_receipt(_build_field())
    assert r1 == r2


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError, match="parent_certification_root must be 64 hex characters"):
        synthesize_coherence_field({"alpha": 1.0}, parent_certification_root="123")

    with pytest.raises(ValueError, match="state_components must not be empty"):
        synthesize_coherence_field({}, parent_certification_root=_hex("a"))

    with pytest.raises(ValueError, match="state component values must be finite"):
        synthesize_coherence_field({"alpha": float("nan")}, parent_certification_root=_hex("a"))

    with pytest.raises(ValueError, match="duplicate state component name"):
        synthesize_coherence_field(
            (("alpha", 1.0), ("alpha", 2.0)),
            parent_certification_root=_hex("a"),
        )

    field = _build_field()
    tampered = replace(field, stable_field_hash=_hex("b"))
    with pytest.raises(ValueError, match="stable_field_hash mismatch"):
        export_coherence_bytes(tampered)
