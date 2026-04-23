from __future__ import annotations

import json

import pytest

from qec.analysis.periodicity_structure_kernel import (
    PeriodicityReceipt,
    STRONG_PERIODIC_THRESHOLD,
    detect_periodicity,
)


def test_deterministic_replay_identical_receipt_and_hash() -> None:
    trace = ["A", "B", "A", "B", "A", "B"]
    a = detect_periodicity(trace)
    b = detect_periodicity(trace)

    assert a.to_dict() == b.to_dict()
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.stable_hash == b.stable_hash
    assert a.stable_hash == a.computed_stable_hash()


def test_simple_periodic_trace_detects_period_two() -> None:
    receipt = detect_periodicity(["A", "B", "A", "B", "A", "B"])

    assert receipt.dominant_period == 2
    assert receipt.classification == "strong_periodic"
    assert receipt.dominant_confidence >= STRONG_PERIODIC_THRESHOLD


def test_two_occurrence_motif_is_detected_for_short_trace() -> None:
    receipt = detect_periodicity(["A", "B", "A", "B"])

    assert receipt.dominant_period == 2
    assert receipt.classification in {"weak_periodic", "strong_periodic"}
    assert any(candidate.gcd_period == 2 for candidate in receipt.candidates)


def test_repetition_count_tracks_occurrences_not_gaps() -> None:
    receipt = detect_periodicity(["A", "B", "A", "B", "A", "B"])

    top = receipt.candidates[0]
    assert top.repetition_count == 3


def test_no_periodicity_returns_aperiodic() -> None:
    receipt = detect_periodicity(["A", "B", "C", "D", "E", "F"])

    assert receipt.classification == "aperiodic"
    assert receipt.dominant_period is None
    assert receipt.candidates == ()


def test_multiple_motifs_have_deterministic_ordering_and_selection() -> None:
    trace = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"]
    receipt = detect_periodicity(trace)

    sorted_by_rule = sorted(
        receipt.candidates,
        key=lambda candidate: (-candidate.confidence, candidate.motif_signature),
    )
    assert list(receipt.candidates) == sorted_by_rule
    assert receipt.dominant_period is not None


def test_edge_case_minimal_valid_trace() -> None:
    receipt = detect_periodicity(["A"])

    assert receipt.trace_length == 1
    assert receipt.classification == "aperiodic"
    assert receipt.candidates == ()


def test_invalid_input_empty_trace_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        detect_periodicity([])


def test_invalid_input_non_canonical_trace_raises() -> None:
    with pytest.raises(ValueError, match="canonical"):
        detect_periodicity(["A", " B"])


def test_float_stability_and_canonical_json_rounding() -> None:
    receipt = detect_periodicity(["A", "B", "A", "B", "A", "B"])

    canonical = receipt.to_canonical_json()
    parsed = json.loads(canonical)
    assert parsed["dominant_confidence"] == round(parsed["dominant_confidence"], 12)
    for candidate in parsed["candidates"]:
        assert candidate["spacing_variance"] == round(candidate["spacing_variance"], 12)
        assert candidate["confidence"] == round(candidate["confidence"], 12)


def test_receipt_rejects_invalid_stable_hash_format() -> None:
    base = detect_periodicity(["A", "B", "A", "B", "A", "B"])
    payload = base.to_dict()
    payload["stable_hash"] = "abc123"

    with pytest.raises(ValueError, match="64-char hex string"):
        PeriodicityReceipt(
            trace_length=payload["trace_length"],
            candidates=base.candidates,
            dominant_period=payload["dominant_period"],
            dominant_confidence=payload["dominant_confidence"],
            classification=payload["classification"],
            stable_hash=payload["stable_hash"],
        )


def test_receipt_rejects_tampered_stable_hash_payload_mismatch() -> None:
    base = detect_periodicity(["A", "B", "A", "B", "A", "B"])
    tampered_hash = "0" * 64 if base.stable_hash != ("0" * 64) else "f" * 64

    with pytest.raises(ValueError, match="stable_hash mismatch"):
        PeriodicityReceipt(
            trace_length=base.trace_length,
            candidates=base.candidates,
            dominant_period=base.dominant_period,
            dominant_confidence=base.dominant_confidence,
            classification=base.classification,
            stable_hash=tampered_hash,
        )
