from __future__ import annotations

import json

import pytest

from qec.analysis.periodicity_structure_kernel import detect_periodicity


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
    assert receipt.dominant_confidence >= 0.3


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
