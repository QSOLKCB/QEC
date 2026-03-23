"""Tests for v86.4.0 — Phase Syndrome Decoder."""

from qec.experiments.phase_syndrome_decoder import (
    analyze_syndrome_patterns,
    classify_syndrome_trajectory,
    compute_decoder_confidence,
    decode_syndrome_trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transitions(encoded):
    """Build transition records from encoded list (mirrors detect_syndrome_transitions)."""
    transitions = []
    for i in range(len(encoded) - 1):
        if encoded[i] != encoded[i + 1]:
            hd = sum(a != b for a, b in zip(encoded[i], encoded[i + 1]))
            transitions.append({
                "index": i,
                "from": encoded[i],
                "to": encoded[i + 1],
                "hamming_distance": hd,
            })
    return transitions


# ---------------------------------------------------------------------------
# Test 1 — Stable case (single syndrome)
# ---------------------------------------------------------------------------

def test_stable_regime():
    encoded = ["1111", "1111", "1111", "1111"]
    transitions = _make_transitions(encoded)
    result = decode_syndrome_trajectory(encoded, transitions)
    assert result["regime_type"] == "stable"
    assert result["dominant_syndrome"] == "1111"
    assert result["n_transitions"] == 0
    assert result["confidence"] == 1.0


# ---------------------------------------------------------------------------
# Test 2 — Oscillatory case (2 alternating, Hamming <= 1)
# ---------------------------------------------------------------------------

def test_oscillatory_regime():
    encoded = ["1101", "1111", "1101", "1111", "1101", "1111"]
    transitions = _make_transitions(encoded)
    result = decode_syndrome_trajectory(encoded, transitions)
    assert result["regime_type"] == "oscillatory"
    assert result["n_transitions"] == 5


# ---------------------------------------------------------------------------
# Test 3 — Chaotic case (3+ unique, mean Hamming >= 2)
# ---------------------------------------------------------------------------

def test_chaotic_regime():
    # 3 unique syndromes with Hamming distances of 2+ between transitions
    encoded = ["1111", "0000", "1100", "0011", "1111", "0000"]
    transitions = _make_transitions(encoded)
    result = decode_syndrome_trajectory(encoded, transitions)
    assert result["regime_type"] == "chaotic"
    assert result["patterns"]["unique_count"] >= 3


# ---------------------------------------------------------------------------
# Test 4 — Boundary case (2+ unique, mean Hamming == 1)
# ---------------------------------------------------------------------------

def test_boundary_regime():
    # 3 unique syndromes but all transitions have Hamming distance 1
    encoded = ["1111", "1110", "1100", "1110", "1111", "1110"]
    transitions = _make_transitions(encoded)
    result = decode_syndrome_trajectory(encoded, transitions)
    assert result["regime_type"] == "boundary"


# ---------------------------------------------------------------------------
# Test 5 — Confidence correctness
# ---------------------------------------------------------------------------

def test_confidence_value():
    encoded = ["1111", "1111", "1111", "1101", "1101"]
    transitions = _make_transitions(encoded)
    result = decode_syndrome_trajectory(encoded, transitions)
    # max_run_length = 3 (first three "1111"), n_steps = 5
    assert result["confidence"] == 3.0 / 5.0


# ---------------------------------------------------------------------------
# Test 6 — Deterministic output
# ---------------------------------------------------------------------------

def test_determinism():
    encoded = ["1101", "1001", "1101", "1001"]
    transitions = _make_transitions(encoded)
    r1 = decode_syndrome_trajectory(encoded, transitions)
    r2 = decode_syndrome_trajectory(encoded, transitions)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Test 7 — Empty input
# ---------------------------------------------------------------------------

def test_empty_input():
    result = decode_syndrome_trajectory([], [])
    assert result["regime_type"] == "undetermined"
    assert result["confidence"] == 0.0
    assert result["dominant_syndrome"] == ""
    assert result["n_transitions"] == 0


# ---------------------------------------------------------------------------
# Test 8 — Single element
# ---------------------------------------------------------------------------

def test_single_element():
    encoded = ["1010"]
    transitions = _make_transitions(encoded)
    result = decode_syndrome_trajectory(encoded, transitions)
    assert result["regime_type"] == "stable"
    assert result["confidence"] == 1.0
    assert result["dominant_syndrome"] == "1010"


# ---------------------------------------------------------------------------
# Test 9 — Pattern extraction correctness
# ---------------------------------------------------------------------------

def test_analyze_patterns():
    encoded = ["1111", "1111", "1101", "1111"]
    transitions = _make_transitions(encoded)
    patterns = analyze_syndrome_patterns(encoded, transitions)
    assert patterns["unique_count"] == 2
    assert patterns["most_common"] == "1111"
    assert patterns["max_run_length"] == 2
    assert patterns["n_transitions"] == 2
    assert patterns["n_steps"] == 4


# ---------------------------------------------------------------------------
# Test 10 — Classification rules cover all branches
# ---------------------------------------------------------------------------

def test_classify_all_branches():
    # stable
    assert classify_syndrome_trajectory({"unique_count": 1, "mean_hamming": 0.0}) == "stable"
    # oscillatory
    assert classify_syndrome_trajectory({"unique_count": 2, "mean_hamming": 1.0}) == "oscillatory"
    # chaotic
    assert classify_syndrome_trajectory({"unique_count": 4, "mean_hamming": 3.0}) == "chaotic"
    # boundary
    assert classify_syndrome_trajectory({"unique_count": 3, "mean_hamming": 1.0}) == "boundary"
    # undetermined (fallback)
    assert classify_syndrome_trajectory({"unique_count": 3, "mean_hamming": 1.5}) == "undetermined"
    # empty
    assert classify_syndrome_trajectory({"unique_count": 0, "mean_hamming": 0.0}) == "undetermined"


# ---------------------------------------------------------------------------
# Test 11 — Confidence clamping
# ---------------------------------------------------------------------------

def test_confidence_clamping():
    # max_run_length can never exceed n_steps in practice,
    # but verify clamping logic works.
    assert compute_decoder_confidence({"max_run_length": 10, "n_steps": 5}) == 1.0
    assert compute_decoder_confidence({"max_run_length": 0, "n_steps": 5}) == 0.0
    assert compute_decoder_confidence({"max_run_length": 0, "n_steps": 0}) == 0.0


# ---------------------------------------------------------------------------
# Test 12 — No input mutation
# ---------------------------------------------------------------------------

def test_no_input_mutation():
    encoded = ["1101", "1001", "1101"]
    transitions = _make_transitions(encoded)
    encoded_copy = list(encoded)
    transitions_copy = [dict(t) for t in transitions]
    decode_syndrome_trajectory(encoded, transitions)
    assert encoded == encoded_copy
    assert transitions == transitions_copy
