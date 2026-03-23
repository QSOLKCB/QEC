"""Tests for v86.3.0 — Phase Syndrome Analysis (Discrete Invariant Signatures).

Covers:
  - deterministic output
  - correct encoding
  - transition detection
  - Hamming distance correctness
  - empty input
  - single element
  - consistency rules
  - no mutation of inputs
  - target_class filtering
  - best_pair unwrapping
  - edge cases for stability_ok fallback
"""

import copy

import pytest

from qec.experiments.phase_syndrome_analysis import (
    _hamming_distance,
    detect_syndrome_transitions,
    encode_syndrome,
    extract_phase_syndrome,
    extract_syndrome_series,
    run_syndrome_analysis,
)


# ---------------------------------------------------------------------------
# Helpers — deterministic test fixtures
# ---------------------------------------------------------------------------


def _make_pair(
    cls="stable",
    phase="stable_region",
    compatibility=0.8,
    score=1.0,
    normalized_score=None,
):
    """Build a minimal best_pair dict."""
    d = {
        "class": cls,
        "phase": phase,
        "compatibility": compatibility,
        "score": score,
    }
    if normalized_score is not None:
        d["normalized_score"] = normalized_score
    return d


def _wrap(pair):
    """Wrap a best_pair in a result dict."""
    return {"best_pair": pair}


# ---------------------------------------------------------------------------
# 1. Deterministic output — same input always yields same syndrome
# ---------------------------------------------------------------------------


def test_deterministic_output():
    pair = _make_pair()
    result = _wrap(pair)
    s1 = extract_phase_syndrome(result)
    s2 = extract_phase_syndrome(result)
    assert s1 == s2


# ---------------------------------------------------------------------------
# 2. Correct encoding — bit ordering [class, phase, structure, stability]
# ---------------------------------------------------------------------------


def test_encode_all_true():
    syndrome = {
        "class_consistent": True,
        "phase_consistent": True,
        "structure_consistent": True,
        "stability_ok": True,
    }
    assert encode_syndrome(syndrome) == "1111"


def test_encode_all_false():
    syndrome = {
        "class_consistent": False,
        "phase_consistent": False,
        "structure_consistent": False,
        "stability_ok": False,
    }
    assert encode_syndrome(syndrome) == "0000"


def test_encode_mixed():
    syndrome = {
        "class_consistent": True,
        "phase_consistent": True,
        "structure_consistent": False,
        "stability_ok": True,
    }
    assert encode_syndrome(syndrome) == "1101"


# ---------------------------------------------------------------------------
# 3. Transition detection — detects changes between consecutive syndromes
# ---------------------------------------------------------------------------


def test_detect_transitions_basic():
    encoded = ["1111", "1111", "1101", "1101", "0000"]
    transitions = detect_syndrome_transitions(encoded)
    assert len(transitions) == 2
    assert transitions[0]["index"] == 1
    assert transitions[0]["from"] == "1111"
    assert transitions[0]["to"] == "1101"
    assert transitions[1]["index"] == 3
    assert transitions[1]["from"] == "1101"
    assert transitions[1]["to"] == "0000"


def test_no_transitions_when_constant():
    encoded = ["1111", "1111", "1111"]
    transitions = detect_syndrome_transitions(encoded)
    assert transitions == []


# ---------------------------------------------------------------------------
# 4. Hamming distance correctness
# ---------------------------------------------------------------------------


def test_hamming_distance_identical():
    assert _hamming_distance("1111", "1111") == 0


def test_hamming_distance_one_flip():
    assert _hamming_distance("1111", "1110") == 1


def test_hamming_distance_all_flipped():
    assert _hamming_distance("1111", "0000") == 4


def test_hamming_in_transitions():
    encoded = ["1111", "0000"]
    transitions = detect_syndrome_transitions(encoded)
    assert transitions[0]["hamming_distance"] == 4


# ---------------------------------------------------------------------------
# 5. Empty input
# ---------------------------------------------------------------------------


def test_empty_results():
    series = extract_syndrome_series([])
    assert series["syndromes"] == []
    assert series["encoded"] == []
    assert series["n_steps"] == 0


def test_empty_transitions():
    transitions = detect_syndrome_transitions([])
    assert transitions == []


# ---------------------------------------------------------------------------
# 6. Single element
# ---------------------------------------------------------------------------


def test_single_result():
    results = [_wrap(_make_pair())]
    series = extract_syndrome_series(results)
    assert series["n_steps"] == 1
    assert len(series["encoded"]) == 1


def test_single_element_no_transitions():
    transitions = detect_syndrome_transitions(["1111"])
    assert transitions == []


# ---------------------------------------------------------------------------
# 7. Consistency rules — syndrome logic
# ---------------------------------------------------------------------------


def test_class_consistent_with_target():
    pair = _make_pair(cls="stable")
    s = extract_phase_syndrome(pair, target_class="stable")
    assert s["class_consistent"] is True


def test_class_inconsistent_with_target():
    pair = _make_pair(cls="chaotic")
    s = extract_phase_syndrome(pair, target_class="stable")
    assert s["class_consistent"] is False


def test_class_consistent_no_target():
    pair = _make_pair(cls="chaotic")
    s = extract_phase_syndrome(pair)
    assert s["class_consistent"] is True  # defaults to True


def test_phase_consistent_stable():
    pair = _make_pair(phase="stable_region")
    assert extract_phase_syndrome(pair)["phase_consistent"] is True


def test_phase_consistent_near_boundary():
    pair = _make_pair(phase="near_boundary")
    assert extract_phase_syndrome(pair)["phase_consistent"] is True


def test_phase_inconsistent_unstable():
    pair = _make_pair(phase="unstable_region")
    assert extract_phase_syndrome(pair)["phase_consistent"] is False


def test_phase_inconsistent_chaotic():
    pair = _make_pair(phase="chaotic_transition")
    assert extract_phase_syndrome(pair)["phase_consistent"] is False


def test_structure_consistent_above_threshold():
    pair = _make_pair(compatibility=0.5)
    assert extract_phase_syndrome(pair)["structure_consistent"] is True


def test_structure_inconsistent_below_threshold():
    pair = _make_pair(compatibility=0.49)
    assert extract_phase_syndrome(pair)["structure_consistent"] is False


def test_stability_ok_with_normalized_score():
    pair = _make_pair(normalized_score=0.5)
    assert extract_phase_syndrome(pair)["stability_ok"] is True

    pair2 = _make_pair(normalized_score=0.49)
    assert extract_phase_syndrome(pair2)["stability_ok"] is False


def test_stability_ok_fallback_to_score():
    pair = _make_pair(score=0.0)
    assert extract_phase_syndrome(pair)["stability_ok"] is True  # >= 0.0

    pair2 = _make_pair(score=-0.1)
    assert extract_phase_syndrome(pair2)["stability_ok"] is False


# ---------------------------------------------------------------------------
# 8. No mutation of inputs
# ---------------------------------------------------------------------------


def test_no_mutation_of_result():
    pair = _make_pair(normalized_score=0.7)
    result = _wrap(pair)
    original = copy.deepcopy(result)
    extract_phase_syndrome(result, target_class="stable")
    assert result == original


def test_no_mutation_of_results_list():
    results = [_wrap(_make_pair()), _wrap(_make_pair(cls="chaotic"))]
    original = copy.deepcopy(results)
    run_syndrome_analysis(results, target_class="stable")
    assert results == original


# ---------------------------------------------------------------------------
# 9. best_pair unwrapping
# ---------------------------------------------------------------------------


def test_unwrap_best_pair():
    result = _wrap(_make_pair(cls="fragile", compatibility=0.3))
    s = extract_phase_syndrome(result, target_class="fragile")
    assert s["class_consistent"] is True
    assert s["structure_consistent"] is False


def test_direct_pair_without_wrapping():
    pair = _make_pair(cls="stable", phase="unstable_region")
    s = extract_phase_syndrome(pair)
    assert s["phase_consistent"] is False


# ---------------------------------------------------------------------------
# 10. Full pipeline — run_syndrome_analysis
# ---------------------------------------------------------------------------


def test_run_syndrome_analysis_integration():
    results = [
        _wrap(_make_pair(cls="stable", phase="stable_region",
                         compatibility=0.8, normalized_score=0.9)),
        _wrap(_make_pair(cls="stable", phase="stable_region",
                         compatibility=0.8, normalized_score=0.9)),
        _wrap(_make_pair(cls="chaotic", phase="chaotic_transition",
                         compatibility=0.2, normalized_score=0.1)),
    ]
    analysis = run_syndrome_analysis(results, target_class="stable")
    assert analysis["series"]["n_steps"] == 3
    assert analysis["series"]["encoded"][0] == "1111"
    assert analysis["series"]["encoded"][1] == "1111"
    # chaotic: class=False, phase=False, structure=False, stability=False
    assert analysis["series"]["encoded"][2] == "0000"
    assert len(analysis["transitions"]) == 1
    assert analysis["transitions"][0]["hamming_distance"] == 4


# ---------------------------------------------------------------------------
# 11. Syndrome series with target_class threading
# ---------------------------------------------------------------------------


def test_syndrome_series_target_class():
    results = [
        _wrap(_make_pair(cls="stable")),
        _wrap(_make_pair(cls="fragile")),
    ]
    series = extract_syndrome_series(results, target_class="stable")
    assert series["syndromes"][0]["class_consistent"] is True
    assert series["syndromes"][1]["class_consistent"] is False


# ---------------------------------------------------------------------------
# 12. Return types are always bool
# ---------------------------------------------------------------------------


def test_return_types_are_bool():
    pair = _make_pair(normalized_score=0.5)
    s = extract_phase_syndrome(pair)
    for key in ("class_consistent", "phase_consistent",
                "structure_consistent", "stability_ok"):
        assert isinstance(s[key], bool), f"{key} should be bool"
