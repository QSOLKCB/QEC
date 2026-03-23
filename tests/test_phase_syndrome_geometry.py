"""Tests for v86.5.0 — Ternary & Geometric Syndrome Extension."""

from __future__ import annotations

import copy
import math

import pytest

from qec.experiments.phase_syndrome_geometry import (
    build_multilayer_syndrome,
    compute_syndrome_distance,
    detect_geometric_transitions,
    encode_ternary_syndrome,
    extract_ternary_series,
    extract_ternary_syndrome,
    run_syndrome_geometry_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    cls: str = "stable",
    phase: str = "stable",
    compatibility: float = 0.8,
    score: float = 0.9,
    normalized_score: float | None = None,
) -> dict:
    """Build a minimal result dict for testing."""
    r: dict = {
        "class": cls,
        "phase": phase,
        "compatibility": compatibility,
        "score": score,
    }
    if normalized_score is not None:
        r["normalized_score"] = normalized_score
    return r


# ---------------------------------------------------------------------------
# Test 1 — Ternary mapping correctness
# ---------------------------------------------------------------------------


class TestExtractTernarySyndrome:
    def test_all_positive(self):
        r = _make_result(
            cls="stable", phase="stable",
            compatibility=0.8, normalized_score=0.9,
        )
        s = extract_ternary_syndrome(r, target_class="stable")
        assert s == {
            "class_state": 1,
            "phase_state": 1,
            "structure_state": 1,
            "stability_state": 1,
        }

    def test_all_negative(self):
        r = _make_result(
            cls="chaotic", phase="chaotic_transition",
            compatibility=0.1, normalized_score=0.2,
        )
        s = extract_ternary_syndrome(r, target_class="stable")
        assert s == {
            "class_state": -1,
            "phase_state": -1,
            "structure_state": -1,
            "stability_state": -1,
        }

    def test_mixed_states(self):
        r = _make_result(
            cls="stable", phase="boundary",
            compatibility=0.5, normalized_score=0.5,
        )
        s = extract_ternary_syndrome(r, target_class="stable")
        assert s["class_state"] == 1
        assert s["phase_state"] == 0   # boundary -> neutral
        assert s["structure_state"] == 0  # 0.5 in [0.4, 0.75)
        assert s["stability_state"] == 0  # 0.5 in [0.4, 0.75)


# ---------------------------------------------------------------------------
# Test 2 — Boundary detection (0 state)
# ---------------------------------------------------------------------------


class TestBoundaryDetection:
    def test_no_target_class_gives_zero(self):
        r = _make_result()
        s = extract_ternary_syndrome(r, target_class=None)
        assert s["class_state"] == 0

    def test_structure_boundary_low(self):
        r = _make_result(compatibility=0.4)
        s = extract_ternary_syndrome(r)
        assert s["structure_state"] == 0

    def test_structure_boundary_high(self):
        r = _make_result(compatibility=0.749)
        s = extract_ternary_syndrome(r)
        assert s["structure_state"] == 0

    def test_structure_above_boundary(self):
        r = _make_result(compatibility=0.75)
        s = extract_ternary_syndrome(r)
        assert s["structure_state"] == 1

    def test_structure_below_boundary(self):
        r = _make_result(compatibility=0.39)
        s = extract_ternary_syndrome(r)
        assert s["structure_state"] == -1

    def test_stability_fallback_to_score(self):
        r = _make_result(score=0.5)
        # No normalized_score -> falls back to score.
        s = extract_ternary_syndrome(r)
        assert s["stability_state"] == 0

    def test_phase_unknown_gives_zero(self):
        r = _make_result(phase="unknown_phase")
        s = extract_ternary_syndrome(r)
        assert s["phase_state"] == 0


# ---------------------------------------------------------------------------
# Test 3 — Encoding
# ---------------------------------------------------------------------------


class TestEncodeTernarySyndrome:
    def test_positive_encoding(self):
        s = {"class_state": 1, "phase_state": 1,
             "structure_state": 1, "stability_state": 1}
        t, c = encode_ternary_syndrome(s)
        assert t == (1, 1, 1, 1)
        assert c == "++++"

    def test_negative_encoding(self):
        s = {"class_state": -1, "phase_state": -1,
             "structure_state": -1, "stability_state": -1}
        t, c = encode_ternary_syndrome(s)
        assert t == (-1, -1, -1, -1)
        assert c == "----"

    def test_mixed_encoding(self):
        s = {"class_state": 1, "phase_state": 0,
             "structure_state": -1, "stability_state": 1}
        t, c = encode_ternary_syndrome(s)
        assert t == (1, 0, -1, 1)
        assert c == "+0-+"


# ---------------------------------------------------------------------------
# Test 4 — Geometric distance correctness
# ---------------------------------------------------------------------------


class TestGeometricDistance:
    def test_identical(self):
        assert compute_syndrome_distance((1, 0, -1, 1), (1, 0, -1, 1)) == 0.0

    def test_one_axis(self):
        # (1,0,0,0) vs (-1,0,0,0) -> distance = 2
        assert compute_syndrome_distance((1, 0, 0, 0), (-1, 0, 0, 0)) == 2.0

    def test_two_axes(self):
        # (1,1,0,0) vs (-1,-1,0,0) -> sqrt(4+4) = sqrt(8)
        d = compute_syndrome_distance((1, 1, 0, 0), (-1, -1, 0, 0))
        assert math.isclose(d, math.sqrt(8))

    def test_all_axes(self):
        # (+1,+1,+1,+1) vs (-1,-1,-1,-1) -> sqrt(16) = 4
        d = compute_syndrome_distance((1, 1, 1, 1), (-1, -1, -1, -1))
        assert d == 4.0

    def test_symmetry(self):
        a = (1, 0, -1, 1)
        b = (-1, 1, 0, -1)
        assert compute_syndrome_distance(a, b) == compute_syndrome_distance(b, a)


# ---------------------------------------------------------------------------
# Test 5 — Transition detection
# ---------------------------------------------------------------------------


class TestGeometricTransitions:
    def test_no_transitions(self):
        encoded = [(1, 0, -1, 1), (1, 0, -1, 1), (1, 0, -1, 1)]
        assert detect_geometric_transitions(encoded) == []

    def test_single_transition(self):
        encoded = [(1, 0, -1, 1), (-1, 0, -1, 1)]
        transitions = detect_geometric_transitions(encoded)
        assert len(transitions) == 1
        assert transitions[0]["index"] == 0
        assert transitions[0]["from"] == (1, 0, -1, 1)
        assert transitions[0]["to"] == (-1, 0, -1, 1)
        assert transitions[0]["distance"] == 2.0

    def test_multiple_transitions(self):
        encoded = [(1, 1, 1, 1), (1, 1, 1, 1), (-1, -1, -1, -1), (0, 0, 0, 0)]
        transitions = detect_geometric_transitions(encoded)
        assert len(transitions) == 2
        assert transitions[0]["index"] == 1
        assert transitions[1]["index"] == 2


# ---------------------------------------------------------------------------
# Test 6 — Backward compatibility (multi-layer)
# ---------------------------------------------------------------------------


class TestMultilayerSyndrome:
    def test_binary_layer_present(self):
        r = _make_result(cls="stable", phase="stable",
                         compatibility=0.8, normalized_score=0.9)
        ml = build_multilayer_syndrome(r, target_class="stable")
        assert "binary" in ml
        assert isinstance(ml["binary"], str)
        assert len(ml["binary"]) == 4
        assert set(ml["binary"]) <= {"0", "1"}

    def test_ternary_layer_present(self):
        r = _make_result()
        ml = build_multilayer_syndrome(r)
        assert "ternary" in ml
        assert isinstance(ml["ternary"], tuple)
        assert len(ml["ternary"]) == 4

    def test_binary_unchanged(self):
        """Binary layer must match v86.3 output exactly."""
        from qec.experiments.phase_syndrome_analysis import (
            encode_syndrome,
            extract_phase_syndrome,
        )
        r = _make_result(cls="stable", phase="stable",
                         compatibility=0.8, normalized_score=0.9)
        expected = encode_syndrome(
            extract_phase_syndrome(r, target_class="stable"))
        ml = build_multilayer_syndrome(r, target_class="stable")
        assert ml["binary"] == expected


# ---------------------------------------------------------------------------
# Test 7 — Deterministic output
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_extraction(self):
        r = _make_result(cls="stable", phase="stable",
                         compatibility=0.6, normalized_score=0.5)
        results = [extract_ternary_syndrome(r, target_class="stable")
                   for _ in range(10)]
        assert all(s == results[0] for s in results)

    def test_series_determinism(self):
        results = [
            _make_result(phase="stable", compatibility=0.8, normalized_score=0.9),
            _make_result(phase="chaotic_transition", compatibility=0.1,
                         normalized_score=0.1),
        ]
        s1 = extract_ternary_series(results)
        s2 = extract_ternary_series(results)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Test 8 — Empty / single input
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_series(self):
        series = extract_ternary_series([])
        assert series["n_steps"] == 0
        assert series["ternary"] == []
        assert series["encoded"] == []

    def test_single_result(self):
        series = extract_ternary_series([_make_result()])
        assert series["n_steps"] == 1

    def test_empty_transitions(self):
        assert detect_geometric_transitions([]) == []

    def test_single_encoded_no_transitions(self):
        assert detect_geometric_transitions([(1, 0, -1, 1)]) == []


# ---------------------------------------------------------------------------
# Test 9 — No mutation
# ---------------------------------------------------------------------------


class TestNoMutation:
    def test_extract_does_not_mutate(self):
        r = _make_result(cls="stable", phase="stable",
                         compatibility=0.8, normalized_score=0.9)
        original = copy.deepcopy(r)
        extract_ternary_syndrome(r, target_class="stable")
        assert r == original

    def test_series_does_not_mutate(self):
        results = [
            _make_result(phase="stable"),
            _make_result(phase="chaotic_transition"),
        ]
        original = copy.deepcopy(results)
        extract_ternary_series(results)
        assert results == original

    def test_multilayer_does_not_mutate(self):
        r = _make_result()
        original = copy.deepcopy(r)
        build_multilayer_syndrome(r)
        assert r == original


# ---------------------------------------------------------------------------
# Test 10 — best_pair unwrapping
# ---------------------------------------------------------------------------


class TestBestPairUnwrap:
    def test_unwrap_best_pair(self):
        wrapped = {"best_pair": _make_result(
            cls="stable", phase="stable",
            compatibility=0.9, normalized_score=0.8,
        )}
        s = extract_ternary_syndrome(wrapped, target_class="stable")
        assert s["class_state"] == 1
        assert s["phase_state"] == 1
        assert s["structure_state"] == 1
        assert s["stability_state"] == 1


# ---------------------------------------------------------------------------
# Test 11 — Full geometry analysis convenience
# ---------------------------------------------------------------------------


class TestRunSyndromeGeometryAnalysis:
    def test_basic(self):
        results = [
            _make_result(phase="stable", compatibility=0.8, normalized_score=0.9),
            _make_result(phase="chaotic_transition", compatibility=0.1,
                         normalized_score=0.1),
            _make_result(phase="stable", compatibility=0.8, normalized_score=0.9),
        ]
        out = run_syndrome_geometry_analysis(results)
        assert "ternary_series" in out
        assert "transitions" in out
        assert out["ternary_series"]["n_steps"] == 3
        assert len(out["transitions"]) > 0

    def test_empty(self):
        out = run_syndrome_geometry_analysis([])
        assert out["ternary_series"]["n_steps"] == 0
        assert out["transitions"] == []


# ---------------------------------------------------------------------------
# Test 12 — Geometric distance vs Hamming
# ---------------------------------------------------------------------------


class TestGeometricVsHamming:
    def test_geometric_captures_magnitude(self):
        """Geometric distance distinguishes (0->1) from (0->-1) transitions
        differently than Hamming would for binary."""
        a = (0, 0, 0, 0)
        b = (1, 0, 0, 0)
        c = (-1, 0, 0, 0)
        # Hamming would treat both as distance 1 in binary.
        # Geometric: a->b = 1.0, a->c = 1.0, but b->c = 2.0
        assert compute_syndrome_distance(a, b) == 1.0
        assert compute_syndrome_distance(a, c) == 1.0
        assert compute_syndrome_distance(b, c) == 2.0
