"""Tests for invariant_registry.py — v105.0.0."""

from __future__ import annotations

import copy

import pytest

from qec.analysis.invariant_registry import (
    ROUND_PRECISION,
    canonicalize_invariant,
    detect_emergent_laws,
    format_invariant_registry,
    init_registry,
    run_invariant_registry_analysis,
    update_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_invariant(
    name: str = "stability_monotonicity",
    inv_type: str = "geometry",
    strength: float = 1.0,
) -> dict:
    return {
        "name": name,
        "type": inv_type,
        "description": f"{name} invariant",
        "strength": strength,
        "support": int(strength * 2),
        "total": 2,
    }


def _make_scored_invariants(invariants: list | None = None) -> dict:
    if invariants is None:
        invariants = [
            _make_invariant("stability_monotonicity", "geometry", 1.0),
            _make_invariant("basin_switch_suppression", "sign", 0.8),
        ]
    return {"scored_invariants": invariants}


def _make_run_result(scored_invariants: list | None = None) -> dict:
    if scored_invariants is None:
        scored_invariants = [
            _make_invariant("stability_monotonicity", "geometry", 1.0),
            _make_invariant("basin_switch_suppression", "sign", 0.8),
        ]
    return {
        "scored_invariants": scored_invariants,
        "global_metrics": {
            "primary_diagnosis": "oscillatory_trap",
            "topology_type": "mixed",
        },
    }


# ---------------------------------------------------------------------------
# Tests: init_registry
# ---------------------------------------------------------------------------


class TestInitRegistry:
    def test_returns_empty_dict(self):
        assert init_registry() == {}

    def test_deterministic(self):
        assert init_registry() == init_registry()


# ---------------------------------------------------------------------------
# Tests: canonicalize_invariant
# ---------------------------------------------------------------------------


class TestCanonicalizeInvariant:
    def test_basic_format(self):
        inv = {"type": "geometry", "name": "spiral_improvement"}
        assert canonicalize_invariant(inv) == "geometry:spiral_improvement"

    def test_missing_type(self):
        inv = {"name": "test"}
        assert canonicalize_invariant(inv) == "unknown:test"

    def test_missing_name(self):
        inv = {"type": "sign"}
        assert canonicalize_invariant(inv) == "sign:unknown"

    def test_deterministic(self):
        inv = {"type": "control", "name": "stability_correlation"}
        assert canonicalize_invariant(inv) == canonicalize_invariant(inv)


# ---------------------------------------------------------------------------
# Tests: update_registry
# ---------------------------------------------------------------------------


class TestUpdateRegistry:
    def test_first_update_creates_entry(self):
        reg = init_registry()
        inv_data = _make_scored_invariants()
        result = update_registry(reg, inv_data, run_id=0)
        assert len(result) == 2
        assert "geometry:stability_monotonicity" in result

    def test_count_increments(self):
        reg = init_registry()
        inv_data = _make_scored_invariants()
        reg = update_registry(reg, inv_data, run_id=0)
        reg = update_registry(reg, inv_data, run_id=1)
        assert reg["geometry:stability_monotonicity"]["count"] == 2

    def test_avg_strength_computed(self):
        reg = init_registry()
        inv1 = _make_scored_invariants([_make_invariant("test", "sign", 1.0)])
        inv2 = _make_scored_invariants([_make_invariant("test", "sign", 0.5)])
        reg = update_registry(reg, inv1, run_id=0)
        reg = update_registry(reg, inv2, run_id=1)
        assert reg["sign:test"]["avg_strength"] == 0.75

    def test_max_strength_tracked(self):
        reg = init_registry()
        inv1 = _make_scored_invariants([_make_invariant("test", "sign", 0.5)])
        inv2 = _make_scored_invariants([_make_invariant("test", "sign", 1.0)])
        reg = update_registry(reg, inv1, run_id=0)
        reg = update_registry(reg, inv2, run_id=1)
        assert reg["sign:test"]["max_strength"] == 1.0

    def test_timestamps_tracked(self):
        reg = init_registry()
        inv_data = _make_scored_invariants()
        reg = update_registry(reg, inv_data, run_id=5)
        reg = update_registry(reg, inv_data, run_id=10)
        key = "geometry:stability_monotonicity"
        assert reg[key]["first_seen"] == 5
        assert reg[key]["last_seen"] == 10

    def test_no_mutation_of_inputs(self):
        reg = init_registry()
        inv_data = _make_scored_invariants()
        reg_orig = copy.deepcopy(reg)
        inv_orig = copy.deepcopy(inv_data)
        update_registry(reg, inv_data, run_id=0)
        assert reg == reg_orig
        assert inv_data == inv_orig

    def test_deterministic(self):
        reg = init_registry()
        inv_data = _make_scored_invariants()
        r1 = update_registry(reg, inv_data, run_id=0)
        r2 = update_registry(reg, inv_data, run_id=0)
        assert r1 == r2

    def test_empty_invariants(self):
        reg = init_registry()
        result = update_registry(reg, {"scored_invariants": []}, run_id=0)
        assert result == {}

    def test_accepts_invariants_key(self):
        reg = init_registry()
        inv_data = {
            "invariants": [
                {
                    "name": "test",
                    "type": "sign",
                    "holds": True,
                    "support": 2,
                    "total": 2,
                },
            ]
        }
        result = update_registry(reg, inv_data, run_id=0)
        assert "sign:test" in result


# ---------------------------------------------------------------------------
# Tests: detect_emergent_laws
# ---------------------------------------------------------------------------


class TestDetectEmergentLaws:
    def test_empty_registry(self):
        assert detect_emergent_laws({}) == []

    def test_insufficient_count(self):
        reg = {
            "sign:test": {
                "count": 1,
                "avg_strength": 1.0,
                "max_strength": 1.0,
                "first_seen": 0,
                "last_seen": 0,
            }
        }
        assert detect_emergent_laws(reg) == []

    def test_detects_strong_frequent_invariant(self):
        reg = {
            "sign:test": {
                "count": 5,
                "avg_strength": 0.9,
                "max_strength": 1.0,
                "first_seen": 0,
                "last_seen": 4,
            }
        }
        laws = detect_emergent_laws(reg)
        assert len(laws) == 1
        assert laws[0]["law"] == "sign:test"
        assert laws[0]["support"] == 5
        assert 0.0 <= laws[0]["confidence"] <= 1.0

    def test_low_strength_not_detected(self):
        reg = {
            "sign:test": {
                "count": 5,
                "avg_strength": 0.2,
                "max_strength": 0.3,
                "first_seen": 0,
                "last_seen": 4,
            }
        }
        assert detect_emergent_laws(reg) == []

    def test_sorted_by_confidence_desc(self):
        reg = {
            "sign:weak": {
                "count": 3,
                "avg_strength": 0.65,
                "max_strength": 0.7,
                "first_seen": 0,
                "last_seen": 2,
            },
            "sign:strong": {
                "count": 5,
                "avg_strength": 0.95,
                "max_strength": 1.0,
                "first_seen": 0,
                "last_seen": 4,
            },
        }
        laws = detect_emergent_laws(reg)
        if len(laws) >= 2:
            assert laws[0]["confidence"] >= laws[1]["confidence"]

    def test_deterministic(self):
        reg = {
            "sign:test": {
                "count": 5,
                "avg_strength": 0.9,
                "max_strength": 1.0,
                "first_seen": 0,
                "last_seen": 4,
            }
        }
        assert detect_emergent_laws(reg) == detect_emergent_laws(reg)


# ---------------------------------------------------------------------------
# Tests: run_invariant_registry_analysis
# ---------------------------------------------------------------------------


class TestRunInvariantRegistryAnalysis:
    def test_returns_required_keys(self):
        results = [_make_run_result() for _ in range(3)]
        result = run_invariant_registry_analysis(results)
        assert "registry" in result
        assert "emergent_laws" in result
        assert "top_invariants" in result

    def test_registry_populated(self):
        results = [_make_run_result() for _ in range(3)]
        result = run_invariant_registry_analysis(results)
        assert len(result["registry"]) > 0

    def test_counts_accumulate(self):
        results = [_make_run_result() for _ in range(5)]
        result = run_invariant_registry_analysis(results)
        for key in result["registry"]:
            assert result["registry"][key]["count"] == 5

    def test_deterministic(self):
        results = [_make_run_result() for _ in range(3)]
        r1 = run_invariant_registry_analysis(results)
        r2 = run_invariant_registry_analysis(results)
        assert r1 == r2

    def test_no_mutation_of_inputs(self):
        results = [_make_run_result() for _ in range(3)]
        orig = copy.deepcopy(results)
        run_invariant_registry_analysis(results)
        assert results == orig

    def test_empty_input(self):
        result = run_invariant_registry_analysis([])
        assert result["registry"] == {}
        assert result["emergent_laws"] == []
        assert result["top_invariants"] == []

    def test_top_invariants_bounded(self):
        results = [_make_run_result() for _ in range(3)]
        result = run_invariant_registry_analysis(results)
        assert len(result["top_invariants"]) <= 10


# ---------------------------------------------------------------------------
# Tests: format_invariant_registry
# ---------------------------------------------------------------------------


class TestFormatInvariantRegistry:
    def test_contains_header(self):
        reg = {
            "sign:test": {
                "count": 3,
                "avg_strength": 0.9,
                "max_strength": 1.0,
                "first_seen": 0,
                "last_seen": 2,
            }
        }
        laws = [{"law": "sign:test", "support": 3, "confidence": 0.85}]
        text = format_invariant_registry(reg, laws)
        assert "Invariant Registry" in text
        assert "sign:test" in text
        assert "Emergent Laws" in text

    def test_empty_registry(self):
        text = format_invariant_registry({}, [])
        assert "No invariants" in text

    def test_no_laws(self):
        reg = {
            "sign:test": {
                "count": 1,
                "avg_strength": 0.5,
                "max_strength": 0.5,
                "first_seen": 0,
                "last_seen": 0,
            }
        }
        text = format_invariant_registry(reg, [])
        assert "No emergent laws" in text
