"""Tests for cross-run invariant drift detection — v105.1.0."""

from __future__ import annotations

import copy

import pytest

from qec.analysis.invariant_registry import (
    detect_invariant_drift,
    init_registry,
    update_registry_incremental,
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
        ]
    return {"scored_invariants": invariants}


def _build_registry_with_streaks() -> dict:
    """Build a registry with varied streak/break patterns."""
    return {
        "geometry:stable": {
            "count": 10,
            "avg_strength": 0.9,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 9,
            "streak": 10,
            "break_count": 0,
            "last_observed": True,
        },
        "sign:drifting": {
            "count": 8,
            "avg_strength": 0.7,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 12,
            "streak": 1,
            "break_count": 3,
            "last_observed": False,
        },
        "topology:new": {
            "count": 2,
            "avg_strength": 0.8,
            "max_strength": 0.9,
            "first_seen": 8,
            "last_seen": 9,
            "streak": 2,
            "break_count": 0,
            "last_observed": True,
        },
        "control:lost": {
            "count": 5,
            "avg_strength": 0.6,
            "max_strength": 0.9,
            "first_seen": 0,
            "last_seen": 5,
            "streak": 0,
            "break_count": 4,
            "last_observed": False,
        },
    }


# ---------------------------------------------------------------------------
# Tests: detect_invariant_drift
# ---------------------------------------------------------------------------


class TestDetectInvariantDrift:
    def test_returns_required_keys(self):
        reg = _build_registry_with_streaks()
        result = detect_invariant_drift(reg)
        assert "drifting_invariants" in result
        assert "stable_invariants" in result
        assert "new_invariants" in result
        assert "lost_invariants" in result

    def test_stable_detected(self):
        reg = _build_registry_with_streaks()
        result = detect_invariant_drift(reg)
        stable_keys = [item["key"] for item in result["stable_invariants"]]
        assert "geometry:stable" in stable_keys

    def test_new_detected(self):
        reg = _build_registry_with_streaks()
        result = detect_invariant_drift(reg)
        new_keys = [item["key"] for item in result["new_invariants"]]
        assert "topology:new" in new_keys

    def test_lost_detected(self):
        reg = _build_registry_with_streaks()
        result = detect_invariant_drift(reg)
        lost_keys = [item["key"] for item in result["lost_invariants"]]
        assert "control:lost" in lost_keys

    def test_drifting_detected(self):
        reg = _build_registry_with_streaks()
        result = detect_invariant_drift(reg)
        drifting_keys = [item["key"] for item in result["drifting_invariants"]]
        assert "sign:drifting" in drifting_keys

    def test_empty_registry(self):
        result = detect_invariant_drift({})
        assert result["drifting_invariants"] == []
        assert result["stable_invariants"] == []
        assert result["new_invariants"] == []
        assert result["lost_invariants"] == []

    def test_all_stable(self):
        reg = {
            "sign:a": {
                "count": 5,
                "avg_strength": 0.9,
                "max_strength": 1.0,
                "first_seen": 0,
                "last_seen": 4,
                "streak": 5,
                "break_count": 0,
                "last_observed": True,
            },
        }
        result = detect_invariant_drift(reg)
        assert len(result["stable_invariants"]) == 1
        assert len(result["drifting_invariants"]) == 0
        assert len(result["lost_invariants"]) == 0
        assert len(result["new_invariants"]) == 0

    def test_deterministic(self):
        reg = _build_registry_with_streaks()
        r1 = detect_invariant_drift(reg)
        r2 = detect_invariant_drift(reg)
        assert r1 == r2

    def test_no_mutation_of_inputs(self):
        reg = _build_registry_with_streaks()
        orig = copy.deepcopy(reg)
        detect_invariant_drift(reg)
        assert reg == orig

    def test_items_include_lifecycle(self):
        reg = _build_registry_with_streaks()
        result = detect_invariant_drift(reg)
        for category in ["drifting_invariants", "stable_invariants",
                         "new_invariants", "lost_invariants"]:
            for item in result[category]:
                assert "key" in item
                assert "entry" in item
                assert "lifecycle" in item
                assert "phase" in item["lifecycle"]
                assert "trend" in item["lifecycle"]

    def test_sorted_deterministically(self):
        reg = _build_registry_with_streaks()
        result = detect_invariant_drift(reg)
        # All lists should be sorted by key (since we iterate sorted keys).
        for category in ["drifting_invariants", "stable_invariants",
                         "new_invariants", "lost_invariants"]:
            keys = [item["key"] for item in result[category]]
            assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Tests: drift detection with incremental updates
# ---------------------------------------------------------------------------


class TestDriftWithIncrementalUpdates:
    def test_drift_after_absence(self):
        reg = init_registry()
        inv = _make_scored_invariants()
        # Present for 5 runs.
        for i in range(5):
            reg = update_registry_incremental(reg, inv, run_id=i)
        # Absent for 5 runs.
        for i in range(5, 10):
            reg = update_registry_incremental(
                reg, {"scored_invariants": []}, run_id=i,
            )
        drift = detect_invariant_drift(reg)
        # Should be drifting or lost.
        all_problem_keys = (
            [item["key"] for item in drift["drifting_invariants"]]
            + [item["key"] for item in drift["lost_invariants"]]
        )
        assert "geometry:stability_monotonicity" in all_problem_keys

    def test_stable_after_consistent_presence(self):
        reg = init_registry()
        inv = _make_scored_invariants()
        for i in range(10):
            reg = update_registry_incremental(reg, inv, run_id=i)
        drift = detect_invariant_drift(reg)
        stable_keys = [item["key"] for item in drift["stable_invariants"]]
        assert "geometry:stability_monotonicity" in stable_keys

    def test_no_false_drift_for_new_invariants(self):
        reg = init_registry()
        inv = _make_scored_invariants()
        reg = update_registry_incremental(reg, inv, run_id=0)
        drift = detect_invariant_drift(reg)
        # Count=1, should be "new", not "drifting".
        new_keys = [item["key"] for item in drift["new_invariants"]]
        drifting_keys = [item["key"] for item in drift["drifting_invariants"]]
        assert "geometry:stability_monotonicity" in new_keys
        assert "geometry:stability_monotonicity" not in drifting_keys
