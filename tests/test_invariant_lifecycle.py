"""Tests for invariant lifecycle tracking — v105.1.0."""

from __future__ import annotations

import copy

import pytest

from qec.analysis.invariant_registry import (
    classify_law,
    init_registry,
    track_invariant_lifecycle,
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
            _make_invariant("basin_switch_suppression", "sign", 0.8),
        ]
    return {"scored_invariants": invariants}


# ---------------------------------------------------------------------------
# Tests: track_invariant_lifecycle
# ---------------------------------------------------------------------------


class TestTrackInvariantLifecycle:
    def test_stable_phase(self):
        entry = {
            "count": 10,
            "streak": 10,
            "break_count": 0,
            "last_observed": True,
            "avg_strength": 0.9,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 9,
        }
        result = track_invariant_lifecycle(entry)
        assert result["phase"] == "stable"
        assert result["trend"] == "strengthening"

    def test_decaying_phase(self):
        entry = {
            "count": 10,
            "streak": 1,
            "break_count": 6,
            "last_observed": False,
            "avg_strength": 0.5,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 15,
        }
        result = track_invariant_lifecycle(entry)
        assert result["phase"] == "decaying"
        assert result["trend"] == "weakening"

    def test_emerging_phase(self):
        entry = {
            "count": 1,
            "streak": 1,
            "break_count": 0,
            "last_observed": True,
            "avg_strength": 0.8,
            "max_strength": 0.8,
            "first_seen": 0,
            "last_seen": 0,
        }
        result = track_invariant_lifecycle(entry)
        assert result["phase"] == "emerging"
        assert result["trend"] == "strengthening"

    def test_unstable_phase(self):
        entry = {
            "count": 5,
            "streak": 1,
            "break_count": 1,
            "last_observed": True,
            "avg_strength": 0.7,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 5,
        }
        result = track_invariant_lifecycle(entry)
        assert result["phase"] == "unstable"

    def test_weakening_trend(self):
        entry = {
            "count": 10,
            "streak": 2,
            "break_count": 4,
            "last_observed": False,
            "avg_strength": 0.6,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 13,
        }
        result = track_invariant_lifecycle(entry)
        assert result["trend"] == "weakening"

    def test_oscillating_trend(self):
        entry = {
            "count": 5,
            "streak": 2,
            "break_count": 1,
            "last_observed": True,
            "avg_strength": 0.7,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 5,
        }
        result = track_invariant_lifecycle(entry)
        assert result["trend"] == "oscillating"

    def test_deterministic(self):
        entry = {
            "count": 5,
            "streak": 3,
            "break_count": 1,
            "last_observed": True,
            "avg_strength": 0.8,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 5,
        }
        r1 = track_invariant_lifecycle(entry)
        r2 = track_invariant_lifecycle(entry)
        assert r1 == r2

    def test_returns_required_keys(self):
        entry = {
            "count": 1,
            "streak": 1,
            "break_count": 0,
            "last_observed": True,
            "avg_strength": 0.5,
            "max_strength": 0.5,
            "first_seen": 0,
            "last_seen": 0,
        }
        result = track_invariant_lifecycle(entry)
        assert "phase" in result
        assert "trend" in result

    def test_valid_phases(self):
        for brk in [0, 1, 3, 6]:
            for streak in [0, 1, 3, 8]:
                entry = {
                    "count": 8,
                    "streak": streak,
                    "break_count": brk,
                    "last_observed": streak > 0,
                    "avg_strength": 0.7,
                    "max_strength": 1.0,
                    "first_seen": 0,
                    "last_seen": 10,
                }
                result = track_invariant_lifecycle(entry)
                assert result["phase"] in {"emerging", "stable", "unstable", "decaying"}
                assert result["trend"] in {"strengthening", "weakening", "oscillating"}


# ---------------------------------------------------------------------------
# Tests: classify_law
# ---------------------------------------------------------------------------


class TestClassifyLaw:
    def test_stable_law(self):
        entry = {
            "count": 10,
            "streak": 10,
            "break_count": 0,
        }
        assert classify_law(entry) == "stable_law"

    def test_fragile_law(self):
        entry = {
            "count": 5,
            "streak": 1,
            "break_count": 2,
        }
        assert classify_law(entry) == "fragile_law"

    def test_emerging_law(self):
        entry = {
            "count": 1,
            "streak": 1,
            "break_count": 0,
        }
        assert classify_law(entry) == "emerging_law"

    def test_decaying_law(self):
        entry = {
            "count": 10,
            "streak": 1,
            "break_count": 5,
        }
        assert classify_law(entry) == "decaying_law"

    def test_deterministic(self):
        entry = {"count": 5, "streak": 3, "break_count": 1}
        assert classify_law(entry) == classify_law(entry)

    def test_valid_classifications(self):
        valid = {"stable_law", "fragile_law", "emerging_law", "decaying_law"}
        for brk in [0, 1, 3, 6]:
            for streak in [0, 1, 3, 8]:
                entry = {"count": 8, "streak": streak, "break_count": brk}
                assert classify_law(entry) in valid


# ---------------------------------------------------------------------------
# Tests: update_registry_incremental streak tracking
# ---------------------------------------------------------------------------


class TestIncrementalStreakTracking:
    def test_streak_increments_on_presence(self):
        reg = init_registry()
        inv = _make_scored_invariants()
        reg = update_registry_incremental(reg, inv, run_id=0)
        reg = update_registry_incremental(reg, inv, run_id=1)
        key = "geometry:stability_monotonicity"
        assert reg[key]["streak"] == 2
        assert reg[key]["last_observed"] is True

    def test_streak_resets_on_absence(self):
        reg = init_registry()
        inv = _make_scored_invariants()
        reg = update_registry_incremental(reg, inv, run_id=0)
        # Update with empty invariants — invariant is absent.
        reg = update_registry_incremental(reg, {"scored_invariants": []}, run_id=1)
        key = "geometry:stability_monotonicity"
        assert reg[key]["streak"] == 0
        assert reg[key]["break_count"] == 1
        assert reg[key]["last_observed"] is False

    def test_break_count_accumulates(self):
        reg = init_registry()
        inv = _make_scored_invariants()
        reg = update_registry_incremental(reg, inv, run_id=0)
        reg = update_registry_incremental(reg, {"scored_invariants": []}, run_id=1)
        reg = update_registry_incremental(reg, {"scored_invariants": []}, run_id=2)
        key = "geometry:stability_monotonicity"
        assert reg[key]["break_count"] == 2

    def test_streak_resumes_after_break(self):
        reg = init_registry()
        inv = _make_scored_invariants()
        reg = update_registry_incremental(reg, inv, run_id=0)
        reg = update_registry_incremental(reg, {"scored_invariants": []}, run_id=1)
        reg = update_registry_incremental(reg, inv, run_id=2)
        key = "geometry:stability_monotonicity"
        assert reg[key]["streak"] == 1
        assert reg[key]["last_observed"] is True
        assert reg[key]["break_count"] == 1

    def test_no_mutation_of_inputs(self):
        reg = init_registry()
        inv = _make_scored_invariants()
        reg = update_registry_incremental(reg, inv, run_id=0)
        reg_orig = copy.deepcopy(reg)
        inv_orig = copy.deepcopy(inv)
        update_registry_incremental(reg, inv, run_id=1)
        assert reg == reg_orig
        assert inv == inv_orig

    def test_deterministic(self):
        reg = init_registry()
        inv = _make_scored_invariants()
        r1 = update_registry_incremental(reg, inv, run_id=0)
        r2 = update_registry_incremental(reg, inv, run_id=0)
        assert r1 == r2

    def test_new_invariants_not_marked_as_broken(self):
        """Invariants not previously in registry should not get breaks."""
        reg = init_registry()
        inv1 = _make_scored_invariants([_make_invariant("a", "sign", 0.9)])
        reg = update_registry_incremental(reg, inv1, run_id=0)
        # Add a different invariant in next run.
        inv2 = _make_scored_invariants([_make_invariant("b", "sign", 0.8)])
        reg = update_registry_incremental(reg, inv2, run_id=1)
        # "a" was absent in run 1, should be marked broken.
        assert reg["sign:a"]["break_count"] == 1
        # "b" is new in run 1, should not be broken.
        assert reg["sign:b"]["break_count"] == 0
