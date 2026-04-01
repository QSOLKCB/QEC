"""Tests for v132.1.0 deterministic DFA supervisor synthesizer."""

from __future__ import annotations

import dataclasses

import pytest

from qec.analysis.dfa_supervisor_synthesizer import (
    DFAStateMachine,
    run_dfa_supervisor_synthesizer,
    synthesize_supervisor,
)


def _make_plant(**kwargs) -> DFAStateMachine:
    defaults = {
        "states": ("s0",),
        "transitions": {},
        "initial_state": "s0",
        "safe_states": ("s0",),
        "controllable_events": (),
        "uncontrollable_events": (),
    }
    defaults.update(kwargs)
    return DFAStateMachine(**defaults)


class TestFrozenImmutability:
    def test_dfa_state_machine_frozen(self):
        plant = _make_plant()
        with pytest.raises(dataclasses.FrozenInstanceError):
            plant.states = ("s1",)


class TestSimpleSafeDFA:
    def test_all_safe(self):
        plant = _make_plant(
            states=("s0", "s1"),
            transitions={("s0", "a"): "s1", ("s1", "a"): "s0"},
            initial_state="s0",
            safe_states=("s0", "s1"),
            controllable_events=("a",),
        )
        result = synthesize_supervisor(plant)
        assert set(result["legal_states"]) == {"s0", "s1"}
        assert result["blocked_transitions"] == ()
        assert result["synthesis_score"] == 0.0
        assert result["synthesis_label"] == "safe"
        assert result["nonblocking_verified"] is True
        assert result["maximally_permissive"] is True


class TestUnsafeDeadEndPruning:
    def test_dead_end_pruned(self):
        # s2 is a dead-end with no path to safe state
        plant = _make_plant(
            states=("s0", "s1", "s2"),
            transitions={
                ("s0", "a"): "s1",
                ("s1", "b"): "s0",
                ("s0", "c"): "s2",
            },
            initial_state="s0",
            safe_states=("s0", "s1"),
            controllable_events=("a", "b", "c"),
        )
        result = synthesize_supervisor(plant)
        assert "s2" not in result["legal_states"]
        assert ("s0", "c") in result["blocked_transitions"]
        assert result["synthesis_score"] == 0.5
        assert result["synthesis_label"] == "warning"


class TestUncontrollableUnsafeEdge:
    def test_uncontrollable_forces_illegal(self):
        # s1 has uncontrollable transition to unsafe dead-end s2
        # so s1 must become illegal too
        plant = _make_plant(
            states=("s0", "s1", "s2"),
            transitions={
                ("s0", "a"): "s1",
                ("s1", "u"): "s2",
            },
            initial_state="s0",
            safe_states=("s0",),
            controllable_events=("a",),
            uncontrollable_events=("u",),
        )
        result = synthesize_supervisor(plant)
        assert "s2" not in result["legal_states"]
        assert "s1" not in result["legal_states"]
        assert ("s0", "a") in result["blocked_transitions"]


class TestControllableEdgeBlocking:
    def test_only_controllable_blocked(self):
        plant = _make_plant(
            states=("s0", "s1", "s2"),
            transitions={
                ("s0", "a"): "s1",
                ("s1", "b"): "s0",
                ("s0", "c"): "s2",
            },
            initial_state="s0",
            safe_states=("s0", "s1"),
            controllable_events=("a", "b", "c"),
        )
        result = synthesize_supervisor(plant)
        # Only controllable transitions are blocked
        for src, evt in result["blocked_transitions"]:
            assert evt in plant.controllable_events


class TestMaximalPermissiveness:
    def test_keeps_safe_transitions(self):
        plant = _make_plant(
            states=("s0", "s1", "s2", "s3"),
            transitions={
                ("s0", "a"): "s1",
                ("s1", "b"): "s0",
                ("s0", "c"): "s2",
                ("s2", "d"): "s0",
                ("s0", "e"): "s3",
            },
            initial_state="s0",
            safe_states=("s0", "s1", "s2"),
            controllable_events=("a", "b", "c", "d", "e"),
        )
        result = synthesize_supervisor(plant)
        # s3 is dead-end, pruned; s0, s1, s2 kept
        assert set(result["legal_states"]) == {"s0", "s1", "s2"}
        # Only e is blocked (leading to s3)
        assert ("s0", "e") in result["blocked_transitions"]
        assert result["maximally_permissive"] is True


class TestNonblockingVerification:
    def test_nonblocking_all_reach_safe(self):
        plant = _make_plant(
            states=("s0", "s1"),
            transitions={("s0", "a"): "s1", ("s1", "a"): "s0"},
            initial_state="s0",
            safe_states=("s0",),
            controllable_events=("a",),
        )
        result = synthesize_supervisor(plant)
        assert result["nonblocking_verified"] is True

    def test_nonblocking_trivial_safe(self):
        plant = _make_plant(
            states=("s0",),
            transitions={},
            initial_state="s0",
            safe_states=("s0",),
        )
        result = synthesize_supervisor(plant)
        assert result["nonblocking_verified"] is True


class TestDeterministicRepeatability:
    def test_repeated_runs_identical(self):
        plant = _make_plant(
            states=("s0", "s1", "s2", "s3"),
            transitions={
                ("s0", "a"): "s1",
                ("s1", "b"): "s2",
                ("s2", "c"): "s0",
                ("s0", "d"): "s3",
            },
            initial_state="s0",
            safe_states=("s0", "s1", "s2"),
            controllable_events=("a", "b", "c", "d"),
        )
        results = [synthesize_supervisor(plant) for _ in range(10)]
        for r in results[1:]:
            assert r == results[0]


class TestExactSchemaStability:
    def test_synthesize_schema(self):
        plant = _make_plant()
        result = synthesize_supervisor(plant)
        expected_keys = {
            "legal_states",
            "blocked_transitions",
            "maximally_permissive",
            "nonblocking_verified",
            "synthesis_score",
            "synthesis_label",
        }
        assert set(result.keys()) == expected_keys

    def test_runner_schema(self):
        plant = _make_plant()
        result = run_dfa_supervisor_synthesizer(plant)
        assert set(result.keys()) == {"plant", "synthesis", "supervisory_ready"}
        assert result["supervisory_ready"] is True
        assert result["plant"] is plant


class TestEmptyDFA:
    def test_empty_states(self):
        plant = DFAStateMachine(
            states=(),
            transitions={},
            initial_state="",
            safe_states=(),
            controllable_events=(),
            uncontrollable_events=(),
        )
        result = synthesize_supervisor(plant)
        assert result["legal_states"] == ()
        assert result["blocked_transitions"] == ()
        assert result["synthesis_score"] == 1.0
        assert result["synthesis_label"] == "critical"

    def test_single_unsafe_state(self):
        plant = _make_plant(
            states=("s0",),
            transitions={},
            initial_state="s0",
            safe_states=(),
        )
        result = synthesize_supervisor(plant)
        assert result["legal_states"] == ()
        assert result["synthesis_score"] == 1.0
        assert result["synthesis_label"] == "critical"


class TestRunnerIntegration:
    def test_runner_returns_synthesis(self):
        plant = _make_plant(
            states=("s0", "s1"),
            transitions={("s0", "a"): "s1", ("s1", "a"): "s0"},
            initial_state="s0",
            safe_states=("s0", "s1"),
            controllable_events=("a",),
        )
        result = run_dfa_supervisor_synthesizer(plant)
        assert result["supervisory_ready"] is True
        assert result["synthesis"]["synthesis_label"] == "safe"
