# SPDX-License-Identifier: MIT
"""Tests for the comparative universe experiment framework — v133.4.0."""

from __future__ import annotations

import pytest

from qec.sims.universe_kernel import UniverseState
from qec.sims.comparative_experiments import (
    UniverseComparison,
    compare_universes,
    run_anti_universe,
    run_lawful_universe,
)


def _make_initial_state() -> UniverseState:
    """Canonical initial state for reproducible tests."""
    return UniverseState(
        field_amplitudes=(1.0, 2.0, 3.0, 4.0),
        qutrit_states=(0, 1, 2),
        timestep=0,
        law_name="lawful",
    )


class TestLawfulUniverse:
    """Deterministic lawful evolution tests."""

    def test_history_length(self) -> None:
        state = _make_initial_state()
        history = run_lawful_universe(state, steps=10)
        assert len(history) == 11

    def test_initial_state_preserved(self) -> None:
        state = _make_initial_state()
        history = run_lawful_universe(state, steps=5)
        assert history[0] is state

    def test_timestep_increments(self) -> None:
        state = _make_initial_state()
        history = run_lawful_universe(state, steps=5)
        for i, s in enumerate(history):
            assert s.timestep == i

    def test_deterministic_replay(self) -> None:
        state = _make_initial_state()
        h1 = run_lawful_universe(state, steps=20)
        h2 = run_lawful_universe(state, steps=20)
        assert h1 == h2

    def test_field_decay(self) -> None:
        state = _make_initial_state()
        history = run_lawful_universe(state, steps=1)
        # After one step, fields should be smaller (decay dominates).
        for orig, evolved in zip(
            state.field_amplitudes, history[1].field_amplitudes
        ):
            assert evolved <= orig

    def test_returns_tuple(self) -> None:
        state = _make_initial_state()
        history = run_lawful_universe(state, steps=3)
        assert isinstance(history, tuple)


class TestAntiUniverse:
    """Deterministic anti-law evolution tests."""

    def test_history_length(self) -> None:
        state = _make_initial_state()
        history = run_anti_universe(state, steps=10)
        assert len(history) == 11

    def test_initial_state_preserved(self) -> None:
        state = _make_initial_state()
        history = run_anti_universe(state, steps=5)
        assert history[0] is state

    def test_timestep_increments(self) -> None:
        state = _make_initial_state()
        history = run_anti_universe(state, steps=5)
        for i, s in enumerate(history):
            assert s.timestep == i

    def test_deterministic_replay(self) -> None:
        state = _make_initial_state()
        h1 = run_anti_universe(state, steps=20)
        h2 = run_anti_universe(state, steps=20)
        assert h1 == h2

    def test_law_name_set(self) -> None:
        state = _make_initial_state()
        history = run_anti_universe(state, steps=1)
        assert history[1].law_name == "anti-law"

    def test_anti_decay_amplifies(self) -> None:
        # With all-neutral qutrits, anti-decay (1.001) should amplify.
        state = UniverseState(
            field_amplitudes=(1.0, 2.0, 3.0),
            qutrit_states=(0, 0, 0),
            timestep=0,
            law_name="lawful",
        )
        history = run_anti_universe(state, steps=1)
        for orig, evolved in zip(
            state.field_amplitudes, history[1].field_amplitudes
        ):
            assert evolved > orig

    def test_returns_tuple(self) -> None:
        state = _make_initial_state()
        history = run_anti_universe(state, steps=3)
        assert isinstance(history, tuple)


class TestDivergenceMetrics:
    """Tests for divergence correctness."""

    def test_divergence_score_computation(self) -> None:
        state = _make_initial_state()
        result = compare_universes(state, steps=50)
        expected = abs(result.lawful_final_energy - result.anti_final_energy)
        assert result.divergence_score == pytest.approx(expected)

    def test_energy_ratio_computation(self) -> None:
        state = _make_initial_state()
        result = compare_universes(state, steps=50)
        expected = result.anti_final_energy / result.lawful_final_energy
        assert result.energy_ratio == pytest.approx(expected)

    def test_divergence_positive_after_steps(self) -> None:
        state = _make_initial_state()
        result = compare_universes(state, steps=100)
        assert result.divergence_score > 0.0

    def test_anti_energy_exceeds_lawful(self) -> None:
        # Anti-law amplifies; lawful decays. Anti energy should be higher.
        state = _make_initial_state()
        result = compare_universes(state, steps=50)
        assert result.anti_final_energy > result.lawful_final_energy
        assert result.energy_ratio > 1.0


class TestCompareUniverses:
    """Tests for the comparison runner."""

    def test_step_count(self) -> None:
        state = _make_initial_state()
        result = compare_universes(state, steps=42)
        assert result.steps == 42

    def test_frozen_dataclass(self) -> None:
        state = _make_initial_state()
        result = compare_universes(state, steps=10)
        assert isinstance(result, UniverseComparison)
        try:
            result.steps = 999  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass

    def test_deterministic_comparison(self) -> None:
        state = _make_initial_state()
        r1 = compare_universes(state, steps=30)
        r2 = compare_universes(state, steps=30)
        assert r1 == r2

    def test_default_steps(self) -> None:
        state = _make_initial_state()
        result = compare_universes(state)
        assert result.steps == 100

    def test_zero_steps(self) -> None:
        state = _make_initial_state()
        result = compare_universes(state, steps=0)
        assert result.divergence_score == pytest.approx(0.0)
        assert result.steps == 0

    def test_zero_energy_ratio(self) -> None:
        """When lawful energy is zero, energy_ratio must be 0.0."""
        state = UniverseState(
            field_amplitudes=(0.0, 0.0, 0.0),
            qutrit_states=(0, 0, 0),
            timestep=0,
            law_name="lawful",
        )
        result = compare_universes(state, steps=10)
        assert result.lawful_final_energy == pytest.approx(0.0)
        assert result.anti_final_energy == pytest.approx(0.0)
        assert result.energy_ratio == 0.0
        assert result.divergence_score == pytest.approx(0.0)
