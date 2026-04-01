# SPDX-License-Identifier: MIT
"""Deterministic tests for the micro-universe simulation kernel (v133.0.0)."""

from __future__ import annotations

import pytest

from qec.sims.universe_kernel import (
    CREATED_BY_RELEASE,
    EXPORT_SCHEMA_VERSION,
    UniverseState,
    evolve_universe,
    to_simulation_export,
)
from qec.sims.observable_probe import UniverseObservation, observe_universe
from qec.simulation.export_codec import (
    export_to_json,
    load_from_json,
    validate_export_replay,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state() -> UniverseState:
    return UniverseState(
        field_amplitudes=(1.0, 2.0, 3.0),
        qutrit_states=(0, 1, 2),
        timestep=0,
        law_name="decay_0.999",
    )


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    def test_universe_state_is_frozen(self):
        state = _make_state()
        with pytest.raises(AttributeError):
            state.timestep = 99  # type: ignore[misc]

    def test_universe_observation_is_frozen(self):
        obs = UniverseObservation(
            mean_field_energy=1.0,
            active_qutrit_count=2,
            stability_score=0.5,
            timestep=0,
        )
        with pytest.raises(AttributeError):
            obs.timestep = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

class TestEvolution:
    def test_timestep_increment(self):
        s0 = _make_state()
        s1 = evolve_universe(s0)
        assert s1.timestep == 1

    def test_field_decay_correctness(self):
        s0 = _make_state()
        s1 = evolve_universe(s0)
        for i, f in enumerate(s1.field_amplitudes):
            assert f == s0.field_amplitudes[i] * 0.999

    def test_qutrit_persistence(self):
        s0 = _make_state()
        s1 = evolve_universe(s0)
        assert s1.qutrit_states == s0.qutrit_states

    def test_law_name_preserved(self):
        s0 = _make_state()
        s1 = evolve_universe(s0)
        assert s1.law_name == s0.law_name

    def test_repeated_evolution_equality(self):
        """Same input must produce identical output on repeated calls."""
        s0 = _make_state()
        s1a = evolve_universe(s0)
        s1b = evolve_universe(s0)
        assert s1a == s1b

    def test_multi_step_determinism(self):
        """Multi-step evolution must be replay-safe."""
        s = _make_state()
        for _ in range(100):
            s = evolve_universe(s)
        # Replay
        s2 = _make_state()
        for _ in range(100):
            s2 = evolve_universe(s2)
        assert s == s2

    def test_empty_fields(self):
        state = UniverseState(
            field_amplitudes=(),
            qutrit_states=(),
            timestep=0,
            law_name="empty",
        )
        s1 = evolve_universe(state)
        assert s1.field_amplitudes == ()
        assert s1.timestep == 1


# ---------------------------------------------------------------------------
# Observable probe
# ---------------------------------------------------------------------------

class TestObservableProbe:
    def test_mean_field_energy(self):
        state = _make_state()
        obs = observe_universe(state)
        expected = (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) / 3
        assert obs.mean_field_energy == pytest.approx(expected)

    def test_active_qutrit_count(self):
        state = _make_state()
        obs = observe_universe(state)
        # qutrit_states = (0, 1, 2) -> active (non-zero) = 2
        assert obs.active_qutrit_count == 2

    def test_stability_score(self):
        state = _make_state()
        obs = observe_universe(state)
        expected_energy = (1.0 + 4.0 + 9.0) / 3
        expected_score = expected_energy / (1 + 2)
        assert obs.stability_score == pytest.approx(expected_score)

    def test_timestep_propagation(self):
        state = _make_state()
        obs = observe_universe(state)
        assert obs.timestep == 0

    def test_observation_determinism(self):
        state = _make_state()
        obs1 = observe_universe(state)
        obs2 = observe_universe(state)
        assert obs1 == obs2

    def test_empty_state_observation(self):
        state = UniverseState(
            field_amplitudes=(),
            qutrit_states=(),
            timestep=0,
            law_name="empty",
        )
        obs = observe_universe(state)
        assert obs.mean_field_energy == 0.0
        assert obs.active_qutrit_count == 0
        assert obs.stability_score == 0.0


# ---------------------------------------------------------------------------
# Replay determinism
# ---------------------------------------------------------------------------

class TestReplayDeterminism:
    def test_evolve_observe_replay(self):
        """Full evolve-then-observe cycle must be deterministic."""
        s = _make_state()
        observations_a = []
        for _ in range(50):
            s = evolve_universe(s)
            observations_a.append(observe_universe(s))

        s = _make_state()
        observations_b = []
        for _ in range(50):
            s = evolve_universe(s)
            observations_b.append(observe_universe(s))

        assert observations_a == observations_b


# ---------------------------------------------------------------------------
# Export bridge compatibility
# ---------------------------------------------------------------------------

class TestExportBridge:
    def test_export_produces_valid_schema(self):
        state = _make_state()
        export = to_simulation_export(state)
        assert export.metadata.schema_version == EXPORT_SCHEMA_VERSION
        assert export.metadata.created_by_release == CREATED_BY_RELEASE
        assert len(export.metadata.trace_hash) == 64  # SHA-256 hex

    def test_export_round_trip(self):
        state = _make_state()
        export = to_simulation_export(state)
        json_str = export_to_json(export)
        reloaded = load_from_json(json_str)
        assert reloaded == export

    def test_export_replay_validation(self):
        state = _make_state()
        export = to_simulation_export(state)
        assert validate_export_replay(export)

    def test_export_determinism(self):
        state = _make_state()
        e1 = to_simulation_export(state)
        e2 = to_simulation_export(state)
        assert e1 == e2
        assert export_to_json(e1) == export_to_json(e2)

    def test_evolved_state_export(self):
        s = _make_state()
        for _ in range(10):
            s = evolve_universe(s)
        export = to_simulation_export(s)
        assert validate_export_replay(export)
        assert export.dwell_events == (10,)
