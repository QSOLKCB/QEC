# SPDX-License-Identifier: MIT
"""Deterministic tests for qutrit-field coupling law (v133.2.0)."""

from __future__ import annotations

import pytest

from qec.sims.universe_kernel import UniverseState
from qec.sims.qutrit_coupling import (
    CouplingObservation,
    apply_qutrit_coupling,
    evolve_universe_coupled,
    observe_coupling,
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
# apply_qutrit_coupling — per-state multipliers
# ---------------------------------------------------------------------------

class TestNeutralState:
    def test_neutral_leaves_field_unchanged(self):
        result = apply_qutrit_coupling((1.0, 2.0, 3.0), (0, 0, 0))
        assert result == (1.0, 2.0, 3.0)


class TestAmplificationState:
    def test_state_1_amplifies(self):
        result = apply_qutrit_coupling((1.0, 2.0, 3.0), (1, 1, 1))
        assert result == (1.0 * 1.001, 2.0 * 1.001, 3.0 * 1.001)


class TestDampingState:
    def test_state_2_damps(self):
        result = apply_qutrit_coupling((1.0, 2.0, 3.0), (2, 2, 2))
        assert result == (1.0 * 0.998, 2.0 * 0.998, 3.0 * 0.998)


class TestMixedStates:
    def test_mixed_coupling(self):
        result = apply_qutrit_coupling((1.0, 2.0, 3.0), (0, 1, 2))
        assert result == (1.0 * 1.000, 2.0 * 1.001, 3.0 * 0.998)


# ---------------------------------------------------------------------------
# Cyclic qutrit repeat
# ---------------------------------------------------------------------------

class TestCyclicRepeat:
    def test_qutrits_shorter_than_fields(self):
        result = apply_qutrit_coupling((1.0, 2.0, 3.0, 4.0, 5.0), (0, 1))
        # indices: 0%2=0, 1%2=1, 2%2=0, 3%2=1, 4%2=0
        expected = (
            1.0 * 1.000,
            2.0 * 1.001,
            3.0 * 1.000,
            4.0 * 1.001,
            5.0 * 1.000,
        )
        assert result == expected

    def test_single_qutrit_broadcast(self):
        result = apply_qutrit_coupling((1.0, 2.0, 3.0), (2,))
        expected = (1.0 * 0.998, 2.0 * 0.998, 3.0 * 0.998)
        assert result == expected

    def test_empty_fields(self):
        result = apply_qutrit_coupling((), (0, 1, 2))
        assert result == ()

    def test_empty_qutrits(self):
        result = apply_qutrit_coupling((1.0, 2.0), ())
        assert result == (1.0, 2.0)


# ---------------------------------------------------------------------------
# Tuple-only output
# ---------------------------------------------------------------------------

class TestTupleOutput:
    def test_coupling_returns_tuple(self):
        result = apply_qutrit_coupling([1.0, 2.0], [0, 1])
        assert isinstance(result, tuple)

    def test_coupled_evolution_returns_tuple_fields(self):
        state = _make_state()
        s1 = evolve_universe_coupled(state)
        assert isinstance(s1.field_amplitudes, tuple)

    def test_coupling_from_lists(self):
        result = apply_qutrit_coupling([1.0, 2.0, 3.0], [0, 1, 2])
        expected = apply_qutrit_coupling((1.0, 2.0, 3.0), (0, 1, 2))
        assert result == expected


# ---------------------------------------------------------------------------
# evolve_universe_coupled
# ---------------------------------------------------------------------------

class TestCoupledEvolution:
    def test_timestep_increment(self):
        s0 = _make_state()
        s1 = evolve_universe_coupled(s0)
        assert s1.timestep == 1

    def test_decay_then_coupling(self):
        s0 = _make_state()
        s1 = evolve_universe_coupled(s0)
        # state 0 -> 1.0 * 0.999 * 1.000
        # state 1 -> 2.0 * 0.999 * 1.001
        # state 2 -> 3.0 * 0.999 * 0.998
        assert s1.field_amplitudes[0] == pytest.approx(1.0 * 0.999 * 1.000)
        assert s1.field_amplitudes[1] == pytest.approx(2.0 * 0.999 * 1.001)
        assert s1.field_amplitudes[2] == pytest.approx(3.0 * 0.999 * 0.998)

    def test_qutrit_states_preserved(self):
        s0 = _make_state()
        s1 = evolve_universe_coupled(s0)
        assert s1.qutrit_states == s0.qutrit_states

    def test_law_name_preserved(self):
        s0 = _make_state()
        s1 = evolve_universe_coupled(s0)
        assert s1.law_name == s0.law_name

    def test_original_state_unchanged(self):
        s0 = _make_state()
        _ = evolve_universe_coupled(s0)
        assert s0.timestep == 0
        assert s0.field_amplitudes == (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------

class TestDeterministicReplay:
    def test_repeated_coupling_equality(self):
        s0 = _make_state()
        s1a = evolve_universe_coupled(s0)
        s1b = evolve_universe_coupled(s0)
        assert s1a == s1b

    def test_multi_step_replay(self):
        s = _make_state()
        for _ in range(100):
            s = evolve_universe_coupled(s)
        s2 = _make_state()
        for _ in range(100):
            s2 = evolve_universe_coupled(s2)
        assert s == s2

    def test_multi_step_field_values_identical(self):
        s = _make_state()
        for _ in range(50):
            s = evolve_universe_coupled(s)
        s2 = _make_state()
        for _ in range(50):
            s2 = evolve_universe_coupled(s2)
        for i in range(len(s.field_amplitudes)):
            assert s.field_amplitudes[i] == s2.field_amplitudes[i]


# ---------------------------------------------------------------------------
# Coupling observation
# ---------------------------------------------------------------------------

class TestCouplingObservation:
    def test_observation_correctness(self):
        state = _make_state()  # qutrits (0, 1, 2) -> multipliers (1.0, 1.001, 0.998)
        obs = observe_coupling(state)
        assert obs.amplified_lanes == 1
        assert obs.damped_lanes == 1
        expected_mean = (1.000 + 1.001 + 0.998) / 3
        assert obs.mean_coupling_gain == pytest.approx(expected_mean)
        assert obs.timestep == 0

    def test_all_neutral(self):
        state = UniverseState(
            field_amplitudes=(1.0, 2.0),
            qutrit_states=(0, 0),
            timestep=5,
            law_name="test",
        )
        obs = observe_coupling(state)
        assert obs.amplified_lanes == 0
        assert obs.damped_lanes == 0
        assert obs.mean_coupling_gain == 1.0
        assert obs.timestep == 5

    def test_all_amplified(self):
        state = UniverseState(
            field_amplitudes=(1.0, 2.0, 3.0),
            qutrit_states=(1, 1, 1),
            timestep=0,
            law_name="test",
        )
        obs = observe_coupling(state)
        assert obs.amplified_lanes == 3
        assert obs.damped_lanes == 0

    def test_all_damped(self):
        state = UniverseState(
            field_amplitudes=(1.0, 2.0, 3.0),
            qutrit_states=(2, 2, 2),
            timestep=0,
            law_name="test",
        )
        obs = observe_coupling(state)
        assert obs.amplified_lanes == 0
        assert obs.damped_lanes == 3

    def test_empty_state(self):
        state = UniverseState(
            field_amplitudes=(),
            qutrit_states=(),
            timestep=0,
            law_name="empty",
        )
        obs = observe_coupling(state)
        assert obs.amplified_lanes == 0
        assert obs.damped_lanes == 0
        assert obs.mean_coupling_gain == 1.0

    def test_observation_determinism(self):
        state = _make_state()
        obs1 = observe_coupling(state)
        obs2 = observe_coupling(state)
        assert obs1 == obs2

    def test_cyclic_observation(self):
        state = UniverseState(
            field_amplitudes=(1.0, 2.0, 3.0, 4.0),
            qutrit_states=(1,),  # all lanes get amplified
            timestep=0,
            law_name="test",
        )
        obs = observe_coupling(state)
        assert obs.amplified_lanes == 4
        assert obs.damped_lanes == 0


# ---------------------------------------------------------------------------
# Invalid qutrit state validation
# ---------------------------------------------------------------------------

class TestInvalidQutritStates:
    def test_negative_state_raises(self):
        with pytest.raises(ValueError, match="qutrit_states must contain only values"):
            apply_qutrit_coupling((1.0, 2.0), (-1, 0))

    def test_state_3_raises(self):
        with pytest.raises(ValueError, match="qutrit_states must contain only values"):
            apply_qutrit_coupling((1.0,), (3,))

    def test_state_99_raises(self):
        with pytest.raises(ValueError, match="qutrit_states must contain only values"):
            apply_qutrit_coupling((1.0,), (99,))

    def test_observe_invalid_raises(self):
        state = UniverseState(
            field_amplitudes=(1.0,),
            qutrit_states=(5,),
            timestep=0,
            law_name="test",
        )
        with pytest.raises(ValueError, match="qutrit_states must contain only values"):
            observe_coupling(state)

    def test_evolve_coupled_invalid_raises(self):
        state = UniverseState(
            field_amplitudes=(1.0,),
            qutrit_states=(-1,),
            timestep=0,
            law_name="test",
        )
        with pytest.raises(ValueError, match="qutrit_states must contain only values"):
            evolve_universe_coupled(state)

    def test_same_invalid_input_same_exception(self):
        for _ in range(3):
            with pytest.raises(ValueError, match="qutrit_states must contain only values"):
                apply_qutrit_coupling((1.0,), (7,))


# ---------------------------------------------------------------------------
# Helper consistency — apply and observe use same lane semantics
# ---------------------------------------------------------------------------

class TestHelperConsistency:
    def test_coupling_and_observation_agree_on_lanes(self):
        state = UniverseState(
            field_amplitudes=(1.0, 2.0, 3.0, 4.0, 5.0),
            qutrit_states=(0, 1, 2),
            timestep=0,
            law_name="test",
        )
        coupled = apply_qutrit_coupling(
            state.field_amplitudes, state.qutrit_states
        )
        obs = observe_coupling(state)
        # Lane 0: state 0 -> 1.0 (neutral)
        # Lane 1: state 1 -> 1.001 (amplified)
        # Lane 2: state 2 -> 0.998 (damped)
        # Lane 3: state 0 -> 1.0 (neutral, cyclic)
        # Lane 4: state 1 -> 1.001 (amplified, cyclic)
        assert coupled[0] == 1.0 * 1.000
        assert coupled[1] == 2.0 * 1.001
        assert coupled[4] == 5.0 * 1.001
        assert obs.amplified_lanes == 2
        assert obs.damped_lanes == 1

    def test_single_qutrit_consistency(self):
        state = UniverseState(
            field_amplitudes=(1.0, 2.0, 3.0),
            qutrit_states=(2,),
            timestep=0,
            law_name="test",
        )
        obs = observe_coupling(state)
        assert obs.damped_lanes == 3
        assert obs.amplified_lanes == 0
        coupled = apply_qutrit_coupling(
            state.field_amplitudes, state.qutrit_states
        )
        assert all(coupled[i] == state.field_amplitudes[i] * 0.998 for i in range(3))


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    def test_coupling_observation_is_frozen(self):
        obs = CouplingObservation(
            mean_coupling_gain=1.0,
            amplified_lanes=1,
            damped_lanes=1,
            timestep=0,
        )
        with pytest.raises(AttributeError):
            obs.timestep = 99  # type: ignore[misc]

    def test_evolved_state_is_frozen(self):
        state = _make_state()
        s1 = evolve_universe_coupled(state)
        with pytest.raises(AttributeError):
            s1.timestep = 99  # type: ignore[misc]
