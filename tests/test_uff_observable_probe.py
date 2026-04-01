# SPDX-License-Identifier: MIT
"""Tests for the UFF rotation-curve observable probe."""

from __future__ import annotations

import math

import pytest

from qec.sims.uff_observable import (
    RotationCurveObservation,
    curve_flatness_score,
    observe_rotation_curve,
    v_circ_uff,
)
from qec.sims.universe_kernel import UniverseState


# -- fixtures ---------------------------------------------------------------

def _make_state(timestep: int = 0) -> UniverseState:
    return UniverseState(
        field_amplitudes=(1.0, 0.5),
        qutrit_states=(0, 1),
        timestep=timestep,
        law_name="uff_test",
    )


_RADII = (1.0, 2.0, 5.0, 10.0, 20.0)
_THETA = (220.0, 2.0, 0.5)


# -- deterministic replay ---------------------------------------------------

class TestDeterministicReplay:

    def test_identical_inputs_give_identical_outputs(self):
        v1 = v_circ_uff(_RADII, _THETA)
        v2 = v_circ_uff(_RADII, _THETA)
        assert v1 == v2

    def test_observation_replay(self):
        state = _make_state()
        obs1 = observe_rotation_curve(state, _RADII, _THETA)
        obs2 = observe_rotation_curve(state, _RADII, _THETA)
        assert obs1 == obs2

    def test_stable_float_formatting(self):
        v = v_circ_uff(_RADII, _THETA)
        formatted1 = tuple(f"{x:.17g}" for x in v)
        formatted2 = tuple(f"{x:.17g}" for x in v)
        assert formatted1 == formatted2


# -- non-negative clipping --------------------------------------------------

class TestNonNegativeClipping:

    def test_all_velocities_non_negative(self):
        v = v_circ_uff(_RADII, _THETA)
        assert all(x >= 0.0 for x in v)

    def test_zero_radius(self):
        v = v_circ_uff((0.0,), _THETA)
        assert v[0] >= 0.0

    def test_extreme_parameters(self):
        v = v_circ_uff((0.0, 0.001, 1000.0), (0.0, 0.0, 0.0))
        assert all(x >= 0.0 for x in v)


# -- peak / mean correctness ------------------------------------------------

class TestPeakMean:

    def test_peak_is_max(self):
        state = _make_state()
        obs = observe_rotation_curve(state, _RADII, _THETA)
        assert obs.peak_velocity == max(obs.velocities_kms)

    def test_mean_correctness(self):
        state = _make_state()
        obs = observe_rotation_curve(state, _RADII, _THETA)
        expected_mean = sum(obs.velocities_kms) / len(obs.velocities_kms)
        assert obs.mean_velocity == expected_mean

    def test_timestep_provenance(self):
        state = _make_state(timestep=42)
        obs = observe_rotation_curve(state, _RADII, _THETA)
        assert obs.timestep == 42


# -- flatness score ----------------------------------------------------------

class TestFlatnessScore:

    def test_flat_curve_has_low_score(self):
        obs = RotationCurveObservation(
            radii_kpc=(1.0, 2.0, 3.0),
            velocities_kms=(200.0, 200.0, 200.0),
            peak_velocity=200.0,
            mean_velocity=200.0,
            timestep=0,
        )
        assert curve_flatness_score(obs) == 0.0

    def test_variable_curve_has_positive_score(self):
        obs = RotationCurveObservation(
            radii_kpc=(1.0, 2.0, 3.0),
            velocities_kms=(100.0, 200.0, 300.0),
            peak_velocity=300.0,
            mean_velocity=200.0,
            timestep=0,
        )
        score = curve_flatness_score(obs)
        assert score > 0.0
        expected_std = math.sqrt(
            ((100.0 - 200.0) ** 2 + (200.0 - 200.0) ** 2 + (300.0 - 200.0) ** 2) / 3
        )
        expected = expected_std / 200.0
        assert abs(score - expected) < 1e-12

    def test_empty_curve(self):
        obs = RotationCurveObservation(
            radii_kpc=(),
            velocities_kms=(),
            peak_velocity=0.0,
            mean_velocity=0.0,
            timestep=0,
        )
        assert curve_flatness_score(obs) == 0.0


# -- tuple-only storage ------------------------------------------------------

class TestTupleStorage:

    def test_radii_is_tuple(self):
        state = _make_state()
        obs = observe_rotation_curve(state, _RADII, _THETA)
        assert isinstance(obs.radii_kpc, tuple)

    def test_velocities_is_tuple(self):
        state = _make_state()
        obs = observe_rotation_curve(state, _RADII, _THETA)
        assert isinstance(obs.velocities_kms, tuple)

    def test_v_circ_returns_tuple(self):
        v = v_circ_uff(_RADII, _THETA)
        assert isinstance(v, tuple)


# -- frozen immutability -----------------------------------------------------

class TestFrozenImmutability:

    def test_observation_is_frozen(self):
        state = _make_state()
        obs = observe_rotation_curve(state, _RADII, _THETA)
        with pytest.raises(AttributeError):
            obs.peak_velocity = 999.0  # type: ignore[misc]

    def test_observation_is_frozen_velocities(self):
        state = _make_state()
        obs = observe_rotation_curve(state, _RADII, _THETA)
        with pytest.raises(AttributeError):
            obs.velocities_kms = ()  # type: ignore[misc]
