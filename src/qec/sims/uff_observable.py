# SPDX-License-Identifier: MIT
"""UFF-style rotation-curve observable probe.

Deterministic circular velocity law and rotation-curve observation
for the micro-universe kernel.  All outputs are replay-safe with
byte-identical results for identical inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from qec.sims.universe_kernel import UniverseState

_EPS = 1e-9


def v_circ_uff(
    radii_kpc: Tuple[float, ...],
    theta: Tuple[float, ...],
) -> Tuple[float, ...]:
    """Compute UFF-style circular velocities.

    Parameters
    ----------
    radii_kpc : Tuple[float, ...]
        Galactocentric radii in kpc.
    theta : Tuple[float, ...]
        Model parameters ``(V0, Rc, beta)``.

    Returns
    -------
    Tuple[float, ...]
        Circular velocities in km/s, clipped non-negative.
    """
    V0, Rc, beta = theta[0], theta[1], theta[2]
    velocities: list[float] = []
    for R in radii_kpc:
        denom = math.sqrt(R * R + Rc * Rc + _EPS)
        base_arg = 1.0 - Rc / denom
        base = V0 * math.sqrt(max(base_arg, 0.0))
        tweak = (R / (R + Rc + _EPS)) ** beta
        v = base * (1.0 + 0.2 * tweak)
        velocities.append(max(v, 0.0))
    return tuple(velocities)


@dataclass(frozen=True)
class RotationCurveObservation:
    """Immutable rotation-curve observation."""

    radii_kpc: Tuple[float, ...]
    velocities_kms: Tuple[float, ...]
    peak_velocity: float
    mean_velocity: float
    timestep: int


def observe_rotation_curve(
    state: UniverseState,
    radii_kpc: Tuple[float, ...],
    theta: Tuple[float, ...],
) -> RotationCurveObservation:
    """Observe a rotation curve from the current universe state.

    Parameters
    ----------
    state : UniverseState
        Universe snapshot (used for timestep provenance).
    radii_kpc : Tuple[float, ...]
        Radii at which to evaluate the velocity law.
    theta : Tuple[float, ...]
        Model parameters ``(V0, Rc, beta)``.

    Returns
    -------
    RotationCurveObservation
        Frozen observation with computed metrics.
    """
    velocities = v_circ_uff(radii_kpc, theta)
    peak_velocity = max(velocities) if velocities else 0.0
    mean_velocity = sum(velocities) / len(velocities) if velocities else 0.0
    return RotationCurveObservation(
        radii_kpc=radii_kpc,
        velocities_kms=velocities,
        peak_velocity=peak_velocity,
        mean_velocity=mean_velocity,
        timestep=state.timestep,
    )


def curve_flatness_score(obs: RotationCurveObservation) -> float:
    """Compute rotation-curve flatness score.

    Defined as ``std(velocities) / max(mean_velocity, eps)``.

    Lower values indicate a flatter curve.

    Parameters
    ----------
    obs : RotationCurveObservation
        The observation to score.

    Returns
    -------
    float
        Deterministic flatness metric.
    """
    n = len(obs.velocities_kms)
    if n == 0:
        return 0.0
    mean = obs.mean_velocity
    variance = sum((v - mean) ** 2 for v in obs.velocities_kms) / n
    std = math.sqrt(variance)
    return std / max(mean, _EPS)
