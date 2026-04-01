# SPDX-License-Identifier: MIT
"""Law sweep engine — v133.5.0.

Deterministic parameter sweep across law configurations.
Runs many universes across decay x coupling parameter spaces
and identifies stability / divergence phase transitions.

All operations are deterministic and replay-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.universe_kernel import UniverseState
from qec.sims.observable_probe import observe_universe


@dataclass(frozen=True)
class LawSweepConfig:
    """Frozen configuration for a law parameter sweep.

    All collection fields use tuples to enforce immutability.
    """

    decay_values: Tuple[float, ...]
    coupling_profiles: Tuple[Tuple[float, float, float], ...]
    steps: int
    label: str


@dataclass(frozen=True)
class LawSweepResult:
    """Frozen result from a single sweep run."""

    decay: float
    coupling_profile: Tuple[float, float, float]
    final_energy: float
    divergence_score: float
    regime_label: str


@dataclass(frozen=True)
class LawSweepSummary:
    """Frozen aggregate summary of a full sweep."""

    num_stable: int
    num_critical: int
    num_divergent: int
    max_divergence: float
    total_runs: int


# Phase classification tolerance.
_CRITICAL_TOLERANCE: float = 1e-9


def _classify_regime(initial_energy: float, final_energy: float) -> str:
    """Deterministic phase classification.

    stable:    final_energy < initial_energy
    critical:  approx equal within tolerance 1e-9
    divergent: final_energy > initial_energy
    """
    diff = final_energy - initial_energy
    if abs(diff) <= _CRITICAL_TOLERANCE:
        return "critical"
    if diff < 0.0:
        return "stable"
    return "divergent"


def _evolve_with_params(
    state: UniverseState,
    decay: float,
    coupling_profile: Tuple[float, float, float],
) -> UniverseState:
    """Evolve one step with parameterized decay and coupling.

    Steps:
        1. Apply decay multiplier per lane
        2. Apply coupling multipliers from profile (cyclic over qutrits)
        3. Increment timestep
    """
    # Step 1: parameterized decay
    decayed = tuple(f * decay for f in state.field_amplitudes)
    # Step 2: parameterized coupling
    n_fields = len(decayed)
    n_qutrits = len(state.qutrit_states)
    if n_fields == 0 or n_qutrits == 0:
        coupled = decayed
    else:
        coupled = tuple(
            decayed[i] * coupling_profile[state.qutrit_states[i % n_qutrits]]
            for i in range(n_fields)
        )
    return UniverseState(
        field_amplitudes=coupled,
        qutrit_states=state.qutrit_states,
        timestep=state.timestep + 1,
        law_name=state.law_name,
    )


def _run_single(
    initial_state: UniverseState,
    decay: float,
    coupling_profile: Tuple[float, float, float],
    steps: int,
    baseline_energy: float,
) -> LawSweepResult:
    """Run a single parameterized evolution and classify the result."""
    state = initial_state
    for _ in range(steps):
        state = _evolve_with_params(state, decay, coupling_profile)
    obs = observe_universe(state)
    final_energy = obs.mean_field_energy
    initial_energy = baseline_energy
    divergence_score = abs(final_energy - initial_energy)
    regime_label = _classify_regime(initial_energy, final_energy)
    return LawSweepResult(
        decay=decay,
        coupling_profile=coupling_profile,
        final_energy=final_energy,
        divergence_score=divergence_score,
        regime_label=regime_label,
    )


def run_law_sweep(
    initial_state: UniverseState,
    config: LawSweepConfig,
) -> Tuple[LawSweepResult, ...]:
    """Run a deterministic sweep across decay x coupling parameter space.

    For each combination of decay value and coupling profile, runs
    a parameterized evolution from the initial state and classifies
    the resulting phase regime.

    Parameters
    ----------
    initial_state : UniverseState
        Starting state for all sweep runs.
    config : LawSweepConfig
        Sweep configuration with parameter ranges.

    Returns
    -------
    Tuple[LawSweepResult, ...]
        One result per (decay, coupling_profile) combination.
        Ordered by decay values then coupling profiles.
    """
    initial_obs = observe_universe(initial_state)
    baseline_energy = initial_obs.mean_field_energy

    results = []
    for decay in config.decay_values:
        for coupling_profile in config.coupling_profiles:
            result = _run_single(
                initial_state, decay, coupling_profile,
                config.steps, baseline_energy,
            )
            results.append(result)
    return tuple(results)


def summarize_sweep(
    results: Tuple[LawSweepResult, ...],
) -> LawSweepSummary:
    """Compute aggregate summary from sweep results.

    Parameters
    ----------
    results : Tuple[LawSweepResult, ...]
        Results from run_law_sweep.

    Returns
    -------
    LawSweepSummary
        Frozen aggregate counts and max divergence.
    """
    num_stable = sum(1 for r in results if r.regime_label == "stable")
    num_critical = sum(1 for r in results if r.regime_label == "critical")
    num_divergent = sum(1 for r in results if r.regime_label == "divergent")
    max_divergence = max(
        (r.divergence_score for r in results), default=0.0,
    )
    return LawSweepSummary(
        num_stable=num_stable,
        num_critical=num_critical,
        num_divergent=num_divergent,
        max_divergence=max_divergence,
        total_runs=len(results),
    )
