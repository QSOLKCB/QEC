"""v110.2.0 — Deterministic multi-regime finite-size scaling diagnostics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats

from qec.analysis.finite_size_scaling import run_finite_size_scaling
from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS

MIN_SCALING_EXPONENT = 0.0
MAX_SCALING_EXPONENT = 10.0
TRANSITION_SCORE_SCALE = 10.0

UNIFORM_SCALING_MAX = 0.15
TRANSITION_SCALING_MAX = 0.35

SCALING_CLASS_UNIFORM = "uniform_scaling"
SCALING_CLASS_TRANSITION = "transition_scaling"
SCALING_CLASS_SHIFTED = "regime_shifted_scaling"


def run_multi_regime_scaling(
    chain_lengths: Sequence[int] | None = None,
    threshold_values: Sequence[float] | None = None,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
) -> dict[str, Any]:
    """Run deterministic piecewise scaling diagnostics over three size regimes."""
    scaling_result = run_finite_size_scaling(
        chain_lengths=chain_lengths,
        threshold_values=threshold_values,
        perturbation_values=perturbation_values,
        diffusion_steps=diffusion_steps,
    )
    ordered_chain_lengths = tuple(int(value) for value in scaling_result["chain_lengths"])
    logical_error_scaling_curve = tuple(
        float(value) for value in scaling_result.get("logical_error_scaling_curve", [])
    )

    boundary_a, boundary_b = _regime_boundaries(len(ordered_chain_lengths))
    regime_boundaries = (boundary_a, boundary_b)

    small_regime_exponent = _regime_exponent(
        ordered_chain_lengths[:boundary_a],
        logical_error_scaling_curve[:boundary_a],
    )
    medium_regime_exponent = _regime_exponent(
        ordered_chain_lengths[boundary_a:boundary_b],
        logical_error_scaling_curve[boundary_a:boundary_b],
    )
    large_regime_exponent = _regime_exponent(
        ordered_chain_lengths[boundary_b:],
        logical_error_scaling_curve[boundary_b:],
    )

    transition_raw = 0.5 * (
        abs(medium_regime_exponent - small_regime_exponent)
        + abs(large_regime_exponent - medium_regime_exponent)
    )
    regime_transition_score = _clamp01(transition_raw / TRANSITION_SCORE_SCALE)

    return {
        "chain_lengths": ordered_chain_lengths,
        "scaling_result": scaling_result,
        "regime_boundaries": regime_boundaries,
        "small_regime_exponent": small_regime_exponent,
        "medium_regime_exponent": medium_regime_exponent,
        "large_regime_exponent": large_regime_exponent,
        "regime_transition_score": regime_transition_score,
        "scaling_regime_class": _scaling_regime_class(regime_transition_score),
    }


def _regime_boundaries(num_points: int) -> tuple[int, int]:
    if num_points <= 1:
        return (1 if num_points == 1 else 0, num_points)
    boundary_a = max(1, num_points // 3)
    boundary_b = max(boundary_a + 1, (2 * num_points) // 3)
    boundary_b = min(boundary_b, num_points)
    return (boundary_a, boundary_b)


def _regime_exponent(
    chain_lengths: Sequence[int],
    logical_error_scaling_curve: Sequence[float],
) -> float:
    x = np.asarray([float(value) for value in chain_lengths], dtype=np.float64)
    y = np.asarray([float(value) for value in logical_error_scaling_curve], dtype=np.float64)

    valid = (x > 0.0) & (y > 0.0)
    if np.count_nonzero(valid) < 2:
        return 0.0

    fit = stats.linregress(np.log(x[valid]), np.log(y[valid]))
    return _clamp(float(abs(fit.slope)), MIN_SCALING_EXPONENT, MAX_SCALING_EXPONENT)


def _scaling_regime_class(regime_transition_score: float) -> str:
    if regime_transition_score <= UNIFORM_SCALING_MAX:
        return SCALING_CLASS_UNIFORM
    if regime_transition_score <= TRANSITION_SCALING_MAX:
        return SCALING_CLASS_TRANSITION
    return SCALING_CLASS_SHIFTED


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


__all__ = ["run_multi_regime_scaling"]
