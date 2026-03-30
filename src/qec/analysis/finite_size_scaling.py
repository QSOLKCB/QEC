"""v110.1.0 — Deterministic finite-size scaling engine."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats

from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS
from qec.analysis.phase_surface_analysis import (
    DEFAULT_CHAIN_LENGTH_SWEEP,
    run_phase_surface_analysis,
)

MIN_SCALING_EXPONENT = 0.0
MAX_SCALING_EXPONENT = 10.0
DEFAULT_PSEUDO_THRESHOLD_CROSSING = 0.5
DEFAULT_CRITICAL_ERROR_CROSSING = 0.5
FIT_QUALITY_GOOD_MIN = 0.8
FAULT_TOLERANT_CRITICAL_MAX = 0.5

SCALING_CLASS_UNDER_THRESHOLD = "under_threshold"
SCALING_CLASS_NEAR_THRESHOLD = "near_threshold"
SCALING_CLASS_FAULT_TOLERANT = "fault_tolerant_regime"


def run_finite_size_scaling(
    chain_lengths: Sequence[int] | None = None,
    threshold_values: Sequence[float] | None = None,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
) -> dict[str, Any]:
    """Run deterministic finite-size scaling analysis from phase-surface outputs."""
    surface_result = run_phase_surface_analysis(
        chain_lengths=chain_lengths,
        threshold_values=threshold_values,
        perturbation_values=perturbation_values,
        diffusion_steps=diffusion_steps,
    )

    ordered_chain_lengths = tuple(
        int(value) for value in (
            DEFAULT_CHAIN_LENGTH_SWEEP if chain_lengths is None else chain_lengths
        )
    )
    chain_stability_curve = [
        _clamp01(float(value)) for value in surface_result.get("chain_stability_curve", [])
    ]
    default_error_proxy = 1.0 - _clamp01(float(surface_result.get("surface_stability_score", 0.0)))

    logical_error_scaling_curve = [
        _clamp01(
            1.0 - chain_stability_curve[index]
            if index < len(chain_stability_curve)
            else default_error_proxy
        )
        for index, _ in enumerate(ordered_chain_lengths)
    ]

    pseudo_threshold_estimate = _pseudo_threshold_estimate(
        ordered_chain_lengths,
        logical_error_scaling_curve,
    )

    fit = _loglog_fit(
        ordered_chain_lengths,
        logical_error_scaling_curve,
        crossing=DEFAULT_CRITICAL_ERROR_CROSSING,
    )
    critical_threshold_estimate = fit["critical_threshold_estimate"]
    scaling_exponent = fit["scaling_exponent"]
    fit_quality_score = fit["fit_quality_score"]

    if critical_threshold_estimate is None and pseudo_threshold_estimate is not None:
        critical_threshold_estimate = float(pseudo_threshold_estimate)

    return {
        "chain_lengths": ordered_chain_lengths,
        "surface_result": surface_result,
        "logical_error_scaling_curve": logical_error_scaling_curve,
        "pseudo_threshold_estimate": pseudo_threshold_estimate,
        "critical_threshold_estimate": critical_threshold_estimate,
        "scaling_exponent": scaling_exponent,
        "fit_quality_score": fit_quality_score,
        "scaling_class": _scaling_class(
            pseudo_threshold_estimate=pseudo_threshold_estimate,
            critical_threshold_estimate=critical_threshold_estimate,
            fit_quality_score=fit_quality_score,
        ),
    }


def _pseudo_threshold_estimate(
    chain_lengths: Sequence[int],
    logical_error_scaling_curve: Sequence[float],
) -> float | None:
    for chain_length, error_proxy in zip(chain_lengths, logical_error_scaling_curve):
        if float(error_proxy) < DEFAULT_PSEUDO_THRESHOLD_CROSSING:
            return _normalized_threshold_proxy(int(chain_length), chain_lengths)
    return None


def _loglog_fit(
    chain_lengths: Sequence[int],
    logical_error_scaling_curve: Sequence[float],
    crossing: float,
) -> dict[str, float | None]:
    x = np.asarray([float(value) for value in chain_lengths], dtype=np.float64)
    y = np.asarray([float(value) for value in logical_error_scaling_curve], dtype=np.float64)

    valid = (x > 0.0) & (y > 0.0)
    if np.count_nonzero(valid) < 2:
        return {
            "critical_threshold_estimate": None,
            "scaling_exponent": 0.0,
            "fit_quality_score": 0.0,
        }

    log_x = np.log(x[valid], dtype=np.float64)
    log_y = np.log(y[valid], dtype=np.float64)

    fit = stats.linregress(log_x, log_y)
    slope = float(fit.slope)
    intercept = float(fit.intercept)

    scaling_exponent = _clamp(float(abs(slope)), MIN_SCALING_EXPONENT, MAX_SCALING_EXPONENT)
    fit_quality_score = _clamp01(float(fit.rvalue) * float(fit.rvalue))

    if np.isclose(slope, 0.0):
        return {
            "critical_threshold_estimate": None,
            "scaling_exponent": scaling_exponent,
            "fit_quality_score": fit_quality_score,
        }

    crossing_chain = float(np.exp((np.log(float(crossing)) - intercept) / slope))
    critical_threshold_estimate = _normalized_threshold_proxy_from_float(crossing_chain, chain_lengths)

    return {
        "critical_threshold_estimate": critical_threshold_estimate,
        "scaling_exponent": scaling_exponent,
        "fit_quality_score": fit_quality_score,
    }


def _normalized_threshold_proxy(chain_length: int, chain_lengths: Sequence[int]) -> float:
    values = [int(value) for value in chain_lengths]
    minimum = float(min(values))
    maximum = float(max(values))
    if maximum <= minimum:
        return 0.0
    return _clamp01((float(chain_length) - minimum) / (maximum - minimum))


def _normalized_threshold_proxy_from_float(chain_length: float, chain_lengths: Sequence[int]) -> float:
    values = [int(value) for value in chain_lengths]
    minimum = float(min(values))
    maximum = float(max(values))
    if maximum <= minimum:
        return 0.0
    return _clamp01((float(chain_length) - minimum) / (maximum - minimum))


def _scaling_class(
    pseudo_threshold_estimate: float | None,
    critical_threshold_estimate: float | None,
    fit_quality_score: float,
) -> str:
    if pseudo_threshold_estimate is None:
        return SCALING_CLASS_UNDER_THRESHOLD
    if (
        critical_threshold_estimate is not None
        and float(critical_threshold_estimate) <= FAULT_TOLERANT_CRITICAL_MAX
        and float(fit_quality_score) >= FIT_QUALITY_GOOD_MIN
    ):
        return SCALING_CLASS_FAULT_TOLERANT
    return SCALING_CLASS_NEAR_THRESHOLD


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


__all__ = ["run_finite_size_scaling"]
