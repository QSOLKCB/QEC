"""v111.0.0 — Deterministic spectral phase-boundary tracker."""

from __future__ import annotations

from statistics import fmean
from typing import Any, Sequence

from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS
from qec.analysis.multi_regime_scaling import run_multi_regime_scaling

BOUNDARY_WEIGHT_ONSET_DRIFT = 0.5
BOUNDARY_WEIGHT_SPECTRAL_DRIFT = 0.5

PHASE_BOUNDARY_STABLE_MIN = 0.8
PHASE_BOUNDARY_DRIFTING_MIN = 0.4

PHASE_CLASS_STABLE = "stable_boundary"
PHASE_CLASS_DRIFTING = "drifting_boundary"
PHASE_CLASS_CRITICAL = "critical_boundary"

DOMINANT_COMPONENT_THRESHOLD = 0.1
DOMINANT_COMPONENT_ONSET = "onset_dominant"
DOMINANT_COMPONENT_SPECTRAL = "spectral_dominant"
DOMINANT_COMPONENT_BALANCED = "balanced_components"

COMPONENT_KEY_ONSET = "onset_component"
COMPONENT_KEY_SPECTRAL = "spectral_component"
COMPONENT_KEY_BALANCE_SCORE = "component_balance_score"
COMPONENT_KEY_DOMINANT = "dominant_component"
COMPONENT_KEYS = {
    COMPONENT_KEY_ONSET,
    COMPONENT_KEY_SPECTRAL,
    COMPONENT_KEY_BALANCE_SCORE,
    COMPONENT_KEY_DOMINANT,
}


def run_spectral_phase_boundary(
    chain_lengths: Sequence[int] | None = None,
    threshold_values: Sequence[float] | None = None,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
    return_components: bool = False,
) -> dict[str, Any]:
    """Run deterministic spectral phase-boundary tracking over chain growth."""
    multi_regime_result = run_multi_regime_scaling(
        chain_lengths=chain_lengths,
        threshold_values=threshold_values,
        perturbation_values=perturbation_values,
        diffusion_steps=diffusion_steps,
    )

    ordered_chain_lengths = tuple(int(value) for value in multi_regime_result["chain_lengths"])
    spectral_gap_curve = _spectral_gap_curve(multi_regime_result, ordered_chain_lengths)

    spectral_gap_drift = _spectral_gap_drift(spectral_gap_curve)
    onset_drift_index = _onset_drift_index(multi_regime_result)
    onset_component = _clamp01(BOUNDARY_WEIGHT_ONSET_DRIFT * onset_drift_index)
    spectral_component = _clamp01(BOUNDARY_WEIGHT_SPECTRAL_DRIFT * spectral_gap_drift)
    boundary_shift_score = _clamp01(
        onset_component + spectral_component
    )
    spectral_stability_score = _clamp01(1.0 - boundary_shift_score)

    result = {
        "chain_lengths": ordered_chain_lengths,
        "multi_regime_result": multi_regime_result,
        "spectral_gap_curve": spectral_gap_curve,
        "spectral_gap_drift": spectral_gap_drift,
        "boundary_shift_score": boundary_shift_score,
        "spectral_stability_score": spectral_stability_score,
        "phase_boundary_class": _phase_boundary_class(spectral_stability_score),
    }
    if return_components:
        result.update({
            COMPONENT_KEY_ONSET: onset_component,
            COMPONENT_KEY_SPECTRAL: spectral_component,
            COMPONENT_KEY_BALANCE_SCORE: _clamp01(1.0 - abs(onset_component - spectral_component)),
            COMPONENT_KEY_DOMINANT: _dominant_component(onset_component, spectral_component),
        })
    return result


def _spectral_gap_curve(multi_regime_result: dict[str, Any], chain_lengths: Sequence[int]) -> list[float]:
    per_chain_scores = _per_chain_regime_scores(multi_regime_result, chain_lengths)
    return [
        _clamp01(1.0 / (1.0 + float(score)))
        for score in per_chain_scores
    ]


def _per_chain_regime_scores(multi_regime_result: dict[str, Any], chain_lengths: Sequence[int]) -> list[float]:
    if len(chain_lengths) == 0:
        return []

    scaling_result = multi_regime_result.get("scaling_result", {})
    logical_error_curve = tuple(float(value) for value in scaling_result.get("logical_error_scaling_curve", []))
    global_transition_score = _clamp01(float(multi_regime_result.get("regime_transition_score", 0.0)))

    if len(logical_error_curve) != len(chain_lengths):
        return [global_transition_score for _ in chain_lengths]

    boundary_a, boundary_b = tuple(int(value) for value in multi_regime_result.get("regime_boundaries", (0, len(chain_lengths))))

    small_exponent = float(multi_regime_result.get("small_regime_exponent", 0.0))
    medium_exponent = float(multi_regime_result.get("medium_regime_exponent", 0.0))
    large_exponent = float(multi_regime_result.get("large_regime_exponent", 0.0))

    regime_exponents: list[float] = []
    for index in range(len(chain_lengths)):
        if index < boundary_a:
            regime_exponents.append(small_exponent)
        elif index < boundary_b:
            regime_exponents.append(medium_exponent)
        else:
            regime_exponents.append(large_exponent)

    scaling_exponent = float(scaling_result.get("scaling_exponent", 0.0))

    return [
        _clamp01(abs(regime_exponent - scaling_exponent) / (1.0 + abs(regime_exponent)))
        for regime_exponent in regime_exponents
    ]


def _spectral_gap_drift(gap_curve: Sequence[float]) -> float:
    if len(gap_curve) <= 1:
        return 0.0
    differences = [
        abs(float(current) - float(previous))
        for previous, current in zip(gap_curve, gap_curve[1:])
    ]
    return _clamp01(float(fmean(differences)))


def _onset_drift_index(multi_regime_result: dict[str, Any]) -> float:
    scaling_result = multi_regime_result.get("scaling_result", {})
    surface_result = scaling_result.get("surface_result", {})
    return _clamp01(float(surface_result.get("onset_drift_index", 0.0)))


def _phase_boundary_class(spectral_stability_score: float) -> str:
    if float(spectral_stability_score) >= PHASE_BOUNDARY_STABLE_MIN:
        return PHASE_CLASS_STABLE
    if float(spectral_stability_score) >= PHASE_BOUNDARY_DRIFTING_MIN:
        return PHASE_CLASS_DRIFTING
    return PHASE_CLASS_CRITICAL


def _dominant_component(onset_component: float, spectral_component: float) -> str:
    difference = float(onset_component) - float(spectral_component)
    if difference > DOMINANT_COMPONENT_THRESHOLD:
        return DOMINANT_COMPONENT_ONSET
    if difference < -DOMINANT_COMPONENT_THRESHOLD:
        return DOMINANT_COMPONENT_SPECTRAL
    return DOMINANT_COMPONENT_BALANCED


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["run_spectral_phase_boundary"]
