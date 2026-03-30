"""v109.2.0 — Deterministic threshold sensitivity / phase-map layer."""

from __future__ import annotations

from typing import Any, Sequence

from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS
from qec.analysis.phase_transition_analysis import run_phase_transition_analysis

DEFAULT_ONSET_THRESHOLD_SWEEP = [0.25, 0.5, 0.75]

UNSTABLE_THRESHOLD_STABILITY_MAX = 0.5
STABLE_THRESHOLD_STABILITY_MIN = 0.75
STABLE_TRANSITION_CONSISTENCY_MIN = 0.75


def run_threshold_phase_map(
    chain_length: int,
    threshold_values: Sequence[float] | None = None,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
) -> dict[str, Any]:
    """Run deterministic threshold sensitivity analysis over phase-transition outputs."""
    ordered_threshold_values = _ordered_threshold_values(threshold_values)

    phase_results: list[dict[str, Any]] = []
    onset_curve: list[float | None] = []

    for threshold in ordered_threshold_values:
        phase_result = run_phase_transition_analysis(
            chain_length=chain_length,
            perturbation_values=perturbation_values,
            diffusion_steps=diffusion_steps,
        )
        phase_result_with_threshold = dict(phase_result)
        phase_result_with_threshold["onset_threshold"] = float(threshold)
        phase_results.append(phase_result_with_threshold)
        onset_curve.append(phase_result_with_threshold["onset_perturbation"])

    threshold_stability_score = _threshold_stability_score(phase_results)
    transition_consistency_score = _transition_consistency_score(phase_results)

    return {
        "chain_length": int(chain_length),
        "threshold_values": ordered_threshold_values,
        "phase_results": phase_results,
        "onset_curve": onset_curve,
        "threshold_stability_score": threshold_stability_score,
        "transition_consistency_score": transition_consistency_score,
        "phase_map_class": _phase_map_class(
            threshold_stability_score=threshold_stability_score,
            transition_consistency_score=transition_consistency_score,
        ),
    }


def _ordered_threshold_values(threshold_values: Sequence[float] | None) -> list[float]:
    values = DEFAULT_ONSET_THRESHOLD_SWEEP if threshold_values is None else threshold_values
    if len(values) == 0:
        raise ValueError("threshold_values must be non-empty")
    ordered_values = [float(value) for value in values]
    for previous, current in zip(ordered_values, ordered_values[1:]):
        if current <= previous:
            raise ValueError("threshold_values must be strictly increasing")
    return ordered_values


def _threshold_stability_score(phase_results: list[dict[str, Any]]) -> float:
    if len(phase_results) <= 1:
        return 1.0
    onset_indices = [result.get("onset_index") for result in phase_results]
    stable_pairs = [1.0 if current == previous else 0.0 for previous, current in zip(onset_indices, onset_indices[1:])]
    return _clamp01(sum(stable_pairs) / float(len(stable_pairs)))


def _transition_consistency_score(phase_results: list[dict[str, Any]]) -> float:
    if len(phase_results) <= 1:
        return 1.0
    transition_classes = [str(result.get("phase_transition_class", "")) for result in phase_results]
    consistent_pairs = [
        1.0 if current == previous else 0.0
        for previous, current in zip(transition_classes, transition_classes[1:])
    ]
    return _clamp01(sum(consistent_pairs) / float(len(consistent_pairs)))


def _phase_map_class(threshold_stability_score: float, transition_consistency_score: float) -> str:
    if float(threshold_stability_score) < UNSTABLE_THRESHOLD_STABILITY_MAX:
        return "unstable_threshold_region"
    if (
        float(threshold_stability_score) >= STABLE_THRESHOLD_STABILITY_MIN
        and float(transition_consistency_score) >= STABLE_TRANSITION_CONSISTENCY_MIN
    ):
        return "stable_threshold_region"
    return "mixed_transition_region"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["run_threshold_phase_map"]
