"""v109.1.0 — Deterministic phase-transition scalar analysis layer."""

from __future__ import annotations

from typing import Any, Sequence

from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS
from qec.analysis.robustness_sweep import run_robustness_sweep

ONSET_THRESHOLD = 0.5
SHARP_TRANSITION_DROP_THRESHOLD = 0.25


def run_phase_transition_analysis(
    chain_length: int,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
    onset_threshold: float = ONSET_THRESHOLD,
) -> dict[str, Any]:
    """Summarize a robustness sweep into deterministic bounded transition metrics."""
    sweep_result = run_robustness_sweep(
        chain_length=chain_length,
        perturbation_values=perturbation_values,
        diffusion_steps=diffusion_steps,
    )

    perturbation_axis = [float(value) for value in sweep_result["perturbation_values"]]
    robustness_curve = [_clamp01(float(score)) for score in sweep_result["robustness_curve"]]

    onset_index = _onset_index(robustness_curve, onset_threshold=onset_threshold)
    onset_perturbation = None if onset_index is None else float(perturbation_axis[onset_index])
    onset_drop_magnitude = _onset_drop_magnitude(robustness_curve, onset_index)

    return {
        "chain_length": int(chain_length),
        "sweep_result": sweep_result,
        "normalized_auc_score": _normalized_auc_score(perturbation_axis, robustness_curve),
        "onset_index": onset_index,
        "onset_perturbation": onset_perturbation,
        "onset_drop_magnitude": onset_drop_magnitude,
        "phase_transition_class": _phase_transition_class(onset_index, onset_drop_magnitude),
    }


def _normalized_auc_score(perturbation_values: list[float], robustness_curve: list[float]) -> float:
    if len(perturbation_values) != len(robustness_curve):
        raise ValueError("perturbation_values and robustness_curve must have equal length")
    if len(perturbation_values) == 0 or len(robustness_curve) == 0:
        return 0.0
    if len(perturbation_values) == 1 or len(robustness_curve) == 1:
        return _clamp01(robustness_curve[0])

    area = 0.0
    for previous_idx in range(len(perturbation_values) - 1):
        current_idx = previous_idx + 1
        delta_x = float(perturbation_values[current_idx] - perturbation_values[previous_idx])
        trapezoid_mean = 0.5 * (
            float(robustness_curve[previous_idx]) + float(robustness_curve[current_idx])
        )
        area += delta_x * trapezoid_mean

    max_possible_area = float(perturbation_values[-1] - perturbation_values[0])
    if max_possible_area <= 0.0:
        return 0.0
    return _clamp01(area / max_possible_area)


def _onset_index(robustness_curve: list[float], onset_threshold: float = ONSET_THRESHOLD) -> int | None:
    for idx, score in enumerate(robustness_curve):
        if float(score) < float(onset_threshold):
            return idx
    return None


def _onset_drop_magnitude(robustness_curve: list[float], onset_index: int | None) -> float:
    if onset_index is None or onset_index == 0:
        return 0.0
    previous_score = float(robustness_curve[onset_index - 1])
    onset_score = float(robustness_curve[onset_index])
    return _clamp01(previous_score - onset_score)


def _phase_transition_class(onset_index: int | None, onset_drop_magnitude: float) -> str:
    if onset_index is None:
        return "no_transition"
    if float(onset_drop_magnitude) >= SHARP_TRANSITION_DROP_THRESHOLD:
        return "sharp_transition"
    return "gradual_transition"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["run_phase_transition_analysis"]
