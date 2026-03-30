"""v109.0.0 — Deterministic robustness perturbation sweep."""

from __future__ import annotations

from typing import Any

from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS
from qec.analysis.protection_metrics import run_protection_metrics

DEFAULT_PERTURBATION_SWEEP = [0.25, 0.5, 1.0, 2.0]

ROBUSTNESS_CLASS_HIGHLY_STABLE_MONOTONICITY_THRESHOLD = 0.95
ROBUSTNESS_CLASS_HIGHLY_STABLE_MEAN_THRESHOLD = 0.67
ROBUSTNESS_CLASS_STABLE_MONOTONICITY_THRESHOLD = 0.75
ROBUSTNESS_CLASS_STABLE_MEAN_THRESHOLD = 0.34


def run_robustness_sweep(
    chain_length: int,
    perturbation_values: list[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
) -> dict[str, Any]:
    """Run a deterministic ordered perturbation sweep over protection metrics."""
    ordered_perturbation_values = _ordered_perturbation_values(perturbation_values)
    sweep_results = [
        run_protection_metrics(
            chain_length=chain_length,
            perturbation_magnitude=perturbation_magnitude,
            diffusion_steps=diffusion_steps,
        )
        for perturbation_magnitude in ordered_perturbation_values
    ]
    robustness_curve = [float(result["robustness_score"]) for result in sweep_results]

    monotonicity_score = _monotonicity_score(robustness_curve)
    curve_stability_score = _curve_stability_score(robustness_curve)
    mean_robustness = _mean_robustness(robustness_curve)

    return {
        "chain_length": int(chain_length),
        "perturbation_values": ordered_perturbation_values,
        "sweep_results": sweep_results,
        "robustness_curve": robustness_curve,
        "monotonicity_score": monotonicity_score,
        "curve_stability_score": curve_stability_score,
        "robustness_class": _robustness_class(monotonicity_score, mean_robustness),
    }


def _ordered_perturbation_values(perturbation_values: list[float] | None) -> list[float]:
    values = DEFAULT_PERTURBATION_SWEEP if perturbation_values is None else perturbation_values
    if len(values) == 0:
        raise ValueError("perturbation_values must be non-empty")
    return [float(value) for value in values]


def _monotonicity_score(robustness_curve: list[float]) -> float:
    if len(robustness_curve) <= 1:
        return 1.0
    transitions = [
        1.0 if current <= previous else 0.0
        for previous, current in zip(robustness_curve, robustness_curve[1:])
    ]
    return _clamp01(sum(transitions) / float(len(transitions)))


def _curve_stability_score(robustness_curve: list[float]) -> float:
    if len(robustness_curve) <= 2:
        return 1.0
    deltas = [current - previous for previous, current in zip(robustness_curve, robustness_curve[1:])]
    delta_variation = [
        abs(current_delta - previous_delta)
        for previous_delta, current_delta in zip(deltas, deltas[1:])
    ]
    mean_delta_variation = sum(delta_variation) / float(len(delta_variation))
    return _clamp01(1.0 / (1.0 + mean_delta_variation))


def _mean_robustness(robustness_curve: list[float]) -> float:
    if len(robustness_curve) == 0:
        return 0.0
    return _clamp01(sum(robustness_curve) / float(len(robustness_curve)))


def _robustness_class(monotonicity_score: float, mean_robustness: float) -> str:
    if (
        monotonicity_score >= ROBUSTNESS_CLASS_HIGHLY_STABLE_MONOTONICITY_THRESHOLD
        and mean_robustness >= ROBUSTNESS_CLASS_HIGHLY_STABLE_MEAN_THRESHOLD
    ):
        return "highly_stable"
    if (
        monotonicity_score >= ROBUSTNESS_CLASS_STABLE_MONOTONICITY_THRESHOLD
        and mean_robustness >= ROBUSTNESS_CLASS_STABLE_MEAN_THRESHOLD
    ):
        return "stable"
    return "fragile"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["run_robustness_sweep"]
