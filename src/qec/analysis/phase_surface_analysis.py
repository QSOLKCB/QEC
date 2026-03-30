"""v110.0.0 — Deterministic chain-length × threshold phase-surface layer."""

from __future__ import annotations

from typing import Any, Sequence

from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS
from qec.analysis.threshold_phase_map import run_threshold_phase_map

DEFAULT_CHAIN_LENGTH_SWEEP = (5, 9, 17)

UNSTABLE_SURFACE_STABILITY_MAX = 0.5
STABLE_SURFACE_STABILITY_MIN = 0.75
UNSTABLE_ONSET_DRIFT_MIN = 0.5
STABLE_ONSET_DRIFT_MAX = 0.25


def run_phase_surface_analysis(
    chain_lengths: Sequence[int] | None = None,
    threshold_values: Sequence[float] | None = None,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
) -> dict[str, Any]:
    """Run deterministic phase-surface analysis across chain lengths and onset thresholds."""
    ordered_chain_lengths = _ordered_chain_lengths(chain_lengths)

    surface_results: list[dict[str, Any]] = []
    chain_stability_curve: list[float] = []

    for chain_length in ordered_chain_lengths:
        phase_map = run_threshold_phase_map(
            chain_length=chain_length,
            threshold_values=threshold_values,
            perturbation_values=perturbation_values,
            diffusion_steps=diffusion_steps,
        )
        surface_results.append(phase_map)
        chain_stability_curve.append(_clamp01(float(phase_map["threshold_stability_score"])))

    threshold_values_metadata = _validated_threshold_values(surface_results)
    max_possible_shift, chain_max_possible_shifts = _onset_shift_metadata(surface_results)
    onset_drift_index = _onset_drift_index(
        surface_results,
        max_possible_shift=max_possible_shift,
        chain_max_possible_shifts=chain_max_possible_shifts,
    )
    surface_stability_score = _surface_stability_score(chain_stability_curve)

    return {
        "chain_lengths": tuple(ordered_chain_lengths),
        "threshold_values": threshold_values_metadata,
        "surface_results": surface_results,
        "onset_drift_index": onset_drift_index,
        "chain_stability_curve": chain_stability_curve,
        "surface_stability_score": surface_stability_score,
        "surface_class": _surface_class(
            surface_stability_score=surface_stability_score,
            onset_drift_index=onset_drift_index,
        ),
    }


def _ordered_chain_lengths(chain_lengths: Sequence[int] | None) -> list[int]:
    values = DEFAULT_CHAIN_LENGTH_SWEEP if chain_lengths is None else chain_lengths
    if len(values) == 0:
        raise ValueError("chain_lengths must be non-empty")
    ordered_values = [int(value) for value in values]
    for previous, current in zip(ordered_values, ordered_values[1:]):
        if current <= previous:
            raise ValueError("chain_lengths must be strictly increasing")
    return ordered_values


def _surface_stability_score(chain_stability_curve: list[float]) -> float:
    if len(chain_stability_curve) == 0:
        return 0.0
    return _clamp01(sum(float(score) for score in chain_stability_curve) / float(len(chain_stability_curve)))


def _validated_threshold_values(surface_results: list[dict[str, Any]]) -> tuple[float, ...]:
    reference = tuple(float(value) for value in surface_results[0]["threshold_values"])
    for surface_result in surface_results[1:]:
        current = tuple(float(value) for value in surface_result["threshold_values"])
        if current != reference:
            raise ValueError("all threshold_values must match across chain lengths")
    return reference


def _onset_drift_index(
    surface_results: list[dict[str, Any]],
    max_possible_shift: int,
    chain_max_possible_shifts: list[int],
) -> float:
    if len(surface_results) <= 1:
        return 0.0

    representative_indices = [
        _representative_onset_index(result, chain_max_possible_shift=chain_max_possible_shift)
        for result, chain_max_possible_shift in zip(surface_results, chain_max_possible_shifts)
    ]
    max_possible_shift_float = float(max_possible_shift)
    if max_possible_shift_float <= 0.0:
        return 0.0

    shifts = [
        abs(float(current) - float(previous))
        for previous, current in zip(representative_indices, representative_indices[1:])
    ]
    return _clamp01((sum(shifts) / float(len(shifts))) / max_possible_shift_float)


def _representative_onset_index(surface_result: dict[str, Any], chain_max_possible_shift: int) -> float:
    phase_results = list(surface_result.get("phase_results", []))
    if len(phase_results) == 0:
        return 0.0
    chain_max_index = float(chain_max_possible_shift)
    onset_indices: list[float] = []
    for result in phase_results:
        onset_index = result.get("onset_index")
        onset_indices.append(chain_max_index if onset_index is None else float(onset_index))
    return sum(onset_indices) / float(len(onset_indices))


def _onset_shift_metadata(surface_results: list[dict[str, Any]]) -> tuple[int, list[int]]:
    global_max_shift = 0
    chain_max_shifts: list[int] = []
    for surface_result in surface_results:
        chain_max_shift = 0
        phase_results = list(surface_result.get("phase_results", []))
        for result in phase_results:
            sweep_result = result.get("sweep_result")
            if isinstance(sweep_result, dict) and "perturbation_values" in sweep_result:
                perturbation_values = list(sweep_result.get("perturbation_values", []))
                chain_max_shift = max(chain_max_shift, max(0, len(perturbation_values) - 1))
            onset_index = result.get("onset_index")
            if onset_index is not None:
                chain_max_shift = max(chain_max_shift, int(onset_index))
        chain_max_shifts.append(chain_max_shift)
        global_max_shift = max(global_max_shift, chain_max_shift)
    return global_max_shift, chain_max_shifts


def _surface_class(surface_stability_score: float, onset_drift_index: float) -> str:
    if (
        float(surface_stability_score) >= STABLE_SURFACE_STABILITY_MIN
        and float(onset_drift_index) <= STABLE_ONSET_DRIFT_MAX
    ):
        return "stable_surface"
    if (
        float(surface_stability_score) < UNSTABLE_SURFACE_STABILITY_MAX
        or float(onset_drift_index) >= UNSTABLE_ONSET_DRIFT_MIN
    ):
        return "unstable_surface"
    return "mixed_surface"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["run_phase_surface_analysis"]
