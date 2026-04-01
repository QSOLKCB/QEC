"""v112.0.0 — Deterministic cellular automata correction field layer."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from qec.analysis.correction_dispatch import (
    ACTION_BOUNDARY_INTERVENE,
    ACTION_HOLD_STATE,
    ACTION_LOCAL_STABILIZE,
    ACTION_SPECTRAL_REBALANCE,
    run_correction_dispatch,
)
from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS

FIELD_CLASS_BY_ACTION = {
    ACTION_HOLD_STATE: "stable_field",
    ACTION_LOCAL_STABILIZE: "adaptive_field",
    ACTION_SPECTRAL_REBALANCE: "adaptive_field",
    ACTION_BOUNDARY_INTERVENE: "intervention_field",
}


def run_cellular_correction_field(
    chain_length: int = 9,
    chain_state: Sequence[float] | None = None,
    chain_lengths: Sequence[int] | None = None,
    threshold_values: Sequence[float] | None = None,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
    automata_steps: int = 3,
    return_trace: bool = False,
) -> dict[str, Any]:
    """Run deterministic cellular correction field updates from dispatch policy."""
    dispatch_result = run_correction_dispatch(
        chain_lengths=chain_lengths,
        threshold_values=threshold_values,
        perturbation_values=perturbation_values,
        diffusion_steps=diffusion_steps,
    )

    resolved_chain_length = _resolve_chain_length(chain_length, chain_state)
    initial_array = _resolve_initial_field(
        chain_length=resolved_chain_length,
        chain_state=chain_state,
        dispatch_urgency_score=float(dispatch_result["dispatch_urgency_score"]),
    )

    correction_action = str(dispatch_result["correction_action"])
    corrected_array = initial_array.copy()
    trace: list[tuple[float, ...]] = []

    for _ in range(max(0, int(automata_steps))):
        corrected_array = _apply_automata_rule(corrected_array, correction_action)
        if return_trace:
            trace.append(tuple(float(value) for value in corrected_array))

    field_drift_score = _clamp01(float(np.mean(np.abs(corrected_array - initial_array))))
    local_stability_score = _clamp01(1.0 - field_drift_score)

    if correction_action not in FIELD_CLASS_BY_ACTION:
        raise KeyError(f"unknown correction_action for field class: {correction_action}")

    result: dict[str, Any] = {
        "chain_length": resolved_chain_length,
        "dispatch_result": dispatch_result,
        "initial_field": tuple(float(value) for value in initial_array),
        "corrected_field": tuple(float(value) for value in corrected_array),
        "field_drift_score": field_drift_score,
        "local_stability_score": local_stability_score,
        "correction_field_class": FIELD_CLASS_BY_ACTION[correction_action],
    }
    if return_trace:
        result["automata_trace"] = trace
    return result


def _resolve_chain_length(chain_length: int, chain_state: Sequence[float] | None) -> int:
    if chain_state is not None:
        flattened_chain_state = _flatten_chain_state(chain_state)
        flattened_length = int(flattened_chain_state.size)
        if int(chain_length) != flattened_length:
            raise ValueError("chain_length must match flattened chain_state length")
        return flattened_length
    return max(1, int(chain_length))


def _resolve_initial_field(
    *,
    chain_length: int,
    chain_state: Sequence[float] | None,
    dispatch_urgency_score: float,
) -> np.ndarray:
    if chain_state is not None:
        return _clamp_field(_flatten_chain_state(chain_state))
    return np.full(chain_length, _clamp01(dispatch_urgency_score), dtype=np.float64)


def _apply_automata_rule(field: np.ndarray, correction_action: str) -> np.ndarray:
    if correction_action == ACTION_HOLD_STATE:
        return field.copy()
    if correction_action == ACTION_LOCAL_STABILIZE:
        return _local_stabilize(field)
    if correction_action == ACTION_SPECTRAL_REBALANCE:
        return _spectral_rebalance(field)
    if correction_action == ACTION_BOUNDARY_INTERVENE:
        return _boundary_intervene(field)
    raise KeyError(f"unknown correction_action: {correction_action}")


def _local_stabilize(field: np.ndarray) -> np.ndarray:
    n = int(field.size)
    out = np.empty(n, dtype=np.float64)
    for index in range(n):
        lo = max(0, index - 1)
        hi = min(n, index + 2)
        out[index] = float(np.mean(field[lo:hi], dtype=np.float64))
    return _clamp_field(out)


def _spectral_rebalance(field: np.ndarray) -> np.ndarray:
    n = int(field.size)
    out = np.empty(n, dtype=np.float64)
    for index in range(n):
        lo = max(0, index - 2)
        hi = min(n, index + 3)
        out[index] = float(np.mean(field[lo:hi], dtype=np.float64))
    return _clamp_field(out)


def _boundary_intervene(field: np.ndarray) -> np.ndarray:
    n = int(field.size)
    if n <= 1:
        return _clamp_field(field.copy())

    out = _local_stabilize(field)
    out[0] = 0.5 * (float(field[0]) + float(field[1]))
    out[n - 1] = 0.5 * (float(field[n - 1]) + float(field[n - 2]))
    return _clamp_field(out)


def _clamp_field(field: np.ndarray) -> np.ndarray:
    return np.clip(field.astype(np.float64, copy=False), 0.0, 1.0)


def _flatten_chain_state(chain_state: Sequence[float]) -> np.ndarray:
    flattened = np.asarray(chain_state, dtype=np.float64).reshape(-1)
    if int(flattened.size) == 0:
        raise ValueError("chain_state must be non-empty")
    return flattened


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["run_cellular_correction_field"]
