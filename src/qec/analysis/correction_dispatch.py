"""v111.2.0 — Deterministic correction dispatch control semantics."""

from __future__ import annotations

from typing import Any, Sequence

from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS
from qec.analysis.spectral_phase_boundary import (
    DOMINANT_COMPONENT_ONSET,
    DOMINANT_COMPONENT_SPECTRAL,
    PHASE_CLASS_CRITICAL,
    PHASE_CLASS_STABLE,
    run_spectral_phase_boundary,
)

ACTION_HOLD_STATE = "hold_state"
ACTION_LOCAL_STABILIZE = "local_stabilize"
ACTION_SPECTRAL_REBALANCE = "spectral_rebalance"
ACTION_BOUNDARY_INTERVENE = "boundary_intervene"

POLICY_MONITOR = "monitor_policy"
POLICY_LOCAL = "local_policy"
POLICY_SPECTRAL = "spectral_policy"
POLICY_INTERVENTION = "intervention_policy"

SPECTRAL_STABILITY_HOLD_MIN = 0.8
BOUNDARY_SHIFT_INTERVENE_MIN = 0.6

CYCLE_BUDGET_THRESHOLD_1 = 0.25
CYCLE_BUDGET_THRESHOLD_2 = 0.50
CYCLE_BUDGET_THRESHOLD_3 = 0.75

ACTION_TABLE = {
    PHASE_CLASS_STABLE: ACTION_HOLD_STATE,
    DOMINANT_COMPONENT_ONSET: ACTION_LOCAL_STABILIZE,
    DOMINANT_COMPONENT_SPECTRAL: ACTION_SPECTRAL_REBALANCE,
    PHASE_CLASS_CRITICAL: ACTION_BOUNDARY_INTERVENE,
}

POLICY_CLASS_BY_ACTION = {
    ACTION_HOLD_STATE: POLICY_MONITOR,
    ACTION_LOCAL_STABILIZE: POLICY_LOCAL,
    ACTION_SPECTRAL_REBALANCE: POLICY_SPECTRAL,
    ACTION_BOUNDARY_INTERVENE: POLICY_INTERVENTION,
}

REQUIRED_SPECTRAL_RESULT_KEYS = (
    "phase_boundary_class",
    "dominant_component",
    "boundary_shift_score",
    "spectral_stability_score",
)


def run_correction_dispatch(
    chain_lengths: Sequence[int] | None = None,
    threshold_values: Sequence[float] | None = None,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
    return_action_table: bool = False,
) -> dict[str, Any]:
    """Run deterministic correction dispatch and emit bounded control action metadata."""
    spectral_result = run_spectral_phase_boundary(
        chain_lengths=chain_lengths,
        threshold_values=threshold_values,
        perturbation_values=perturbation_values,
        diffusion_steps=diffusion_steps,
        return_components=True,
    )
    for required_key in REQUIRED_SPECTRAL_RESULT_KEYS:
        if required_key not in spectral_result:
            raise KeyError(f"spectral_result missing required key: {required_key}")

    phase_boundary_class = str(spectral_result["phase_boundary_class"])
    dominant_component = str(spectral_result["dominant_component"])
    spectral_stability_score = _clamp01(float(spectral_result["spectral_stability_score"]))
    boundary_shift_score = _clamp01(float(spectral_result["boundary_shift_score"]))

    correction_action = _select_correction_action(
        phase_boundary_class=phase_boundary_class,
        dominant_component=dominant_component,
        spectral_stability_score=spectral_stability_score,
        boundary_shift_score=boundary_shift_score,
    )
    dispatch_urgency_score = _clamp01(boundary_shift_score)
    dispatch_cycle_budget = _dispatch_cycle_budget(dispatch_urgency_score)
    correction_policy_class = POLICY_CLASS_BY_ACTION[correction_action]
    action_stability_score = _action_stability_score(
        correction_action=correction_action,
        phase_boundary_class=phase_boundary_class,
        dominant_component=dominant_component,
        boundary_shift_score=boundary_shift_score,
    )

    result: dict[str, Any] = {
        "chain_lengths": tuple(int(value) for value in spectral_result.get("chain_lengths", ())),
        "spectral_result": spectral_result,
        "correction_action": correction_action,
        "dispatch_urgency_score": dispatch_urgency_score,
        "dispatch_cycle_budget": dispatch_cycle_budget,
        "correction_policy_class": correction_policy_class,
        "action_stability_score": action_stability_score,
    }

    if return_action_table:
        result["action_table"] = dict(ACTION_TABLE)

    return result


def _select_correction_action(
    *,
    phase_boundary_class: str,
    dominant_component: str,
    spectral_stability_score: float,
    boundary_shift_score: float,
) -> str:
    if phase_boundary_class == PHASE_CLASS_STABLE and spectral_stability_score >= SPECTRAL_STABILITY_HOLD_MIN:
        return ACTION_HOLD_STATE
    if phase_boundary_class == PHASE_CLASS_CRITICAL or _should_intervene(boundary_shift_score):
        return ACTION_BOUNDARY_INTERVENE
    if dominant_component == DOMINANT_COMPONENT_ONSET:
        return ACTION_LOCAL_STABILIZE
    if dominant_component == DOMINANT_COMPONENT_SPECTRAL:
        return ACTION_SPECTRAL_REBALANCE
    return ACTION_HOLD_STATE


def _dispatch_cycle_budget(urgency: float) -> int:
    if urgency < CYCLE_BUDGET_THRESHOLD_1:
        return 1
    if urgency < CYCLE_BUDGET_THRESHOLD_2:
        return 2
    if urgency < CYCLE_BUDGET_THRESHOLD_3:
        return 3
    return 4


def _should_intervene(boundary_shift_score: float) -> bool:
    return float(boundary_shift_score) >= BOUNDARY_SHIFT_INTERVENE_MIN


def _action_stability_score(
    *,
    correction_action: str,
    phase_boundary_class: str,
    dominant_component: str,
    boundary_shift_score: float,
) -> float:
    if correction_action == ACTION_HOLD_STATE and phase_boundary_class == PHASE_CLASS_STABLE:
        return 1.0
    if correction_action == ACTION_LOCAL_STABILIZE and dominant_component == DOMINANT_COMPONENT_ONSET:
        return 1.0
    if correction_action == ACTION_SPECTRAL_REBALANCE and dominant_component == DOMINANT_COMPONENT_SPECTRAL:
        return 1.0
    if (
        correction_action == ACTION_BOUNDARY_INTERVENE
        and (phase_boundary_class == PHASE_CLASS_CRITICAL or _should_intervene(boundary_shift_score))
    ):
        return 1.0
    return _clamp01(0.5)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["run_correction_dispatch"]
