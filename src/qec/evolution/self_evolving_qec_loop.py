"""
Deterministic Self-Evolving QEC Loop — v136.8.5

Closed-loop adaptation layer that uses prior orchestration results
to deterministically improve routing decisions.

Loop law:
    State → Observe → Fingerprint → Route → Evaluate → Persist → Improve → Re-route

Integrates with:
    - v136.8.4 orchestration (OrchestratorDecision)
    - v136.8.3 audio cognition (CognitionCycleResult)
    - v136.8.1 snapshot schema (ControllerSnapshot)

All outputs are deterministic. Same input → same output → same bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from qec.ai.controller_snapshot_schema import (
    ControllerSnapshot,
    serialize_snapshot,
)
from qec.audio.audio_cognition_engine import CognitionCycleResult
from qec.orchestration.decoder_portfolio_orchestrator import (
    OrchestratorDecision,
)

# ---------------------------------------------------------------------------
# Valid adaptation actions (deterministic, rule-based)
# ---------------------------------------------------------------------------

VALID_ADAPTATION_ACTIONS: Tuple[str, ...] = (
    "RETAIN_PRIOR_ROUTE",
    "ESCALATE_PORTFOLIO",
    "REINITIALIZE_LATTICE",
    "SWITCH_CODE_FAMILY_PATH",
    "REDUCE_ROUTE_PRIORITY",
)

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvolutionStep:
    """A single evolution observation within the loop."""

    step_index: int
    prior_action: str
    observed_outcome: str
    confidence_delta: float
    improvement_score: float
    action_taken: str


@dataclass(frozen=True)
class EvolutionLedger:
    """Immutable, ordered ledger of evolution steps."""

    steps: Tuple[EvolutionStep, ...]
    cumulative_improvement: float
    stable_hash: str


@dataclass(frozen=True)
class EvolutionDecision:
    """Deterministic evolution decision produced by the loop."""

    selected_action: str
    confidence: float
    rationale: str
    improvement_applied: bool


@dataclass(frozen=True)
class EvolutionCycleResult:
    """Full result bundle from a single evolution cycle."""

    decision: EvolutionDecision
    ledger: EvolutionLedger
    snapshot_hash: str
    orchestrator_decision_action: str
    cognition_confidence: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_improvement_score(
    prior_confidence: float,
    new_confidence: float,
    invariant_passed: bool,
) -> float:
    """Compute deterministic improvement score.

    Uses only confidence delta and invariant status.
    Result is clamped to [0.0, 1.0].
    """
    if not invariant_passed:
        return 0.0
    delta = new_confidence - prior_confidence
    # Normalize delta from [-1, 1] to [0, 1]
    raw = (delta + 1.0) / 2.0
    return max(0.0, min(1.0, raw))


def compute_evolution_hash(ledger: EvolutionLedger) -> str:
    """Compute SHA-256 hash of a ledger using canonical JSON.

    Deterministic: same ledger → same hash.
    """
    steps_data = tuple(
        {
            "action_taken": s.action_taken,
            "confidence_delta": s.confidence_delta,
            "improvement_score": s.improvement_score,
            "observed_outcome": s.observed_outcome,
            "prior_action": s.prior_action,
            "step_index": s.step_index,
        }
        for s in sorted(ledger.steps, key=lambda s: s.step_index)
    )
    canonical = json.dumps(
        {
            "steps": steps_data,
            "cumulative_improvement": ledger.cumulative_improvement,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_evolution_ledger(ledger: EvolutionLedger) -> bool:
    """Validate ledger structural invariants."""
    # Steps must be sorted by step_index
    for i in range(len(ledger.steps)):
        if ledger.steps[i].step_index != i:
            return False

    # Cumulative improvement must be in bounds
    if not (0.0 <= ledger.cumulative_improvement <= 1.0):
        return False

    # Cumulative improvement must match recomputed clamped mean
    if len(ledger.steps) > 0:
        total = sum(s.improvement_score for s in ledger.steps)
        expected = max(0.0, min(1.0, total / len(ledger.steps)))
        if abs(ledger.cumulative_improvement - expected) > 1e-12:
            return False

    # Hash must match recomputed hash
    if ledger.stable_hash != compute_evolution_hash(ledger):
        return False

    # All improvement scores must be in bounds
    for step in ledger.steps:
        if not (0.0 <= step.improvement_score <= 1.0):
            return False

    return True


def record_evolution_step(
    step: EvolutionStep,
    ledger: Optional[EvolutionLedger] = None,
) -> EvolutionLedger:
    """Append a step to the ledger, returning a new immutable ledger."""
    if ledger is None:
        steps = (step,)
    else:
        steps = ledger.steps + (step,)

    # Recompute cumulative improvement as clamped mean
    total = sum(s.improvement_score for s in steps)
    cumulative = max(0.0, min(1.0, total / len(steps)))

    partial = EvolutionLedger(
        steps=steps,
        cumulative_improvement=cumulative,
        stable_hash="",
    )
    stable_hash = compute_evolution_hash(partial)
    return EvolutionLedger(
        steps=steps,
        cumulative_improvement=cumulative,
        stable_hash=stable_hash,
    )


def _determine_outcome(
    orchestrator_decision: OrchestratorDecision,
    snapshot: ControllerSnapshot,
) -> str:
    """Determine observed outcome from orchestrator decision and snapshot."""
    if not snapshot.invariant_passed:
        return "INVARIANT_FAILED"
    if orchestrator_decision.confidence >= 0.8:
        return "HIGH_CONFIDENCE_ROUTE"
    if orchestrator_decision.confidence >= 0.5:
        return "MODERATE_CONFIDENCE_ROUTE"
    return "LOW_CONFIDENCE_ROUTE"


def build_next_action(
    code_family: str,
    prior_decision: Optional[EvolutionDecision],
    ledger: EvolutionLedger,
    orchestrator_confidence: float = 0.5,
) -> str:
    """Canonical rule engine for adaptation action selection.

    Single source of truth for all action decisions.
    Pure rule-based. No randomness. No ML.
    """
    if len(ledger.steps) == 0:
        return "RETAIN_PRIOR_ROUTE"

    last_step = ledger.steps[-1]

    # If outcome was invariant failure, reinitialize (highest priority)
    if last_step.observed_outcome == "INVARIANT_FAILED":
        return "REINITIALIZE_LATTICE"

    # If cumulative improvement is high, retain
    if ledger.cumulative_improvement >= 0.65:
        return "RETAIN_PRIOR_ROUTE"

    # If last step showed poor improvement, apply family-aware escalation
    if last_step.improvement_score < 0.3:
        # QLDPC codes prefer switching family path on low improvement
        if code_family == "qldpc":
            return "SWITCH_CODE_FAMILY_PATH"
        if prior_decision is not None and prior_decision.selected_action == "ESCALATE_PORTFOLIO":
            return "SWITCH_CODE_FAMILY_PATH"
        return "ESCALATE_PORTFOLIO"

    # If prior was ESCALATE and confidence still weak, switch family path
    if (
        prior_decision is not None
        and prior_decision.selected_action == "ESCALATE_PORTFOLIO"
        and orchestrator_confidence < 0.4
    ):
        return "SWITCH_CODE_FAMILY_PATH"

    # Surface codes prefer reducing priority when orchestrator confidence is weak
    if code_family == "surface" and orchestrator_confidence < 0.4:
        return "REDUCE_ROUTE_PRIORITY"

    if orchestrator_confidence < 0.4:
        return "REDUCE_ROUTE_PRIORITY"

    return "RETAIN_PRIOR_ROUTE"


def run_evolution_cycle(
    orchestrator_decision: OrchestratorDecision,
    cognition_result: CognitionCycleResult,
    snapshot: ControllerSnapshot,
    prior_ledger: Optional[EvolutionLedger] = None,
    code_family: str = "surface",
) -> EvolutionCycleResult:
    """Run one full evolution cycle.

    Deterministic: same inputs → same result → same bytes.

    Parameters
    ----------
    orchestrator_decision : OrchestratorDecision
        Decision from the v136.8.4 portfolio orchestrator.
    cognition_result : CognitionCycleResult
        Result from the v136.8.3 audio cognition engine.
    snapshot : ControllerSnapshot
        Snapshot from the v136.8.1 schema.
    prior_ledger : EvolutionLedger, optional
        Previous ledger for continuation. None starts fresh.
    code_family : str
        Code family identifier for family-aware adaptation rules.
    """
    # Determine prior confidence from ledger
    if prior_ledger is not None and len(prior_ledger.steps) > 0:
        prior_confidence = prior_ledger.cumulative_improvement
    else:
        prior_confidence = 0.5  # neutral starting point

    new_confidence = orchestrator_decision.confidence
    cognition_confidence = cognition_result.match.confidence

    # Blend orchestrator and cognition confidence deterministically
    blended_confidence = (new_confidence + cognition_confidence) / 2.0

    improvement = compute_improvement_score(
        prior_confidence=prior_confidence,
        new_confidence=blended_confidence,
        invariant_passed=snapshot.invariant_passed,
    )

    # Determine outcome
    outcome = _determine_outcome(orchestrator_decision, snapshot)

    # Thread prior_action from previous cycle's action_taken
    step_index = 0 if prior_ledger is None else len(prior_ledger.steps)
    prior_action = "NONE"
    if prior_ledger is not None and len(prior_ledger.steps) > 0:
        prior_action = prior_ledger.steps[-1].action_taken

    # Reconstruct prior_decision from previous cycle for escalation logic
    prior_decision: Optional[EvolutionDecision] = None
    if prior_ledger is not None and len(prior_ledger.steps) > 0:
        last_prior = prior_ledger.steps[-1]
        prior_decision = EvolutionDecision(
            selected_action=last_prior.action_taken,
            confidence=blended_confidence,
            rationale="prior",
            improvement_applied=last_prior.improvement_score > 0.5,
        )

    # Create preliminary step for action computation
    preliminary_step = EvolutionStep(
        step_index=step_index,
        prior_action=prior_action,
        observed_outcome=outcome,
        confidence_delta=blended_confidence - prior_confidence,
        improvement_score=improvement,
        action_taken="",
    )
    temp_ledger = record_evolution_step(preliminary_step, prior_ledger)

    # Select adaptation action via single canonical rule engine
    adaptation_action = build_next_action(
        code_family=code_family,
        prior_decision=prior_decision,
        ledger=temp_ledger,
        orchestrator_confidence=orchestrator_decision.confidence,
    )

    # Create final step with action_taken
    step = EvolutionStep(
        step_index=step_index,
        prior_action=prior_action,
        observed_outcome=outcome,
        confidence_delta=blended_confidence - prior_confidence,
        improvement_score=improvement,
        action_taken=adaptation_action,
    )

    # Record final step
    ledger = record_evolution_step(step, prior_ledger)

    # Build rationale deterministically
    rationale = (
        f"action={adaptation_action} "
        f"improvement={improvement:.6f} "
        f"cumulative={ledger.cumulative_improvement:.6f} "
        f"outcome={outcome} "
        f"invariant={'PASS' if snapshot.invariant_passed else 'FAIL'}"
    )

    decision = EvolutionDecision(
        selected_action=adaptation_action,
        confidence=blended_confidence,
        rationale=rationale,
        improvement_applied=improvement > 0.5,
    )

    snapshot_hash = hashlib.sha256(
        serialize_snapshot(snapshot).encode("utf-8")
    ).hexdigest()

    return EvolutionCycleResult(
        decision=decision,
        ledger=ledger,
        snapshot_hash=snapshot_hash,
        orchestrator_decision_action=orchestrator_decision.policy_action,
        cognition_confidence=cognition_confidence,
    )


def export_evolution_bundle(result: EvolutionCycleResult) -> Dict[str, Any]:
    """Export evolution result as a deterministic dictionary.

    Canonical JSON-safe output. Same result → same bytes.
    """
    return {
        "decision": {
            "selected_action": result.decision.selected_action,
            "confidence": result.decision.confidence,
            "rationale": result.decision.rationale,
            "improvement_applied": result.decision.improvement_applied,
        },
        "ledger": {
            "steps": tuple(
                {
                    "action_taken": s.action_taken,
                    "confidence_delta": s.confidence_delta,
                    "improvement_score": s.improvement_score,
                    "observed_outcome": s.observed_outcome,
                    "prior_action": s.prior_action,
                    "step_index": s.step_index,
                }
                for s in result.ledger.steps
            ),
            "cumulative_improvement": result.ledger.cumulative_improvement,
            "stable_hash": result.ledger.stable_hash,
        },
        "snapshot_hash": result.snapshot_hash,
        "orchestrator_decision_action": result.orchestrator_decision_action,
        "cognition_confidence": result.cognition_confidence,
    }
