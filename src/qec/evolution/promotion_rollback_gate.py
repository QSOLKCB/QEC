"""
v136.8.6 — Deterministic Promotion & Rollback Gate

Rule-based governance layer that decides whether a newly evolved route
may be promoted, must rollback, or requires further evidence.

Sits on top of:
  - v136.8.5 self-evolving loop
  - v136.8.4 portfolio orchestrator
  - v136.8.3 audio cognition
  - v136.8.1 snapshot schema

All decisions are deterministic.  Same input → same verdict → same bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

VERDICT_PROMOTE = "PROMOTE"
VERDICT_HOLD = "HOLD"
VERDICT_ROLLBACK = "ROLLBACK"
VERDICT_BLOCKED_BY_INVARIANT = "BLOCKED_BY_INVARIANT"
VERDICT_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

VALID_VERDICTS: Tuple[str, ...] = (
    VERDICT_PROMOTE,
    VERDICT_HOLD,
    VERDICT_ROLLBACK,
    VERDICT_BLOCKED_BY_INVARIANT,
    VERDICT_INSUFFICIENT_EVIDENCE,
)

# ---------------------------------------------------------------------------
# Thresholds (deterministic, no ML)
# ---------------------------------------------------------------------------

PROMOTION_IMPROVEMENT_THRESHOLD = 0.60
PROMOTION_CONFIDENCE_THRESHOLD = 0.75
PROMOTION_COGNITION_THRESHOLD = 0.75
PROMOTION_EVIDENCE_THRESHOLD = 0.60

ROLLBACK_CONFIDENCE_LOW = 0.35
ROLLBACK_COGNITION_LOW = 0.35
ROLLBACK_WEAK_IMPROVEMENT_THRESHOLD = 0.10
ROLLBACK_WEAK_STREAK_LENGTH = 3


# ---------------------------------------------------------------------------
# Dataclasses (frozen — immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromotionCandidate:
    """A candidate route evaluation assembled from upstream layers."""
    action: str
    confidence: float
    improvement_score: float
    invariant_passed: bool
    cognition_confidence: float
    evidence_score: float
    snapshot_hash: str


@dataclass(frozen=True)
class GateDecision:
    """The deterministic verdict for a single gate evaluation."""
    verdict: str
    promoted_action: str
    rollback_action: str
    rationale: str
    confidence: float


@dataclass(frozen=True)
class GateLedger:
    """Immutable, ordered record of all gate decisions."""
    decisions: Tuple[GateDecision, ...]
    cumulative_promotions: int
    cumulative_rollbacks: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_promotion_candidate(
    evolution_result: Any,
    orchestrator_decision: Any,
    cognition_result: Any,
    snapshot: Any,
) -> PromotionCandidate:
    """Assemble a PromotionCandidate from upstream layer outputs.

    Parameters
    ----------
    evolution_result : EvolutionCycleResult
        From self_evolving_qec_loop.run_evolution_cycle
    orchestrator_decision : OrchestratorDecision
        From decoder_portfolio_orchestrator.run_orchestration_cycle
    cognition_result : CognitionCycleResult
        From audio_cognition_engine.run_cognition_cycle
    snapshot : ControllerSnapshot
        From controller_snapshot_schema
    """
    # Extract action from evolution decision
    action = evolution_result.decision.selected_action

    # Confidence is the minimum of evolution and orchestrator confidence
    confidence = min(
        evolution_result.decision.confidence,
        orchestrator_decision.confidence,
    )

    # Improvement from evolution ledger
    improvement_score = evolution_result.ledger.cumulative_improvement

    # Invariant from snapshot
    invariant_passed = snapshot.invariant_passed

    # Cognition confidence from the match score
    cognition_confidence = cognition_result.match.score

    # Evidence score from snapshot
    evidence_score = snapshot.evidence_score

    # Snapshot hash
    snapshot_hash = evolution_result.snapshot_hash

    return PromotionCandidate(
        action=action,
        confidence=confidence,
        improvement_score=improvement_score,
        invariant_passed=invariant_passed,
        cognition_confidence=cognition_confidence,
        evidence_score=evidence_score,
        snapshot_hash=snapshot_hash,
    )


# ---------------------------------------------------------------------------
# Gate Evaluation — Promotion
# ---------------------------------------------------------------------------

def evaluate_promotion_gate(candidate: PromotionCandidate) -> GateDecision:
    """Evaluate a candidate against deterministic promotion / rejection rules.

    Decision priority (checked in order):
    1. invariant_passed == False  → BLOCKED_BY_INVARIANT
    2. evidence_score < threshold → INSUFFICIENT_EVIDENCE
    3. all thresholds met         → PROMOTE
    4. confidence & cognition both very low → ROLLBACK
    5. otherwise                  → HOLD
    """
    # --- BLOCKED_BY_INVARIANT ---
    if not candidate.invariant_passed:
        return GateDecision(
            verdict=VERDICT_BLOCKED_BY_INVARIANT,
            promoted_action="",
            rollback_action=candidate.action,
            rationale="Invariant check failed; promotion blocked.",
            confidence=candidate.confidence,
        )

    # --- INSUFFICIENT_EVIDENCE ---
    if candidate.evidence_score < PROMOTION_EVIDENCE_THRESHOLD:
        return GateDecision(
            verdict=VERDICT_INSUFFICIENT_EVIDENCE,
            promoted_action="",
            rollback_action="",
            rationale=(
                f"Evidence score {candidate.evidence_score:.4f} "
                f"below threshold {PROMOTION_EVIDENCE_THRESHOLD}."
            ),
            confidence=candidate.confidence,
        )

    # --- PROMOTE ---
    if (
        candidate.improvement_score >= PROMOTION_IMPROVEMENT_THRESHOLD
        and candidate.confidence >= PROMOTION_CONFIDENCE_THRESHOLD
        and candidate.cognition_confidence >= PROMOTION_COGNITION_THRESHOLD
    ):
        return GateDecision(
            verdict=VERDICT_PROMOTE,
            promoted_action=candidate.action,
            rollback_action="",
            rationale="All promotion thresholds met.",
            confidence=candidate.confidence,
        )

    # --- ROLLBACK (both confidences very low) ---
    if (
        candidate.confidence < ROLLBACK_CONFIDENCE_LOW
        and candidate.cognition_confidence < ROLLBACK_COGNITION_LOW
    ):
        return GateDecision(
            verdict=VERDICT_ROLLBACK,
            promoted_action="",
            rollback_action=candidate.action,
            rationale=(
                f"Confidence {candidate.confidence:.4f} and "
                f"cognition {candidate.cognition_confidence:.4f} "
                f"both below rollback threshold."
            ),
            confidence=candidate.confidence,
        )

    # --- HOLD (default) ---
    return GateDecision(
        verdict=VERDICT_HOLD,
        promoted_action="",
        rollback_action="",
        rationale="Thresholds not fully met; holding current route.",
        confidence=candidate.confidence,
    )


# ---------------------------------------------------------------------------
# Gate Evaluation — Rollback
# ---------------------------------------------------------------------------

def evaluate_rollback_gate(
    candidate: PromotionCandidate,
    prior_decision: Optional[GateDecision] = None,
) -> GateDecision:
    """Dedicated rollback evaluator with streak-aware logic.

    Returns a ROLLBACK decision when rollback conditions are met,
    otherwise delegates to evaluate_promotion_gate.
    """
    # Invariant failure always triggers rollback
    if not candidate.invariant_passed:
        return GateDecision(
            verdict=VERDICT_ROLLBACK,
            promoted_action="",
            rollback_action=candidate.action,
            rationale="Invariant failed; rollback required.",
            confidence=candidate.confidence,
        )

    # Both confidences very low
    if (
        candidate.confidence < ROLLBACK_CONFIDENCE_LOW
        and candidate.cognition_confidence < ROLLBACK_COGNITION_LOW
    ):
        return GateDecision(
            verdict=VERDICT_ROLLBACK,
            promoted_action="",
            rollback_action=candidate.action,
            rationale=(
                "Both confidence and cognition below rollback threshold."
            ),
            confidence=candidate.confidence,
        )

    # If prior decision was already ROLLBACK or HOLD and improvement is weak
    if prior_decision is not None and prior_decision.verdict in (
        VERDICT_ROLLBACK,
        VERDICT_HOLD,
    ):
        if candidate.improvement_score < ROLLBACK_WEAK_IMPROVEMENT_THRESHOLD:
            return GateDecision(
                verdict=VERDICT_ROLLBACK,
                promoted_action="",
                rollback_action=candidate.action,
                rationale=(
                    "Weak improvement after prior negative decision; "
                    "rollback enforced."
                ),
                confidence=candidate.confidence,
            )

    # Otherwise, fall through to standard gate
    return evaluate_promotion_gate(candidate)


# ---------------------------------------------------------------------------
# Ledger Operations
# ---------------------------------------------------------------------------

def _canonical_decision_dict(d: GateDecision) -> Dict[str, Any]:
    """Convert a GateDecision to a canonical dict for serialization."""
    return {
        "confidence": d.confidence,
        "promoted_action": d.promoted_action,
        "rationale": d.rationale,
        "rollback_action": d.rollback_action,
        "verdict": d.verdict,
    }


def _canonical_ledger_dict(ledger: GateLedger) -> Dict[str, Any]:
    """Convert a GateLedger to a canonical dict for hashing."""
    return {
        "cumulative_promotions": ledger.cumulative_promotions,
        "cumulative_rollbacks": ledger.cumulative_rollbacks,
        "decisions": [_canonical_decision_dict(d) for d in ledger.decisions],
    }


def compute_gate_hash(ledger: GateLedger) -> str:
    """Compute a deterministic SHA-256 hash of the ledger.

    Uses canonical JSON with sorted keys and no extra whitespace.
    """
    payload = _canonical_ledger_dict(ledger)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_gate_decision(
    decision: GateDecision,
    ledger: Optional[GateLedger] = None,
) -> GateLedger:
    """Append a decision to the ledger, returning a new immutable ledger."""
    if ledger is None:
        prior_decisions: Tuple[GateDecision, ...] = ()
        prior_promotions = 0
        prior_rollbacks = 0
    else:
        prior_decisions = ledger.decisions
        prior_promotions = ledger.cumulative_promotions
        prior_rollbacks = ledger.cumulative_rollbacks

    new_decisions = prior_decisions + (decision,)
    new_promotions = prior_promotions + (
        1 if decision.verdict == VERDICT_PROMOTE else 0
    )
    new_rollbacks = prior_rollbacks + (
        1 if decision.verdict == VERDICT_ROLLBACK else 0
    )

    # Build ledger with placeholder hash, then compute real hash
    temp_ledger = GateLedger(
        decisions=new_decisions,
        cumulative_promotions=new_promotions,
        cumulative_rollbacks=new_rollbacks,
        stable_hash="",
    )
    stable_hash = compute_gate_hash(temp_ledger)

    return GateLedger(
        decisions=new_decisions,
        cumulative_promotions=new_promotions,
        cumulative_rollbacks=new_rollbacks,
        stable_hash=stable_hash,
    )


def validate_gate_ledger(ledger: GateLedger) -> bool:
    """Validate structural invariants of a gate ledger."""
    # All decisions must have valid verdicts
    for d in ledger.decisions:
        if d.verdict not in VALID_VERDICTS:
            return False

    # Count consistency
    promotions = sum(
        1 for d in ledger.decisions if d.verdict == VERDICT_PROMOTE
    )
    rollbacks = sum(
        1 for d in ledger.decisions if d.verdict == VERDICT_ROLLBACK
    )
    if promotions != ledger.cumulative_promotions:
        return False
    if rollbacks != ledger.cumulative_rollbacks:
        return False

    # Hash stability — recompute and compare
    # The hash is computed with stable_hash="" so we reconstruct
    temp_ledger = GateLedger(
        decisions=ledger.decisions,
        cumulative_promotions=ledger.cumulative_promotions,
        cumulative_rollbacks=ledger.cumulative_rollbacks,
        stable_hash="",
    )
    expected_hash = compute_gate_hash(temp_ledger)
    if ledger.stable_hash != expected_hash:
        return False

    return True


# ---------------------------------------------------------------------------
# Streak Detection (for rollback with repeated weak improvement)
# ---------------------------------------------------------------------------

def _detect_weak_streak(ledger: GateLedger) -> bool:
    """Check if the last N decisions show a weak streak (HOLD/ROLLBACK)."""
    if len(ledger.decisions) < ROLLBACK_WEAK_STREAK_LENGTH:
        return False
    recent = ledger.decisions[-ROLLBACK_WEAK_STREAK_LENGTH:]
    return all(
        d.verdict in (VERDICT_HOLD, VERDICT_ROLLBACK) for d in recent
    )


# ---------------------------------------------------------------------------
# Full Gate Cycle
# ---------------------------------------------------------------------------

def run_gate_cycle(
    evolution_result: Any,
    orchestrator_decision: Any,
    cognition_result: Any,
    snapshot: Any,
    prior_ledger: Optional[GateLedger] = None,
) -> Dict[str, Any]:
    """Execute one full promotion/rollback gate cycle.

    Returns a deterministic bundle containing the decision, updated ledger,
    and metadata.
    """
    # 1. Build candidate
    candidate = build_promotion_candidate(
        evolution_result,
        orchestrator_decision,
        cognition_result,
        snapshot,
    )

    # 2. Determine prior decision (if any)
    prior_decision: Optional[GateDecision] = None
    if prior_ledger is not None and len(prior_ledger.decisions) > 0:
        prior_decision = prior_ledger.decisions[-1]

    # 3. Evaluate with rollback-aware gate
    decision = evaluate_rollback_gate(candidate, prior_decision)

    # 4. Streak-based rollback override
    if (
        prior_ledger is not None
        and decision.verdict == VERDICT_HOLD
        and _detect_weak_streak(prior_ledger)
    ):
        decision = GateDecision(
            verdict=VERDICT_ROLLBACK,
            promoted_action="",
            rollback_action=candidate.action,
            rationale=(
                f"Weak streak of {ROLLBACK_WEAK_STREAK_LENGTH} "
                f"consecutive non-promote decisions; rollback enforced."
            ),
            confidence=candidate.confidence,
        )

    # 5. Record decision
    updated_ledger = record_gate_decision(decision, prior_ledger)

    return {
        "candidate": candidate,
        "decision": decision,
        "ledger": updated_ledger,
        "snapshot_hash": candidate.snapshot_hash,
        "gate_hash": updated_ledger.stable_hash,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_gate_bundle(result: Dict[str, Any]) -> Dict[str, Any]:
    """Export a gate cycle result as a JSON-serializable dict."""
    candidate: PromotionCandidate = result["candidate"]
    decision: GateDecision = result["decision"]
    ledger: GateLedger = result["ledger"]

    return {
        "candidate": {
            "action": candidate.action,
            "confidence": candidate.confidence,
            "improvement_score": candidate.improvement_score,
            "invariant_passed": candidate.invariant_passed,
            "cognition_confidence": candidate.cognition_confidence,
            "evidence_score": candidate.evidence_score,
            "snapshot_hash": candidate.snapshot_hash,
        },
        "decision": _canonical_decision_dict(decision),
        "ledger": {
            "cumulative_promotions": ledger.cumulative_promotions,
            "cumulative_rollbacks": ledger.cumulative_rollbacks,
            "decision_count": len(ledger.decisions),
            "stable_hash": ledger.stable_hash,
        },
        "snapshot_hash": result["snapshot_hash"],
        "gate_hash": result["gate_hash"],
    }
