"""
v136.8.7 — Deterministic Promotion History & Policy Drift Monitor

Tracks and analyzes the history of promotion / rollback decisions
across cycles.  Provides:

  - promotion lineage
  - rollback streak analysis
  - drift detection
  - policy regression warning
  - promotion stability scoring

Sits on top of:
  - v136.8.6 promotion_rollback_gate
  - v136.8.5 self_evolving_qec_loop
  - v136.8.4 orchestrator

All computations are deterministic.  Same input → same output → same bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

SEVERITY_NONE = "NONE"
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

VALID_SEVERITIES: Tuple[str, ...] = (
    SEVERITY_NONE,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL,
)

# ---------------------------------------------------------------------------
# Drift thresholds
# ---------------------------------------------------------------------------

DRIFT_CRITICAL_THRESHOLD = 0.80
DRIFT_HIGH_THRESHOLD = 0.60
DRIFT_MEDIUM_THRESHOLD = 0.40
DRIFT_LOW_THRESHOLD = 0.20

# ---------------------------------------------------------------------------
# Valid verdicts (mirrored from gate layer)
# ---------------------------------------------------------------------------

VALID_VERDICTS: Tuple[str, ...] = (
    "PROMOTE",
    "HOLD",
    "ROLLBACK",
    "BLOCKED_BY_INVARIANT",
    "INSUFFICIENT_EVIDENCE",
)

VALID_ACTIONS: Tuple[str, ...] = (
    "promote",
    "rollback",
    "hold",
)


# ---------------------------------------------------------------------------
# Dataclasses (all frozen)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromotionHistoryEntry:
    """Single promotion/rollback event in the history."""
    cycle_index: int
    verdict: str
    action: str
    confidence: float
    improvement_score: float
    snapshot_hash: str


@dataclass(frozen=True)
class DriftAlert:
    """Policy drift detection result."""
    drift_detected: bool
    drift_score: float
    rationale: str
    severity: str


@dataclass(frozen=True)
class PromotionHistoryLedger:
    """Immutable ledger of all promotion history entries."""
    entries: Tuple[PromotionHistoryEntry, ...]
    promotion_rate: float
    rollback_rate: float
    drift_score: float
    stable_hash: str


# ---------------------------------------------------------------------------
# History recording
# ---------------------------------------------------------------------------

def record_promotion_history(
    entry: PromotionHistoryEntry,
    ledger: Optional[PromotionHistoryLedger] = None,
) -> PromotionHistoryLedger:
    """Append an entry to the history ledger deterministically."""
    if ledger is None:
        entries: Tuple[PromotionHistoryEntry, ...] = (entry,)
    else:
        entries = ledger.entries + (entry,)

    promotion_rate, rollback_rate = _compute_rates(entries)
    drift_score = _compute_drift(entries)
    stable_hash = _hash_entries(entries)

    return PromotionHistoryLedger(
        entries=entries,
        promotion_rate=promotion_rate,
        rollback_rate=rollback_rate,
        drift_score=drift_score,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Rate computation
# ---------------------------------------------------------------------------

def compute_promotion_rates(ledger: PromotionHistoryLedger) -> Dict[str, float]:
    """Return promotion and rollback rates for the ledger."""
    promotion_rate, rollback_rate = _compute_rates(ledger.entries)
    return {
        "promotion_rate": promotion_rate,
        "rollback_rate": rollback_rate,
        "hold_rate": round(1.0 - promotion_rate - rollback_rate, 10),
    }


def _compute_rates(
    entries: Tuple[PromotionHistoryEntry, ...],
) -> Tuple[float, float]:
    """Deterministic rate computation from entry tuple."""
    if len(entries) == 0:
        return 0.0, 0.0
    total = len(entries)
    promotions = sum(1 for e in entries if e.action == "promote")
    rollbacks = sum(1 for e in entries if e.action == "rollback")
    return round(promotions / total, 10), round(rollbacks / total, 10)


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------

def compute_policy_drift_score(ledger: PromotionHistoryLedger) -> float:
    """Compute the policy drift score for a ledger."""
    return _compute_drift(ledger.entries)


def _compute_drift(entries: Tuple[PromotionHistoryEntry, ...]) -> float:
    """Deterministic drift computation.

    Drift is based on four factors:
      1. Promotion frequency shift (first half vs second half)
      2. Rollback streak length (max consecutive rollbacks / total)
      3. Confidence trend delta (mean of first half vs second half)
      4. Verdict transition instability (fraction of adjacent verdict changes)

    Each factor contributes equally (0.25 weight).
    Result is clamped to [0.0, 1.0].
    """
    if len(entries) < 2:
        return 0.0

    n = len(entries)
    mid = n // 2
    first_half = entries[:mid]
    second_half = entries[mid:]

    # 1. Promotion frequency shift
    first_promo_rate = (
        sum(1 for e in first_half if e.action == "promote") / len(first_half)
        if len(first_half) > 0 else 0.0
    )
    second_promo_rate = (
        sum(1 for e in second_half if e.action == "promote") / len(second_half)
        if len(second_half) > 0 else 0.0
    )
    freq_shift = abs(second_promo_rate - first_promo_rate)

    # 2. Rollback streak length
    max_streak = 0
    current_streak = 0
    for e in entries:
        if e.action == "rollback":
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    streak_ratio = min(max_streak / n, 1.0)

    # 3. Confidence trend delta
    first_conf = (
        sum(e.confidence for e in first_half) / len(first_half)
        if len(first_half) > 0 else 0.0
    )
    second_conf = (
        sum(e.confidence for e in second_half) / len(second_half)
        if len(second_half) > 0 else 0.0
    )
    conf_delta = abs(second_conf - first_conf)

    # 4. Verdict transition instability
    transitions = 0
    for i in range(1, n):
        if entries[i].verdict != entries[i - 1].verdict:
            transitions += 1
    instability = transitions / (n - 1)

    raw_score = 0.25 * freq_shift + 0.25 * streak_ratio + 0.25 * conf_delta + 0.25 * instability
    return round(max(0.0, min(1.0, raw_score)), 10)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def detect_policy_drift(ledger: PromotionHistoryLedger) -> DriftAlert:
    """Evaluate the ledger and return a deterministic drift alert."""
    score = _compute_drift(ledger.entries)
    severity = _score_to_severity(score)
    detected = severity != SEVERITY_NONE

    rationale_parts = []
    if detected:
        rationale_parts.append(f"Drift score {score:.4f} exceeds threshold.")
        n = len(ledger.entries)
        if n >= 2:
            mid = n // 2
            first_half = ledger.entries[:mid]
            second_half = ledger.entries[mid:]
            first_promo = (
                sum(1 for e in first_half if e.action == "promote") / len(first_half)
                if len(first_half) > 0 else 0.0
            )
            second_promo = (
                sum(1 for e in second_half if e.action == "promote") / len(second_half)
                if len(second_half) > 0 else 0.0
            )
            rationale_parts.append(
                f"Promotion rate shifted from {first_promo:.4f} to {second_promo:.4f}."
            )
    else:
        rationale_parts.append("No significant policy drift detected.")

    return DriftAlert(
        drift_detected=detected,
        drift_score=score,
        rationale=" ".join(rationale_parts),
        severity=severity,
    )


def _score_to_severity(score: float) -> str:
    """Map drift score to severity level deterministically."""
    if score >= DRIFT_CRITICAL_THRESHOLD:
        return SEVERITY_CRITICAL
    if score >= DRIFT_HIGH_THRESHOLD:
        return SEVERITY_HIGH
    if score >= DRIFT_MEDIUM_THRESHOLD:
        return SEVERITY_MEDIUM
    if score >= DRIFT_LOW_THRESHOLD:
        return SEVERITY_LOW
    return SEVERITY_NONE


# ---------------------------------------------------------------------------
# Ledger validation
# ---------------------------------------------------------------------------

def validate_history_ledger(ledger: PromotionHistoryLedger) -> bool:
    """Validate ledger integrity: hash, rates, and drift consistency."""
    expected_hash = _hash_entries(ledger.entries)
    if ledger.stable_hash != expected_hash:
        return False

    expected_promo, expected_rb = _compute_rates(ledger.entries)
    if ledger.promotion_rate != expected_promo:
        return False
    if ledger.rollback_rate != expected_rb:
        return False

    expected_drift = _compute_drift(ledger.entries)
    if ledger.drift_score != expected_drift:
        return False

    for i, entry in enumerate(ledger.entries):
        if entry.verdict not in VALID_VERDICTS:
            return False
        if entry.action not in VALID_ACTIONS:
            return False
        if entry.action != _verdict_to_action(entry.verdict):
            return False
        if entry.cycle_index != i:
            return False

    return True


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------

def compute_history_hash(ledger: PromotionHistoryLedger) -> str:
    """Compute deterministic SHA-256 hash for a ledger."""
    return _hash_entries(ledger.entries)


def _hash_entries(entries: Tuple[PromotionHistoryEntry, ...]) -> str:
    """Canonical JSON → SHA-256 hash of entry tuple."""
    canonical = json.dumps(
        [
            {
                "cycle_index": e.cycle_index,
                "verdict": e.verdict,
                "action": e.action,
                "confidence": e.confidence,
                "improvement_score": e.improvement_score,
                "snapshot_hash": e.snapshot_hash,
            }
            for e in entries
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Monitor cycle
# ---------------------------------------------------------------------------

def run_history_monitor_cycle(
    gate_result: Dict[str, Any],
    prior_ledger: Optional[PromotionHistoryLedger] = None,
) -> Dict[str, Any]:
    """Execute one full history monitor cycle.

    Parameters
    ----------
    gate_result : dict
        Output from ``run_gate_cycle`` (v136.8.6).
    prior_ledger : optional
        Previous history ledger for continuity.

    Returns
    -------
    dict
        Deterministic bundle with entry, ledger, alert, and hashes.
    """
    decision = gate_result["decision"]
    candidate = gate_result["candidate"]

    action = _verdict_to_action(decision.verdict)

    cycle_index = 0
    if prior_ledger is not None and len(prior_ledger.entries) > 0:
        cycle_index = prior_ledger.entries[-1].cycle_index + 1

    entry = PromotionHistoryEntry(
        cycle_index=cycle_index,
        verdict=decision.verdict,
        action=action,
        confidence=decision.confidence,
        improvement_score=candidate.improvement_score,
        snapshot_hash=gate_result["snapshot_hash"],
    )

    ledger = record_promotion_history(entry, prior_ledger)
    alert = detect_policy_drift(ledger)
    rates = compute_promotion_rates(ledger)

    return {
        "entry": entry,
        "ledger": ledger,
        "alert": alert,
        "rates": rates,
        "history_hash": ledger.stable_hash,
        "snapshot_hash": gate_result["snapshot_hash"],
    }


def _verdict_to_action(verdict: str) -> str:
    """Map gate verdict to history action deterministically."""
    if verdict == "PROMOTE":
        return "promote"
    if verdict in ("ROLLBACK", "BLOCKED_BY_INVARIANT"):
        return "rollback"
    if verdict in ("HOLD", "INSUFFICIENT_EVIDENCE"):
        return "hold"
    raise ValueError(f"Unknown gate verdict: {verdict!r}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_history_bundle(result: Dict[str, Any]) -> Dict[str, Any]:
    """Export a history monitor result as a JSON-serializable dict."""
    entry: PromotionHistoryEntry = result["entry"]
    ledger: PromotionHistoryLedger = result["ledger"]
    alert: DriftAlert = result["alert"]

    return {
        "entry": {
            "cycle_index": entry.cycle_index,
            "verdict": entry.verdict,
            "action": entry.action,
            "confidence": entry.confidence,
            "improvement_score": entry.improvement_score,
            "snapshot_hash": entry.snapshot_hash,
        },
        "ledger": {
            "entry_count": len(ledger.entries),
            "promotion_rate": ledger.promotion_rate,
            "rollback_rate": ledger.rollback_rate,
            "drift_score": ledger.drift_score,
            "stable_hash": ledger.stable_hash,
        },
        "alert": {
            "drift_detected": alert.drift_detected,
            "drift_score": alert.drift_score,
            "rationale": alert.rationale,
            "severity": alert.severity,
        },
        "rates": result["rates"],
        "history_hash": result["history_hash"],
        "snapshot_hash": result["snapshot_hash"],
    }
