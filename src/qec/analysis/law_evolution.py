"""Deterministic Law Evolution & Theory Consistency Engine (v97.6.0).

Extends the Law system with:
- conflict detection between laws via interval overlap analysis
- deterministic conflict resolution
- confidence evolution over time
- append-only history tracking
- global theory consistency computation
- law pruning (weak / redundant removal)

Pipeline: laws -> evolve -> resolve conflicts -> prune -> consistent theory

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Optional, Tuple
import math

from qec.analysis.law_promotion import Condition, Law

# ---------------------------------------------------------------------------
# FLOAT PRECISION GUARD
# ---------------------------------------------------------------------------

_NORM_DIGITS = 12


def _norm(x: float) -> float:
    return round(float(x), _NORM_DIGITS)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CONFIDENCE_INCREMENT = 0.01
CONFIDENCE_DECREMENT = 0.05
CONFLICT_LOSS_FACTOR = 0.9
DECAY_FACTOR = 0.99
PRUNE_THRESHOLD = 0.1
CONFLICT_LOSS_PRUNE_COUNT = 3

# ---------------------------------------------------------------------------
# STEP 1 — INTERVAL EXTRACTION
# ---------------------------------------------------------------------------

_POS_INF = float("inf")
_NEG_INF = float("-inf")


def extract_intervals(law: Law) -> Dict[str, Tuple[float, float]]:
    """Convert law conditions into per-metric intervals.

    Each condition maps to an interval on its metric axis:
      gt/gte  -> (value, +inf)
      lt/lte  -> (-inf, value)
      eq      -> (value, value)

    Multiple conditions on the same metric are intersected.
    Returns {metric: (low, high)}.
    """
    intervals: Dict[str, Tuple[float, float]] = {}
    for cond in law.conditions:
        if cond.operator in ("gt", "gte"):
            iv = (cond.value, _POS_INF)
        elif cond.operator in ("lt", "lte"):
            iv = (_NEG_INF, cond.value)
        elif cond.operator == "eq":
            iv = (cond.value, cond.value)
        else:
            # neq — cannot be represented as a single interval; skip
            continue

        if cond.metric in intervals:
            old_lo, old_hi = intervals[cond.metric]
            new_lo = max(old_lo, iv[0])
            new_hi = min(old_hi, iv[1])
            intervals[cond.metric] = (new_lo, new_hi)
        else:
            intervals[cond.metric] = iv
    return intervals


# ---------------------------------------------------------------------------
# STEP 2 — INTERVAL OVERLAP
# ---------------------------------------------------------------------------


def intervals_overlap(
    interval_a: Tuple[float, float],
    interval_b: Tuple[float, float],
) -> str:
    """Classify overlap between two intervals.

    Returns one of: "disjoint", "subset", "partial", "equal".
    """
    lo_a, hi_a = interval_a
    lo_b, hi_b = interval_b

    # Equal check first
    if lo_a == lo_b and hi_a == hi_b:
        return "equal"

    # Intersection bounds
    lo = max(lo_a, lo_b)
    hi = min(hi_a, hi_b)

    if lo > hi:
        return "disjoint"

    # Check if one is a subset of the other
    # A subset of B: lo_b <= lo_a and hi_a <= hi_b
    a_in_b = lo_b <= lo_a and hi_a <= hi_b
    # B subset of A: lo_a <= lo_b and hi_b <= hi_a
    b_in_a = lo_a <= lo_b and hi_b <= hi_a

    if a_in_b or b_in_a:
        return "subset"

    return "partial"


# ---------------------------------------------------------------------------
# STEP 3 — LAW CONFLICT DETECTION
# ---------------------------------------------------------------------------


def _domains_overlap(law_a: Law, law_b: Law) -> bool:
    """Check if two laws have overlapping domains across all shared metrics."""
    ivs_a = extract_intervals(law_a)
    ivs_b = extract_intervals(law_b)

    shared_metrics = set(ivs_a.keys()) & set(ivs_b.keys())

    if not shared_metrics:
        # No shared metrics — domains could overlap anywhere on
        # the unshared dimensions. Treat as overlapping.
        return True

    for metric in sorted(shared_metrics):
        rel = intervals_overlap(ivs_a[metric], ivs_b[metric])
        if rel == "disjoint":
            return False

    return True


def laws_conflict(law_a: Law, law_b: Law) -> bool:
    """Two laws conflict iff their domains overlap AND actions differ."""
    if law_a.action == law_b.action:
        return False
    return _domains_overlap(law_a, law_b)


# ---------------------------------------------------------------------------
# STEP 4 — CONFLICT RESOLUTION
# ---------------------------------------------------------------------------


def _law_confidence(law: Law) -> float:
    """Return the confidence value for a law."""
    return getattr(law, "confidence", law.scores.get("consistency", 1.0))


def resolve_conflict(law_a: Law, law_b: Law) -> Law:
    """Deterministic conflict resolution.

    Priority:
      1. Higher confidence wins
      2. Tie → more conditions (higher specificity) wins
      3. Tie → lexicographically smaller law.id wins
    """
    conf_a = _law_confidence(law_a)
    conf_b = _law_confidence(law_b)

    if conf_a != conf_b:
        return law_a if conf_a > conf_b else law_b

    cnt_a = law_a.condition_count()
    cnt_b = law_b.condition_count()

    if cnt_a != cnt_b:
        return law_a if cnt_a > cnt_b else law_b

    return law_a if law_a.id <= law_b.id else law_b


# ---------------------------------------------------------------------------
# STEP 5 — CONFIDENCE EVOLUTION
# ---------------------------------------------------------------------------


def update_confidence(law: Law, outcome: str, timestamp: int = 0) -> Law:
    """Update law confidence based on outcome. Returns a new Law copy.

    Outcomes:
      "correct"       -> +0.01 (cap 1.0)
      "incorrect"     -> -0.05 (floor 0.0)
      "conflict_loss" -> *0.9
      "decay"         -> *0.99
    """
    conf = _law_confidence(law)
    history = list(getattr(law, "history", []))

    if outcome == "correct":
        new_conf = min(1.0, _norm(conf + CONFIDENCE_INCREMENT))
        event = "validated"
    elif outcome == "incorrect":
        new_conf = max(0.0, _norm(conf - CONFIDENCE_DECREMENT))
        event = "contradicted"
    elif outcome == "conflict_loss":
        new_conf = _norm(conf * CONFLICT_LOSS_FACTOR)
        event = "conflict_loss"
    elif outcome == "decay":
        new_conf = _norm(conf * DECAY_FACTOR)
        event = "decay"
    else:
        raise ValueError(f"Unknown outcome: {outcome!r}")

    history.append({
        "event": event,
        "timestamp": timestamp,
        "details": {"old_confidence": conf, "new_confidence": new_conf},
    })

    return _copy_law(law, confidence=new_conf, history=history)


# ---------------------------------------------------------------------------
# STEP 6 — HISTORY TRACKING (append-only via _copy_law)
# ---------------------------------------------------------------------------


def _copy_law(
    law: Law,
    confidence: Optional[float] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Law:
    """Create a copy of a law with optional overrides. Never mutates input."""
    new_law = Law(
        law_id=law.id,
        version=law.version,
        conditions=list(law.conditions),
        action=law.action,
        evidence=list(law.evidence),
        scores=dict(law.scores),
        created_at=law.created_at,
    )
    new_law.confidence = confidence if confidence is not None else _law_confidence(law)
    new_law.history = list(history) if history is not None else list(getattr(law, "history", []))
    return new_law


def init_law(law: Law, confidence: float = 1.0, timestamp: int = 0) -> Law:
    """Initialize a law with confidence and history tracking."""
    history = [{
        "event": "created",
        "timestamp": timestamp,
        "details": {"initial_confidence": confidence},
    }]
    return _copy_law(law, confidence=confidence, history=history)


# ---------------------------------------------------------------------------
# STEP 7 — THEORY CONSISTENCY
# ---------------------------------------------------------------------------


def compute_theory_consistency(
    laws: List[Law],
    sample_points: List[Dict[str, float]],
) -> float:
    """Compute global theory consistency score.

    For each sample point:
      - find applicable laws
      - if multiple laws give different actions -> conflict

    Returns 1 - (conflicting_points / covered_points).
    If no points are covered, returns 1.0.
    """
    covered = 0
    conflicting = 0

    for point in sample_points:
        applicable = [law for law in laws if law.evaluate(point)]
        if not applicable:
            continue
        covered += 1
        actions = set(law.action for law in applicable)
        if len(actions) > 1:
            conflicting += 1

    if covered == 0:
        return 1.0
    return _norm(1.0 - (conflicting / covered))


# ---------------------------------------------------------------------------
# STEP 8 — LAW PRUNING
# ---------------------------------------------------------------------------


def prune_laws(laws: List[Law]) -> List[Law]:
    """Remove weak or redundant laws.

    Removes a law if:
      - confidence < PRUNE_THRESHOLD
      - redundant: same domain + same action, lower confidence
      - repeatedly lost conflicts (>= CONFLICT_LOSS_PRUNE_COUNT losses in history)
    """
    # First pass: remove low confidence and repeated conflict losers
    surviving = []
    for law in laws:
        conf = _law_confidence(law)
        if conf < PRUNE_THRESHOLD:
            continue

        history = getattr(law, "history", [])
        loss_count = sum(1 for h in history if h.get("event") == "conflict_loss")
        if loss_count >= CONFLICT_LOSS_PRUNE_COUNT:
            continue

        surviving.append(law)

    # Second pass: remove redundant laws (same domain + same action, lower confidence)
    # Sort deterministically by confidence desc, then id asc
    surviving.sort(key=lambda l: (-_law_confidence(l), l.id))

    final: List[Law] = []
    for law in surviving:
        redundant = False
        ivs = extract_intervals(law)
        for kept in final:
            if kept.action != law.action:
                continue
            kept_ivs = extract_intervals(kept)
            if ivs == kept_ivs:
                # Same domain, same action — lower confidence is redundant
                redundant = True
                break
        if not redundant:
            final.append(law)

    return final


# ---------------------------------------------------------------------------
# STEP 9 — EVOLUTION LOOP
# ---------------------------------------------------------------------------


def evolve_laws(
    laws: List[Law],
    new_results: List[Dict[str, Any]],
    timestamp: int = 0,
) -> List[Law]:
    """Run one evolution cycle on the law set.

    For each result (dict with "metrics" and "observed_action"):
      - find applicable laws
      - update confidence based on prediction correctness
      - resolve conflicts between applicable laws

    Then: apply decay to all laws, prune.

    Returns a new list of laws (never mutates input).
    """
    # Work on copies
    law_map: Dict[str, Law] = {}
    for law in laws:
        copied = _copy_law(law)
        law_map[copied.id] = copied

    # Process each result
    for result in new_results:
        metrics = result.get("metrics", {})
        observed = result.get("observed_action", None)

        applicable_ids = [
            lid for lid, law in sorted(law_map.items())
            if law.evaluate(metrics)
        ]

        # Update confidence for applicable laws
        for lid in applicable_ids:
            law = law_map[lid]
            if law.action == observed:
                law_map[lid] = update_confidence(law, "correct", timestamp)
            else:
                law_map[lid] = update_confidence(law, "incorrect", timestamp)

        # Resolve conflicts among applicable laws
        if len(applicable_ids) > 1:
            applicable_laws = [law_map[lid] for lid in applicable_ids]
            actions = set(l.action for l in applicable_laws)
            if len(actions) > 1:
                # Group by action, pick winner from each group, then resolve across groups
                winner = applicable_laws[0]
                for other in applicable_laws[1:]:
                    if laws_conflict(winner, other):
                        resolved = resolve_conflict(winner, other)
                        loser_id = other.id if resolved.id == winner.id else winner.id
                        law_map[loser_id] = update_confidence(
                            law_map[loser_id], "conflict_loss", timestamp
                        )
                        winner = law_map[resolved.id]

    # Apply decay to all laws
    decayed: List[Law] = []
    for lid in sorted(law_map.keys()):
        decayed.append(update_confidence(law_map[lid], "decay", timestamp))

    # Prune
    return prune_laws(decayed)
