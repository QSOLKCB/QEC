"""Deterministic Law Promotion Engine (v97.5.0).

Converts validated conjectures into formal laws via deterministic
condition evaluation, scoring, minimization, and conflict resolution.

Pipeline: conjectures -> validated -> promoted laws -> registry

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import time


# ---------------------------------------------------------------------------
# PART 1 — CONDITION
# ---------------------------------------------------------------------------

_OPERATORS = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
}


class Condition:
    """A single deterministic condition on a metric."""

    __slots__ = ("metric", "operator", "value")

    def __init__(self, metric: str, operator: str, value: float) -> None:
        if operator not in _OPERATORS:
            raise ValueError(f"Unknown operator: {operator!r}")
        self.metric = metric
        self.operator = operator
        self.value = float(value)

    def evaluate(self, metrics_dict: Dict[str, float]) -> bool:
        """Evaluate this condition against a metrics dictionary."""
        if self.metric not in metrics_dict:
            return False
        return _OPERATORS[self.operator](metrics_dict[self.metric], self.value)

    def to_dict(self) -> Dict[str, Any]:
        return {"metric": self.metric, "operator": self.operator, "value": self.value}

    def sort_key(self) -> str:
        """Deterministic string key for lexicographic ordering."""
        return f"{self.metric}:{self.operator}:{self.value}"

    def __repr__(self) -> str:
        return f"Condition({self.metric!r}, {self.operator!r}, {self.value})"


# ---------------------------------------------------------------------------
# PART 2 — LAW
# ---------------------------------------------------------------------------


class Law:
    """A promoted, validated law with conditions, action, and scores."""

    def __init__(
        self,
        law_id: str,
        version: int,
        conditions: List[Condition],
        action: str,
        evidence: List[str],
        scores: Dict[str, float],
        created_at: float,
    ) -> None:
        self.id = law_id
        self.version = version
        self.conditions = list(conditions)
        self.action = action
        self.evidence = list(evidence)
        self.scores = dict(scores)
        self.created_at = created_at

    def evaluate(self, metrics_dict: Dict[str, float]) -> bool:
        """True iff ALL conditions pass."""
        return all(c.evaluate(metrics_dict) for c in self.conditions)

    def condition_count(self) -> int:
        return len(self.conditions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "conditions": [c.to_dict() for c in self.conditions],
            "action": self.action,
            "evidence": list(self.evidence),
            "scores": dict(self.scores),
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# PART 3 — SCORING
# ---------------------------------------------------------------------------


def compute_metrics(
    conditions: List[Condition],
    action: str,
    validation_runs: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute coverage and consistency for conditions+action over runs.

    Each run is a dict with at least ``"metrics"`` and ``"action"`` keys.
    Returns ``{"coverage": float, "consistency": float}``.
    """
    total = len(validation_runs)
    if total == 0:
        return {"coverage": 0.0, "consistency": 0.0}

    condition_matches = 0
    condition_and_action = 0

    for run in validation_runs:
        metrics = run.get("metrics", {})
        all_pass = all(c.evaluate(metrics) for c in conditions)
        if all_pass:
            condition_matches += 1
            if run.get("action") == action:
                condition_and_action += 1

    coverage = condition_matches / total
    consistency = (
        condition_and_action / condition_matches if condition_matches > 0 else 0.0
    )
    return {"coverage": coverage, "consistency": consistency}


def compute_simplicity(num_conditions: int) -> float:
    """simplicity = 1 / (1 + num_conditions)."""
    return 1.0 / (1.0 + num_conditions)


def compute_law_score(coverage: float, simplicity: float) -> float:
    """law_score = coverage * simplicity."""
    return coverage * simplicity


# ---------------------------------------------------------------------------
# PART 4 — CONDITION MINIMIZATION
# ---------------------------------------------------------------------------


def minimize_conditions(
    conditions: List[Condition],
    action: str,
    validation_runs: List[Dict[str, Any]],
) -> List[Condition]:
    """Greedy removal of redundant conditions while preserving consistency==1.0.

    For each condition in deterministic order, try removing it.
    If consistency remains 1.0 after removal, keep it removed.
    Repeat until stable.
    """
    # Sort conditions deterministically for reproducibility
    current = sorted(conditions, key=lambda c: c.sort_key())

    changed = True
    while changed:
        changed = False
        for i in range(len(current)):
            candidate = current[:i] + current[i + 1 :]
            if len(candidate) == 0:
                continue
            m = compute_metrics(candidate, action, validation_runs)
            if m["consistency"] == 1.0 and m["coverage"] > 0.0:
                current = candidate
                changed = True
                break

    return current


# ---------------------------------------------------------------------------
# PART 5 — LAW REGISTRY
# ---------------------------------------------------------------------------


def _conditions_key(conditions: List[Condition]) -> str:
    """Deterministic string representation of a condition set."""
    return "|".join(sorted(c.sort_key() for c in conditions))


class LawRegistry:
    """Registry of promoted laws with conflict detection."""

    def __init__(self) -> None:
        self.laws: List[Law] = []

    def _find_conflicts(
        self, new_law: Law, validation_runs: List[Dict[str, Any]]
    ) -> List[Law]:
        """Find existing laws that conflict with new_law.

        Two laws conflict if both conditions are true on the same run
        and their actions differ.
        """
        conflicts = []
        for existing in self.laws:
            if existing.action == new_law.action:
                continue
            for run in validation_runs:
                metrics = run.get("metrics", {})
                if existing.evaluate(metrics) and new_law.evaluate(metrics):
                    conflicts.append(existing)
                    break
        return conflicts

    def _resolve_conflict(self, new_law: Law, existing: Law) -> bool:
        """Return True if new_law wins over existing.

        Resolution order:
        1. Higher law_score wins
        2. More specific (more conditions) wins
        3. Lexicographically smaller condition string wins
        """
        new_score = new_law.scores.get("law_score", 0.0)
        old_score = existing.scores.get("law_score", 0.0)

        if new_score != old_score:
            return new_score > old_score

        new_count = new_law.condition_count()
        old_count = existing.condition_count()
        if new_count != old_count:
            return new_count > old_count

        new_key = _conditions_key(new_law.conditions)
        old_key = _conditions_key(existing.conditions)
        return new_key < old_key

    def add_law(
        self, law: Law, validation_runs: List[Dict[str, Any]]
    ) -> bool:
        """Attempt to add a law. Returns True if added, False if rejected."""
        conflicts = self._find_conflicts(law, validation_runs)

        for conflict in conflicts:
            if not self._resolve_conflict(law, conflict):
                return False

        # New law wins all conflicts — remove losers
        for conflict in conflicts:
            self.laws = [l for l in self.laws if l.id != conflict.id]

        self.laws.append(law)
        return True


# ---------------------------------------------------------------------------
# PART 6 — LAW PROMOTER
# ---------------------------------------------------------------------------


def _make_law_id(conditions: List[Condition], action: str) -> str:
    """Deterministic law ID via SHA-256."""
    key = _conditions_key(conditions) + "=>" + action
    return "law_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class LawPromoter:
    """Promotes validated conjectures into laws."""

    def __init__(
        self,
        registry: LawRegistry,
        validation_runs: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> None:
        self.registry = registry
        self.validation_runs = validation_runs
        self.threshold = threshold

    def promote(self, conjecture: Dict[str, Any]) -> Optional[Law]:
        """Attempt to promote a conjecture into a law.

        Conjecture must have:
          - "conditions": list of dicts with metric/operator/value
          - "action": str
          - "evidence": list of str (optional)

        Returns the promoted Law or None if rejected.
        """
        raw_conditions = [
            Condition(c["metric"], c["operator"], c["value"])
            for c in conjecture["conditions"]
        ]
        action = conjecture["action"]
        evidence = list(conjecture.get("evidence", []))

        # Step 1: minimize conditions
        minimized = minimize_conditions(
            raw_conditions, action, self.validation_runs
        )

        # Step 2: compute metrics
        m = compute_metrics(minimized, action, self.validation_runs)

        # Step 3: reject if consistency != 1.0
        if m["consistency"] != 1.0:
            return None

        # Step 4: compute simplicity
        simplicity = compute_simplicity(len(minimized))

        # Step 5: compute law_score
        law_score = compute_law_score(m["coverage"], simplicity)

        # Step 6: reject if below threshold
        if law_score < self.threshold:
            return None

        # Step 7: create Law
        law_id = _make_law_id(minimized, action)
        scores = {
            "consistency": m["consistency"],
            "coverage": m["coverage"],
            "simplicity": simplicity,
            "law_score": law_score,
        }
        law = Law(
            law_id=law_id,
            version=1,
            conditions=minimized,
            action=action,
            evidence=evidence,
            scores=scores,
            created_at=conjecture.get("created_at", 0.0),
        )

        # Step 8: attempt registry insertion
        if not self.registry.add_law(law, self.validation_runs):
            return None

        # Step 9: return law
        return law
