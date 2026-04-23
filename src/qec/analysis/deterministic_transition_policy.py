"""v144.3 — deterministic transition policy selection layer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import string

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.state_conditioned_filter_mesh import FilterMeshReceipt, OrderingScore

CLEAR_MARGIN_THRESHOLD = 0.15
WEAK_MARGIN_THRESHOLD = 0.05
STABLE_CONFIDENCE_THRESHOLD = 0.5

DECISION_TYPE_CLEAR_WINNER = "clear_winner"
DECISION_TYPE_NARROW_MARGIN = "narrow_margin"
DECISION_TYPE_TIE_BREAK = "tie_break"

CLASSIFICATION_STABLE_TRANSITION = "stable_transition"
CLASSIFICATION_UNCERTAIN_TRANSITION = "uncertain_transition"

SELECTION_RULE = "ordered_scores_margin_dominance_v1"

_ALLOWED_DECISION_TYPES = frozenset(
    {DECISION_TYPE_CLEAR_WINNER, DECISION_TYPE_NARROW_MARGIN, DECISION_TYPE_TIE_BREAK}
)
_ALLOWED_CLASSIFICATIONS = frozenset({CLASSIFICATION_STABLE_TRANSITION, CLASSIFICATION_UNCERTAIN_TRANSITION})

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _normalize_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    if normalized != value:
        raise ValueError(f"{field_name} must not include leading/trailing whitespace")
    return normalized


def _validate_unit_interval(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric in [0,1]")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0,1]")
    return numeric


def _validate_sha256_hex(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be 64-char hex string")
    if any(ch not in string.hexdigits for ch in value) or value.lower() != value:
        raise ValueError(f"{field_name} must be 64-char hex string")
    return value


def _round12(value: float) -> float:
    return round(float(value), 12)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class TransitionDecision:
    selected_ordering_signature: str
    selected_score: float
    decision_rank: int
    margin_to_next: float
    decision_confidence: float
    decision_type: str
    stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_ordering_signature",
            _normalize_string(self.selected_ordering_signature, "selected_ordering_signature"),
        )
        object.__setattr__(self, "selected_score", _validate_unit_interval(self.selected_score, "selected_score"))
        if isinstance(self.decision_rank, bool) or not isinstance(self.decision_rank, int) or self.decision_rank < 1:
            raise ValueError("decision_rank must be int >= 1")
        object.__setattr__(self, "margin_to_next", _validate_unit_interval(self.margin_to_next, "margin_to_next"))
        object.__setattr__(
            self,
            "decision_confidence",
            _validate_unit_interval(self.decision_confidence, "decision_confidence"),
        )
        if self.decision_type not in _ALLOWED_DECISION_TYPES:
            raise ValueError("decision_type is invalid")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical decision payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "selected_ordering_signature": self.selected_ordering_signature,
            "selected_score": _round12(self.selected_score),
            "decision_rank": self.decision_rank,
            "margin_to_next": _round12(self.margin_to_next),
            "decision_confidence": _round12(self.decision_confidence),
            "decision_type": self.decision_type,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


@dataclass(frozen=True)
class TransitionPolicyReceipt:
    input_receipt_hash: str
    candidate_count: int
    selected_decision: TransitionDecision
    selection_rule: str
    classification: str
    stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_receipt_hash",
            _validate_sha256_hex(_normalize_string(self.input_receipt_hash, "input_receipt_hash"), "input_receipt_hash"),
        )
        if isinstance(self.candidate_count, bool) or not isinstance(self.candidate_count, int) or self.candidate_count < 1:
            raise ValueError("candidate_count must be int >= 1")
        if not isinstance(self.selected_decision, TransitionDecision):
            raise ValueError("selected_decision must be TransitionDecision")
        object.__setattr__(self, "selection_rule", _normalize_string(self.selection_rule, "selection_rule"))
        if self.classification not in _ALLOWED_CLASSIFICATIONS:
            raise ValueError("classification is invalid")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical transition policy payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "input_receipt_hash": self.input_receipt_hash,
            "candidate_count": self.candidate_count,
            "selected_decision": self.selected_decision.to_dict(),
            "selection_rule": self.selection_rule,
            "classification": self.classification,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


def select_deterministic_transition(receipt: FilterMeshReceipt) -> TransitionPolicyReceipt:
    if not isinstance(receipt, FilterMeshReceipt):
        raise ValueError("receipt must be a FilterMeshReceipt")
    if receipt.stable_hash != receipt.computed_stable_hash():
        raise ValueError("receipt stable_hash is invalid")
    if isinstance(receipt.candidate_count, bool) or not isinstance(receipt.candidate_count, int):
        raise ValueError("candidate_count must be int")
    if receipt.candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")
    if not isinstance(receipt.ordered_scores, tuple):
        raise ValueError("ordered_scores must be tuple")
    if any(not isinstance(item, OrderingScore) for item in receipt.ordered_scores):
        raise ValueError("ordered_scores must contain OrderingScore entries")
    if len(receipt.ordered_scores) != receipt.candidate_count:
        raise ValueError("candidate_count mismatch")
    if not receipt.ordered_scores:
        raise ValueError("ordered_scores must not be empty")

    expected_sort = tuple(
        sorted(
            receipt.ordered_scores,
            key=lambda item: (-_round12(item.total_score), item.ordering_signature, item.stable_hash),
        )
    )
    if expected_sort != receipt.ordered_scores:
        raise ValueError("ordered_scores must be sorted canonically")

    seen_signatures: set[str] = set()
    for expected_rank, score in enumerate(receipt.ordered_scores, start=1):
        if score.rank != expected_rank:
            raise ValueError("invalid ranking state")
        _validate_unit_interval(score.total_score, "total_score")
        if score.ordering_signature in seen_signatures:
            raise ValueError("duplicate ordering signatures")
        seen_signatures.add(score.ordering_signature)

    first = receipt.ordered_scores[0]
    if receipt.dominant_ordering_signature != first.ordering_signature:
        raise ValueError("dominant_ordering_signature must match ordered_scores[0]")
    if _round12(receipt.dominant_score) != _round12(first.total_score):
        raise ValueError("dominant_score must match ordered_scores[0]")
    first_score = _round12(first.total_score)
    second_score = _round12(receipt.ordered_scores[1].total_score) if receipt.candidate_count > 1 else 0.0
    margin = 1.0 if receipt.candidate_count == 1 else _round12(first_score - second_score)
    if margin == -0.0:
        margin = 0.0

    if receipt.candidate_count == 1 or margin >= CLEAR_MARGIN_THRESHOLD:
        decision_type = DECISION_TYPE_CLEAR_WINNER
    elif margin >= WEAK_MARGIN_THRESHOLD:
        decision_type = DECISION_TYPE_NARROW_MARGIN
    else:
        decision_type = DECISION_TYPE_TIE_BREAK

    confidence = _round12(_clamp01(first_score * (1.0 + margin)))
    classification = (
        CLASSIFICATION_STABLE_TRANSITION
        if confidence >= STABLE_CONFIDENCE_THRESHOLD
        else CLASSIFICATION_UNCERTAIN_TRANSITION
    )

    decision_payload = {
        "selected_ordering_signature": first.ordering_signature,
        "selected_score": first_score,
        "decision_rank": first.rank,
        "margin_to_next": _round12(margin),
        "decision_confidence": confidence,
        "decision_type": decision_type,
    }
    selected_decision = TransitionDecision(
        selected_ordering_signature=first.ordering_signature,
        selected_score=first_score,
        decision_rank=first.rank,
        margin_to_next=margin,
        decision_confidence=confidence,
        decision_type=decision_type,
        stable_hash=sha256_hex(decision_payload),
    )

    policy_payload = {
        "input_receipt_hash": receipt.stable_hash,
        "candidate_count": receipt.candidate_count,
        "selected_decision": selected_decision.to_dict(),
        "selection_rule": SELECTION_RULE,
        "classification": classification,
    }
    return TransitionPolicyReceipt(
        input_receipt_hash=receipt.stable_hash,
        candidate_count=receipt.candidate_count,
        selected_decision=selected_decision,
        selection_rule=SELECTION_RULE,
        classification=classification,
        stable_hash=sha256_hex(policy_payload),
    )


__all__ = [
    "CLASSIFICATION_STABLE_TRANSITION",
    "CLASSIFICATION_UNCERTAIN_TRANSITION",
    "CLEAR_MARGIN_THRESHOLD",
    "DECISION_TYPE_CLEAR_WINNER",
    "DECISION_TYPE_NARROW_MARGIN",
    "DECISION_TYPE_TIE_BREAK",
    "SELECTION_RULE",
    "STABLE_CONFIDENCE_THRESHOLD",
    "TransitionDecision",
    "TransitionPolicyReceipt",
    "WEAK_MARGIN_THRESHOLD",
    "select_deterministic_transition",
]
