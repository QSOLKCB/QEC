"""v145.0 — Governed Orchestration Layer (GOL)."""

from __future__ import annotations

from dataclasses import dataclass
import math

from qec.analysis.bounded_refinement_kernel import RefinementReceipt
from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.deterministic_transition_policy import TransitionPolicyReceipt

VERDICT_ALLOW = "allow"
VERDICT_HOLD = "hold"
VERDICT_REJECT = "reject"

REASON_OK = "within_policy"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_TIE_BREAK_DISALLOWED = "tie_break_disallowed"
REASON_UNSTABLE_TRANSITION = "unstable_transition"
REASON_LOW_CONVERGENCE = "low_convergence"
REASON_NO_IMPROVEMENT = "no_improvement_disallowed"
REASON_MARGIN_TOO_LOW = "margin_too_low"
REASON_SCORE_TOO_LOW = "score_too_low"

CHECK_SELECTED_SCORE = "selected_score_check"
CHECK_CONFIDENCE = "confidence_check"
CHECK_MARGIN = "margin_check"
CHECK_TIE_BREAK = "tie_break_check"
CHECK_STABLE_TRANSITION = "stable_transition_check"
CHECK_CONVERGENCE = "convergence_check"
CHECK_NO_IMPROVEMENT = "no_improvement_check"

_ALLOWED_VERDICTS = frozenset({VERDICT_ALLOW, VERDICT_HOLD, VERDICT_REJECT})
_ALLOWED_REASONS = frozenset(
    {
        REASON_OK,
        REASON_LOW_CONFIDENCE,
        REASON_TIE_BREAK_DISALLOWED,
        REASON_UNSTABLE_TRANSITION,
        REASON_LOW_CONVERGENCE,
        REASON_NO_IMPROVEMENT,
        REASON_MARGIN_TOO_LOW,
        REASON_SCORE_TOO_LOW,
    }
)
_HARD_REJECT_REASON_ORDER = (
    REASON_SCORE_TOO_LOW,
    REASON_MARGIN_TOO_LOW,
    REASON_TIE_BREAK_DISALLOWED,
)
_SOFT_HOLD_REASON_ORDER = (
    REASON_LOW_CONFIDENCE,
    REASON_UNSTABLE_TRANSITION,
    REASON_LOW_CONVERGENCE,
    REASON_NO_IMPROVEMENT,
)
_CHECK_EVALUATION_ORDER = (
    CHECK_SELECTED_SCORE,
    CHECK_CONFIDENCE,
    CHECK_MARGIN,
    CHECK_TIE_BREAK,
    CHECK_STABLE_TRANSITION,
    CHECK_CONVERGENCE,
    CHECK_NO_IMPROVEMENT,
)

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _round12(value: float) -> float:
    return round(float(value), 12)


def _validate_sha256_hex(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{field_name} must be 64-char lowercase SHA-256 hex")
    return value


def _normalize_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    if value != normalized:
        raise ValueError(f"{field_name} must not include leading/trailing whitespace")
    return normalized


def _validate_unit_interval(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric in [0,1]")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0,1]")
    return numeric


def _validate_nested_stable_hash(value: object, field_name: str) -> None:
    stable_hash = getattr(value, "stable_hash", None)
    computed = getattr(value, "computed_stable_hash", None)
    if not isinstance(stable_hash, str) or not callable(computed):
        raise ValueError(f"{field_name} must expose stable_hash/computed_stable_hash")
    _validate_sha256_hex(stable_hash, f"{field_name}.stable_hash")
    if stable_hash != computed():
        raise ValueError(f"{field_name} stable_hash is invalid")


@dataclass(frozen=True)
class GovernancePolicy:
    min_required_score: float
    min_required_confidence: float
    min_required_margin: float
    min_required_convergence: float
    allow_tie_break: bool
    allow_no_improvement: bool
    require_stable_transition: bool
    stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_required_score",
            _validate_unit_interval(self.min_required_score, "min_required_score"),
        )
        object.__setattr__(
            self,
            "min_required_confidence",
            _validate_unit_interval(self.min_required_confidence, "min_required_confidence"),
        )
        object.__setattr__(
            self,
            "min_required_margin",
            _validate_unit_interval(self.min_required_margin, "min_required_margin"),
        )
        object.__setattr__(
            self,
            "min_required_convergence",
            _validate_unit_interval(self.min_required_convergence, "min_required_convergence"),
        )
        if not isinstance(self.allow_tie_break, bool):
            raise ValueError("allow_tie_break must be bool")
        if not isinstance(self.allow_no_improvement, bool):
            raise ValueError("allow_no_improvement must be bool")
        if not isinstance(self.require_stable_transition, bool):
            raise ValueError("require_stable_transition must be bool")
        _validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical governance policy payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "min_required_score": _round12(self.min_required_score),
            "min_required_confidence": _round12(self.min_required_confidence),
            "min_required_margin": _round12(self.min_required_margin),
            "min_required_convergence": _round12(self.min_required_convergence),
            "allow_tie_break": self.allow_tie_break,
            "allow_no_improvement": self.allow_no_improvement,
            "require_stable_transition": self.require_stable_transition,
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
class GovernanceCheck:
    check_name: str
    passed: bool
    observed_value: float | None
    threshold_value: float | None
    message: str
    stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_name", _normalize_string(self.check_name, "check_name"))
        if self.check_name not in _CHECK_EVALUATION_ORDER:
            raise ValueError("check_name is invalid")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be bool")
        if self.observed_value is not None:
            object.__setattr__(self, "observed_value", _validate_unit_interval(self.observed_value, "observed_value"))
        if self.threshold_value is not None:
            object.__setattr__(self, "threshold_value", _validate_unit_interval(self.threshold_value, "threshold_value"))
        object.__setattr__(self, "message", _normalize_string(self.message, "message"))
        _validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical governance check payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "observed_value": None if self.observed_value is None else _round12(self.observed_value),
            "threshold_value": None if self.threshold_value is None else _round12(self.threshold_value),
            "message": self.message,
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
class OrchestrationVerdict:
    verdict: str
    admissible: bool
    reason_code: str
    selected_ordering_signature: str
    decision_type: str
    transition_classification: str
    refinement_classification: str
    stable_hash: str

    def __post_init__(self) -> None:
        if self.verdict not in _ALLOWED_VERDICTS:
            raise ValueError("verdict is invalid")
        if not isinstance(self.admissible, bool):
            raise ValueError("admissible must be bool")
        if self.reason_code not in _ALLOWED_REASONS:
            raise ValueError("reason_code is invalid")
        object.__setattr__(
            self,
            "selected_ordering_signature",
            _normalize_string(self.selected_ordering_signature, "selected_ordering_signature"),
        )
        object.__setattr__(self, "decision_type", _normalize_string(self.decision_type, "decision_type"))
        if self.decision_type not in {"clear_winner", "narrow_margin", "tie_break"}:
            raise ValueError("decision_type is invalid")
        if self.transition_classification not in {"stable_transition", "uncertain_transition"}:
            raise ValueError("transition_classification is invalid")
        if self.refinement_classification not in {"converged", "bounded", "no_improvement"}:
            raise ValueError("refinement_classification is invalid")
        _validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical orchestration verdict payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "verdict": self.verdict,
            "admissible": self.admissible,
            "reason_code": self.reason_code,
            "selected_ordering_signature": self.selected_ordering_signature,
            "decision_type": self.decision_type,
            "transition_classification": self.transition_classification,
            "refinement_classification": self.refinement_classification,
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
class GovernedOrchestrationReceipt:
    policy: GovernancePolicy
    input_transition_hash: str
    input_refinement_hash: str
    checks: tuple[GovernanceCheck, ...]
    verdict: OrchestrationVerdict
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy, GovernancePolicy):
            raise ValueError("policy must be GovernancePolicy")
        _validate_sha256_hex(self.input_transition_hash, "input_transition_hash")
        _validate_sha256_hex(self.input_refinement_hash, "input_refinement_hash")
        if not isinstance(self.checks, tuple) or any(not isinstance(item, GovernanceCheck) for item in self.checks):
            raise ValueError("checks must be tuple[GovernanceCheck, ...]")
        if tuple(item.check_name for item in self.checks) != _CHECK_EVALUATION_ORDER:
            raise ValueError("checks must follow fixed deterministic ordering")
        if not isinstance(self.verdict, OrchestrationVerdict):
            raise ValueError("verdict must be OrchestrationVerdict")
        _validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash must match canonical governed orchestration receipt payload")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "policy": self.policy.to_dict(),
            "input_transition_hash": self.input_transition_hash,
            "input_refinement_hash": self.input_refinement_hash,
            "checks": tuple(check.to_dict() for check in self.checks),
            "verdict": self.verdict.to_dict(),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())


def _make_check(
    check_name: str,
    passed: bool,
    observed_value: float | None,
    threshold_value: float | None,
    message: str,
) -> GovernanceCheck:
    payload = {
        "check_name": check_name,
        "passed": passed,
        "observed_value": None if observed_value is None else _round12(observed_value),
        "threshold_value": None if threshold_value is None else _round12(threshold_value),
        "message": message,
    }
    return GovernanceCheck(
        check_name=check_name,
        passed=passed,
        observed_value=observed_value,
        threshold_value=threshold_value,
        message=message,
        stable_hash=sha256_hex(payload),
    )


def evaluate_governed_orchestration(
    policy: GovernancePolicy,
    transition_receipt: TransitionPolicyReceipt,
    refinement_receipt: RefinementReceipt,
) -> GovernedOrchestrationReceipt:
    if not isinstance(policy, GovernancePolicy):
        raise ValueError("policy must be GovernancePolicy")
    if not isinstance(transition_receipt, TransitionPolicyReceipt):
        raise ValueError("transition_receipt must be TransitionPolicyReceipt")
    if not isinstance(refinement_receipt, RefinementReceipt):
        raise ValueError("refinement_receipt must be RefinementReceipt")

    _validate_nested_stable_hash(policy, "policy")
    _validate_nested_stable_hash(transition_receipt, "transition_receipt")
    _validate_nested_stable_hash(transition_receipt.selected_decision, "transition_receipt.selected_decision")
    _validate_nested_stable_hash(refinement_receipt, "refinement_receipt")
    for index, step in enumerate(refinement_receipt.steps):
        _validate_nested_stable_hash(step, f"refinement_receipt.steps[{index}]")

    if refinement_receipt.input_policy_hash != transition_receipt.stable_hash:
        raise ValueError("refinement input_policy_hash must match transition_receipt.stable_hash")

    score = transition_receipt.selected_decision.selected_score
    confidence = transition_receipt.selected_decision.decision_confidence
    margin = transition_receipt.selected_decision.margin_to_next
    convergence = refinement_receipt.convergence_metric

    checks = (
        _make_check(
            check_name=CHECK_SELECTED_SCORE,
            passed=score >= policy.min_required_score,
            observed_value=score,
            threshold_value=policy.min_required_score,
            message="score meets minimum requirement"
            if score >= policy.min_required_score
            else "score below required minimum",
        ),
        _make_check(
            check_name=CHECK_CONFIDENCE,
            passed=confidence >= policy.min_required_confidence,
            observed_value=confidence,
            threshold_value=policy.min_required_confidence,
            message="decision confidence meets minimum" if confidence >= policy.min_required_confidence else "decision confidence below minimum",
        ),
        _make_check(
            check_name=CHECK_MARGIN,
            passed=margin >= policy.min_required_margin,
            observed_value=margin,
            threshold_value=policy.min_required_margin,
            message="margin meets minimum requirement"
            if margin >= policy.min_required_margin
            else "margin below required minimum",
        ),
        _make_check(
            check_name=CHECK_TIE_BREAK,
            passed=(transition_receipt.selected_decision.decision_type != "tie_break") or policy.allow_tie_break,
            observed_value=None,
            threshold_value=None,
            message="tie-break policy satisfied"
            if (transition_receipt.selected_decision.decision_type != "tie_break") or policy.allow_tie_break
            else "tie-break decision is disallowed by policy",
        ),
        _make_check(
            check_name=CHECK_STABLE_TRANSITION,
            passed=(not policy.require_stable_transition)
            or (transition_receipt.classification == "stable_transition"),
            observed_value=None,
            threshold_value=None,
            message="transition stability requirement satisfied"
            if (not policy.require_stable_transition) or (transition_receipt.classification == "stable_transition")
            else "transition is not stable while policy requires stable transitions",
        ),
        _make_check(
            check_name=CHECK_CONVERGENCE,
            passed=convergence >= policy.min_required_convergence,
            observed_value=convergence,
            threshold_value=policy.min_required_convergence,
            message="convergence meets minimum" if convergence >= policy.min_required_convergence else "convergence below minimum",
        ),
        _make_check(
            check_name=CHECK_NO_IMPROVEMENT,
            passed=(refinement_receipt.classification != "no_improvement") or policy.allow_no_improvement,
            observed_value=None,
            threshold_value=None,
            message="no-improvement policy satisfied"
            if (refinement_receipt.classification != "no_improvement") or policy.allow_no_improvement
            else "no-improvement refinement is disallowed by policy",
        ),
    )

    failures = {check.check_name for check in checks if not check.passed}
    reason_by_check = {
        CHECK_SELECTED_SCORE: REASON_SCORE_TOO_LOW,
        CHECK_CONFIDENCE: REASON_LOW_CONFIDENCE,
        CHECK_MARGIN: REASON_MARGIN_TOO_LOW,
        CHECK_TIE_BREAK: REASON_TIE_BREAK_DISALLOWED,
        CHECK_STABLE_TRANSITION: REASON_UNSTABLE_TRANSITION,
        CHECK_CONVERGENCE: REASON_LOW_CONVERGENCE,
        CHECK_NO_IMPROVEMENT: REASON_NO_IMPROVEMENT,
    }

    failed_reasons_in_order = tuple(reason_by_check[name] for name in _CHECK_EVALUATION_ORDER if name in failures)
    hard_fail_reasons = tuple(reason for reason in _HARD_REJECT_REASON_ORDER if reason in failed_reasons_in_order)
    soft_fail_reasons = tuple(reason for reason in _SOFT_HOLD_REASON_ORDER if reason in failed_reasons_in_order)

    if hard_fail_reasons:
        verdict_value = VERDICT_REJECT
        admissible = False
        reason_code = hard_fail_reasons[0]
    elif soft_fail_reasons:
        verdict_value = VERDICT_HOLD
        admissible = False
        reason_code = soft_fail_reasons[0]
    else:
        verdict_value = VERDICT_ALLOW
        admissible = True
        reason_code = REASON_OK

    verdict_payload = {
        "verdict": verdict_value,
        "admissible": admissible,
        "reason_code": reason_code,
        "selected_ordering_signature": transition_receipt.selected_decision.selected_ordering_signature,
        "decision_type": transition_receipt.selected_decision.decision_type,
        "transition_classification": transition_receipt.classification,
        "refinement_classification": refinement_receipt.classification,
    }
    verdict = OrchestrationVerdict(
        verdict=verdict_value,
        admissible=admissible,
        reason_code=reason_code,
        selected_ordering_signature=transition_receipt.selected_decision.selected_ordering_signature,
        decision_type=transition_receipt.selected_decision.decision_type,
        transition_classification=transition_receipt.classification,
        refinement_classification=refinement_receipt.classification,
        stable_hash=sha256_hex(verdict_payload),
    )

    receipt_payload = {
        "policy": policy.to_dict(),
        "input_transition_hash": transition_receipt.stable_hash,
        "input_refinement_hash": refinement_receipt.stable_hash,
        "checks": tuple(check.to_dict() for check in checks),
        "verdict": verdict.to_dict(),
    }
    return GovernedOrchestrationReceipt(
        policy=policy,
        input_transition_hash=transition_receipt.stable_hash,
        input_refinement_hash=refinement_receipt.stable_hash,
        checks=checks,
        verdict=verdict,
        stable_hash=sha256_hex(receipt_payload),
    )


__all__ = [
    "CHECK_CONFIDENCE",
    "CHECK_CONVERGENCE",
    "CHECK_MARGIN",
    "CHECK_NO_IMPROVEMENT",
    "CHECK_SELECTED_SCORE",
    "CHECK_STABLE_TRANSITION",
    "CHECK_TIE_BREAK",
    "GovernanceCheck",
    "GovernancePolicy",
    "GovernedOrchestrationReceipt",
    "OrchestrationVerdict",
    "REASON_LOW_CONFIDENCE",
    "REASON_LOW_CONVERGENCE",
    "REASON_MARGIN_TOO_LOW",
    "REASON_NO_IMPROVEMENT",
    "REASON_OK",
    "REASON_SCORE_TOO_LOW",
    "REASON_TIE_BREAK_DISALLOWED",
    "REASON_UNSTABLE_TRANSITION",
    "VERDICT_ALLOW",
    "VERDICT_HOLD",
    "VERDICT_REJECT",
    "evaluate_governed_orchestration",
]
