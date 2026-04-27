# SPDX-License-Identifier: MIT
"""v148.10 — deterministic promotion gate over EvaluationPackReceipt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.evaluation_pack import EvaluationPackReceipt

SCHEMA_VERSION = "v148.10"
MODULE_VERSION = "v148.10"

CHECK_DETERMINISTIC_INTEGRITY = "DETERMINISTIC_INTEGRITY"
CHECK_GOVERNANCE_STABILITY = "GOVERNANCE_STABILITY"
CHECK_REPAIR_REASONING_CONSISTENCY = "REPAIR_REASONING_CONSISTENCY"
CHECK_FAILURES_BOUNDED_CLASSIFIED = "FAILURES_BOUNDED_CLASSIFIED"
CHECK_MEASURABLE_BENEFIT = "MEASURABLE_BENEFIT"

CANONICAL_CHECK_ORDER: tuple[str, ...] = (
    CHECK_DETERMINISTIC_INTEGRITY,
    CHECK_GOVERNANCE_STABILITY,
    CHECK_REPAIR_REASONING_CONSISTENCY,
    CHECK_FAILURES_BOUNDED_CLASSIFIED,
    CHECK_MEASURABLE_BENEFIT,
)

CHECK_STATUS_PASS = "PASS"
CHECK_STATUS_FAIL = "FAIL"

PROMOTION_STATUS_PROMOTE = "PROMOTE"
PROMOTION_STATUS_STOP = "STOP"

ITEM_TYPE_GOVERNANCE_VALIDATION = "GOVERNANCE_VALIDATION"
ITEM_TYPE_ISSUE_NORMALIZATION = "ISSUE_NORMALIZATION"
ITEM_TYPE_FIX_PROPOSAL = "FIX_PROPOSAL"
ITEM_TYPE_FIX_VALIDATION = "FIX_VALIDATION"
ITEM_TYPE_COUNTERFACTUAL_REPLAY = "COUNTERFACTUAL_REPLAY"
ITEM_TYPE_FAILURE_LEDGER = "FAILURE_LEDGER"
ITEM_TYPE_DETERMINISTIC_WORKLOAD = "DETERMINISTIC_WORKLOAD"

_ALLOWED_CHECK_NAMES = frozenset(CANONICAL_CHECK_ORDER)
_ALLOWED_CHECK_STATUSES = frozenset({CHECK_STATUS_PASS, CHECK_STATUS_FAIL})
_ALLOWED_PROMOTION_STATUSES = frozenset({PROMOTION_STATUS_PROMOTE, PROMOTION_STATUS_STOP})

_GOVERNANCE_ACCEPTABLE_STATUSES = frozenset({"VALIDATED"})
_REPAIR_FAILURE_STATUSES = frozenset(
    {
        "INVALID",
        "UNSAFE",
        "INSUFFICIENT",
        "HAS_INVALID",
        "HAS_UNSAFE",
        "HAS_INSUFFICIENT",
        "HAS_UNRESOLVED",
    }
)


@dataclass(frozen=True)
class PromotionGateCheck:
    check_name: str
    check_status: str
    reason: str

    def __post_init__(self) -> None:
        if self.check_name not in _ALLOWED_CHECK_NAMES:
            raise ValueError("invalid check_name")
        if self.check_status not in _ALLOWED_CHECK_STATUSES:
            raise ValueError("invalid check_status")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")

    def _payload(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "check_status": self.check_status,
            "reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "stable_hash": self.stable_hash()}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self._payload())


@dataclass(frozen=True)
class PromotionGateReceipt:
    schema_version: str
    module_version: str
    promotion_status: str
    input_pack_hash: str
    checks: tuple[PromotionGateCheck, ...]
    pass_count: int
    fail_count: int
    stop_reason: str

    def __post_init__(self) -> None:
        if self.promotion_status not in _ALLOWED_PROMOTION_STATUSES:
            raise ValueError("invalid promotion_status")
        if not isinstance(self.input_pack_hash, str) or len(self.input_pack_hash) != 64:
            raise ValueError("input_pack_hash must be a SHA-256 hex string")
        if not isinstance(self.stop_reason, str) or not self.stop_reason:
            raise ValueError("stop_reason must be a non-empty string")
        if self.pass_count < 0 or self.fail_count < 0:
            raise ValueError("pass_count/fail_count must be non-negative")
        if len(self.checks) != len(CANONICAL_CHECK_ORDER):
            raise ValueError("checks must contain all canonical checks")
        if tuple(check.check_name for check in self.checks) != CANONICAL_CHECK_ORDER:
            raise ValueError("checks must follow canonical check order")

        actual_fail_checks = tuple(
            check.check_name for check in self.checks if check.check_status == CHECK_STATUS_FAIL
        )
        actual_fail_count = len(actual_fail_checks)
        actual_pass_count = sum(1 for check in self.checks if check.check_status == CHECK_STATUS_PASS)

        if self.pass_count != actual_pass_count or self.fail_count != actual_fail_count:
            raise ValueError("pass_count/fail_count must match check statuses")
        if self.pass_count + self.fail_count != len(self.checks):
            raise ValueError("pass_count + fail_count must equal number of checks")

        if self.promotion_status == PROMOTION_STATUS_PROMOTE:
            if self.fail_count != 0:
                raise ValueError("PROMOTE receipts must have fail_count == 0")
            if self.stop_reason != "NONE":
                raise ValueError("PROMOTE receipts must have stop_reason == 'NONE'")
        else:
            if self.fail_count == 0:
                raise ValueError("STOP receipts must have fail_count > 0")
            if self.stop_reason == "NONE":
                raise ValueError("STOP receipts must not have stop_reason == 'NONE'")
            if self.stop_reason not in actual_fail_checks:
                raise ValueError("STOP receipts must use a failed check name as stop_reason")
    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "promotion_status": self.promotion_status,
            "input_pack_hash": self.input_pack_hash,
            "checks": [check.to_dict() for check in self.checks],
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "stop_reason": self.stop_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "stable_hash": self.stable_hash()}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self._payload())


def _build_check(check_name: str, passed: bool, reason: str) -> PromotionGateCheck:
    return PromotionGateCheck(
        check_name=check_name,
        check_status=CHECK_STATUS_PASS if passed else CHECK_STATUS_FAIL,
        reason=reason,
    )


def _evaluate_deterministic_integrity(evaluation_pack: EvaluationPackReceipt) -> PromotionGateCheck:
    passed = bool(evaluation_pack.summary.determinism_preserved)
    reason = (
        "determinism_preserved is True"
        if passed
        else "determinism_preserved is not True"
    )
    return _build_check(CHECK_DETERMINISTIC_INTEGRITY, passed, reason)


def _evaluate_governance_stability(evaluation_pack: EvaluationPackReceipt) -> PromotionGateCheck:
    governance_items = tuple(item for item in evaluation_pack.items if item.item_type == ITEM_TYPE_GOVERNANCE_VALIDATION)
    if int(evaluation_pack.summary.type_counts.get(ITEM_TYPE_GOVERNANCE_VALIDATION, 0)) <= 0 or not governance_items:
        return _build_check(CHECK_GOVERNANCE_STABILITY, False, "GOVERNANCE_VALIDATION missing")
    invalid_statuses = tuple(
        sorted({item.status for item in governance_items if item.status not in _GOVERNANCE_ACCEPTABLE_STATUSES})
    )
    if invalid_statuses:
        return _build_check(
            CHECK_GOVERNANCE_STABILITY,
            False,
            f"governance status not acceptable: {', '.join(invalid_statuses)}",
        )
    return _build_check(CHECK_GOVERNANCE_STABILITY, True, "governance statuses validated")


def _evaluate_repair_reasoning_consistency(evaluation_pack: EvaluationPackReceipt) -> PromotionGateCheck:
    required_types = (
        ITEM_TYPE_ISSUE_NORMALIZATION,
        ITEM_TYPE_FIX_PROPOSAL,
        ITEM_TYPE_FIX_VALIDATION,
        ITEM_TYPE_COUNTERFACTUAL_REPLAY,
    )
    missing = tuple(
        item_type for item_type in required_types if int(evaluation_pack.summary.type_counts.get(item_type, 0)) <= 0
    )
    if missing:
        return _build_check(
            CHECK_REPAIR_REASONING_CONSISTENCY,
            False,
            f"repair item types missing: {', '.join(missing)}",
        )

    repair_items = tuple(item for item in evaluation_pack.items if item.item_type in required_types)
    failure_statuses = tuple(sorted({item.status for item in repair_items if item.status in _REPAIR_FAILURE_STATUSES}))
    if failure_statuses:
        return _build_check(
            CHECK_REPAIR_REASONING_CONSISTENCY,
            False,
            f"repair failure statuses present: {', '.join(failure_statuses)}",
        )

    return _build_check(CHECK_REPAIR_REASONING_CONSISTENCY, True, "repair reasoning checks consistent")


def _evaluate_failures_bounded_classified(evaluation_pack: EvaluationPackReceipt) -> PromotionGateCheck:
    if int(evaluation_pack.summary.failure_count) > 0:
        return _build_check(CHECK_FAILURES_BOUNDED_CLASSIFIED, False, "failure_count > 0")
    if int(evaluation_pack.summary.type_counts.get(ITEM_TYPE_FAILURE_LEDGER, 0)) <= 0:
        return _build_check(CHECK_FAILURES_BOUNDED_CLASSIFIED, False, "FAILURE_LEDGER missing")
    return _build_check(CHECK_FAILURES_BOUNDED_CLASSIFIED, True, "failure_count == 0 and FAILURE_LEDGER present")


def _evaluate_measurable_benefit(evaluation_pack: EvaluationPackReceipt) -> PromotionGateCheck:
    workload_items = tuple(item for item in evaluation_pack.items if item.item_type == ITEM_TYPE_DETERMINISTIC_WORKLOAD)
    if int(evaluation_pack.summary.type_counts.get(ITEM_TYPE_DETERMINISTIC_WORKLOAD, 0)) <= 0 or not workload_items:
        return _build_check(CHECK_MEASURABLE_BENEFIT, False, "DETERMINISTIC_WORKLOAD missing")
    has_evaluated = any(item.status == "EVALUATED" for item in workload_items)
    if not has_evaluated:
        return _build_check(CHECK_MEASURABLE_BENEFIT, False, "no workload status EVALUATED")
    return _build_check(CHECK_MEASURABLE_BENEFIT, True, "deterministic workload evaluated")


def evaluate_promotion_gate(evaluation_pack: EvaluationPackReceipt) -> PromotionGateReceipt:
    if not isinstance(evaluation_pack, EvaluationPackReceipt):
        raise ValueError("evaluation_pack must be an EvaluationPackReceipt")

    checks = (
        _evaluate_deterministic_integrity(evaluation_pack),
        _evaluate_governance_stability(evaluation_pack),
        _evaluate_repair_reasoning_consistency(evaluation_pack),
        _evaluate_failures_bounded_classified(evaluation_pack),
        _evaluate_measurable_benefit(evaluation_pack),
    )
    fail_count = sum(1 for check in checks if check.check_status == CHECK_STATUS_FAIL)
    pass_count = len(checks) - fail_count

    if fail_count > 0:
        promotion_status = PROMOTION_STATUS_STOP
        stop_reason = next(check.check_name for check in checks if check.check_status == CHECK_STATUS_FAIL)
    else:
        promotion_status = PROMOTION_STATUS_PROMOTE
        stop_reason = "NONE"

    return PromotionGateReceipt(
        schema_version=SCHEMA_VERSION,
        module_version=MODULE_VERSION,
        promotion_status=promotion_status,
        input_pack_hash=evaluation_pack.stable_hash(),
        checks=checks,
        pass_count=pass_count,
        fail_count=fail_count,
        stop_reason=stop_reason,
    )


__all__ = [
    "CANONICAL_CHECK_ORDER",
    "PromotionGateCheck",
    "PromotionGateReceipt",
    "evaluate_promotion_gate",
]
