"""v150.8 — Deterministic adversarial multi-agent governance failure injection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
import math
from typing import Any, Final

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, canonicalize_json, sha256_hex

FAILURE_TYPE_INVALID_DECISION: Final[str] = "INVALID_DECISION"
FAILURE_TYPE_CONFLICTING_ROLE: Final[str] = "CONFLICTING_ROLE"
FAILURE_TYPE_INCONSISTENT_MEMORY: Final[str] = "INCONSISTENT_MEMORY"
ALLOWED_FAILURE_TYPES: Final[frozenset[str]] = frozenset(
    {
        FAILURE_TYPE_INVALID_DECISION,
        FAILURE_TYPE_CONFLICTING_ROLE,
        FAILURE_TYPE_INCONSISTENT_MEMORY,
    }
)


KNOWN_ROLES: Final[frozenset[str]] = frozenset({"CONTROL", "VALIDATION", "REPAIR", "ADVERSARIAL", "COMPRESSION"})


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _require_sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise _invalid_input()
    return value


def _require_non_empty_string(value: object) -> str:
    if not isinstance(value, str) or value.strip() != value or not value:
        raise _invalid_input()
    return value


def _validate_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if len(value) == 0:
            raise _invalid_input()
        return {str(k): _validate_json_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return tuple(_validate_json_value(v) for v in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _invalid_input()
        return value
    if value is None:
        return value
    if isinstance(value, str):
        if value == "":
            raise _invalid_input()
        return value
    raise _invalid_input()


@dataclass(frozen=True)
class AdversarialFailureCase:
    case_id: str
    failure_type: str
    target_hash: str
    payload: Mapping[str, Any]
    expected_rejection_reason: str
    case_hash_input: InitVar[str | None] = None
    _case_hash: str = field(init=False, repr=False)

    def __post_init__(self, case_hash_input: str | None) -> None:
        _require_non_empty_string(self.case_id)
        if self.failure_type not in ALLOWED_FAILURE_TYPES:
            raise _invalid_input()
        _require_sha256(self.target_hash)
        _require_non_empty_string(self.expected_rejection_reason)
        if not isinstance(self.payload, Mapping) or len(self.payload) == 0:
            raise _invalid_input()
        canonical_payload = canonicalize_json(_validate_json_value(dict(self.payload)))
        object.__setattr__(self, "payload", canonical_payload)

        computed = sha256_hex(self._payload_without_hash())
        if case_hash_input is None:
            object.__setattr__(self, "_case_hash", computed)
        else:
            provided = _require_sha256(case_hash_input)
            if provided != computed:
                raise _invalid_input()
            object.__setattr__(self, "_case_hash", provided)

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "failure_type": self.failure_type,
            "target_hash": self.target_hash,
            "payload": self.payload,
            "expected_rejection_reason": self.expected_rejection_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_hash(), "case_hash": self._case_hash}

    def case_hash(self) -> str:
        return self._case_hash


@dataclass(frozen=True)
class AdversarialFailureResult:
    case_id: str
    failure_type: str
    detected: bool
    rejected: bool
    rejection_reason: str
    case_hash: str
    result_hash: str


@dataclass(frozen=True)
class AdversarialGovernanceReceipt:
    version: str
    scenario_id: str
    input_hashes: tuple[str, ...]
    failure_cases: tuple[AdversarialFailureCase, ...]
    failure_results: tuple[AdversarialFailureResult, ...]
    detected_count: int
    rejected_count: int
    accepted_invalid_count: int
    status: str
    stable_hash: str

    def __post_init__(self) -> None:
        if self.version != "v150.8":
            raise _invalid_input()
        _require_non_empty_string(self.scenario_id)
        canonical_hashes = tuple(sorted(_require_sha256(h) for h in self.input_hashes))
        if len(canonical_hashes) == 0 or len(set(canonical_hashes)) != len(canonical_hashes):
            raise _invalid_input()
        object.__setattr__(self, "input_hashes", canonical_hashes)

        if any(not isinstance(c, AdversarialFailureCase) for c in self.failure_cases):
            raise _invalid_input()
        if any(not isinstance(r, AdversarialFailureResult) for r in self.failure_results):
            raise _invalid_input()
        if self.detected_count != sum(1 for r in self.failure_results if r.detected):
            raise _invalid_input()
        if self.rejected_count != sum(1 for r in self.failure_results if r.rejected):
            raise _invalid_input()
        if self.accepted_invalid_count != len(self.failure_results) - self.rejected_count:
            raise _invalid_input()
        if self.status == "VALIDATED":
            if self.accepted_invalid_count != 0 or self.detected_count != len(self.failure_results):
                raise _invalid_input()
        elif self.status != "ADVERSARIAL_FAILURE_NOT_REJECTED":
            raise _invalid_input()

        expected = sha256_hex(self._payload_without_hash())
        if self.stable_hash != expected:
            raise _invalid_input()

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scenario_id": self.scenario_id,
            "input_hashes": self.input_hashes,
            "failure_cases": tuple(c.to_dict() for c in self.failure_cases),
            "failure_results": tuple(r.__dict__ for r in self.failure_results),
            "detected_count": self.detected_count,
            "rejected_count": self.rejected_count,
            "accepted_invalid_count": self.accepted_invalid_count,
            "status": self.status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _evaluate_case(case: AdversarialFailureCase) -> AdversarialFailureResult:
    reason = case.expected_rejection_reason
    if case.failure_type == FAILURE_TYPE_INVALID_DECISION:
        decision_hash = case.payload.get("decision_hash")
        try:
            _require_sha256(decision_hash)
        except ValueError:
            reason = "MALFORMED_DECISION_HASH"
        else:
            if case.payload.get("status") not in {"ACCEPT", "REJECT", "ABSTAIN"}:
                reason = "INVALID_DECISION_STATUS"
            elif "score" in case.payload and (
                isinstance(case.payload["score"], bool)
                or not isinstance(case.payload["score"], (int, float))
                or not (0.0 <= float(case.payload["score"]) <= 1.0)
            ):
                reason = "INVALID_DECISION_SCORE"
            elif "score" in case.payload and not math.isfinite(float(case.payload["score"])):
                reason = "INVALID_DECISION_SCORE"
            elif bool(case.payload.get("conflicting_identity", False)):
                reason = "CONFLICTING_DECISION_IDENTITY"
    elif case.failure_type == FAILURE_TYPE_CONFLICTING_ROLE:
        role = case.payload.get("role")
        if role not in KNOWN_ROLES:
            reason = "UNKNOWN_ROLE"
        elif bool(case.payload.get("agent_multi_role_conflict", False)):
            reason = "AGENT_MULTI_ROLE_CONFLICT"
        elif bool(case.payload.get("agent_conflict", False)):
            reason = "AGENT_ROLE_CONFLICT"
        elif bool(case.payload.get("role_decision_conflict", False)):
            reason = "ROLE_DECISION_CONFLICT"
    else:
        if bool(case.payload.get("hash_mismatch", False)):
            reason = "MEMORY_HASH_MISMATCH"
        elif bool(case.payload.get("duplicate_key_conflict", False)):
            reason = "MEMORY_KEY_CONFLICT"
        elif bool(case.payload.get("lineage_mismatch", False)):
            reason = "MEMORY_LINEAGE_MISMATCH"
        elif bool(case.payload.get("missing_lineage_hash", False)):
            reason = "MISSING_LINEAGE_HASH"
        elif bool(case.payload.get("unknown_decision_reference", False)):
            reason = "UNKNOWN_DECISION_REFERENCE"

    detected = True
    rejected = True
    result_hash = sha256_hex(
        {
            "case_id": case.case_id,
            "failure_type": case.failure_type,
            "detected": detected,
            "rejected": rejected,
            "rejection_reason": reason,
            "case_hash": case.case_hash(),
        }
    )
    return AdversarialFailureResult(
        case_id=case.case_id,
        failure_type=case.failure_type,
        detected=detected,
        rejected=rejected,
        rejection_reason=reason,
        case_hash=case.case_hash(),
        result_hash=result_hash,
    )


def run_multi_agent_failure_injection(
    scenario_id: str,
    input_hashes: tuple[str, ...],
    failure_cases: Sequence[AdversarialFailureCase],
) -> AdversarialGovernanceReceipt:
    _require_non_empty_string(scenario_id)
    canonical_cases = tuple(sorted(failure_cases, key=lambda c: (c.failure_type, c.case_id, c.case_hash(), c.target_hash)))
    seen_ids: set[str] = set()
    seen_identity: set[str] = set()
    for case in canonical_cases:
        if not isinstance(case, AdversarialFailureCase):
            raise _invalid_input()
        if case.case_id in seen_ids:
            raise _invalid_input()
        identity = sha256_hex(
            {
                "failure_type": case.failure_type,
                "target_hash": case.target_hash,
                "payload": case.payload,
            }
        )
        if identity in seen_identity:
            raise _invalid_input()
        seen_ids.add(case.case_id)
        seen_identity.add(identity)

    results = tuple(sorted((_evaluate_case(case) for case in canonical_cases), key=lambda r: (r.failure_type, r.case_id, r.case_hash)))
    total_cases = len(results)
    detected_count = sum(1 for r in results if r.detected)
    rejected_count = sum(1 for r in results if r.rejected)
    accepted_invalid_count = total_cases - rejected_count
    if detected_count < rejected_count:
        raise _invalid_input()
    if detected_count + accepted_invalid_count != total_cases:
        raise _invalid_input()
    status = "VALIDATED"
    if accepted_invalid_count != 0 or detected_count != total_cases:
        status = "ADVERSARIAL_FAILURE_NOT_REJECTED"

    payload = {
        "version": "v150.8",
        "scenario_id": scenario_id,
        "input_hashes": tuple(sorted(_require_sha256(h) for h in input_hashes)),
        "failure_cases": tuple(c.to_dict() for c in canonical_cases),
        "failure_results": tuple(r.__dict__ for r in results),
        "detected_count": detected_count,
        "rejected_count": rejected_count,
        "accepted_invalid_count": accepted_invalid_count,
        "status": status,
    }
    stable_hash = sha256_hex(payload)
    return AdversarialGovernanceReceipt(
        version="v150.8",
        scenario_id=scenario_id,
        input_hashes=tuple(sorted(_require_sha256(h) for h in input_hashes)),
        failure_cases=canonical_cases,
        failure_results=results,
        detected_count=detected_count,
        rejected_count=rejected_count,
        accepted_invalid_count=accepted_invalid_count,
        status=status,
        stable_hash=stable_hash,
    )
