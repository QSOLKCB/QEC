"""v137.18.0 — Proof-Carrying Agent Action Capsule.

Deterministic, replay-safe data contract for packaging an intended agent
action together with its proof obligations, invariants, and receipts. This
module is the bridge between certified orchestration (v137.17.x) and the
future bounded agentic governance layers.

This release is intentionally descriptive and additive. It does NOT execute
actions, enforce policy, or mutate any decoder / supervisory state. It is a
data-contract + validation primitive only.

Hard constraints (per CLAUDE.md / v137.18.0):

* no decoder coupling
* no policy engine
* no auto-remediation
* no autonomy escalation
* no self-modifying behavior
* no async behavior / wall-clock dependence
* no hidden randomness
* no external I/O
* deterministic ordering everywhere
* stable canonical JSON + SHA-256 hashing
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------

#: Narrow, symbolic, non-mutating action vocabulary. Ordered for determinism.
SUPPORTED_ACTION_TYPES: Tuple[str, ...] = (
    "certify",
    "observe",
    "summarize",
    "traverse",
    "validate",
)

#: Proof obligation kinds. Ordered for determinism.
SUPPORTED_OBLIGATION_KINDS: Tuple[str, ...] = (
    "determinism",
    "invariant",
    "postcondition",
    "precondition",
    "replay_identity",
    "schema_stability",
)

#: Validation flags recognized on a capsule. Ordered for determinism.
SUPPORTED_VALIDATION_FLAGS: Tuple[str, ...] = (
    "canonical_bytes_required",
    "deterministic_only",
    "no_external_io",
    "no_mutation",
    "replay_safe",
    "schema_stable",
)


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------


def _canonical_json(data: Any) -> str:
    """Return canonical JSON (sorted keys, no whitespace, strict ASCII, no NaN)."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonicalize_payload(payload: Any) -> Dict[str, Any]:
    """Deterministically canonicalize the action payload.

    Payloads must be JSON-serializable mappings. Nested structures are
    converted into lists / dicts with stable ordering so that canonical JSON
    emission is byte-stable across replays.
    """
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("action_payload must be a mapping")
    return _canonicalize_mapping(payload)


def _canonicalize_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw_key in sorted(mapping.keys(), key=str):
        key = str(raw_key)
        result[key] = _canonicalize_value(mapping[raw_key])
    return result


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _canonicalize_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"unsupported payload value type: {type(value).__name__}")


def _normalize_str_tuple(values: Iterable[Any], *, field: str) -> Tuple[str, ...]:
    normalized: List[str] = []
    for raw in values:
        if raw is None:
            raise ValueError(f"{field} entries must be non-empty strings")
        text = str(raw).strip()
        if not text:
            raise ValueError(f"{field} entries must be non-empty strings")
        normalized.append(text)
    return tuple(sorted(normalized))


def _require_non_empty(value: Any, *, field: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


# ---------------------------------------------------------------------------
# Proof obligation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentActionProofObligation:
    """A single proof obligation attached to an agent action capsule."""

    obligation_id: str
    obligation_kind: str
    obligation_statement: str
    obligation_scope: str
    obligation_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "obligation_kind": self.obligation_kind,
            "obligation_statement": self.obligation_statement,
            "obligation_scope": self.obligation_scope,
            "obligation_epoch": self.obligation_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


def _normalize_obligation(raw: Union["AgentActionProofObligation", Mapping[str, Any]]) -> AgentActionProofObligation:
    if isinstance(raw, AgentActionProofObligation):
        obligation_id = _require_non_empty(raw.obligation_id, field="obligation_id")
        obligation_kind = _require_non_empty(raw.obligation_kind, field="obligation_kind")
        obligation_statement = _require_non_empty(raw.obligation_statement, field="obligation_statement")
        obligation_scope = _require_non_empty(raw.obligation_scope, field="obligation_scope")
        epoch = raw.obligation_epoch
    elif isinstance(raw, Mapping):
        obligation_id = _require_non_empty(raw.get("obligation_id"), field="obligation_id")
        obligation_kind = _require_non_empty(raw.get("obligation_kind"), field="obligation_kind")
        obligation_statement = _require_non_empty(raw.get("obligation_statement"), field="obligation_statement")
        obligation_scope = _require_non_empty(raw.get("obligation_scope"), field="obligation_scope")
        epoch = raw.get("obligation_epoch", 0)
    else:
        raise ValueError("proof obligation must be mapping or AgentActionProofObligation")

    if obligation_kind not in SUPPORTED_OBLIGATION_KINDS:
        raise ValueError(f"unsupported obligation_kind: {obligation_kind}")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise ValueError("obligation_epoch must be a non-negative integer")
    if epoch < 0:
        raise ValueError("obligation_epoch must be a non-negative integer")

    return AgentActionProofObligation(
        obligation_id=obligation_id,
        obligation_kind=obligation_kind,
        obligation_statement=obligation_statement,
        obligation_scope=obligation_scope,
        obligation_epoch=epoch,
    )


def _normalize_obligations(
    raw_obligations: Iterable[Union[AgentActionProofObligation, Mapping[str, Any]]],
) -> Tuple[AgentActionProofObligation, ...]:
    normalized = [_normalize_obligation(item) for item in raw_obligations]
    seen_ids = set()
    for obligation in normalized:
        if obligation.obligation_id in seen_ids:
            raise ValueError(f"duplicate proof obligation id: {obligation.obligation_id}")
        seen_ids.add(obligation.obligation_id)
    normalized.sort(
        key=lambda o: (o.obligation_epoch, o.obligation_kind, o.obligation_id)
    )
    return tuple(normalized)


# ---------------------------------------------------------------------------
# Proof receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentActionProofReceipt:
    """Deterministic receipt attesting to a capsule's proof obligations."""

    receipt_id: str
    action_id: str
    capsule_hash: str
    obligation_hashes: Tuple[str, ...]
    receipt_epoch: int
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "action_id": self.action_id,
            "capsule_hash": self.capsule_hash,
            "obligation_hashes": list(self.obligation_hashes),
            "receipt_epoch": self.receipt_epoch,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


def build_action_proof_receipt(
    action_id: str,
    capsule_hash: str,
    obligations: Sequence[AgentActionProofObligation],
    *,
    receipt_epoch: int = 0,
) -> AgentActionProofReceipt:
    """Build a deterministic proof receipt for a capsule snapshot."""
    normalized_action_id = _require_non_empty(action_id, field="action_id")
    normalized_capsule_hash = _require_non_empty(capsule_hash, field="capsule_hash")
    if not isinstance(receipt_epoch, int) or isinstance(receipt_epoch, bool):
        raise ValueError("receipt_epoch must be a non-negative integer")
    if receipt_epoch < 0:
        raise ValueError("receipt_epoch must be a non-negative integer")

    normalized_obligations = _normalize_obligations(obligations)
    obligation_hashes = tuple(sorted(ob.stable_hash() for ob in normalized_obligations))

    receipt_id = f"agent-action-receipt::{normalized_capsule_hash[:16]}::{receipt_epoch}"

    unsigned_payload = {
        "receipt_id": receipt_id,
        "action_id": normalized_action_id,
        "capsule_hash": normalized_capsule_hash,
        "obligation_hashes": list(obligation_hashes),
        "receipt_epoch": receipt_epoch,
    }
    receipt_hash = _sha256_hex(_canonical_json(unsigned_payload).encode("utf-8"))

    return AgentActionProofReceipt(
        receipt_id=receipt_id,
        action_id=normalized_action_id,
        capsule_hash=normalized_capsule_hash,
        obligation_hashes=obligation_hashes,
        receipt_epoch=receipt_epoch,
        receipt_hash=receipt_hash,
    )


def _normalize_receipt(
    raw: Union[AgentActionProofReceipt, Mapping[str, Any]],
) -> AgentActionProofReceipt:
    if isinstance(raw, AgentActionProofReceipt):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("receipt must be mapping or AgentActionProofReceipt")
    obligation_hashes = tuple(str(h).strip() for h in raw.get("obligation_hashes", ()))
    return AgentActionProofReceipt(
        receipt_id=_require_non_empty(raw.get("receipt_id"), field="receipt_id"),
        action_id=_require_non_empty(raw.get("action_id"), field="action_id"),
        capsule_hash=_require_non_empty(raw.get("capsule_hash"), field="capsule_hash"),
        obligation_hashes=obligation_hashes,
        receipt_epoch=int(raw.get("receipt_epoch", 0)),
        receipt_hash=_require_non_empty(raw.get("receipt_hash"), field="receipt_hash"),
    )


def _normalize_receipt_chain(
    receipts: Iterable[Union[AgentActionProofReceipt, Mapping[str, Any]]],
) -> Tuple[AgentActionProofReceipt, ...]:
    normalized = [_normalize_receipt(item) for item in receipts]
    # Receipt chains must be monotonically non-decreasing by epoch, with
    # deterministic tie-breaking by receipt_id for stable ordering.
    sorted_chain = sorted(normalized, key=lambda r: (r.receipt_epoch, r.receipt_id))
    if list(sorted_chain) != list(normalized):
        raise ValueError("unsorted proof receipt chain")
    seen_ids = set()
    for receipt in sorted_chain:
        if receipt.receipt_id in seen_ids:
            raise ValueError(f"duplicate receipt id in chain: {receipt.receipt_id}")
        seen_ids.add(receipt.receipt_id)
    return tuple(sorted_chain)


# ---------------------------------------------------------------------------
# Capsule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofCarryingAgentActionCapsule:
    """Deterministic proof-carrying capsule for an intended agent action."""

    action_id: str
    action_type: str
    action_scope: str
    action_payload: Mapping[str, Any]
    preconditions: Tuple[str, ...]
    invariants: Tuple[str, ...]
    proof_obligations: Tuple[AgentActionProofObligation, ...]
    validation_flags: Tuple[str, ...]
    receipt_chain: Tuple[AgentActionProofReceipt, ...]
    replay_identity: str
    capsule_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_scope": self.action_scope,
            "action_payload": _canonicalize_mapping(self.action_payload),
            "preconditions": list(self.preconditions),
            "invariants": list(self.invariants),
            "proof_obligations": [ob.to_dict() for ob in self.proof_obligations],
            "validation_flags": list(self.validation_flags),
            "receipt_chain": [r.to_dict() for r in self.receipt_chain],
            "replay_identity": self.replay_identity,
            "capsule_hash": self.capsule_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.capsule_hash


def _capsule_body_payload(
    *,
    action_id: str,
    action_type: str,
    action_scope: str,
    action_payload: Mapping[str, Any],
    preconditions: Tuple[str, ...],
    invariants: Tuple[str, ...],
    proof_obligations: Tuple[AgentActionProofObligation, ...],
    validation_flags: Tuple[str, ...],
) -> Dict[str, Any]:
    """Canonical body used to compute the capsule hash.

    The receipt chain is intentionally excluded so that receipts can be
    appended to a capsule without invalidating the hash they attest to.
    Receipts themselves are validated separately against this body hash.
    """
    return {
        "action_id": action_id,
        "action_type": action_type,
        "action_scope": action_scope,
        "action_payload": _canonicalize_mapping(action_payload),
        "preconditions": list(preconditions),
        "invariants": list(invariants),
        "proof_obligations": [ob.to_dict() for ob in proof_obligations],
        "validation_flags": list(validation_flags),
    }


def build_proof_carrying_agent_action_capsule(
    action_id: str,
    action_type: str,
    action_scope: str,
    action_payload: Mapping[str, Any],
    *,
    preconditions: Sequence[str] = (),
    invariants: Sequence[str] = (),
    proof_obligations: Sequence[Union[AgentActionProofObligation, Mapping[str, Any]]] = (),
    validation_flags: Sequence[str] = (),
    receipt_chain: Sequence[Union[AgentActionProofReceipt, Mapping[str, Any]]] = (),
) -> ProofCarryingAgentActionCapsule:
    """Deterministically construct a proof-carrying agent action capsule."""
    normalized_action_id = _require_non_empty(action_id, field="action_id")
    normalized_action_type = _require_non_empty(action_type, field="action_type")
    if normalized_action_type not in SUPPORTED_ACTION_TYPES:
        raise ValueError(f"unsupported action_type: {normalized_action_type}")
    normalized_action_scope = _require_non_empty(action_scope, field="action_scope")
    normalized_payload = _canonicalize_payload(action_payload)

    normalized_preconditions = _normalize_str_tuple(preconditions, field="preconditions")
    normalized_invariants = _normalize_str_tuple(invariants, field="invariants")
    for invariant in normalized_invariants:
        if ":" not in invariant:
            raise ValueError(
                f"invalid invariant structure (expected 'kind:statement'): {invariant}"
            )

    normalized_flags = _normalize_str_tuple(validation_flags, field="validation_flags")
    for flag in normalized_flags:
        if flag not in SUPPORTED_VALIDATION_FLAGS:
            raise ValueError(f"unsupported validation_flag: {flag}")

    normalized_obligations = _normalize_obligations(proof_obligations)
    normalized_chain = _normalize_receipt_chain(receipt_chain)

    body = _capsule_body_payload(
        action_id=normalized_action_id,
        action_type=normalized_action_type,
        action_scope=normalized_action_scope,
        action_payload=normalized_payload,
        preconditions=normalized_preconditions,
        invariants=normalized_invariants,
        proof_obligations=normalized_obligations,
        validation_flags=normalized_flags,
    )
    capsule_hash = _sha256_hex(_canonical_json(body).encode("utf-8"))

    # Validate any provided receipts against the newly-computed capsule hash
    # so that receipt_chain injection cannot silently bypass binding.
    for receipt in normalized_chain:
        if receipt.action_id != normalized_action_id:
            raise ValueError(f"receipt mismatch: action_id for {receipt.receipt_id}")
        if receipt.capsule_hash != capsule_hash:
            raise ValueError(f"receipt mismatch: capsule_hash for {receipt.receipt_id}")
        expected = build_action_proof_receipt(
            normalized_action_id,
            capsule_hash,
            normalized_obligations,
            receipt_epoch=receipt.receipt_epoch,
        )
        if receipt.receipt_hash != expected.receipt_hash:
            raise ValueError(f"receipt mismatch: receipt_hash for {receipt.receipt_id}")
        if receipt.obligation_hashes != expected.obligation_hashes:
            raise ValueError(f"receipt mismatch: obligation_hashes for {receipt.receipt_id}")
    replay_identity = f"agent-action::{normalized_action_type}::{capsule_hash[:16]}"

    return ProofCarryingAgentActionCapsule(
        action_id=normalized_action_id,
        action_type=normalized_action_type,
        action_scope=normalized_action_scope,
        action_payload=normalized_payload,
        preconditions=normalized_preconditions,
        invariants=normalized_invariants,
        proof_obligations=normalized_obligations,
        validation_flags=normalized_flags,
        receipt_chain=normalized_chain,
        replay_identity=replay_identity,
        capsule_hash=capsule_hash,
    )


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentActionValidationReport:
    """Deterministic validation report for a capsule."""

    action_id: str
    capsule_hash: str
    is_valid: bool
    violations: Tuple[str, ...]
    report_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "capsule_hash": self.capsule_hash,
            "is_valid": self.is_valid,
            "violations": list(self.violations),
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.report_hash


def validate_agent_action_capsule(
    capsule: ProofCarryingAgentActionCapsule,
) -> AgentActionValidationReport:
    """Deterministically validate a capsule and return a validation report."""
    violations: List[str] = []

    if not capsule.action_id:
        violations.append("empty action_id")
    if capsule.action_type not in SUPPORTED_ACTION_TYPES:
        violations.append(f"unsupported action_type: {capsule.action_type}")
    if not capsule.action_scope:
        violations.append("empty action_scope")
    if not isinstance(capsule.action_payload, Mapping):
        violations.append("malformed payload: not a mapping")

    # Obligation checks
    obligation_ids = [ob.obligation_id for ob in capsule.proof_obligations]
    if len(obligation_ids) != len(set(obligation_ids)):
        violations.append("duplicate proof obligations")
    sorted_obligations = sorted(
        capsule.proof_obligations,
        key=lambda o: (o.obligation_epoch, o.obligation_kind, o.obligation_id),
    )
    if list(sorted_obligations) != list(capsule.proof_obligations):
        violations.append("unsorted proof obligations")
    for obligation in capsule.proof_obligations:
        if obligation.obligation_kind not in SUPPORTED_OBLIGATION_KINDS:
            violations.append(f"unsupported obligation_kind: {obligation.obligation_kind}")
        if obligation.obligation_epoch < 0:
            violations.append(f"negative obligation_epoch: {obligation.obligation_id}")

    # Invariants structural check
    for invariant in capsule.invariants:
        if ":" not in invariant:
            violations.append(f"invalid invariant structure: {invariant}")

    # Validation flags check
    for flag in capsule.validation_flags:
        if flag not in SUPPORTED_VALIDATION_FLAGS:
            violations.append(f"unsupported validation_flag: {flag}")

    # Receipt chain checks
    sorted_chain = sorted(
        capsule.receipt_chain, key=lambda r: (r.receipt_epoch, r.receipt_id)
    )
    if list(sorted_chain) != list(capsule.receipt_chain):
        violations.append("unsorted proof chain")
    seen_receipt_ids: set = set()
    for receipt in capsule.receipt_chain:
        if receipt.receipt_id in seen_receipt_ids:
            violations.append(f"duplicate receipt id: {receipt.receipt_id}")
        seen_receipt_ids.add(receipt.receipt_id)
        if receipt.action_id != capsule.action_id:
            violations.append(f"receipt mismatch: action_id for {receipt.receipt_id}")
        if receipt.capsule_hash != capsule.capsule_hash:
            violations.append(f"receipt mismatch: capsule_hash for {receipt.receipt_id}")
        expected = build_action_proof_receipt(
            capsule.action_id,
            capsule.capsule_hash,
            capsule.proof_obligations,
            receipt_epoch=receipt.receipt_epoch,
        )
        if receipt.receipt_hash != expected.receipt_hash:
            violations.append(f"receipt mismatch: receipt_hash for {receipt.receipt_id}")
        if receipt.obligation_hashes != expected.obligation_hashes:
            violations.append(f"receipt mismatch: obligation_hashes for {receipt.receipt_id}")

    # Replay identity check (body excludes receipt_chain by design).
    body = _capsule_body_payload(
        action_id=capsule.action_id,
        action_type=capsule.action_type,
        action_scope=capsule.action_scope,
        action_payload=capsule.action_payload,
        preconditions=capsule.preconditions,
        invariants=capsule.invariants,
        proof_obligations=capsule.proof_obligations,
        validation_flags=capsule.validation_flags,
    )
    recomputed_hash = _sha256_hex(_canonical_json(body).encode("utf-8"))
    if recomputed_hash != capsule.capsule_hash:
        violations.append("replay drift: capsule_hash mismatch")
    expected_identity = f"agent-action::{capsule.action_type}::{capsule.capsule_hash[:16]}"
    if capsule.replay_identity != expected_identity:
        violations.append("replay drift: replay_identity mismatch")

    ordered_violations = tuple(sorted(violations))
    is_valid = len(ordered_violations) == 0
    report_payload = {
        "action_id": capsule.action_id,
        "capsule_hash": capsule.capsule_hash,
        "is_valid": is_valid,
        "violations": list(ordered_violations),
    }
    report_hash = _sha256_hex(_canonical_json(report_payload).encode("utf-8"))

    return AgentActionValidationReport(
        action_id=capsule.action_id,
        capsule_hash=capsule.capsule_hash,
        is_valid=is_valid,
        violations=ordered_violations,
        report_hash=report_hash,
    )


# ---------------------------------------------------------------------------
# Certification + replay comparison
# ---------------------------------------------------------------------------


def certify_agent_action_capsule(
    capsule: ProofCarryingAgentActionCapsule,
) -> AgentActionValidationReport:
    """Validate a capsule and raise ValueError on any violation.

    Returns the validation report on success.
    """
    report = validate_agent_action_capsule(capsule)
    if not report.is_valid:
        raise ValueError(
            f"agent action capsule failed certification: {','.join(report.violations)}"
        )
    return report


def compare_action_capsule_replay(
    capsule_a: ProofCarryingAgentActionCapsule,
    capsule_b: ProofCarryingAgentActionCapsule,
) -> bool:
    """Return True if two capsules are byte-identical under canonical JSON.

    Raises ValueError with a deterministic reason on replay drift.
    """
    bytes_a = capsule_a.to_canonical_json().encode("utf-8")
    bytes_b = capsule_b.to_canonical_json().encode("utf-8")
    if bytes_a != bytes_b:
        raise ValueError(
            f"replay drift: canonical bytes differ ({_sha256_hex(bytes_a)} != {_sha256_hex(bytes_b)})"
        )
    if capsule_a.capsule_hash != capsule_b.capsule_hash:
        raise ValueError("replay drift: capsule_hash mismatch")
    if capsule_a.replay_identity != capsule_b.replay_identity:
        raise ValueError("replay drift: replay_identity mismatch")
    return True


__all__ = (
    "SUPPORTED_ACTION_TYPES",
    "SUPPORTED_OBLIGATION_KINDS",
    "SUPPORTED_VALIDATION_FLAGS",
    "AgentActionProofObligation",
    "AgentActionProofReceipt",
    "ProofCarryingAgentActionCapsule",
    "AgentActionValidationReport",
    "build_proof_carrying_agent_action_capsule",
    "validate_agent_action_capsule",
    "certify_agent_action_capsule",
    "build_action_proof_receipt",
    "compare_action_capsule_replay",
)
