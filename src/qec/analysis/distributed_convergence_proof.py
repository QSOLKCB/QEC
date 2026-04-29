"""v150.9 — Deterministic distributed convergence proof."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Final

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, canonicalize_json, sha256_hex

VALIDATED: Final[str] = "VALIDATED"
DISTRIBUTED_CONVERGENCE_MISMATCH: Final[str] = "DISTRIBUTED_CONVERGENCE_MISMATCH"
FINAL_PROOF_HASH_MISMATCH: Final[str] = "FINAL_PROOF_HASH_MISMATCH"
EXPECTED_FINAL_PROOF_HASH_MISMATCH: Final[str] = "EXPECTED_FINAL_PROOF_HASH_MISMATCH"
ALLOWED_MISMATCH_REASONS: Final[frozenset[str]] = frozenset(
    {FINAL_PROOF_HASH_MISMATCH, EXPECTED_FINAL_PROOF_HASH_MISMATCH}
)


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _require_non_empty_string(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or value.strip() != value or not value:
        raise _invalid_input()
    return value


def _require_sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise _invalid_input()
    return value


def _validate_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        validated: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key == "":
                raise _invalid_input()
            validated[key] = _validate_json_value(item)
        return validated
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
        return value
    raise _invalid_input()


def _node_key(evidence: "DistributedNodeConvergenceEvidence") -> tuple[str, str, str, str]:
    return (evidence.node_id, evidence.node_role, evidence.final_proof_hash, evidence.evidence_hash)


def _mismatch_key(mismatch: "DistributedConvergenceMismatch") -> tuple[str, str, str, str]:
    return (
        mismatch.reason,
        mismatch.observed_final_proof_hash,
        mismatch.reference_final_proof_hash,
        mismatch.mismatch_hash,
    )


@dataclass(frozen=True)
class DistributedNodeConvergenceEvidence:
    node_id: str
    node_role: str
    convergence_hash: str
    governance_hash: str
    adversarial_hash: str
    final_proof_hash: str
    metadata: Mapping[str, Any]
    evidence_hash: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.node_id)
        _require_non_empty_string(self.node_role)
        object.__setattr__(self, "convergence_hash", _require_sha256(self.convergence_hash))
        object.__setattr__(self, "governance_hash", _require_sha256(self.governance_hash))
        object.__setattr__(self, "adversarial_hash", _require_sha256(self.adversarial_hash))
        object.__setattr__(self, "final_proof_hash", _require_sha256(self.final_proof_hash))
        if not isinstance(self.metadata, Mapping):
            raise _invalid_input()
        canonical_metadata = canonicalize_json(_validate_json_value(dict(self.metadata)))
        object.__setattr__(self, "metadata", canonical_metadata)
        validated_hash = _require_sha256(self.evidence_hash)
        if validated_hash != self.computed_stable_hash():
            raise _invalid_input()

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_role": self.node_role,
            "convergence_hash": self.convergence_hash,
            "governance_hash": self.governance_hash,
            "adversarial_hash": self.adversarial_hash,
            "final_proof_hash": self.final_proof_hash,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "evidence_hash": self.evidence_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._hash_payload())


@dataclass(frozen=True)
class DistributedConvergenceMismatch:
    reference_final_proof_hash: str
    observed_final_proof_hash: str
    node_ids: tuple[str, ...]
    reason: str
    mismatch_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_final_proof_hash", _require_sha256(self.reference_final_proof_hash))
        object.__setattr__(self, "observed_final_proof_hash", _require_sha256(self.observed_final_proof_hash))
        canonical_node_ids = tuple(self.node_ids)
        if not canonical_node_ids:
            raise _invalid_input()
        if any(isinstance(node_id, bool) or not isinstance(node_id, str) or not node_id for node_id in canonical_node_ids):
            raise _invalid_input()
        sorted_node_ids = tuple(sorted(canonical_node_ids))
        if sorted_node_ids != canonical_node_ids or len(set(canonical_node_ids)) != len(canonical_node_ids):
            raise _invalid_input()
        if self.reason not in ALLOWED_MISMATCH_REASONS:
            raise _invalid_input()
        validated_hash = _require_sha256(self.mismatch_hash)
        if validated_hash != self.computed_stable_hash():
            raise _invalid_input()

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "reference_final_proof_hash": self.reference_final_proof_hash,
            "observed_final_proof_hash": self.observed_final_proof_hash,
            "node_ids": list(self.node_ids),
            "reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "mismatch_hash": self.mismatch_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._hash_payload())


@dataclass(frozen=True)
class DistributedConvergenceReceipt:
    version: str
    scenario_id: str
    node_evidence: tuple[DistributedNodeConvergenceEvidence, ...]
    mismatches: tuple[DistributedConvergenceMismatch, ...]
    expected_final_proof_hash: str | None
    reference_final_proof_hash: str
    node_count: int
    agreement_count: int
    mismatch_count: int
    status: str
    stable_hash: str

    def __post_init__(self) -> None:
        if self.version != "v150.9":
            raise _invalid_input()
        _require_non_empty_string(self.scenario_id)
        if self.expected_final_proof_hash is not None:
            object.__setattr__(self, "expected_final_proof_hash", _require_sha256(self.expected_final_proof_hash))
        object.__setattr__(self, "reference_final_proof_hash", _require_sha256(self.reference_final_proof_hash))

        canonical_evidence = tuple(self.node_evidence)
        if any(not isinstance(e, DistributedNodeConvergenceEvidence) for e in canonical_evidence):
            raise _invalid_input()
        sorted_evidence = tuple(sorted(canonical_evidence, key=_node_key))
        if canonical_evidence != sorted_evidence or not canonical_evidence:
            raise _invalid_input()

        node_ids = tuple(e.node_id for e in canonical_evidence)
        if len(set(node_ids)) != len(node_ids):
            raise _invalid_input()

        canonical_mismatches = tuple(self.mismatches)
        if any(not isinstance(m, DistributedConvergenceMismatch) for m in canonical_mismatches):
            raise _invalid_input()
        sorted_mismatches = tuple(sorted(canonical_mismatches, key=_mismatch_key))
        if canonical_mismatches != sorted_mismatches:
            raise _invalid_input()

        if self.node_count != len(canonical_evidence):
            raise _invalid_input()
        recomputed_agreement = sum(1 for e in canonical_evidence if e.final_proof_hash == self.reference_final_proof_hash)
        recomputed_mismatch = self.node_count - recomputed_agreement
        if self.agreement_count != recomputed_agreement or self.mismatch_count != recomputed_mismatch:
            raise _invalid_input()

        if self.status == VALIDATED:
            if self.node_count <= 0 or self.mismatch_count != 0 or self.agreement_count != self.node_count:
                raise _invalid_input()
            if any(e.final_proof_hash != self.reference_final_proof_hash for e in canonical_evidence):
                raise _invalid_input()
            if self.expected_final_proof_hash is not None and self.expected_final_proof_hash != self.reference_final_proof_hash:
                raise _invalid_input()
        elif self.status == DISTRIBUTED_CONVERGENCE_MISMATCH:
            if self.mismatch_count == 0 and self.agreement_count == self.node_count:
                raise _invalid_input()
        else:
            raise _invalid_input()

        if _require_sha256(self.stable_hash) != self.computed_stable_hash():
            raise _invalid_input()

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scenario_id": self.scenario_id,
            "node_evidence": [e.to_dict() for e in self.node_evidence],
            "mismatches": [m.to_dict() for m in self.mismatches],
            "expected_final_proof_hash": self.expected_final_proof_hash,
            "reference_final_proof_hash": self.reference_final_proof_hash,
            "node_count": self.node_count,
            "agreement_count": self.agreement_count,
            "mismatch_count": self.mismatch_count,
            "status": self.status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._hash_payload())


def _derive_reference_final_hash(
    canonical_evidence: tuple[DistributedNodeConvergenceEvidence, ...],
    expected_final_proof_hash: str | None,
) -> str:
    if expected_final_proof_hash is not None:
        return expected_final_proof_hash
    counts = Counter(e.final_proof_hash for e in canonical_evidence)
    max_count = max(counts.values())
    return min(hash_value for hash_value, count in counts.items() if count == max_count)


def run_distributed_convergence_proof(
    scenario_id: str,
    node_evidence: Sequence[DistributedNodeConvergenceEvidence],
    expected_final_proof_hash: str | None = None,
) -> DistributedConvergenceReceipt:
    _require_non_empty_string(scenario_id)
    validated_expected = None if expected_final_proof_hash is None else _require_sha256(expected_final_proof_hash)

    validated_node_evidence = tuple(node_evidence)
    if not validated_node_evidence or any(not isinstance(e, DistributedNodeConvergenceEvidence) for e in validated_node_evidence):
        raise _invalid_input()

    canonical_node_evidence = tuple(sorted(validated_node_evidence, key=_node_key))
    node_ids = tuple(e.node_id for e in canonical_node_evidence)
    if len(set(node_ids)) != len(node_ids):
        raise _invalid_input()

    reference_final_proof_hash = _derive_reference_final_hash(canonical_node_evidence, validated_expected)

    agreement_count = sum(1 for e in canonical_node_evidence if e.final_proof_hash == reference_final_proof_hash)
    node_count = len(canonical_node_evidence)
    mismatch_count = node_count - agreement_count

    mismatches: list[DistributedConvergenceMismatch] = []
    grouped_node_ids: dict[str, list[str]] = {}
    for evidence in canonical_node_evidence:
        if evidence.final_proof_hash != reference_final_proof_hash:
            grouped_node_ids.setdefault(evidence.final_proof_hash, []).append(evidence.node_id)

    for observed_hash in sorted(grouped_node_ids):
        node_id_group = tuple(sorted(grouped_node_ids[observed_hash]))
        mismatch_payload = {
            "reference_final_proof_hash": reference_final_proof_hash,
            "observed_final_proof_hash": observed_hash,
            "node_ids": list(node_id_group),
            "reason": FINAL_PROOF_HASH_MISMATCH,
        }
        mismatches.append(
            DistributedConvergenceMismatch(
                reference_final_proof_hash=reference_final_proof_hash,
                observed_final_proof_hash=observed_hash,
                node_ids=node_id_group,
                reason=FINAL_PROOF_HASH_MISMATCH,
                mismatch_hash=sha256_hex(mismatch_payload),
            )
        )



    observed_hashes = tuple(sorted(set(e.final_proof_hash for e in canonical_node_evidence)))
    if (
        validated_expected is not None
        and len(observed_hashes) == 1
        and observed_hashes[0] != validated_expected
    ):
        expected_mismatch_payload = {
            "reference_final_proof_hash": reference_final_proof_hash,
            "observed_final_proof_hash": observed_hashes[0],
            "node_ids": list(node_ids),
            "reason": EXPECTED_FINAL_PROOF_HASH_MISMATCH,
        }
        mismatches.append(
            DistributedConvergenceMismatch(
                reference_final_proof_hash=reference_final_proof_hash,
                observed_final_proof_hash=observed_hashes[0],
                node_ids=node_ids,
                reason=EXPECTED_FINAL_PROOF_HASH_MISMATCH,
                mismatch_hash=sha256_hex(expected_mismatch_payload),
            )
        )

    status = VALIDATED if mismatch_count == 0 else DISTRIBUTED_CONVERGENCE_MISMATCH
    if status == VALIDATED:
        if mismatch_count != 0 or agreement_count != node_count:
            raise _invalid_input()
    elif status == DISTRIBUTED_CONVERGENCE_MISMATCH:
        if mismatch_count == 0 and agreement_count == node_count:
            raise _invalid_input()
    else:
        raise _invalid_input()

    canonical_mismatches = tuple(sorted(mismatches, key=_mismatch_key))
    payload = {
        "version": "v150.9",
        "scenario_id": scenario_id,
        "node_evidence": [e.to_dict() for e in canonical_node_evidence],
        "mismatches": [m.to_dict() for m in canonical_mismatches],
        "expected_final_proof_hash": validated_expected,
        "reference_final_proof_hash": reference_final_proof_hash,
        "node_count": node_count,
        "agreement_count": agreement_count,
        "mismatch_count": mismatch_count,
        "status": status,
    }
    return DistributedConvergenceReceipt(
        version="v150.9",
        scenario_id=scenario_id,
        node_evidence=canonical_node_evidence,
        mismatches=canonical_mismatches,
        expected_final_proof_hash=validated_expected,
        reference_final_proof_hash=reference_final_proof_hash,
        node_count=node_count,
        agreement_count=agreement_count,
        mismatch_count=mismatch_count,
        status=status,
        stable_hash=sha256_hex(payload),
    )


__all__ = [
    "DistributedNodeConvergenceEvidence",
    "DistributedConvergenceMismatch",
    "DistributedConvergenceReceipt",
    "run_distributed_convergence_proof",
]
