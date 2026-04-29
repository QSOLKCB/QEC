"""This module adheres to the QEC identity and hashing surface contract.

See: qec.analysis.identity_hashing_contract.get_identity_hashing_contract()
"""


from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.identity_contract import get_identity_contract
from qec.analysis.canonical_identity import _invalid_input, _require_sha256_hex, canonical_hash_identity


def _sorted_reports(reports: Sequence[NodeProofReport]) -> tuple[NodeProofReport, ...]:
    return tuple(sorted(reports, key=lambda report: (report.proof_hash, report.governance_hash, report.node_id)))


def _consistency_hash_payload(
    distributed_state: DistributedProofState,
    input_memory_hashes: tuple[str, ...],
    governance_hashes: tuple[str, ...],
    node_proof_hashes: tuple[str, ...],
) -> dict[str, object]:
    return {
        "distributed_state": distributed_state.to_dict(),
        "input_memory_hashes": list(input_memory_hashes),
        "governance_hashes": list(governance_hashes),
        "node_proof_hashes": list(node_proof_hashes),
    }


@dataclass(frozen=True)
class NodeProofReport:
    node_id: str
    governance_hash: str
    proof_hash: str
    input_memory_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.node_id, bool) or not isinstance(self.node_id, str) or not self.node_id:
            raise _invalid_input()
        object.__setattr__(self, "governance_hash", _require_sha256_hex(self.governance_hash))
        object.__setattr__(self, "proof_hash", _require_sha256_hex(self.proof_hash))
        object.__setattr__(self, "input_memory_hashes", canonical_hash_identity(self.input_memory_hashes))

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "governance_hash": self.governance_hash,
            "proof_hash": self.proof_hash,
            "input_memory_hashes": list(self.input_memory_hashes),
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class DistributedProofState:
    reports: tuple[NodeProofReport, ...]
    agreed_proof_hash: str
    consistent: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "reports", tuple(self.reports))
        ordered_reports = _sorted_reports(self.reports)
        if ordered_reports != self.reports:
            raise _invalid_input()
        if not ordered_reports:
            raise _invalid_input()

        seen_node_ids: set[str] = set()
        proof_hashes: set[str] = set()
        baseline_input_hashes = ordered_reports[0].input_memory_hashes

        for report in ordered_reports:
            if report.node_id in seen_node_ids:
                raise _invalid_input()
            seen_node_ids.add(report.node_id)
            if report.input_memory_hashes != baseline_input_hashes:
                raise _invalid_input()
            proof_hashes.add(report.proof_hash)

        validated_agreed = _require_sha256_hex(self.agreed_proof_hash)
        object.__setattr__(self, "agreed_proof_hash", validated_agreed)

        expected_consistent = len(proof_hashes) == 1
        if self.consistent is not expected_consistent:
            raise _invalid_input()

        expected_agreed = sorted(proof_hashes)[0]
        if validated_agreed != expected_agreed:
            raise _invalid_input()

    def to_dict(self) -> dict[str, object]:
        return {
            "reports": [report.to_dict() for report in self.reports],
            "agreed_proof_hash": self.agreed_proof_hash,
            "consistent": self.consistent,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class DistributedProofConsistencyReceipt:
    distributed_state: DistributedProofState
    input_memory_hashes: tuple[str, ...]
    governance_hashes: tuple[str, ...]
    node_proof_hashes: tuple[str, ...]
    consistency_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.distributed_state, DistributedProofState):
            raise _invalid_input()

        validated_input_hashes = canonical_hash_identity(self.input_memory_hashes)
        validated_governance_hashes = canonical_hash_identity(self.governance_hashes)
        validated_proof_hashes = canonical_hash_identity(self.node_proof_hashes)
        validated_consistency_hash = _require_sha256_hex(self.consistency_hash)

        object.__setattr__(self, "input_memory_hashes", validated_input_hashes)
        object.__setattr__(self, "governance_hashes", validated_governance_hashes)
        object.__setattr__(self, "node_proof_hashes", validated_proof_hashes)
        object.__setattr__(self, "consistency_hash", validated_consistency_hash)

        reports = self.distributed_state.reports
        if not reports:
            raise _invalid_input()

        expected_input_hashes = reports[0].input_memory_hashes
        if validated_input_hashes != expected_input_hashes:
            raise _invalid_input()

        expected_governance_hashes = canonical_hash_identity(tuple(sorted(set(report.governance_hash for report in reports))))
        expected_proof_hashes = canonical_hash_identity(tuple(sorted(set(report.proof_hash for report in reports))))
        if validated_governance_hashes != expected_governance_hashes:
            raise _invalid_input()
        if validated_proof_hashes != expected_proof_hashes:
            raise _invalid_input()

        if validated_consistency_hash != self.computed_stable_hash():
            raise _invalid_input()

    def to_dict(self) -> dict[str, object]:
        return {
            "distributed_state": self.distributed_state.to_dict(),
            "input_memory_hashes": list(self.input_memory_hashes),
            "governance_hashes": list(self.governance_hashes),
            "node_proof_hashes": list(self.node_proof_hashes),
            "consistency_hash": self.consistency_hash,
        }

    def _hash_payload(self) -> dict[str, object]:
        return _consistency_hash_payload(
            distributed_state=self.distributed_state,
            input_memory_hashes=self.input_memory_hashes,
            governance_hashes=self.governance_hashes,
            node_proof_hashes=self.node_proof_hashes,
        )

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._hash_payload())


def verify_distributed_proof_consistency(
    reports: Sequence[NodeProofReport],
) -> DistributedProofConsistencyReceipt:
    canonical_reports = tuple(reports)
    if not canonical_reports:
        raise _invalid_input()
    if any(not isinstance(report, NodeProofReport) for report in canonical_reports):
        raise _invalid_input()

    seen_node_ids: set[str] = set()
    baseline_input_hashes = canonical_reports[0].input_memory_hashes
    for report in canonical_reports:
        if report.node_id in seen_node_ids:
            raise _invalid_input()
        seen_node_ids.add(report.node_id)
        if report.input_memory_hashes != baseline_input_hashes:
            raise _invalid_input()

    sorted_reports = _sorted_reports(canonical_reports)
    proof_hashes = tuple(report.proof_hash for report in sorted_reports)
    node_proof_hashes = canonical_hash_identity(tuple(sorted(set(proof_hashes))))

    consistent = len(node_proof_hashes) == 1
    agreed_proof_hash = node_proof_hashes[0]

    distributed_state = DistributedProofState(
        reports=sorted_reports,
        agreed_proof_hash=agreed_proof_hash,
        consistent=consistent,
    )

    governance_hashes = canonical_hash_identity(tuple(sorted(set(report.governance_hash for report in sorted_reports))))
    input_memory_hashes = canonical_hash_identity(baseline_input_hashes)

    consistency_hash = sha256_hex(
        _consistency_hash_payload(
            distributed_state=distributed_state,
            input_memory_hashes=input_memory_hashes,
            governance_hashes=governance_hashes,
            node_proof_hashes=node_proof_hashes,
        )
    )
    return DistributedProofConsistencyReceipt(
        distributed_state=distributed_state,
        input_memory_hashes=input_memory_hashes,
        governance_hashes=governance_hashes,
        node_proof_hashes=node_proof_hashes,
        consistency_hash=consistency_hash,
    )


__all__ = [
    "NodeProofReport",
    "DistributedProofState",
    "DistributedProofConsistencyReceipt",
    "verify_distributed_proof_consistency",
]
