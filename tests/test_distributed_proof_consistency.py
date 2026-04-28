from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import itertools
import json

import pytest

from qec.analysis.canonical_identity import canonical_hash_identity
from qec.analysis.distributed_proof_consistency import (
    DistributedProofConsistencyReceipt,
    DistributedProofState,
    NodeProofReport,
    verify_distributed_proof_consistency,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _report(node_id: str, governance: str, proof: str, memory: tuple[str, ...]) -> NodeProofReport:
    return NodeProofReport(
        node_id=node_id,
        governance_hash=governance,
        proof_hash=proof,
        input_memory_hashes=memory,
    )


def _canonical_hashes(*labels: str) -> tuple[str, ...]:
    return tuple(sorted(_h(label) for label in labels))


def _base_reports() -> tuple[NodeProofReport, ...]:
    memory = canonical_hash_identity(_canonical_hashes("m0", "m1", "m2"))
    proof = _h("proof/shared")
    return (
        _report("node-a", _h("gov/a"), proof, memory),
        _report("node-b", _h("gov/b"), proof, memory),
        _report("node-c", _h("gov/c"), proof, memory),
    )


def test_deterministic_replay_100_runs() -> None:
    reports = _base_reports()
    first = verify_distributed_proof_consistency(reports)
    for _ in range(100):
        current = verify_distributed_proof_consistency(reports)
        assert current.to_canonical_bytes() == first.to_canonical_bytes()


def test_permutation_invariance() -> None:
    reports = _base_reports()
    expected = verify_distributed_proof_consistency(reports)
    for perm in itertools.permutations(reports):
        actual = verify_distributed_proof_consistency(perm)
        assert actual.to_canonical_bytes() == expected.to_canonical_bytes()


def test_consistent_proof_agreement() -> None:
    receipt = verify_distributed_proof_consistency(_base_reports())
    assert receipt.distributed_state.consistent is True
    assert receipt.distributed_state.agreed_proof_hash == _h("proof/shared")


def test_divergence_detection_uses_sorted_first_proof_hash() -> None:
    memory = canonical_hash_identity(_canonical_hashes("m0", "m1"))
    reports = (
        _report("node-a", _h("gov/a"), _h("proof/z"), memory),
        _report("node-b", _h("gov/b"), _h("proof/a"), memory),
        _report("node-c", _h("gov/c"), _h("proof/m"), memory),
    )

    receipt = verify_distributed_proof_consistency(reports)
    expected = sorted((_h("proof/z"), _h("proof/a"), _h("proof/m")))[0]
    assert receipt.distributed_state.consistent is False
    assert receipt.distributed_state.agreed_proof_hash == expected


def test_canonical_identity_helper_rules() -> None:
    valid = canonical_hash_identity(_canonical_hashes("a", "b", "c"))
    assert valid == tuple(sorted(valid))

    sorted_pair = _canonical_hashes("a", "b")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        canonical_hash_identity((sorted_pair[1], sorted_pair[0]))

    duplicate = _h("dup")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        canonical_hash_identity((duplicate, duplicate))

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        canonical_hash_identity((_h("ok"), "not-a-sha256"))


def test_duplicate_node_rejected() -> None:
    memory = canonical_hash_identity(_canonical_hashes("m0"))
    reports = (
        _report("node-x", _h("gov/a"), _h("proof"), memory),
        _report("node-x", _h("gov/b"), _h("proof"), memory),
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        verify_distributed_proof_consistency(reports)


def test_input_memory_identity_mismatch_rejected() -> None:
    reports = (
        _report("node-a", _h("gov/a"), _h("proof"), canonical_hash_identity(_canonical_hashes("m0"))),
        _report("node-b", _h("gov/b"), _h("proof"), canonical_hash_identity(_canonical_hashes("m1"))),
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        verify_distributed_proof_consistency(reports)


def test_receipt_hash_stability_recomputes_exactly() -> None:
    receipt = verify_distributed_proof_consistency(_base_reports())
    payload = receipt.to_dict()
    payload.pop("consistency_hash")
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert recomputed == receipt.consistency_hash
    assert recomputed == receipt.computed_stable_hash()


def test_immutability() -> None:
    report = _base_reports()[0]
    with pytest.raises(FrozenInstanceError):
        report.node_id = "mutate"  # type: ignore[misc]

    receipt = verify_distributed_proof_consistency(_base_reports())
    with pytest.raises(FrozenInstanceError):
        receipt.consistency_hash = "0" * 64  # type: ignore[misc]


def test_invalid_report_inputs_rejected() -> None:
    memory = canonical_hash_identity(_canonical_hashes("m0", "m1"))

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _report("", _h("gov"), _h("proof"), memory)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _report("node", "G" * 64, _h("proof"), memory)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _report("node", _h("gov"), "Z" * 64, memory)

    non_canonical_memory = tuple(reversed(memory))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _report("node", _h("gov"), _h("proof"), non_canonical_memory)


def test_receipt_constructor_validation_rejects_mismatched_consistency_hash() -> None:
    receipt = verify_distributed_proof_consistency(_base_reports())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DistributedProofConsistencyReceipt(
            distributed_state=receipt.distributed_state,
            input_memory_hashes=receipt.input_memory_hashes,
            governance_hashes=receipt.governance_hashes,
            node_proof_hashes=receipt.node_proof_hashes,
            consistency_hash="f" * 64,
        )


def test_distributed_state_validation_rejects_unsorted_reports() -> None:
    reports = _base_reports()
    unsorted_reports = (reports[2], reports[0], reports[1])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DistributedProofState(
            reports=unsorted_reports,
            agreed_proof_hash=reports[0].proof_hash,
            consistent=True,
        )
