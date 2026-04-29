import json
from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.distributed_convergence_proof import (
    DISTRIBUTED_CONVERGENCE_MISMATCH,
    EXPECTED_FINAL_PROOF_HASH_MISMATCH,
    FINAL_PROOF_HASH_MISMATCH,
    VALIDATED,
    DistributedConvergenceReceipt,
    DistributedNodeConvergenceEvidence,
    run_distributed_convergence_proof,
)


def _h(label: str) -> str:
    return sha256_hex({"label": label})


def _evidence(node_id: str, final_hash: str, metadata: dict | None = None) -> DistributedNodeConvergenceEvidence:
    payload = {
        "node_id": node_id,
        "node_role": "CONTROL",
        "convergence_hash": _h(f"conv:{node_id}"),
        "governance_hash": _h(f"gov:{node_id}"),
        "adversarial_hash": _h(f"adv:{node_id}"),
        "final_proof_hash": final_hash,
        "metadata": {} if metadata is None else metadata,
    }
    try:
        evidence_hash = sha256_hex(payload)
    except Exception:
        evidence_hash = "0" * 64
    return DistributedNodeConvergenceEvidence(**payload, evidence_hash=evidence_hash)


def test_valid_distributed_convergence_and_expected_match() -> None:
    shared = _h("shared")
    nodes = (_evidence("n1", shared), _evidence("n2", shared), _evidence("n3", shared))

    receipt = run_distributed_convergence_proof("s1", nodes)
    receipt_expected = run_distributed_convergence_proof("s1", nodes, expected_final_proof_hash=shared)

    assert receipt.status == VALIDATED
    assert receipt.agreement_count == receipt.node_count == 3
    assert receipt.mismatch_count == 0
    assert receipt.reference_final_proof_hash == shared
    assert receipt_expected.status == VALIDATED


def test_expected_hash_mismatch() -> None:
    shared = _h("shared")
    expected = _h("different")
    receipt = run_distributed_convergence_proof("s2", (_evidence("n1", shared), _evidence("n2", shared)), expected)

    assert receipt.status == DISTRIBUTED_CONVERGENCE_MISMATCH
    assert receipt.mismatch_count == 2
    reasons = {m.reason for m in receipt.mismatches}
    assert EXPECTED_FINAL_PROOF_HASH_MISMATCH in reasons
    assert FINAL_PROOF_HASH_MISMATCH in reasons


def test_node_disagreement_majority_tie_and_reorder_invariance() -> None:
    a = _h("a")
    b = _h("b")
    c = _h("c")
    majority = run_distributed_convergence_proof("s3", (_evidence("n1", a), _evidence("n2", a), _evidence("n3", b)))

    assert majority.status == DISTRIBUTED_CONVERGENCE_MISMATCH
    assert majority.reference_final_proof_hash == a
    assert majority.mismatch_count == 1
    assert majority.mismatches[0].node_ids == ("n3",)
    assert majority.mismatches[0].reason == FINAL_PROOF_HASH_MISMATCH

    tie = run_distributed_convergence_proof("s4", (_evidence("n1", b), _evidence("n2", c)))
    assert tie.reference_final_proof_hash == min(b, c)

    original = run_distributed_convergence_proof("s5", (_evidence("n1", a), _evidence("n2", b), _evidence("n3", a)))
    reversed_order = run_distributed_convergence_proof("s5", tuple(reversed((_evidence("n1", a), _evidence("n2", b), _evidence("n3", a)))))
    assert original.stable_hash == reversed_order.stable_hash
    assert original.to_canonical_json() == reversed_order.to_canonical_json()
    assert original.to_canonical_bytes() == reversed_order.to_canonical_bytes()
    assert run_distributed_convergence_proof("s5", (_evidence("n1", a), _evidence("n2", b), _evidence("n3", a))).stable_hash == original.stable_hash


def test_invalid_hash_and_duplicates() -> None:
    shared = _h("shared")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_distributed_convergence_proof("s6", (_evidence("n1", shared), _evidence("n1", shared)))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _evidence("n1", "abc")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _evidence("n1", _h("x").upper())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _evidence("n1", "g" * 64)


def test_metadata_validation_immutability_and_json() -> None:
    shared = _h("shared")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _evidence("n1", shared, {1: "x"})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _evidence("n1", shared, {"": "x"})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _evidence("n1", shared, {"x": float("nan")})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _evidence("n1", shared, {"x": float("inf")})

    evidence = _evidence("n2", shared, {"score": 1.25, "enabled": True, "flags": [1, 2, 3]})
    receipt = run_distributed_convergence_proof("s7", (evidence, _evidence("n3", shared)))
    json.dumps(evidence.to_dict())
    json.dumps(receipt.to_dict())

    with pytest.raises(FrozenInstanceError):
        evidence.node_id = "mutated"
    with pytest.raises(FrozenInstanceError):
        receipt.status = "X"


def test_receipt_integrity_and_branch_reachability() -> None:
    a = _h("a")
    b = _h("b")
    n1 = _evidence("n1", a)
    n2 = _evidence("n2", b)
    validated_receipt = run_distributed_convergence_proof("s8", (n1, _evidence("n3", a)))
    mismatch_receipt = run_distributed_convergence_proof("s9", (n1, n2))

    assert validated_receipt.status == "VALIDATED"
    assert mismatch_receipt.status == "DISTRIBUTED_CONVERGENCE_MISMATCH"
    assert validated_receipt.computed_stable_hash() == validated_receipt.stable_hash

    unsorted_nodes = tuple(sorted((n1, n2), key=lambda e: e.node_id, reverse=True))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DistributedConvergenceReceipt(
            version="v150.9",
            scenario_id="bad1",
            node_evidence=unsorted_nodes,
            mismatches=tuple(),
            expected_final_proof_hash=None,
            reference_final_proof_hash=a,
            node_count=2,
            agreement_count=1,
            mismatch_count=1,
            status=DISTRIBUTED_CONVERGENCE_MISMATCH,
            stable_hash="0" * 64,
        )

    canonical_nodes = tuple(sorted((n1, n2), key=lambda e: (e.node_id, e.node_role, e.final_proof_hash, e.evidence_hash)))
    payload = {
        "version": "v150.9",
        "scenario_id": "bad2",
        "node_evidence": [e.to_dict() for e in canonical_nodes],
        "mismatches": [],
        "expected_final_proof_hash": None,
        "reference_final_proof_hash": a,
        "node_count": 2,
        "agreement_count": 2,
        "mismatch_count": 0,
        "status": VALIDATED,
    }
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DistributedConvergenceReceipt(
            version="v150.9",
            scenario_id="bad2",
            node_evidence=canonical_nodes,
            mismatches=tuple(),
            expected_final_proof_hash=None,
            reference_final_proof_hash=a,
            node_count=2,
            agreement_count=2,
            mismatch_count=0,
            status=VALIDATED,
            stable_hash=sha256_hex(payload),
        )

    payload["scenario_id"] = "bad3"
    payload["agreement_count"] = 1
    payload["mismatch_count"] = 1
    payload["status"] = "INVALID_STATUS"
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DistributedConvergenceReceipt(
            version="v150.9",
            scenario_id="bad3",
            node_evidence=canonical_nodes,
            mismatches=tuple(),
            expected_final_proof_hash=None,
            reference_final_proof_hash=a,
            node_count=2,
            agreement_count=1,
            mismatch_count=1,
            status="INVALID_STATUS",
            stable_hash=sha256_hex(payload),
        )
