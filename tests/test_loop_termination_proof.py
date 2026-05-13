from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.loop_termination_contract import (
    TERMINATION_POLICY_BOUNDED_DIVERGENCE_COUNT,
    TERMINATION_POLICY_FIXED_POINT_HASH,
    TERMINATION_POLICY_MAX_DEPTH_ONLY,
    TERMINATION_POLICY_STATUS_FIELD_MATCH,
    TERMINATION_POLICY_TARGET_HASH_REACHED,
    build_loop_termination_contract,
)
from qec.analysis.loop_termination_proof import (
    build_loop_termination_proof,
    build_ouroboric_convergence_receipt,
    validate_loop_termination_proof,
    validate_loop_termination_proof_with_receipts,
    validate_ouroboric_convergence_receipt,
    validate_ouroboric_convergence_receipt_with_receipts,
)
from qec.analysis.recursive_proof_receipt import build_loop_iteration_record, build_recursive_proof_receipt


def _h(v: str) -> str:
    return sha256_hex({"v": v})


def _contract(policy: str, max_depth: int = 3, params: dict | None = None):
    return build_loop_termination_contract(
        source_artifact_type="SubstrateStateReceipt",
        source_artifact_hash=_h("src"),
        loop_label="LOOP_A",
        max_depth=max_depth,
        input_receipt_hash_field="in_hash",
        output_receipt_hash_field="out_hash",
        termination_policy=policy,
        termination_parameters=params,
    )


def _receipt(c, seq):
    recs = [build_loop_iteration_record(c, i, _h(a), _h(b), s) for i, (a, b, s) in enumerate(seq)]
    return build_recursive_proof_receipt(c, recs)


def test_basics_and_hash_validation():
    c = _contract(TERMINATION_POLICY_MAX_DEPTH_ONLY, max_depth=2)
    r = _receipt(c, [("a", "b", "ITERATION_CONTINUED"), ("b", "c", "ITERATION_MAX_DEPTH_REACHED")])
    p1 = build_loop_termination_proof(c, r)
    p2 = build_loop_termination_proof(c, r)
    assert p1.loop_termination_proof_hash == p2.loop_termination_proof_hash
    assert p1.to_canonical_json() == p2.to_canonical_json()
    assert p1.to_canonical_bytes() == p2.to_canonical_bytes()
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_loop_termination_proof(replace(p1, loop_termination_proof_hash="A" * 64))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_loop_termination_proof(replace(p1, loop_termination_proof_hash="0" * 64))
    with pytest.raises(FrozenInstanceError):
        p1.iteration_count = 8


def test_termination_policies_and_complete_validators():
    c = _contract(TERMINATION_POLICY_FIXED_POINT_HASH, max_depth=2, params={"stable_hash_field": "out_hash"})
    r = _receipt(c, [("a", "a", "ITERATION_FIXED_POINT")])
    p = build_loop_termination_proof(c, r)
    assert p.termination_satisfied is True and p.termination_class == "LOOP_TERMINATED_BY_FIXED_POINT"
    assert validate_loop_termination_proof_with_receipts(p, c, r)

    c2 = _contract(TERMINATION_POLICY_TARGET_HASH_REACHED, params={"target_hash": _h("t")})
    r2 = _receipt(c2, [("a", "t", "ITERATION_TARGET_REACHED")])
    assert build_loop_termination_proof(c2, r2).termination_class == "LOOP_TERMINATED_BY_TARGET_HASH"

    c3 = _contract(TERMINATION_POLICY_STATUS_FIELD_MATCH, params={"status_field": "status", "terminal_statuses": ["DONE"]})
    r3 = _receipt(c3, [("a", "b", "ITERATION_STATUS_TERMINAL")])
    assert build_loop_termination_proof(c3, r3).termination_class == "LOOP_TERMINATED_BY_STATUS"

    c4 = _contract(TERMINATION_POLICY_BOUNDED_DIVERGENCE_COUNT, params={"max_divergence_count": 1})
    r4 = _receipt(c4, [("a", "b", "ITERATION_CONTINUED"), ("b", "b", "ITERATION_DIVERGENCE_LIMIT")])
    assert build_loop_termination_proof(c4, r4).termination_class == "LOOP_TERMINATED_BY_DIVERGENCE_BOUND"


def test_ouroboric_semantics_and_validation():
    c = _contract(TERMINATION_POLICY_MAX_DEPTH_ONLY, max_depth=2)
    r = _receipt(c, [("a", "b", "ITERATION_CONTINUED"), ("b", "a", "ITERATION_MAX_DEPTH_REACHED")])
    p = build_loop_termination_proof(c, r)
    o = build_ouroboric_convergence_receipt(c, r, p)
    assert o.convergence_class == "OUROBORIC_CLOSED_LOOP" and o.convergence_stable
    assert validate_ouroboric_convergence_receipt_with_receipts(o, c, r, p)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_ouroboric_convergence_receipt(replace(o, ouroboric_convergence_receipt_hash="A" * 64))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_ouroboric_convergence_receipt(replace(o, ouroboric_convergence_receipt_hash="0" * 64))


def test_boundaries_and_scope_scan():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_loop_termination_proof(object())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_ouroboric_convergence_receipt(object())
    prod = open("src/qec/analysis/loop_termination_proof.py", encoding="utf-8").read()
    for bad in [
        "RouterLoopReceipt", "ReadoutLoopReceipt", "MarkovLoopStabilityReceipt", "RealityLoopProofReceipt", "GlobalTruthReceipt",
        "while True", "recursive_execution", "gameplay", "render", "step_world", "execute_action", "run_game", "importlib",
        "__import__(", "subprocess", "exec(", "eval(", "random", "time.time", "datetime.now", "probability", "probabilistic",
        "neural", "learned_policy", "global_truth",
    ]:
        assert bad not in prod
