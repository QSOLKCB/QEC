from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.loop_termination_contract import TERMINATION_POLICY_MAX_DEPTH_ONLY, build_loop_termination_contract
from qec.analysis.recursive_proof_receipt import (
    LoopIterationRecord,
    RecursiveProofReceipt,
    build_loop_iteration_record,
    build_recursive_proof_receipt,
    get_allowed_loop_iteration_statuses,
    validate_loop_iteration_record,
    validate_loop_iteration_record_with_contract,
    validate_recursive_proof_receipt,
    validate_recursive_proof_receipt_with_contract,
)


def _hash(ch: str = "a") -> str:
    return ch * 64


def _contract(max_depth: int = 4):
    return build_loop_termination_contract(
        source_artifact_type="SubstrateStateReceipt",
        source_artifact_hash=_hash("a"),
        loop_label="PROOF_LOOP",
        max_depth=max_depth,
        input_receipt_hash_field="input_hash",
        output_receipt_hash_field="output_hash",
        termination_policy=TERMINATION_POLICY_MAX_DEPTH_ONLY,
        termination_parameters={},
    )


def _record(contract, i, inp="a", out="b", status="ITERATION_CONTINUED"):
    return build_loop_iteration_record(contract, i, _hash(inp), _hash(out), status)


def test_loop_iteration_record_basics():
    c = _contract()
    r1 = _record(c, 0, "a", "b")
    r2 = _record(c, 0, "a", "b")
    assert r1.loop_iteration_record_hash == r2.loop_iteration_record_hash
    assert r1.changed is True
    assert _record(c, 1, "a", "a").changed is False
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        _record(c, 0, inp="x")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        _record(c, 0, out="x")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_loop_iteration_record(c, 0, "A" * 64, _hash("b"), "ITERATION_CONTINUED")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_loop_iteration_record(replace(r1, loop_iteration_record_hash="BAD"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_loop_iteration_record(replace(r1, loop_iteration_record_hash=_hash("c")))
    with pytest.raises(ValueError, match="ITERATION_INDEX_OUT_OF_BOUNDS"):
        _record(c, True)
    with pytest.raises(ValueError, match="ITERATION_INDEX_OUT_OF_BOUNDS"):
        _record(c, 10_000)
    with pytest.raises(ValueError, match="INVALID_ITERATION_STATUS"):
        _record(c, 0, status="BAD")
    with pytest.raises(ValueError, match="CHANGED_FLAG_MISMATCH"):
        validate_loop_iteration_record(replace(r1, changed=False))
    with pytest.raises(FrozenInstanceError):
        r1.iteration_index = 1  # type: ignore[misc]
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.to_canonical_bytes() == r2.to_canonical_bytes()


def test_recursive_proof_receipt_basics(monkeypatch):
    c = _contract()
    recs = [_record(c, 0, "a", "b"), _record(c, 1, "b", "b", "ITERATION_MAX_DEPTH_REACHED")]
    p1 = build_recursive_proof_receipt(c, recs)
    p2 = build_recursive_proof_receipt(c, list(reversed(recs)))
    assert p1.recursive_proof_receipt_hash == p2.recursive_proof_receipt_hash
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        RecursiveProofReceipt(**{**p1.to_dict(), "iteration_records": [recs[1], recs[0]]})
    with pytest.raises(ValueError, match="DUPLICATE_ITERATION"):
        build_recursive_proof_receipt(c, [recs[0], recs[0]])
    with pytest.raises(ValueError, match="ITERATION_ORDER_MISMATCH"):
        build_recursive_proof_receipt(c, [_record(c, 0), _record(c, 2, status="ITERATION_MAX_DEPTH_REACHED")])
    with pytest.raises(ValueError, match="ITERATION_COUNT_MISMATCH"):
        build_recursive_proof_receipt(c, [])
    monkeypatch.setattr("qec.analysis.recursive_proof_receipt._MAX_ITERATION_RECORDS", 1)
    with pytest.raises(ValueError, match="ITERATION_COUNT_MISMATCH"):
        build_recursive_proof_receipt(c, recs)


def test_receipt_validator_edges():
    c = _contract()
    recs = (_record(c, 0, "a", "b"), _record(c, 1, "b", "b", "ITERATION_MAX_DEPTH_REACHED"))
    p = build_recursive_proof_receipt(c, recs)
    with pytest.raises(ValueError, match="ITERATION_COUNT_MISMATCH"):
        validate_recursive_proof_receipt(replace(p, iteration_count=9))
    with pytest.raises(ValueError, match="ITERATION_COUNT_MISMATCH"):
        validate_recursive_proof_receipt(replace(p, iteration_count=True))
    with pytest.raises(ValueError, match="ITERATION_COUNT_MISMATCH"):
        validate_recursive_proof_receipt(replace(p, terminal_iteration_index=True))
    with pytest.raises(ValueError, match="TERMINAL_ITERATION_MISMATCH"):
        validate_recursive_proof_receipt(replace(p, terminal_iteration_index=0))
    with pytest.raises(ValueError, match="FINAL_OUTPUT_HASH_MISMATCH"):
        validate_recursive_proof_receipt(replace(p, final_output_receipt_hash=_hash("c")))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_recursive_proof_receipt(replace(p, recursive_proof_receipt_hash="BAD"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_recursive_proof_receipt(replace(p, recursive_proof_receipt_hash=_hash("d")))
    with pytest.raises(ValueError, match="CHANGED_FLAG_MISMATCH"):
        LoopIterationRecord(**{**recs[0].to_dict(), "changed": False, "loop_iteration_record_hash": "c" * 64})
    with pytest.raises(FrozenInstanceError):
        p.iteration_count = 0  # type: ignore[misc]
    assert p.to_canonical_json() == build_recursive_proof_receipt(c, recs).to_canonical_json()
    assert p.to_canonical_bytes() == build_recursive_proof_receipt(c, recs).to_canonical_bytes()


def test_terminal_rules_and_contract_validators():
    c = _contract(max_depth=2)
    ok = build_recursive_proof_receipt(c, [_record(c, 0, "a", "b"), _record(c, 1, "b", "b", "ITERATION_MAX_DEPTH_REACHED")])
    assert ok
    with pytest.raises(ValueError, match="TERMINAL_ITERATION_MISMATCH"):
        build_recursive_proof_receipt(c, [_record(c, 0, "a", "b"), _record(c, 1, "b", "c", "ITERATION_CONTINUED")])
    with pytest.raises(ValueError, match="TERMINAL_ITERATION_MISMATCH"):
        build_recursive_proof_receipt(c, [_record(c, 0, "a", "b", "ITERATION_FIXED_POINT"), _record(c, 1, "b", "c", "ITERATION_MAX_DEPTH_REACHED")])
    with pytest.raises(ValueError, match="TERMINAL_ITERATION_MISMATCH"):
        build_recursive_proof_receipt(c, [_record(c, 0, "a", "b", "ITERATION_TARGET_REACHED"), _record(c, 1, "b", "c", "ITERATION_FIXED_POINT")])
    c2 = _contract(max_depth=3)
    assert build_recursive_proof_receipt(c2, [_record(c2, 0, "a", "b"), _record(c2, 1, "b", "b", "ITERATION_MAX_DEPTH_REACHED")])
    with pytest.raises(ValueError, match="MAX_DEPTH_EXCEEDED"):
        build_recursive_proof_receipt(c, [_record(c, 0), _record(c, 1), _record(c, 2, status="ITERATION_MAX_DEPTH_REACHED")])

    rec = _record(c2, 0, status="ITERATION_FIXED_POINT", out="a")
    assert validate_loop_iteration_record_with_contract(rec, c2)
    rcp = build_recursive_proof_receipt(c2, [rec])
    assert validate_recursive_proof_receipt_with_contract(rcp, c2)
    c3 = build_loop_termination_contract(
        source_artifact_type="SubstrateStateReceipt",
        source_artifact_hash=_hash("a"),
        loop_label="OTHER",
        max_depth=3,
        input_receipt_hash_field="input_hash",
        output_receipt_hash_field="output_hash",
        termination_policy=TERMINATION_POLICY_MAX_DEPTH_ONLY,
        termination_parameters={},
    )
    with pytest.raises(ValueError, match="LOOP_CONTRACT_MISMATCH"):
        validate_loop_iteration_record_with_contract(build_loop_iteration_record(c3, 0, _hash("a"), _hash("a"), "ITERATION_FIXED_POINT"), c2)
    c4 = build_loop_termination_contract(
        source_artifact_type="SubstrateStateReceipt",
        source_artifact_hash=_hash("a"),
        loop_label="PROOF_LOOP",
        max_depth=3,
        input_receipt_hash_field="in_hash",
        output_receipt_hash_field="out_hash",
        termination_policy=TERMINATION_POLICY_MAX_DEPTH_ONLY,
        termination_parameters={},
    )
    with pytest.raises(ValueError, match="LOOP_CONTRACT_MISMATCH"):
        validate_loop_iteration_record_with_contract(build_loop_iteration_record(c4, 0, _hash("a"), _hash("a"), "ITERATION_FIXED_POINT"), c2)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_recursive_proof_receipt_with_contract(replace(rcp, source_artifact_hash=_hash("f")), c2)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_recursive_proof_receipt_with_contract(replace(rcp, loop_label="OTHER"), c2)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_recursive_proof_receipt_with_contract(replace(rcp, max_depth=7), c2)


def test_boundary_and_scope_scan():
    assert "ITERATION_CONTINUED" in get_allowed_loop_iteration_statuses()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_loop_iteration_record(object())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_recursive_proof_receipt(object())
    with pytest.raises(TypeError):
        LoopIterationRecord(**{"bad": 1})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RecursiveProofReceipt(**{"bad": 1})  # type: ignore[arg-type]

    blocked = [
        "LoopTerminationProof", "OuroboricConvergenceReceipt", "RouterLoopReceipt", "ReadoutLoopReceipt",
        "MarkovLoopStabilityReceipt", "RealityLoopProofReceipt", "GlobalTruthReceipt", "while True",
        "recursive_execution", "gameplay", "render", "step_world", "execute_action", "run_game",
        "importlib", "__import__(", "subprocess", "exec(", "eval(", "random", "time.time",
        "datetime.now", "probability", "probabilistic", "neural", "learned_policy", "global_truth",
    ]
    with open("src/qec/analysis/recursive_proof_receipt.py", encoding="utf-8") as f:
        text = f.read()
    for token in blocked:
        assert token not in text
