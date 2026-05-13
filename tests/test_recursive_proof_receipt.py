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
    with pytest.raises(ValueError, match="ITERATION_INDEX_OUT_OF_BOUNDS"):
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


def test_hash_chain_continuity():
    """Test P1: Enforce hash-chain continuity across iteration records."""
    c = _contract()
    # Valid chain: a→b, b→c
    valid_recs = [_record(c, 0, "a", "b"), _record(c, 1, "b", "c", "ITERATION_MAX_DEPTH_REACHED")]
    p = build_recursive_proof_receipt(c, valid_recs)
    assert p.recursive_proof_receipt_hash
    
    # Invalid chain: a→b, c→d (broken continuity)
    broken_recs = [_record(c, 0, "a", "b"), _record(c, 1, "c", "d", "ITERATION_MAX_DEPTH_REACHED")]
    with pytest.raises(ValueError, match="HASH_CHAIN_CONTINUITY_BROKEN"):
        build_recursive_proof_receipt(c, broken_recs)
    
    # Test validation path as well - constructor will call validate
    with pytest.raises(ValueError, match="HASH_CHAIN_CONTINUITY_BROKEN"):
        RecursiveProofReceipt(
            loop_termination_contract_hash=c.loop_termination_contract_hash,
            source_artifact_type=c.source_artifact_type,
            source_artifact_hash=c.source_artifact_hash,
            loop_label=c.loop_label,
            loop_mode=c.loop_mode,
            max_depth=c.max_depth,
            input_receipt_hash_field=c.input_receipt_hash_field,
            output_receipt_hash_field=c.output_receipt_hash_field,
            iteration_records=tuple(broken_recs),
            iteration_count=len(broken_recs),
            terminal_iteration_index=broken_recs[-1].iteration_index,
            final_output_receipt_hash=broken_recs[-1].output_receipt_hash,
            recursive_proof_receipt_hash="0" * 64,
        )


def test_per_record_contract_hash_consistency():
    """Test P2: Verify per-record contract hash matches receipt header."""
    c1 = _contract()
    c2 = build_loop_termination_contract(
        source_artifact_type="SubstrateStateReceipt",
        source_artifact_hash=_hash("b"),
        loop_label="PROOF_LOOP",
        max_depth=4,
        input_receipt_hash_field="input_hash",
        output_receipt_hash_field="output_hash",
        termination_policy=TERMINATION_POLICY_MAX_DEPTH_ONLY,
        termination_parameters={},
    )
    
    # Create records with different contract hashes
    rec1 = _record(c1, 0, "a", "b")
    rec2 = _record(c2, 1, "b", "c", "ITERATION_MAX_DEPTH_REACHED")
    
    # Try to build receipt with mixed contract hashes
    with pytest.raises(ValueError, match="LOOP_CONTRACT_MISMATCH"):
        build_recursive_proof_receipt(c1, [rec1, rec2])
    
    # Test validation path: construct a receipt with mismatched contract hash
    valid_rec = _record(c1, 0, "a", "a", "ITERATION_FIXED_POINT")
    with pytest.raises(ValueError, match="LOOP_CONTRACT_MISMATCH"):
        RecursiveProofReceipt(
            loop_termination_contract_hash=c2.loop_termination_contract_hash,  # Different contract
            source_artifact_type=c1.source_artifact_type,
            source_artifact_hash=c1.source_artifact_hash,
            loop_label=c1.loop_label,
            loop_mode=c1.loop_mode,
            max_depth=c1.max_depth,
            input_receipt_hash_field=c1.input_receipt_hash_field,
            output_receipt_hash_field=c1.output_receipt_hash_field,
            iteration_records=(valid_rec,),
            iteration_count=1,
            terminal_iteration_index=0,
            final_output_receipt_hash=valid_rec.output_receipt_hash,
            recursive_proof_receipt_hash="0" * 64,
        )


def test_receipt_hash_field_distinctness():
    """Test that input and output receipt hash fields must be distinct."""
    c = _contract()
    rec = _record(c, 0, "a", "b", "ITERATION_FIXED_POINT")
    
    # Try to create a record with same input/output field names
    with pytest.raises(ValueError, match="RECEIPT_HASH_FIELD_MISMATCH"):
        replace(rec, input_receipt_hash_field="same_field", output_receipt_hash_field="same_field")
    
    # Test at receipt level
    valid_recs = [_record(c, 0, "a", "a", "ITERATION_FIXED_POINT")]
    p = build_recursive_proof_receipt(c, valid_recs)
    with pytest.raises(ValueError, match="RECEIPT_HASH_FIELD_MISMATCH"):
        replace(p, input_receipt_hash_field="same", output_receipt_hash_field="same")


def test_iteration_index_within_contract_max_depth():
    """Test that iteration_index must be within contract max_depth bounds."""
    c = _contract(max_depth=2)
    
    # Valid: indices 0, 1 are within max_depth=2
    assert build_loop_iteration_record(c, 0, _hash("a"), _hash("b"), "ITERATION_CONTINUED")
    assert build_loop_iteration_record(c, 1, _hash("b"), _hash("c"), "ITERATION_MAX_DEPTH_REACHED")
    
    # Invalid: index 2 >= max_depth=2
    with pytest.raises(ValueError, match="ITERATION_INDEX_OUT_OF_BOUNDS"):
        build_loop_iteration_record(c, 2, _hash("c"), _hash("d"), "ITERATION_MAX_DEPTH_REACHED")
    
    # Test in validate_loop_iteration_record_with_contract
    c2 = _contract(max_depth=5)
    rec = build_loop_iteration_record(c2, 3, _hash("a"), _hash("b"), "ITERATION_CONTINUED")
    
    # Valid with c2 (max_depth=5)
    assert validate_loop_iteration_record_with_contract(rec, c2)
    
    # Invalid with c (max_depth=2)
    with pytest.raises(ValueError, match="ITERATION_INDEX_OUT_OF_BOUNDS"):
        validate_loop_iteration_record_with_contract(rec, c)

