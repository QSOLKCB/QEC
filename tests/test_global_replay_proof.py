from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.global_replay_proof import (
    GlobalReplayProof,
    ReplayRecord,
    build_global_replay_proof,
    build_replay_record,
    get_global_replay_record_plan,
    validate_global_replay_proof,
    validate_global_replay_proof_with_artifacts,
    validate_replay_record,
)
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.global_truth_receipt import build_global_threshold_contract, build_global_truth_receipt
from qec.analysis.global_validation_index import build_global_validation_index, get_global_validation_entry_definitions


def _h(i: int) -> str:
    return f"{i:064x}"[-64:]


def _mapping() -> dict[str, str]:
    return {field: _h(i + 1) for i, _, field in get_global_validation_entry_definitions()}


def _artifacts():
    idx = build_global_validation_index(_mapping())
    contract = build_global_threshold_contract(idx)
    truth = build_global_truth_receipt(idx, contract)
    return idx, contract, truth


def test_replay_record_basics():
    plan = get_global_replay_record_plan()
    assert plan == ((0, "REPLAY_GLOBAL_THRESHOLD_CONTRACT", "GLOBAL_THRESHOLD_CONTRACT_REPLAY"), (1, "REPLAY_GLOBAL_TRUTH_RECEIPT", "GLOBAL_TRUTH_RECEIPT_REPLAY"))
    r1 = build_replay_record(0, "GLOBAL_THRESHOLD_CONTRACT_REPLAY", _h(1), _h(1))
    r2 = build_replay_record(0, "GLOBAL_THRESHOLD_CONTRACT_REPLAY", _h(1), _h(1))
    assert r1.replay_record_hash == r2.replay_record_hash
    assert r1.replay_status == "REPLAY_MATCHED"
    r3 = build_replay_record(1, "GLOBAL_TRUTH_RECEIPT_REPLAY", _h(1), _h(2))
    assert r3.replay_status == "REPLAY_MISMATCHED"
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_replay_record(0, "GLOBAL_THRESHOLD_CONTRACT_REPLAY", "abc", _h(1))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_replay_record(0, "GLOBAL_THRESHOLD_CONTRACT_REPLAY", _h(1), "ABC")
    with pytest.raises(ValueError, match="INVALID_REPLAY_INDEX"):
        build_replay_record(True, "GLOBAL_THRESHOLD_CONTRACT_REPLAY", _h(1), _h(1))
    with pytest.raises(ValueError, match="INVALID_REPLAY_INDEX"):
        build_replay_record(2, "GLOBAL_THRESHOLD_CONTRACT_REPLAY", _h(1), _h(1))
    with pytest.raises(ValueError, match="INVALID_REPLAY_KIND"):
        build_replay_record(0, "BAD", _h(1), _h(1))
    with pytest.raises(ValueError, match="REPLAY_RECORD_PLAN_MISMATCH"):
        build_replay_record(0, "GLOBAL_TRUTH_RECEIPT_REPLAY", _h(1), _h(1))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_replay_record(replace(r1, replay_record_hash="xyz"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_replay_record(replace(r1, replay_record_hash=_h(999)))
    with pytest.raises(ValueError, match="INVALID_REPLAY_LABEL"):
        validate_replay_record(replace(r1, replay_label="bad"))
    with pytest.raises(ValueError, match="REPLAY_STATUS_MISMATCH"):
        validate_replay_record(replace(r1, replay_status="REPLAY_MISMATCHED"))
    with pytest.raises(FrozenInstanceError):
        r1.replay_kind = "x"
    assert r1.to_canonical_json() == r1.to_canonical_json()
    assert r1.to_canonical_bytes() == r1.to_canonical_bytes()


def test_global_replay_proof_basics():
    idx, contract, truth = _artifacts()
    p1 = build_global_replay_proof(idx, contract, truth)
    p2 = build_global_replay_proof(idx, contract, truth)
    assert p1.global_replay_proof_hash == p2.global_replay_proof_hash
    assert p1.global_replay_class == "GLOBAL_REPLAY_CONFIRMED"
    assert p1.replay_record_count == 2 and p1.replay_match_count == 2 and p1.replay_mismatch_count == 0
    assert p1.all_replay_records_matched is True
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_global_replay_proof(replace(p1, global_replay_proof_hash="ABC"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_global_replay_proof(replace(p1, global_replay_proof_hash=_h(777)))
    with pytest.raises(ValueError, match="INVALID_REPLAY_MODE"):
        validate_global_replay_proof(replace(p1, replay_mode="BAD"))
    with pytest.raises(ValueError, match="INVALID_GLOBAL_REPLAY_CLASS"):
        validate_global_replay_proof(replace(p1, global_replay_class="BAD"))
    with pytest.raises(ValueError):
        validate_global_replay_proof(replace(p1, replay_record_count=True))
    with pytest.raises(ValueError):
        validate_global_replay_proof(replace(p1, replay_match_count=True))
    with pytest.raises(ValueError):
        validate_global_replay_proof(replace(p1, replay_mismatch_count=True))
    with pytest.raises(ValueError):
        validate_global_replay_proof(replace(p1, replay_match_count=1))
    with pytest.raises(ValueError):
        validate_global_replay_proof(replace(p1, all_replay_records_matched=False))
    with pytest.raises(ValueError):
        validate_global_replay_proof(replace(p1, global_replay_class="GLOBAL_REPLAY_DIVERGED"))
    with pytest.raises(ValueError):
        validate_global_replay_proof(replace(p1, replay_records=(replace(p1.replay_records[0], replay_status="REPLAY_MISMATCHED"), p1.replay_records[1])))
    dup = (p1.replay_records[0], p1.replay_records[0])
    with pytest.raises(ValueError, match="DUPLICATE_REPLAY_RECORD"):
        validate_global_replay_proof(replace(p1, replay_records=dup))
    unsorted = (p1.replay_records[1], p1.replay_records[0])
    with pytest.raises(ValueError, match="REPLAY_RECORD_ORDER_MISMATCH"):
        validate_global_replay_proof(replace(p1, replay_records=unsorted))
    with pytest.raises(ValueError, match="THRESHOLD_CONTRACT_ARTIFACT_MISMATCH"):
        r0_bad = build_replay_record(0, "GLOBAL_THRESHOLD_CONTRACT_REPLAY", _h(444), _h(444))
        bad_threshold = replace(p1, replay_records=(r0_bad, p1.replay_records[1]))
        bad_threshold = replace(bad_threshold, global_replay_proof_hash=sha256_hex(bad_threshold.to_dict()))
        validate_global_replay_proof(bad_threshold)
    with pytest.raises(ValueError, match="TRUTH_RECEIPT_ARTIFACT_MISMATCH"):
        r1_bad = build_replay_record(1, "GLOBAL_TRUTH_RECEIPT_REPLAY", _h(555), _h(555))
        bad_truth = replace(p1, replay_records=(p1.replay_records[0], r1_bad))
        bad_truth = replace(bad_truth, global_replay_proof_hash=sha256_hex(bad_truth.to_dict()))
        validate_global_replay_proof(bad_truth)
    with pytest.raises(FrozenInstanceError):
        p1.replay_mode = "x"
    assert p1.to_canonical_json() == p1.to_canonical_json()
    assert p1.to_canonical_bytes() == p1.to_canonical_bytes()


def test_divergence_and_complete_validator():
    idx, contract, truth = _artifacts()
    p = build_global_replay_proof(idx, contract, truth)
    r_mis = build_replay_record(1, "GLOBAL_TRUTH_RECEIPT_REPLAY", p.global_truth_receipt_hash, _h(2))
    assert validate_replay_record(r_mis) is True and r_mis.replay_status == "REPLAY_MISMATCHED"
    payload = p.to_dict()
    payload["replay_records"][1] = r_mis.to_dict()
    payload["replay_match_count"] = 1
    payload["replay_mismatch_count"] = 1
    payload["all_replay_records_matched"] = False
    payload["global_replay_class"] = "GLOBAL_REPLAY_DIVERGED"
    records = (p.replay_records[0], r_mis)
    d_payload = {
        "global_validation_index_hash": p.global_validation_index_hash,
        "global_threshold_contract_hash": p.global_threshold_contract_hash,
        "global_truth_receipt_hash": p.global_truth_receipt_hash,
        "replay_mode": p.replay_mode,
        "replay_records": records,
        "replay_record_count": 2,
        "replay_match_count": 1,
        "replay_mismatch_count": 1,
        "all_replay_records_matched": False,
        "global_replay_class": "GLOBAL_REPLAY_DIVERGED",
    }
    d_hash_payload = dict(d_payload, replay_records=[records[0].to_dict(), records[1].to_dict()])
    diverged = GlobalReplayProof(**d_payload, global_replay_proof_hash=sha256_hex(d_hash_payload))
    assert validate_global_replay_proof(diverged) is True
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_global_replay_proof_with_artifacts(diverged, idx, contract, truth)
    assert validate_global_replay_proof_with_artifacts(p, idx, contract, truth) is True
    idx2 = build_global_validation_index(dict(_mapping(), canonical_hash=_h(999)))
    c2 = build_global_threshold_contract(idx2)
    t2 = build_global_truth_receipt(idx2, c2)
    with pytest.raises(ValueError):
        validate_global_replay_proof_with_artifacts(p, idx2, contract, truth)
    with pytest.raises(ValueError):
        validate_global_replay_proof_with_artifacts(p, idx, c2, truth)
    with pytest.raises(ValueError):
        validate_global_replay_proof_with_artifacts(p, idx, contract, t2)


def test_boundaries_and_scope_scan():
    idx, contract, truth = _artifacts()
    p = build_global_replay_proof(idx, contract, truth)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_replay_record(object())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_global_replay_proof(object())
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_replay_record(replace(p.replay_records[0], expected_artifact_hash="A" * 64))

    with open("src/qec/analysis/global_replay_proof.py", "r", encoding="utf-8") as fh:
        text = fh.read()
    forbidden = ["semantic_truth", "philosophical_truth", "metaphysical_truth", "omniscience", "runtime_replay_execution", "gameplay", "render", "step_world", "execute_action", "run_game", "importlib", "__import__(", "subprocess", "exec(", "eval(", "random", "time.time", "datetime.now", "probability", "probabilistic", "neural", "learned_policy"]
    for token in forbidden:
        assert token not in text
