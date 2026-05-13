from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.global_truth_receipt import (
    GlobalThresholdContract,
    GlobalTruthReceipt,
    build_global_threshold_contract,
    build_global_truth_receipt,
    get_allowed_global_truth_classes,
    validate_global_threshold_contract,
    validate_global_threshold_contract_with_index,
    validate_global_truth_receipt,
    validate_global_truth_receipt_with_artifacts,
)
from qec.analysis.global_validation_index import build_global_validation_index, get_global_validation_entry_definitions


def _h(i: int) -> str:
    return f"{i:064x}"[-64:]


def _mapping() -> dict[str, str]:
    return {field: _h(i + 1) for i, _, field in get_global_validation_entry_definitions()}


def test_global_threshold_contract_basics():
    idx = build_global_validation_index(_mapping())
    c1 = build_global_threshold_contract(idx)
    c2 = build_global_threshold_contract(idx)
    assert c1.global_threshold_contract_hash == c2.global_threshold_contract_hash
    assert '"require_all_entries":true' in c1.canonical_threshold_parameters
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_global_threshold_contract(replace(c1, global_validation_index_hash="ABC"))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_global_threshold_contract(replace(c1, required_final_receipt_hash="ABC"))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_global_threshold_contract(replace(c1, threshold_parameters_hash="ABC"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_global_threshold_contract(replace(c1, threshold_parameters_hash=_h(2)))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_global_threshold_contract(replace(c1, global_threshold_contract_hash="ABC"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_global_threshold_contract(replace(c1, global_threshold_contract_hash=_h(3)))
    with pytest.raises(ValueError):
        validate_global_threshold_contract(replace(c1, threshold_mode="BAD"))
    with pytest.raises(ValueError):
        build_global_threshold_contract(idx, minimum_required_entries=True)
    with pytest.raises(ValueError):
        build_global_threshold_contract(idx, minimum_required_entries=0)
    with pytest.raises(ValueError):
        build_global_threshold_contract(idx, minimum_required_entries=49)
    with pytest.raises(ValueError):
        validate_global_threshold_contract(replace(c1, required_entry_count=47))
    with pytest.raises(ValueError):
        validate_global_threshold_contract(replace(c1, required_final_receipt_field_name="Bad"))
    with pytest.raises(ValueError):
        validate_global_threshold_contract(replace(c1, canonical_threshold_parameters='{"require_all_entries": true}'))
    with pytest.raises(ValueError):
        validate_global_threshold_contract(replace(c1, canonical_threshold_parameters='"' + ("x" * 9000) + '"'))
    for bad in ((1, 2), 1.25, b"x", {1, 2}, object()):
        with pytest.raises(ValueError):
            build_global_threshold_contract(idx, threshold_parameters=bad)
    with pytest.raises(FrozenInstanceError):
        c1.required_entry_count = 1
    assert c1.to_canonical_json() == c1.to_canonical_json()
    assert c1.to_canonical_bytes() == c1.to_canonical_bytes()


def test_threshold_parameter_semantics():
    idx = build_global_validation_index(_mapping())
    good = {"require_all_entries": True, "require_final_anchor_match": True, "require_fixed_index_mode": True}
    assert validate_global_threshold_contract(build_global_threshold_contract(idx, threshold_parameters=good))
    for key in good:
        bad = dict(good)
        bad[key] = 1
        with pytest.raises(ValueError):
            build_global_threshold_contract(idx, threshold_parameters=bad)
    with pytest.raises(ValueError):
        build_global_threshold_contract(idx, threshold_parameters={"require_all_entries": True})
    with pytest.raises(ValueError):
        build_global_threshold_contract(idx, threshold_parameters=dict(good, extra=True))
    loose = build_global_threshold_contract(
        idx,
        threshold_parameters={"require_all_entries": False, "require_final_anchor_match": False, "require_fixed_index_mode": False},
    )
    r = build_global_truth_receipt(idx, loose)
    assert r.threshold_contract_satisfied is True
    with pytest.raises(ValueError, match="THRESHOLD_CONTRACT_MISMATCH"):
        validate_global_threshold_contract_with_index(loose, idx, threshold_parameters=good)


def test_global_truth_receipt_basics_and_derivation():
    idx = build_global_validation_index(_mapping())
    c = build_global_threshold_contract(idx)
    r1 = build_global_truth_receipt(idx, c)
    r2 = build_global_truth_receipt(idx, c)
    assert r1.global_truth_receipt_hash == r2.global_truth_receipt_hash
    assert r1.global_truth_class == "GLOBAL_TRUTH_REGISTERED"
    assert r1.entry_count_threshold_satisfied is True
    assert r1.final_anchor_present is True
    assert r1.final_anchor_hash_matches is True
    assert r1.threshold_contract_satisfied is True
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_global_truth_receipt(replace(r1, global_truth_receipt_hash="ABC"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_global_truth_receipt(replace(r1, global_truth_receipt_hash=_h(5)))
    with pytest.raises(ValueError):
        validate_global_truth_receipt(replace(r1, truth_mode="BAD"))
    with pytest.raises(ValueError):
        validate_global_truth_receipt(replace(r1, global_truth_class="BAD"))
    with pytest.raises(ValueError):
        validate_global_truth_receipt(replace(r1, registered_entry_count=True))
    with pytest.raises(ValueError):
        validate_global_truth_receipt(replace(r1, global_truth_class="GLOBAL_TRUTH_INVALID"))
    with pytest.raises(ValueError):
        validate_global_truth_receipt(replace(r1, threshold_contract_satisfied=False))
    with pytest.raises(ValueError):
        validate_global_truth_receipt(replace(r1, final_anchor_hash="f" * 64, final_anchor_hash_matches=False))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_global_truth_receipt(replace(r1, final_anchor_present=False, final_anchor_hash_matches=True))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_global_truth_receipt(replace(r1, registered_entry_count=0, entry_count_threshold_satisfied=False, threshold_contract_satisfied=True))
    with pytest.raises(FrozenInstanceError):
        r1.global_truth_class = "x"
    assert r1.to_canonical_json() == r1.to_canonical_json()
    assert r1.to_canonical_bytes() == r1.to_canonical_bytes()

    with pytest.raises(ValueError, match="TRUTH_CLASS_MISMATCH"):
        validate_global_truth_receipt(replace(r1, entry_count_threshold_satisfied=False, registered_entry_count=0, threshold_contract_satisfied=False, global_truth_class="GLOBAL_TRUTH_INVALID", global_truth_receipt_hash=r1.global_truth_receipt_hash))
    with pytest.raises(ValueError, match="TRUTH_CLASS_MISMATCH"):
        validate_global_truth_receipt(replace(r1, final_anchor_hash_matches=False, threshold_contract_satisfied=False, global_truth_class="GLOBAL_TRUTH_REGISTERED", global_truth_receipt_hash=r1.global_truth_receipt_hash))
    assert "GLOBAL_TRUTH_INVALID" in get_allowed_global_truth_classes()


def test_complete_validators_and_boundaries_and_scope_scan():
    idx = build_global_validation_index(_mapping())
    c = build_global_threshold_contract(idx)
    r = build_global_truth_receipt(idx, c)
    assert validate_global_threshold_contract_with_index(c, idx) is True
    assert validate_global_truth_receipt_with_artifacts(r, idx, c) is True
    idx2 = build_global_validation_index(dict(_mapping(), canonical_hash=_h(999)))
    with pytest.raises(ValueError, match="THRESHOLD_CONTRACT_MISMATCH"):
        validate_global_threshold_contract_with_index(c, idx2)
    c2 = build_global_threshold_contract(idx2)
    with pytest.raises(ValueError, match="GLOBAL_TRUTH_RECEIPT_MISMATCH"):
        validate_global_truth_receipt_with_artifacts(r, idx, c2)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_global_threshold_contract(object())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_global_truth_receipt(object())

    with open("src/qec/analysis/global_truth_receipt.py", "r", encoding="utf-8") as fh:
        text = fh.read()
    forbidden = ["GlobalReplayProof", "replay_record_hash", "global_replay_proof_hash", "runtime_replay_execution", "semantic_truth", "philosophical_truth", "metaphysical_truth", "omniscience", "gameplay", "render", "step_world", "execute_action", "run_game", "importlib", "__import__(", "subprocess", "exec(", "eval(", "random", "time.time", "datetime.now", "probability", "probabilistic", "neural", "learned_policy"]
    for token in forbidden:
        assert token not in text
