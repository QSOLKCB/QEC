from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.decay_checkpoint_contract import (
    build_decay_checkpoint,
    build_decay_checkpoint_set,
)
from qec.analysis.digital_decay_signature import (
    build_decay_threshold_contract,
    build_digital_decay_signature,
)
from qec.analysis.entropy_drift_receipt import (
    EntropyDriftReceipt,
    build_entropy_drift_receipt,
    build_layer_decay_receipt,
    build_mask_collision_decay_receipt,
    build_readout_projection_decay_receipt,
    build_router_decay_receipt,
    build_shift_decay_receipt,
    validate_entropy_drift_receipt,
    validate_layer_decay_receipt,
    validate_mask_collision_decay_receipt,
    validate_readout_projection_decay_receipt,
    validate_router_decay_receipt,
    validate_shift_decay_receipt,
)

H = [c * 64 for c in "0123456789abcdef"]


def _sig(score: int):
    cps = build_decay_checkpoint_set(
        [
            build_decay_checkpoint(f"p{i}", H[0], H[1] if i < score else H[0])
            for i in range(8)
        ]
    )
    return build_digital_decay_signature(
        cps,
        build_decay_threshold_contract(2, 4),
        [f"p{i}" for i in range(score)],
    )


def _bundle():
    s1, s2, s3, s0, s4 = _sig(1), _sig(2), _sig(3), _sig(0), _sig(4)
    return (
        build_layer_decay_receipt(H[2], H[3], s1),
        build_router_decay_receipt(H[4], H[5], s2),
        build_mask_collision_decay_receipt(H[6], H[7], "NO_COLLISION", s3),
        build_shift_decay_receipt(H[8], H[9], s0),
        build_readout_projection_decay_receipt(H[10], H[11], s4),
        [s1, s2, s3, s0, s4],
    )


def test_subsystem_determinism_and_mask_collision_values():
    assert len({build_layer_decay_receipt(H[2], H[3], _sig(1)).layer_decay_receipt_hash for _ in range(10)}) == 1
    assert len({build_router_decay_receipt(H[4], H[5], _sig(2)).router_decay_receipt_hash for _ in range(10)}) == 1
    assert len({build_mask_collision_decay_receipt(H[6], H[7], "NO_COLLISION", _sig(3)).mask_collision_decay_receipt_hash for _ in range(10)}) == 1
    assert len({build_shift_decay_receipt(H[8], H[9], _sig(0)).shift_decay_receipt_hash for _ in range(10)}) == 1
    assert len({build_readout_projection_decay_receipt(H[10], H[11], _sig(4)).readout_projection_decay_receipt_hash for _ in range(10)}) == 1
    s = _sig(1)
    for c in ("NO_COLLISION", "KNOWN_EQUIVALENT_COLLISION", "INVALID_COLLISION"):
        assert build_mask_collision_decay_receipt(H[6], H[7], c, s).collision_type == c
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_mask_collision_decay_receipt(H[6], H[7], "BAD", s)


@pytest.mark.parametrize("builder,args", [
    (build_layer_decay_receipt, ("bad", H[3], _sig(1))),
    (build_layer_decay_receipt, (H[2], "bad", _sig(1))),
    (build_router_decay_receipt, ("bad", H[5], _sig(1))),
    (build_router_decay_receipt, (H[4], "bad", _sig(1))),
])
def test_subsystem_malformed_input_hashes_rejected(builder, args):
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        builder(*args)


def test_subsystem_stored_hash_format_and_tamper_rejected():
    s = _sig(1)
    l = build_layer_decay_receipt(H[2], H[3], s)
    object.__setattr__(l, "layer_decay_receipt_hash", "x")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_layer_decay_receipt(l)

    r = build_router_decay_receipt(H[4], H[5], s)
    object.__setattr__(r, "router_decay_receipt_hash", "a" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_router_decay_receipt(r)


def test_entropy_drift_receipt_sorts_subsystem_receipts_and_determinism():
    l, r, m, sh, ro, sigs = _bundle()
    e1 = build_entropy_drift_receipt([l], [r], [m], [sh], [ro], sigs)
    e2 = build_entropy_drift_receipt([l], [r], [m], [sh], [ro], sigs)
    assert e1.entropy_drift_receipt_hash == e2.entropy_drift_receipt_hash
    assert e1.to_canonical_json() == e2.to_canonical_json()
    assert e1.to_canonical_bytes() == e2.to_canonical_bytes()


def test_entropy_drift_receipt_all_subsystems_required_and_invalid_items():
    l, r, m, sh, ro, sigs = _bundle()
    for args in [([], [r], [m], [sh], [ro]), ([l], [], [m], [sh], [ro]), ([l], [r], [], [sh], [ro]), ([l], [r], [m], [], [ro]), ([l], [r], [m], [sh], [])]:
        with pytest.raises(ValueError, match="MISSING_SUBSYSTEM_DECAY_RECEIPTS"):
            build_entropy_drift_receipt(*args, sigs)
    for bad in ([1], [[]], ["not-a-receipt"]):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            build_entropy_drift_receipt(bad, [r], [m], [sh], [ro], sigs)


def test_deep_child_validation_before_parent_binding_and_validation():
    l, r, m, sh, ro, sigs = _bundle()
    for child, field in [
        (l, "layer_decay_receipt_hash"),
        (r, "router_decay_receipt_hash"),
        (m, "mask_collision_decay_receipt_hash"),
        (sh, "shift_decay_receipt_hash"),
        (ro, "readout_projection_decay_receipt_hash"),
    ]:
        tampered = child
        object.__setattr__(tampered, field, "a" * 64)
        with pytest.raises(ValueError, match="HASH_MISMATCH"):
            build_entropy_drift_receipt([l], [r], [m], [sh], [ro], sigs)
        object.__setattr__(tampered, field, "x")
        with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
            validate_entropy_drift_receipt(
                EntropyDriftReceipt((l,), (r,), (m,), (sh,), (ro,), 10, "a" * 64),
                sigs,
            )
        break


def test_aggregate_and_registry_rules():
    l, r, m, sh, ro, sigs = _bundle()
    e = build_entropy_drift_receipt([l], [r], [m], [sh], [ro], sigs)
    assert e.aggregate_decay_score == 10
    s = _sig(2)
    e2 = build_entropy_drift_receipt(
        [build_layer_decay_receipt(H[2], H[3], s)],
        [build_router_decay_receipt(H[4], H[5], s)],
        [build_mask_collision_decay_receipt(H[6], H[7], "NO_COLLISION", _sig(0))],
        [build_shift_decay_receipt(H[8], H[9], _sig(0))],
        [build_readout_projection_decay_receipt(H[10], H[11], _sig(0))],
        [s, _sig(0)],
    )
    assert e2.aggregate_decay_score == 4
    payload = {
        "layer_decay_receipts": [e.to_dict()["layer_decay_receipts"][0]],
        "router_decay_receipts": [e.to_dict()["router_decay_receipts"][0]],
        "mask_collision_decay_receipts": [e.to_dict()["mask_collision_decay_receipts"][0]],
        "shift_decay_receipts": [e.to_dict()["shift_decay_receipts"][0]],
        "readout_projection_decay_receipts": [e.to_dict()["readout_projection_decay_receipts"][0]],
        "aggregate_decay_score": 11,
    }
    import hashlib, json
    wrong_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
    bad = EntropyDriftReceipt((l,), (r,), (m,), (sh,), (ro,), 11, wrong_hash)
    with pytest.raises(ValueError, match="AGGREGATE_SCORE_MISMATCH"):
        validate_entropy_drift_receipt(bad, sigs)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_entropy_drift_receipt(e2, [_sig(0)])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_entropy_drift_receipt(e2, [_sig(0), _sig(0)])


def test_entropy_hash_and_direct_construction_rules():
    l, r, m, sh, ro, sigs = _bundle()
    e = build_entropy_drift_receipt([l], [r], [m], [sh], [ro], sigs)
    object.__setattr__(e, "entropy_drift_receipt_hash", "a" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_entropy_drift_receipt(e, sigs)
    object.__setattr__(e, "entropy_drift_receipt_hash", "x")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_entropy_drift_receipt(e, sigs)

    l2 = build_layer_decay_receipt(H[3], H[2], _sig(1))
    a, b = sorted([l, l2], key=lambda x: x.layer_decay_receipt_hash)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        EntropyDriftReceipt((b, a), (r,), (m,), (sh,), (ro,), 10, "a" * 64)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        EntropyDriftReceipt((a, a), (r,), (m,), (sh,), (ro,), 10, "a" * 64)


def test_validators_frozen_scope_and_typeerror_leaks():
    l, r, m, sh, ro, sigs = _bundle()
    assert validate_layer_decay_receipt(l) is True
    assert validate_router_decay_receipt(r) is True
    assert validate_mask_collision_decay_receipt(m) is True
    assert validate_shift_decay_receipt(sh) is True
    assert validate_readout_projection_decay_receipt(ro) is True
    e = build_entropy_drift_receipt([l], [r], [m], [sh], [ro], sigs)
    assert validate_entropy_drift_receipt(e, sigs) is True
    with pytest.raises(FrozenInstanceError):
        e.aggregate_decay_score = 0
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        EntropyDriftReceipt((l, 1), (r,), (m,), (sh,), (ro,), 10, "a" * 64)
    for missing in (
        "decay_resistance_proof_hash",
        "replay_proof_hash",
        "perturbation_contract_hash",
        "perturbation_stability_proof_hash",
        "substrate_state_receipt_hash",
        "probability",
        "entropy_rate",
    ):
        assert not hasattr(e, missing)
