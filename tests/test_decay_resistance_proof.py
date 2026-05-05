from dataclasses import FrozenInstanceError, dataclass
import subprocess
import sys

import pytest

from qec.analysis.decay_checkpoint_contract import (
    build_decay_checkpoint,
    build_decay_checkpoint_set,
)
from qec.analysis.decay_resistance_proof import (
    DecayResistanceProof,
    build_decay_resistance_proof,
    validate_decay_resistance_proof,
)
from qec.analysis.digital_decay_signature import (
    build_decay_threshold_contract,
    build_digital_decay_signature,
)
from qec.analysis.entropy_drift_receipt import (
    build_entropy_drift_receipt,
    build_layer_decay_receipt,
    build_mask_collision_decay_receipt,
    build_readout_projection_decay_receipt,
    build_router_decay_receipt,
    build_shift_decay_receipt,
)

H = [c * 64 for c in "0123456789abcdef"]


def _different_hash(original: str) -> str:
    candidate = "0" * 64
    if candidate != original:
        return candidate
    return "1" * 64


def _sig(score: int):
    cps = build_decay_checkpoint_set(
        [build_decay_checkpoint(f"p{i}", H[0], H[1] if i < score else H[0]) for i in range(8)]
    )
    return build_digital_decay_signature(
        cps,
        build_decay_threshold_contract(2, 4),
        [f"p{i}" for i in range(score)],
    )


def _clean_bundle():
    s = _sig(0)
    e = build_entropy_drift_receipt(
        [build_layer_decay_receipt(H[2], H[3], s)],
        [build_router_decay_receipt(H[4], H[5], s)],
        [build_mask_collision_decay_receipt(H[6], H[7], "NO_COLLISION", s)],
        [build_shift_decay_receipt(H[8], H[9], s)],
        [build_readout_projection_decay_receipt(H[10], H[11], s)],
        [s],
    )
    return e, [s]


def _other_clean_sig():
    cps = build_decay_checkpoint_set(
        [build_decay_checkpoint(f"q{i}", H[2], H[2]) for i in range(8)]
    )
    return build_digital_decay_signature(
        cps,
        build_decay_threshold_contract(2, 4),
        [],
    )


@dataclass(frozen=True)
class ReplayProofStub:
    lattice_replay_proof_hash: str


def test_decay_resistance_proof_determinism():
    e, sigs = _clean_bundle()
    ps = [build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12])) for _ in range(10)]
    assert len({p.decay_resistance_proof_hash for p in ps}) == 1
    assert len({p.to_canonical_json() for p in ps}) == 1
    assert len({p.to_canonical_bytes() for p in ps}) == 1


def test_validate_decay_resistance_proof_returns_true_for_valid_proof():
    e, sigs = _clean_bundle()
    assert validate_decay_resistance_proof(build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))) is True


def test_invalid_entropy_receipt_type_rejected():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_decay_resistance_proof({}, [], ReplayProofStub(H[12]))


def test_decay_resistance_proof_requires_clean():
    s = _sig(1)
    e = build_entropy_drift_receipt(
        [build_layer_decay_receipt(H[2], H[3], s)],
        [build_router_decay_receipt(H[4], H[5], s)],
        [build_mask_collision_decay_receipt(H[6], H[7], "NO_COLLISION", s)],
        [build_shift_decay_receipt(H[8], H[9], s)],
        [build_readout_projection_decay_receipt(H[10], H[11], s)],
        [s],
    )
    with pytest.raises(ValueError, match="ADVERSARIAL_POSITIONS_PRESENT"):
        build_decay_resistance_proof(e, [s], ReplayProofStub(H[12]))


def test_referenced_signature_hashes_cover_all_subsystems():
    e, _ = _clean_bundle()
    hashes = (
        e.layer_decay_receipts[0].decay_signature_hash,
        e.router_decay_receipts[0].decay_signature_hash,
        e.mask_collision_decay_receipts[0].decay_signature_hash,
        e.shift_decay_receipts[0].decay_signature_hash,
        e.readout_projection_decay_receipts[0].decay_signature_hash,
    )
    assert len(hashes) == 5
    assert len(set(hashes)) == 1


def test_referenced_signature_with_adversarial_positions_rejected():
    e, _ = _clean_bundle()
    bad = _sig(1)
    # Valid non-clean v155.1 signatures may carry adversarial_positions;
    # v155.3 intentionally rejects those via ADVERSARIAL_POSITIONS_PRESENT first.
    with pytest.raises(ValueError, match="ADVERSARIAL_POSITIONS_PRESENT"):
        build_decay_resistance_proof(e, [bad], ReplayProofStub(H[12]))


def test_entropy_receipt_exposed_adversarial_state_rejected():
    # Simulates deserialized external objects carrying an exposed adversarial field;
    # v155.2 EntropyDriftReceipt does not define this field.
    e, sigs = _clean_bundle()
    object.__setattr__(e, "adversarial_positions", ("p0",))
    with pytest.raises(ValueError, match="ADVERSARIAL_POSITIONS_PRESENT"):
        build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))


def test_direct_non_clean_decay_class_rejected():
    with pytest.raises(ValueError, match="DECAY_RESISTANCE_IMPOSSIBLE"):
        DecayResistanceProof(H[0], H[1], "DEGRADED", (), H[2])


def test_adversarial_positions_present_rejected():
    with pytest.raises(ValueError, match="ADVERSARIAL_POSITIONS_PRESENT"):
        DecayResistanceProof(H[0], H[1], "CLEAN", ("p0",), H[2])


@pytest.mark.parametrize("bad", [["p0"], "p0", None])
def test_adversarial_positions_non_tuple_rejected(bad):
    e, sigs = _clean_bundle()
    p = build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))
    object.__setattr__(p, "adversarial_positions_at_proof", bad)
    with pytest.raises(ValueError, match="INVALID_INPUT|ADVERSARIAL_POSITIONS_PRESENT"):
        validate_decay_resistance_proof(p)


def test_malformed_entropy_hash_rejected():
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        DecayResistanceProof("x", H[1], "CLEAN", (), H[2])


def test_malformed_replay_hash_rejected():
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        DecayResistanceProof(H[0], "x", "CLEAN", (), H[2])


def test_malformed_decay_resistance_hash_rejected():
    e, sigs = _clean_bundle()
    p = build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))
    object.__setattr__(p, "decay_resistance_proof_hash", "x")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_decay_resistance_proof(p)


def test_decay_resistance_hash_tamper_detected():
    e, sigs = _clean_bundle()
    p = build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))
    object.__setattr__(p, "decay_resistance_proof_hash", _different_hash(p.decay_resistance_proof_hash))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_decay_resistance_proof(p)


def test_tamper_detection():
    e, sigs = _clean_bundle()
    p = build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))
    for field, value, pattern in (
        ("entropy_drift_receipt_hash", "x", "INVALID_HASH_FORMAT"),
        ("replay_proof_hash", "x", "INVALID_HASH_FORMAT"),
        ("decay_class_at_proof", "DEGRADED", "DECAY_RESISTANCE_IMPOSSIBLE"),
        ("adversarial_positions_at_proof", ("p0",), "ADVERSARIAL_POSITIONS_PRESENT"),
        ("decay_resistance_proof_hash", _different_hash(p.decay_resistance_proof_hash), "HASH_MISMATCH"),
    ):
        t = build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))
        object.__setattr__(t, field, value)
        with pytest.raises(ValueError, match=pattern):
            validate_decay_resistance_proof(t)


def test_replay_proof_hash_extraction():
    e, sigs = _clean_bundle()
    assert build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12])).replay_proof_hash == H[12]


def test_replay_proof_hash_attribute_precedence_and_conflict():
    e, sigs = _clean_bundle()
    both_ok = type("R", (), {"lattice_replay_proof_hash": H[12], "replay_proof_hash": H[12]})()
    assert build_decay_resistance_proof(e, sigs, both_ok).replay_proof_hash == H[12]
    both_bad = type("R", (), {"lattice_replay_proof_hash": H[12], "replay_proof_hash": H[11]})()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_decay_resistance_proof(e, sigs, both_bad)


def test_invalid_replay_proof_rejected():
    e, sigs = _clean_bundle()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_decay_resistance_proof(e, sigs, object())
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_decay_resistance_proof(e, sigs, ReplayProofStub("x"))


def test_entropy_receipt_validated_against_signature_registry():
    e, sigs = _clean_bundle()
    p = build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))
    assert p.entropy_drift_receipt_hash == e.entropy_drift_receipt_hash
    assert p.replay_proof_hash == H[12]


def test_missing_signature_registry_entry_rejected():
    e, _ = _clean_bundle()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_decay_resistance_proof(e, [], ReplayProofStub(H[12]))


def test_duplicate_signature_registry_hash_rejected():
    e, sigs = _clean_bundle()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_decay_resistance_proof(e, [sigs[0], sigs[0]], ReplayProofStub(H[12]))


def test_registry_field_extraction_missing_referenced_hash_rejected():
    e, _ = _clean_bundle()
    other = _other_clean_sig()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_decay_resistance_proof(e, [other], ReplayProofStub(H[12]))


def test_tampered_entropy_receipt_rejected_before_proof_build():
    e, sigs = _clean_bundle()
    object.__setattr__(e, "entropy_drift_receipt_hash", _different_hash(e.entropy_drift_receipt_hash))
    with pytest.raises(ValueError, match="HASH_MISMATCH|AGGREGATE_SCORE_MISMATCH|INVALID_HASH_FORMAT|INVALID_INPUT"):
        build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))


def test_v155_artifact_chain_extension():
    cp = build_decay_checkpoint("p0", H[0], H[0])
    cps = build_decay_checkpoint_set([cp])
    contract = build_decay_threshold_contract(2, 4)
    sig = build_digital_decay_signature(cps, contract, [])
    e = build_entropy_drift_receipt(
        [build_layer_decay_receipt(H[2], H[3], sig)],
        [build_router_decay_receipt(H[4], H[5], sig)],
        [build_mask_collision_decay_receipt(H[6], H[7], "NO_COLLISION", sig)],
        [build_shift_decay_receipt(H[8], H[9], sig)],
        [build_readout_projection_decay_receipt(H[10], H[11], sig)],
        [sig],
    )
    p = build_decay_resistance_proof(e, [sig], ReplayProofStub(H[12]))
    chain = [
            cp.checkpoint_hash,
        cps.checkpoint_set_hash,
        contract.threshold_contract_hash,
        sig.digital_decay_signature_hash,
        e.layer_decay_receipts[0].layer_decay_receipt_hash,
        e.router_decay_receipts[0].router_decay_receipt_hash,
        e.mask_collision_decay_receipts[0].mask_collision_decay_receipt_hash,
        e.shift_decay_receipts[0].shift_decay_receipt_hash,
        e.readout_projection_decay_receipts[0].readout_projection_decay_receipt_hash,
        e.entropy_drift_receipt_hash,
        p.decay_resistance_proof_hash,
    ]
    assert all(isinstance(h, str) and len(h) == 64 and h == h.lower() for h in chain)
    assert p.decay_resistance_proof_hash == build_decay_resistance_proof(e, [sig], ReplayProofStub(H[12])).decay_resistance_proof_hash


def test_cross_environment_invariance():
    code = """
from dataclasses import dataclass
from qec.analysis.decay_checkpoint_contract import build_decay_checkpoint, build_decay_checkpoint_set
from qec.analysis.digital_decay_signature import build_decay_threshold_contract, build_digital_decay_signature
from qec.analysis.entropy_drift_receipt import build_entropy_drift_receipt, build_layer_decay_receipt, build_router_decay_receipt, build_mask_collision_decay_receipt, build_shift_decay_receipt, build_readout_projection_decay_receipt
from qec.analysis.decay_resistance_proof import build_decay_resistance_proof
H=[c*64 for c in '0123456789abcdef']
cp=build_decay_checkpoint('p0',H[0],H[0]); cps=build_decay_checkpoint_set([cp]); ct=build_decay_threshold_contract(2,4)
s=build_digital_decay_signature(cps,ct,[])
e=build_entropy_drift_receipt([build_layer_decay_receipt(H[2],H[3],s)],[build_router_decay_receipt(H[4],H[5],s)],[build_mask_collision_decay_receipt(H[6],H[7],'NO_COLLISION',s)],[build_shift_decay_receipt(H[8],H[9],s)],[build_readout_projection_decay_receipt(H[10],H[11],s)],[s])
@dataclass(frozen=True)
class ReplayProofStub: lattice_replay_proof_hash: str
print(build_decay_resistance_proof(e,[s],ReplayProofStub(H[12])).decay_resistance_proof_hash)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={"PYTHONPATH": "src", "PYTHONHASHSEED": "0"})
    cp = build_decay_checkpoint("p0", H[0], H[0]); cps = build_decay_checkpoint_set([cp]); contract = build_decay_threshold_contract(2, 4)
    sig = build_digital_decay_signature(cps, contract, [])
    e = build_entropy_drift_receipt([build_layer_decay_receipt(H[2], H[3], sig)], [build_router_decay_receipt(H[4], H[5], sig)], [build_mask_collision_decay_receipt(H[6], H[7], "NO_COLLISION", sig)], [build_shift_decay_receipt(H[8], H[9], sig)], [build_readout_projection_decay_receipt(H[10], H[11], sig)], [sig])
    parent = build_decay_resistance_proof(e, [sig], ReplayProofStub(H[12]))
    assert out.stdout.strip() == parent.decay_resistance_proof_hash


def test_no_v156_or_global_truth_scope_creep_fields():
    e, sigs = _clean_bundle(); p = build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))
    for name in ("perturbation_contract_hash", "perturbation_stability_proof_hash", "substrate_state_receipt_hash", "recursive_proof_receipt_hash", "global_truth_receipt_hash", "probability", "entropy_rate"):
        assert not hasattr(p, name)


def test_decay_resistance_proof_is_frozen():
    e, sigs = _clean_bundle(); p = build_decay_resistance_proof(e, sigs, ReplayProofStub(H[12]))
    with pytest.raises(FrozenInstanceError):
        p.replay_proof_hash = H[0]
