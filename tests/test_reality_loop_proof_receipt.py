from dataclasses import FrozenInstanceError, replace
import inspect

import pytest

from qec.analysis.cross_arc_identity_link import build_cross_arc_identity_link_receipt
from qec.analysis.reality_loop_composition_spec import build_reality_loop_composition_spec, get_reality_loop_slot_definitions
from qec.analysis.canonical_hashing import sha256_hex
import qec.analysis.reality_loop_proof_receipt as rlpr_module
from qec.analysis.reality_loop_proof_receipt import (
    _reality_loop_proof_receipt_payload,
    RealityLoopProofReceipt,
    build_reality_loop_proof_receipt,
    validate_reality_loop_proof_receipt,
    validate_reality_loop_proof_receipt_with_artifacts,
)


def _h(i: int) -> str:
    return f"{i:064x}"[-64:]


def _spec(offset: int = 1):
    mapping = {field: _h(i + offset) for i, (_, _, field) in enumerate(get_reality_loop_slot_definitions())}
    return build_reality_loop_composition_spec(mapping)


def _artifacts(offset: int = 1):
    spec = _spec(offset)
    link_receipt = build_cross_arc_identity_link_receipt(spec)
    proof = build_reality_loop_proof_receipt(spec, link_receipt)
    return spec, link_receipt, proof




def _rehash(receipt, **changes):
    d = receipt.to_dict()
    d.update(changes)
    payload = _reality_loop_proof_receipt_payload(
        d["composition_spec_hash"],
        d["cross_arc_identity_link_receipt_hash"],
        d["composition_mode"],
        d["proof_mode"],
        d["slot_count"],
        d["required_slot_count"],
        d["link_count"],
        d["required_link_count"],
        d["first_composition_slot_hash"],
        d["final_composition_slot_hash"],
        tuple(d["composition_slot_hashes"]),
        tuple(d["cross_arc_identity_link_hashes"]),
        d["slots_complete"],
        d["links_complete"],
        d["slot_link_topology_aligned"],
        d["reality_loop_complete"],
        d["reality_loop_proof_class"],
    )
    return RealityLoopProofReceipt(**{**d, "composition_slot_hashes": tuple(d["composition_slot_hashes"]), "cross_arc_identity_link_hashes": tuple(d["cross_arc_identity_link_hashes"]), "reality_loop_proof_receipt_hash": sha256_hex(payload)})



def _unchecked(receipt, **changes):
    values = {**receipt.__dict__, **changes}
    obj = object.__new__(type(receipt))
    for k, v in values.items():
        object.__setattr__(obj, k, v)
    return obj

def test_reality_loop_proof_receipt_basics():
    spec, link_receipt, receipt = _artifacts()
    receipt2 = build_reality_loop_proof_receipt(spec, link_receipt)
    assert receipt.reality_loop_proof_receipt_hash == receipt2.reality_loop_proof_receipt_hash
    assert receipt.reality_loop_proof_class == "REALITY_LOOP_PROOF_COMPLETE"
    assert receipt.slots_complete is True
    assert receipt.links_complete is True
    assert receipt.slot_link_topology_aligned is True
    assert receipt.reality_loop_complete is True
    assert len(receipt.composition_slot_hashes) == 19
    assert len(receipt.cross_arc_identity_link_hashes) == 18
    assert receipt.first_composition_slot_hash == receipt.composition_slot_hashes[0]
    assert receipt.final_composition_slot_hash == receipt.composition_slot_hashes[18]

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_reality_loop_proof_receipt(replace(receipt, reality_loop_proof_receipt_hash="bad"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, reality_loop_proof_receipt_hash=_h(9999)))

    with pytest.raises(FrozenInstanceError):
        receipt.slot_count = 10

    assert receipt.to_canonical_json() == receipt2.to_canonical_json()
    assert receipt.to_canonical_bytes() == receipt2.to_canonical_bytes()


def test_intrinsic_validation():
    _, _, receipt = _artifacts()
    with pytest.raises(ValueError, match="INVALID_PROOF_MODE"):
        validate_reality_loop_proof_receipt(replace(receipt, proof_mode="BAD"))
    with pytest.raises(ValueError, match="INVALID_COMPOSITION_MODE"):
        validate_reality_loop_proof_receipt(replace(receipt, composition_mode="BAD"))
    with pytest.raises(ValueError, match="INVALID_REALITY_LOOP_PROOF_CLASS"):
        validate_reality_loop_proof_receipt(replace(receipt, reality_loop_proof_class="BAD"))
    with pytest.raises(ValueError, match="SLOT_COUNT_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, slot_count=True))
    with pytest.raises(ValueError, match="SLOT_COUNT_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, required_slot_count=True))
    with pytest.raises(ValueError, match="LINK_COUNT_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, link_count=True))
    with pytest.raises(ValueError, match="LINK_COUNT_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, required_link_count=True))

    with pytest.raises(ValueError, match="SLOT_COUNT_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, slot_count=18))
    with pytest.raises(ValueError, match="SLOT_COUNT_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, required_slot_count=20))
    with pytest.raises(ValueError, match="LINK_COUNT_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, link_count=17))
    with pytest.raises(ValueError, match="LINK_COUNT_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, required_link_count=19))

    dup_slot = _unchecked(receipt, composition_slot_hashes=(receipt.composition_slot_hashes[0], *receipt.composition_slot_hashes[1:-1], receipt.composition_slot_hashes[0]))
    with pytest.raises(ValueError, match="DUPLICATE_SLOT_HASH"):
        validate_reality_loop_proof_receipt(dup_slot)
    dup_link = _unchecked(receipt, cross_arc_identity_link_hashes=(receipt.cross_arc_identity_link_hashes[0], *receipt.cross_arc_identity_link_hashes[1:-1], receipt.cross_arc_identity_link_hashes[0]))
    with pytest.raises(ValueError, match="DUPLICATE_LINK_HASH"):
        validate_reality_loop_proof_receipt(dup_link)

    with pytest.raises(ValueError, match="SLOT_COUNT_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, composition_slot_hashes=list(receipt.composition_slot_hashes)))
    with pytest.raises(ValueError, match="LINK_COUNT_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, cross_arc_identity_link_hashes=list(receipt.cross_arc_identity_link_hashes)))

    with pytest.raises(ValueError, match="SLOT_LINK_TOPOLOGY_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, first_composition_slot_hash=_h(1234)))
    with pytest.raises(ValueError, match="SLOT_LINK_TOPOLOGY_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, final_composition_slot_hash=_h(1234)))
    # slots_complete and links_complete are now validated against derived values from counts
    # Setting them to False when counts match required counts raises INVALID_INPUT
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_reality_loop_proof_receipt(replace(receipt, slots_complete=False))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_reality_loop_proof_receipt(replace(receipt, links_complete=False))
    # slot_link_topology_aligned=False with complete counts triggers REALITY_LOOP_COMPLETION_MISMATCH
    # because derived completion would be False but receipt claims reality_loop_complete=True
    with pytest.raises(ValueError, match="REALITY_LOOP_COMPLETION_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, slot_link_topology_aligned=False))
    with pytest.raises(ValueError, match="REALITY_LOOP_COMPLETION_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, reality_loop_complete=False))
    with pytest.raises(ValueError, match="REALITY_LOOP_PROOF_CLASS_MISMATCH"):
        validate_reality_loop_proof_receipt(replace(receipt, reality_loop_proof_class="REALITY_LOOP_PROOF_INVALID"))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_reality_loop_proof_receipt(replace(receipt, composition_spec_hash="A" * 64))


def test_complete_validator():
    spec, link_receipt, receipt = _artifacts()
    assert validate_reality_loop_proof_receipt_with_artifacts(receipt, spec, link_receipt)

    other_spec, other_links, _ = _artifacts(200)
    with pytest.raises(ValueError):
        validate_reality_loop_proof_receipt_with_artifacts(receipt, other_spec, link_receipt)
    with pytest.raises(ValueError):
        validate_reality_loop_proof_receipt_with_artifacts(receipt, spec, other_links)

    for bad in [
        _unchecked(receipt, composition_slot_hashes=(_h(1), *receipt.composition_slot_hashes[1:])),
        _unchecked(receipt, cross_arc_identity_link_hashes=(_h(2), *receipt.cross_arc_identity_link_hashes[1:])),
        _unchecked(receipt, first_composition_slot_hash=_h(3), composition_slot_hashes=(_h(3), *receipt.composition_slot_hashes[1:])),
        _unchecked(receipt, final_composition_slot_hash=_h(4), composition_slot_hashes=((*receipt.composition_slot_hashes[:-1], _h(4)))),
        _unchecked(receipt, reality_loop_complete=False),
        _unchecked(receipt, reality_loop_proof_class="REALITY_LOOP_PROOF_INVALID"),
    ]:
        with pytest.raises(ValueError):
            validate_reality_loop_proof_receipt_with_artifacts(bad, spec, link_receipt)


def test_boundary_and_scope_scan():
    _, _, receipt = _artifacts()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_reality_loop_proof_receipt(object())

    malformed_child = _unchecked(receipt, composition_slot_hashes=(object(), *receipt.composition_slot_hashes[1:]))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_reality_loop_proof_receipt(malformed_child)

    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_reality_loop_proof_receipt(replace(receipt, first_composition_slot_hash="A" * 64))

    # Use inspect to get source file path instead of hard-coded path
    source_file = inspect.getsourcefile(rlpr_module)
    assert source_file is not None, "Could not find source file for module"
    with open(source_file, "r", encoding="utf-8") as f:
        text = f.read()
    forbidden = [
        "GlobalTruthReceipt", "GlobalValidationIndex", "GlobalThresholdContract", "GlobalReplayProof",
        "global_truth", "global_validation", "runtime_replay_execution", "recursive_execution",
        "gameplay", "render", "step_world", "execute_action", "run_game", "importlib", "__import__(",
        "subprocess", "exec(", "eval(", "random", "time.time", "datetime.now", "probability",
        "probabilistic", "neural", "learned_policy",
    ]
    for token in forbidden:
        assert token not in text
    assert "RealityLoopProofReceipt" in text
    assert "reality_loop_proof_receipt_hash" in text
