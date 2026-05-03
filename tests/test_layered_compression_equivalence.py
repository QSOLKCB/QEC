from __future__ import annotations

from types import MappingProxyType

import pytest

from qec.analysis.layer_removal_proof import build_layer_removal_receipt
from qec.analysis.layer_spec_contract import LayerCompatibilityConstraint, LayerInvariantSet, LayerSpec
from qec.analysis.layered_compression_equivalence import (
    CompressedLayeredProof,
    LayeredCompressionContract,
    build_compressed_layered_proof,
    build_layer_equivalence_receipt,
    validate_layer_equivalence_receipt,
)
from qec.analysis.layered_state_receipt import BaseStateReference, LayeredReceipt, build_layered_receipt


def _spec() -> LayerSpec:
    return LayerSpec(
        layer_id="layer-a",
        layer_version="152.3",
        invariant_set=LayerInvariantSet(invariants=("A", "B")),
        activation_rules={"enabled": True},
        removal_rules={"reversible": True},
        compatibility_constraints=(LayerCompatibilityConstraint("c1", "router_path", {"path": "root/a"}),),
    )


def _layered() -> LayeredReceipt:
    base = BaseStateReference(base_hash="base-hash-1", base_type="canonical-doc", base_metadata={"v": 1})
    return build_layered_receipt(base, _spec(), {"x": {"y": [1, 2]}})


def _contract(fields: tuple[str, ...] | None = None) -> LayeredCompressionContract:
    preserved = fields or (
        "base_hash",
        "layer_spec_hash",
        "layer_payload_hash",
        "layered_hash",
        "removal_receipt_hash",
        "return_path_hash",
        "boundary_integrity_hash",
        "compression_contract_hash",
    )
    return LayeredCompressionContract(
        compression_id="v152.3-structural",
        compression_version="1",
        compression_rules={"mode": "structural", "runtime": False},
        preserved_fields=preserved,
        equivalence_rules={"preserve_base_layer_boundary": True},
    )


def test_compression_preserves_base_layer_payload_and_removal_identities():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    proof = build_compressed_layered_proof(layered, removal, _contract())
    assert proof.preserved_identity_hashes["base_hash"] == layered.base_hash
    assert proof.preserved_identity_hashes["layer_payload_hash"] == layered.layer_payload_hash
    assert proof.preserved_identity_hashes["removal_receipt_hash"] == removal.receipt_hash


def test_same_inputs_produce_same_compressed_proof_hash():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    c = _contract()
    assert build_compressed_layered_proof(layered, removal, c).compressed_proof_hash == build_compressed_layered_proof(layered, removal, c).compressed_proof_hash


def test_same_inputs_produce_same_equivalence_receipt_hash():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    proof = build_compressed_layered_proof(layered, removal, _contract())
    assert build_layer_equivalence_receipt(proof, layered, removal).receipt_hash == build_layer_equivalence_receipt(proof, layered, removal).receipt_hash


def test_missing_preserved_identity_field_raises():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_compressed_layered_proof(layered, removal, _contract(fields=("base_hash",)))


def test_duplicate_preserved_fields_rejected():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _contract(fields=("base_hash", "base_hash"))


def test_empty_compression_id_rejected():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        LayeredCompressionContract("", "1", {}, tuple(), {})


def test_tampered_upstream_receipt_rejected_before_equivalence_validation():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    tampered = LayeredReceipt(
        base_hash=layered.base_hash,
        layer_spec_hash=layered.layer_spec_hash,
        layer_payload_hash=layered.layer_payload_hash,
        layered_hash=layered.layered_hash,
        receipt_hash="0" * 64,
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_compressed_layered_proof(tampered, removal, _contract())


def test_tampered_removal_receipt_rejected():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    tampered = type(removal)(
        base_hash=removal.base_hash,
        layered_hash=removal.layered_hash,
        layer_spec_hash=removal.layer_spec_hash,
        layer_payload_hash=removal.layer_payload_hash,
        return_path_proof=removal.return_path_proof,
        boundary_integrity_receipt=removal.boundary_integrity_receipt,
        removal_metadata=removal.removal_metadata,
        receipt_hash="0" * 64,
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_compressed_layered_proof(layered, tampered, _contract())


def test_compressed_proof_mismatch_rejected():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    proof = build_compressed_layered_proof(layered, removal, _contract())
    bad = CompressedLayeredProof(
        compression_contract_hash=proof.compression_contract_hash,
        source_layered_receipt_hash="0" * 64,
        source_removal_receipt_hash=proof.source_removal_receipt_hash,
        preserved_identity_hashes=proof.preserved_identity_hashes,
        compressed_proof_hash="",
    )
    bad = CompressedLayeredProof(
        compression_contract_hash=bad.compression_contract_hash,
        source_layered_receipt_hash=bad.source_layered_receipt_hash,
        source_removal_receipt_hash=bad.source_removal_receipt_hash,
        preserved_identity_hashes=bad.preserved_identity_hashes,
        compressed_proof_hash=bad.stable_hash(),
    )
    eq = build_layer_equivalence_receipt(proof, layered, removal)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_layer_equivalence_receipt(eq, bad, layered, removal)


def test_equivalence_receipt_tamper_detection():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    proof = build_compressed_layered_proof(layered, removal, _contract())
    eq = build_layer_equivalence_receipt(proof, layered, removal)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        type(eq)(**{**eq.to_dict(), "receipt_hash": "0" * 64})


def test_equivalence_rejects_mismatched_identity_fields():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    proof = build_compressed_layered_proof(layered, removal, _contract())
    eq = build_layer_equivalence_receipt(proof, layered, removal)
    mismatched = type(eq)(
        compressed_proof_hash=eq.compressed_proof_hash,
        layered_receipt_hash=eq.layered_receipt_hash,
        removal_receipt_hash=eq.removal_receipt_hash,
        base_hash="0" * 64,
        layered_hash=eq.layered_hash,
        layer_spec_hash=eq.layer_spec_hash,
        layer_payload_hash=eq.layer_payload_hash,
        equivalence_hash="",
        receipt_hash="",
    )
    mismatched = type(eq)(
        compressed_proof_hash=mismatched.compressed_proof_hash,
        layered_receipt_hash=mismatched.layered_receipt_hash,
        removal_receipt_hash=mismatched.removal_receipt_hash,
        base_hash=mismatched.base_hash,
        layered_hash=mismatched.layered_hash,
        layer_spec_hash=mismatched.layer_spec_hash,
        layer_payload_hash=mismatched.layer_payload_hash,
        equivalence_hash=mismatched._equivalence_hash(),
        receipt_hash="",
    )
    mismatched = type(eq)(
        compressed_proof_hash=mismatched.compressed_proof_hash,
        layered_receipt_hash=mismatched.layered_receipt_hash,
        removal_receipt_hash=mismatched.removal_receipt_hash,
        base_hash=mismatched.base_hash,
        layered_hash=mismatched.layered_hash,
        layer_spec_hash=mismatched.layer_spec_hash,
        layer_payload_hash=mismatched.layer_payload_hash,
        equivalence_hash=mismatched.equivalence_hash,
        receipt_hash=mismatched.stable_hash(),
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_layer_equivalence_receipt(mismatched, proof, layered, removal)


def test_equivalence_receipt_hash_recomputation():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    proof = build_compressed_layered_proof(layered, removal, _contract())
    eq = build_layer_equivalence_receipt(proof, layered, removal)
    validate_layer_equivalence_receipt(eq, proof, layered, removal)


def test_no_base_mutation():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    proof = build_compressed_layered_proof(layered, removal, _contract())
    assert proof.preserved_identity_hashes["base_hash"] == layered.base_hash


def test_compressed_proof_is_structural_not_runtime_compression():
    contract = _contract()
    assert contract.compression_rules["mode"] == "structural"


def test_immutability_enforcement():
    proof = build_compressed_layered_proof(_layered(), build_layer_removal_receipt(_layered()), _contract())
    assert isinstance(proof.preserved_identity_hashes, MappingProxyType)
    with pytest.raises(TypeError):
        proof.preserved_identity_hashes["x"] = "y"


def test_json_safety_rejection():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        LayeredCompressionContract("c", "1", {"bad": float("nan")}, ("base_hash",), {})


def test_to_canonical_json_self_hash_exclusion_behavior():
    layered = _layered()
    removal = build_layer_removal_receipt(layered)
    proof = build_compressed_layered_proof(layered, removal, _contract())
    canonical_json_text = proof.to_canonical_json()
    assert '"compressed_proof_hash"' not in canonical_json_text
    assert "compressed_proof_hash" in proof.to_dict()


def test_no_v1524_fractal_behavior():
    contract = _contract()
    assert "fractal" not in contract.to_canonical_json().lower()
