from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from qec.analysis.fractal_invariant_compression import (
    _REQUIRED_PRESERVED_FIELDS,
    FractalInvariantContract,
    FractalInvariantEquivalenceReceipt,
    InvariantPatternNode,
    build_fractal_invariant_compression_receipt,
    build_fractal_invariant_equivalence_receipt,
    validate_fractal_invariant_equivalence_receipt,
)
from qec.analysis.layer_removal_proof import build_layer_removal_receipt
from qec.analysis.layer_spec_contract import LayerCompatibilityConstraint, LayerInvariantSet, LayerSpec
from qec.analysis.layered_compression_equivalence import (
    LayeredCompressionContract,
    build_compressed_layered_proof,
    build_layer_equivalence_receipt,
)
from qec.analysis.layered_state_receipt import BaseStateReference, build_layered_receipt


def _spec() -> LayerSpec:
    return LayerSpec("layer-a", "152.4", LayerInvariantSet(("A", "B")), {"enabled": True}, {"reversible": True}, (LayerCompatibilityConstraint("c1", "router_path", {"path": "root/a"}),))


def _contract() -> LayeredCompressionContract:
    return LayeredCompressionContract("v152.3", "1", {"mode": "structural"}, (
        "base_hash", "layer_spec_hash", "layer_payload_hash", "layered_hash", "removal_receipt_hash", "return_path_hash", "boundary_integrity_hash", "compression_contract_hash",
    ), {"preserve": True})


def _upstream():
    base = BaseStateReference("base-hash-1", "canonical-doc", {"v": 1})
    layered = build_layered_receipt(base, _spec(), {"x": {"y": [1, 2]}})
    removal = build_layer_removal_receipt(layered)
    proof = build_compressed_layered_proof(layered, removal, _contract())
    eq = build_layer_equivalence_receipt(proof, layered, removal)
    return proof, eq


def _fractal_contract(fields: tuple[str, ...] | None = None) -> FractalInvariantContract:
    return FractalInvariantContract("fractal-v152.4", "1", {"deterministic": True}, {"model": "structural"}, fields or _REQUIRED_PRESERVED_FIELDS, {"equivalent": True})


def test_fractal_compression_preserves_all_layered_equivalence_identities():
    proof, eq = _upstream()
    r = build_fractal_invariant_compression_receipt(proof, eq, _fractal_contract())
    node = r.pattern_nodes[0]
    assert set(node.identity_fields.keys()) == set(_REQUIRED_PRESERVED_FIELDS)


def test_same_inputs_produce_same_fractal_pattern_hash_and_receipt_hash():
    proof, eq = _upstream()
    c = _fractal_contract()
    r1 = build_fractal_invariant_compression_receipt(proof, eq, c)
    r2 = build_fractal_invariant_compression_receipt(proof, eq, c)
    assert r1.fractal_pattern_hash == r2.fractal_pattern_hash
    assert r1.receipt_hash == r2.receipt_hash


def test_same_inputs_produce_same_fractal_equivalence_receipt_hash():
    proof, eq = _upstream()
    fr = build_fractal_invariant_compression_receipt(proof, eq, _fractal_contract())
    e1 = build_fractal_invariant_equivalence_receipt(fr, proof, eq)
    e2 = build_fractal_invariant_equivalence_receipt(fr, proof, eq)
    assert e1.receipt_hash == e2.receipt_hash


def test_contract_field_validation_and_empty_id():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _fractal_contract(tuple(_REQUIRED_PRESERVED_FIELDS[:-1]))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _fractal_contract(_REQUIRED_PRESERVED_FIELDS + ("extra",))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _fractal_contract(("base_hash", "base_hash"))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        FractalInvariantContract("", "1", {}, {}, _REQUIRED_PRESERVED_FIELDS, {})


def test_duplicate_pattern_identity_at_same_scale_rejected():
    proof, eq = _upstream()
    fields = {**dict(proof.preserved_identity_hashes), "compressed_proof_hash": proof.compressed_proof_hash, "equivalence_hash": eq.equivalence_hash, "equivalence_receipt_hash": eq.receipt_hash}
    node = InvariantPatternNode("p", 0, fields, "")
    node = InvariantPatternNode("p", 0, fields, node.stable_hash())
    from qec.analysis.fractal_invariant_compression import FractalInvariantCompressionReceipt

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        FractalInvariantCompressionReceipt("a", proof.compressed_proof_hash, eq.receipt_hash, (node, node), "0" * 64, "0" * 64)


def test_negative_and_non_integer_scale_rejected():
    proof, eq = _upstream()
    fields = {**dict(proof.preserved_identity_hashes), "compressed_proof_hash": proof.compressed_proof_hash, "equivalence_hash": eq.equivalence_hash, "equivalence_receipt_hash": eq.receipt_hash}
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        InvariantPatternNode("p", -1, fields, "")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        InvariantPatternNode("p", 1.1, fields, "")


def test_tampered_compressed_or_equivalence_source_rejected():
    proof, eq = _upstream()
    c = _fractal_contract()
    bad_proof = build_compressed_layered_proof(build_layered_receipt(BaseStateReference("base-hash-2", "canonical-doc", {"v": 2}), _spec(), {"x": {"y": [3]}}), build_layer_removal_receipt(build_layered_receipt(BaseStateReference("base-hash-2", "canonical-doc", {"v": 2}), _spec(), {"x": {"y": [3]}})), _contract())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_fractal_invariant_compression_receipt(bad_proof, eq, c)
    bad_eq = type(eq)(**eq.to_dict())
    object.__setattr__(bad_eq, "receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_fractal_invariant_compression_receipt(proof, bad_eq, c)



def test_mismatched_compressed_and_equivalence_identities_rejected():
    proof, eq = _upstream()
    contract = _fractal_contract()
    tampered = replace(eq, base_hash="fake", equivalence_hash="", receipt_hash="")
    tampered = replace(tampered, equivalence_hash=tampered._equivalence_hash())
    tampered = replace(tampered, receipt_hash=tampered.stable_hash())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_fractal_invariant_compression_receipt(proof, tampered, contract)

def test_fractal_identity_mismatch_and_hash_recomputation_enforced():
    proof, eq = _upstream()
    fr = build_fractal_invariant_compression_receipt(proof, eq, _fractal_contract())
    fer = build_fractal_invariant_equivalence_receipt(fr, proof, eq)
    bad = FractalInvariantEquivalenceReceipt(**{**fer.to_dict(), "base_hash": "0" * 64, "equivalence_hash": "", "receipt_hash": ""})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_fractal_invariant_equivalence_receipt(bad, fr, proof, eq)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        FractalInvariantEquivalenceReceipt(**{**fer.to_dict(), "equivalence_hash": "0" * 64})


def test_fractal_compression_is_structural_not_runtime_compression_or_rendering():
    c = _fractal_contract()
    assert "runtime" not in c.to_canonical_json().lower()
    assert "render" not in c.to_canonical_json().lower()
    assert "traverse" not in c.to_canonical_json().lower()


def test_immutability_json_safety_and_self_hash_exclusion_behavior():
    proof, eq = _upstream()
    fr = build_fractal_invariant_compression_receipt(proof, eq, _fractal_contract())
    assert isinstance(fr.pattern_nodes[0].identity_fields, MappingProxyType)
    with pytest.raises(TypeError):
        fr.pattern_nodes[0].identity_fields["x"] = "y"
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        FractalInvariantContract("f", "1", {"bad": float("nan")}, {}, _REQUIRED_PRESERVED_FIELDS, {})
    assert "receipt_hash" in fr.to_dict()
    assert '"receipt_hash"' not in fr.to_canonical_json()
