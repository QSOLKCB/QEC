from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from qec.analysis.fractal_invariant_compression import (
    _REQUIRED_PRESERVED_FIELDS,
    FractalInvariantCompressionReceipt,
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
    CompressedLayeredProof,
    LayerEquivalenceReceipt,
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


def test_to_dict_includes_pattern_hash_per_node():
    """to_dict() must expose each node's pattern_hash (Issue 3: public schema round-trip)."""
    proof, eq = _upstream()
    fr = build_fractal_invariant_compression_receipt(proof, eq, _fractal_contract())
    node_dict = fr.to_dict()["pattern_nodes"][0]
    assert "pattern_hash" in node_dict
    assert node_dict["pattern_hash"] == fr.pattern_nodes[0].pattern_hash
    # canonical JSON (used for hashing) must NOT contain pattern_hash (self-referential exclusion)
    assert '"pattern_hash"' not in fr.to_canonical_json()


def _upstream2():
    """Second independent upstream chain with different base data."""
    base = BaseStateReference("base-hash-2", "canonical-doc", {"v": 2})
    layered = build_layered_receipt(base, _spec(), {"x": {"y": [3, 4]}})
    removal = build_layer_removal_receipt(layered)
    proof = build_compressed_layered_proof(layered, removal, _contract())
    eq = build_layer_equivalence_receipt(proof, layered, removal)
    return proof, eq


def test_empty_equivalence_hash_in_receipt_rejected():
    """A LayerEquivalenceReceipt with equivalence_hash='' must be rejected (Issue 4).

    LayerEquivalenceReceipt.__post_init__ only validates equivalence_hash when it is
    truthy, so an instance with equivalence_hash='' and a self-consistent receipt_hash
    can be constructed without raising.  The new guard in _validated_identity_fields
    must reject such a receipt before it is accepted into a fractal receipt.
    """
    proof, eq = _upstream()
    c = _fractal_contract()
    # Construct a receipt where equivalence_hash is deliberately left empty.
    # LayerEquivalenceReceipt allows this in __post_init__ (the guard is only applied when
    # equivalence_hash is truthy), so we have to compute a self-consistent receipt_hash.
    empty_eq = LayerEquivalenceReceipt(
        compressed_proof_hash=eq.compressed_proof_hash,
        layered_receipt_hash=eq.layered_receipt_hash,
        removal_receipt_hash=eq.removal_receipt_hash,
        base_hash=eq.base_hash,
        layered_hash=eq.layered_hash,
        layer_spec_hash=eq.layer_spec_hash,
        layer_payload_hash=eq.layer_payload_hash,
        equivalence_hash="",
        receipt_hash="",
    )
    empty_eq = replace(empty_eq, receipt_hash=empty_eq.stable_hash())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_fractal_invariant_compression_receipt(proof, empty_eq, c)


def test_forged_source_layered_receipt_hash_in_proof_rejected():
    """A CompressedLayeredProof whose source_layered_receipt_hash doesn't match the
    equivalence receipt's layered_receipt_hash must be rejected (Issue 5)."""
    proof, eq = _upstream()
    c = _fractal_contract()
    # Construct a proof with a forged source_layered_receipt_hash.  We keep all
    # preserved_identity_hashes identical so only the source link differs.
    forged_base = CompressedLayeredProof(
        compression_contract_hash=proof.compression_contract_hash,
        source_layered_receipt_hash="0" * 64,
        source_removal_receipt_hash=proof.source_removal_receipt_hash,
        preserved_identity_hashes=dict(proof.preserved_identity_hashes),
        compressed_proof_hash="",
    )
    forged_proof = CompressedLayeredProof(
        compression_contract_hash=forged_base.compression_contract_hash,
        source_layered_receipt_hash=forged_base.source_layered_receipt_hash,
        source_removal_receipt_hash=forged_base.source_removal_receipt_hash,
        preserved_identity_hashes=dict(forged_base.preserved_identity_hashes),
        compressed_proof_hash=forged_base.stable_hash(),
    )
    # Build a self-consistent equivalence receipt that references forged_proof
    forged_eq = replace(
        eq,
        compressed_proof_hash=forged_proof.compressed_proof_hash,
        equivalence_hash="",
        receipt_hash="",
    )
    forged_eq = replace(forged_eq, equivalence_hash=forged_eq._equivalence_hash())
    forged_eq = replace(forged_eq, receipt_hash=forged_eq.stable_hash())
    # forged_proof.source_layered_receipt_hash ("0"*64) != forged_eq.layered_receipt_hash
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_fractal_invariant_compression_receipt(forged_proof, forged_eq, c)


def test_validator_rejects_equivalence_receipt_with_wrong_compressed_proof_hash():
    """validate_fractal_invariant_equivalence_receipt must check that
    equivalence_receipt.compressed_proof_hash matches the compressed_proof argument (P1)."""
    proof, eq = _upstream()
    proof2, eq2 = _upstream2()
    fr = build_fractal_invariant_compression_receipt(proof, eq, _fractal_contract())
    fer = build_fractal_invariant_equivalence_receipt(fr, proof, eq)
    fr2 = build_fractal_invariant_compression_receipt(proof2, eq2, _fractal_contract())
    fer2 = build_fractal_invariant_equivalence_receipt(fr2, proof2, eq2)

    # fer2/fr2 are valid for proof2/eq2.  Passing proof (chain 1) as compressed_proof
    # means eq2.compressed_proof_hash != proof.compressed_proof_hash -> must fail.
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_fractal_invariant_equivalence_receipt(fer2, fr2, proof, eq2)

    # Symmetric: passing eq (chain 1) as equivalence_receipt means
    # eq.compressed_proof_hash != proof2.compressed_proof_hash -> must fail.
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_fractal_invariant_equivalence_receipt(fer2, fr2, proof2, eq)


def test_validator_recomputes_fractal_pattern_hash_from_upstream():
    """The validator must independently recompute fractal_pattern_hash from the upstream
    inputs and reject a fractal_receipt that claims different pattern nodes (Issue 1)."""
    proof, eq = _upstream()
    proof2, eq2 = _upstream2()
    fr2 = build_fractal_invariant_compression_receipt(proof2, eq2, _fractal_contract())

    # Construct a fractal compression receipt that is internally self-consistent but
    # declares source_compressed_proof_hash/source_equivalence_receipt_hash from chain 1
    # while carrying pattern_nodes (and fractal_pattern_hash) derived from chain 2.
    forged_fr_base = FractalInvariantCompressionReceipt(
        fractal_contract_hash=fr2.fractal_contract_hash,
        source_compressed_proof_hash=proof.compressed_proof_hash,
        source_equivalence_receipt_hash=eq.receipt_hash,
        pattern_nodes=fr2.pattern_nodes,
        fractal_pattern_hash=fr2.fractal_pattern_hash,
        receipt_hash="",
    )
    forged_fr = FractalInvariantCompressionReceipt(
        fractal_contract_hash=forged_fr_base.fractal_contract_hash,
        source_compressed_proof_hash=forged_fr_base.source_compressed_proof_hash,
        source_equivalence_receipt_hash=forged_fr_base.source_equivalence_receipt_hash,
        pattern_nodes=forged_fr_base.pattern_nodes,
        fractal_pattern_hash=forged_fr_base.fractal_pattern_hash,
        receipt_hash=forged_fr_base.stable_hash(),
    )
    # forged_fr.source_compressed_proof_hash == proof.compressed_proof_hash and
    # forged_fr.source_equivalence_receipt_hash == eq.receipt_hash, so the receipt-hash
    # source-link checks would pass — but the pattern nodes are from chain 2.
    # The validator must recompute fractal_pattern_hash from proof/eq and reject the mismatch.
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_fractal_invariant_equivalence_receipt(
            # Build a minimal equivalence receipt for forged_fr directly so we can
            # exercise the validator's recomputation path without going through the builder.
            FractalInvariantEquivalenceReceipt(
                fractal_receipt_hash=forged_fr.receipt_hash,
                compressed_proof_hash=proof.compressed_proof_hash,
                layer_equivalence_receipt_hash=eq.receipt_hash,
                base_hash=eq.base_hash,
                layered_hash=eq.layered_hash,
                layer_spec_hash=eq.layer_spec_hash,
                layer_payload_hash=eq.layer_payload_hash,
                removal_receipt_hash=eq.removal_receipt_hash,
                return_path_hash=dict(proof.preserved_identity_hashes)["return_path_hash"],
                boundary_integrity_hash=dict(proof.preserved_identity_hashes)["boundary_integrity_hash"],
                compression_contract_hash=proof.compression_contract_hash,
                fractal_pattern_hash=forged_fr.fractal_pattern_hash,
                equivalence_hash="",
                receipt_hash="",
            ),
            forged_fr,
            proof,
            eq,
        )
