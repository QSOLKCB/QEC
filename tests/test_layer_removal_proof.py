from __future__ import annotations

from types import MappingProxyType

import pytest

from qec.analysis.layer_removal_proof import (
    BoundaryIntegrityReceipt,
    ReturnPathProof,
    build_layer_removal_receipt,
    validate_layer_removal_receipt,
)
from qec.analysis.layer_spec_contract import (
    LayerCompatibilityConstraint,
    LayerInvariantSet,
    LayerSpec,
)
from qec.analysis.layered_state_receipt import BaseStateReference, LayeredReceipt, build_layered_receipt


def _spec() -> LayerSpec:
    return LayerSpec(
        layer_id="layer-a",
        layer_version="152.2",
        invariant_set=LayerInvariantSet(invariants=("A", "B")),
        activation_rules={"enabled": True},
        removal_rules={"reversible": True},
        compatibility_constraints=(
            LayerCompatibilityConstraint("c1", "router_path", {"path": "root/a"}),
        ),
    )


def _layered_receipt() -> LayeredReceipt:
    base = BaseStateReference(base_hash="base-hash-1", base_type="canonical-doc", base_metadata={"v": 1})
    return build_layered_receipt(base, _spec(), {"x": {"y": [1, 2]}})


def test_layer_removal_preserves_base_identity():
    layered = _layered_receipt()
    removal = build_layer_removal_receipt(layered)

    assert removal.base_hash == layered.base_hash


def test_identical_layered_states_produce_identical_removal_receipts():
    layered = _layered_receipt()
    r1 = build_layer_removal_receipt(layered)
    r2 = build_layer_removal_receipt(layered)

    assert r1.stable_hash() == r2.stable_hash()


def test_tampered_layered_hash_fails_validation() -> None:
    layered = _layered_receipt()
    removal = build_layer_removal_receipt(layered)
    tampered_layered = type(layered)(
        base_hash=layered.base_hash,
        layer_spec_hash=layered.layer_spec_hash,
        layer_payload_hash=layered.layer_payload_hash,
        layered_hash="0" * 64,
        receipt_hash=layered.receipt_hash,
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_layer_removal_receipt(removal, tampered_layered)


def test_tampered_base_hash_fails_validation() -> None:
    layered = _layered_receipt()
    removal = build_layer_removal_receipt(layered)
    tampered = type(removal)(
        base_hash="0" * 64,
        layered_hash=removal.layered_hash,
        layer_spec_hash=removal.layer_spec_hash,
        layer_payload_hash=removal.layer_payload_hash,
        return_path_proof=removal.return_path_proof,
        boundary_integrity_receipt=removal.boundary_integrity_receipt,
        removal_metadata=removal.removal_metadata,
        receipt_hash=removal.receipt_hash,
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_layer_removal_receipt(tampered, layered)


def test_return_path_proof_determinism() -> None:
    layered = _layered_receipt()
    removal_a = build_layer_removal_receipt(layered)
    removal_b = build_layer_removal_receipt(layered)
    assert isinstance(removal_a.return_path_proof, ReturnPathProof)
    assert removal_a.return_path_proof.stable_hash() == removal_b.return_path_proof.stable_hash()


def test_boundary_integrity_validation() -> None:
    layered = _layered_receipt()
    removal = build_layer_removal_receipt(layered)
    assert isinstance(removal.boundary_integrity_receipt, BoundaryIntegrityReceipt)
    validate_layer_removal_receipt(removal, layered)


def test_immutability_enforcement() -> None:
    layered = _layered_receipt()
    removal = build_layer_removal_receipt(layered)
    assert isinstance(removal.removal_metadata, MappingProxyType)
    with pytest.raises(TypeError):
        removal.removal_metadata["x"] = 1


def test_json_safety_rejection() -> None:
    layered = _layered_receipt()
    removal = build_layer_removal_receipt(layered)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        type(removal)(
            base_hash=removal.base_hash,
            layered_hash=removal.layered_hash,
            layer_spec_hash=removal.layer_spec_hash,
            layer_payload_hash=removal.layer_payload_hash,
            return_path_proof=removal.return_path_proof,
            boundary_integrity_receipt=removal.boundary_integrity_receipt,
            removal_metadata={"bad": float("nan")},
            receipt_hash=removal.receipt_hash,
        )


def test_layer_removal_canonical_json_excludes_receipt_hash() -> None:
    removal = build_layer_removal_receipt(_layered_receipt())
    canonical_json_text = removal.to_canonical_json()
    assert '"receipt_hash"' not in canonical_json_text


def test_layer_removal_to_dict_includes_receipt_hash_but_canonical_json_does_not() -> None:
    removal = build_layer_removal_receipt(_layered_receipt())
    payload_dict = removal.to_dict()
    canonical_json_text = removal.to_canonical_json()
    assert "receipt_hash" in payload_dict
    assert payload_dict["receipt_hash"] == removal.receipt_hash
    assert '"receipt_hash"' not in canonical_json_text
