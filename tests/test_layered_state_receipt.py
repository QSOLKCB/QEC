from types import MappingProxyType

import pytest

from qec.analysis.layer_spec_contract import (
    LayerCompatibilityConstraint,
    LayerInvariantSet,
    LayerSpec,
)
from qec.analysis.layered_state_receipt import (
    BaseStateReference,
    LayeredReceipt,
    LayeredState,
    _canonical_layer_payload,
    build_layered_receipt,
    validate_layered_receipt,
)


def _spec() -> LayerSpec:
    return LayerSpec(
        layer_id="layer-a",
        layer_version="152.1",
        invariant_set=LayerInvariantSet(invariants=("A", "B")),
        activation_rules={"enabled": True},
        removal_rules={"reversible": True},
        compatibility_constraints=(
            LayerCompatibilityConstraint("c1", "router_path", {"path": "root/a"}),
        ),
    )


def _base() -> BaseStateReference:
    return BaseStateReference(base_hash="base-hash-1", base_type="canonical-doc", base_metadata={"v": 1})


def test_layered_state_does_not_mutate_base():
    base = _base()
    receipt = build_layered_receipt(base, _spec(), {"x": 1})
    layered = LayeredState(
        base_hash=receipt.base_hash,
        layer_spec_hash=receipt.layer_spec_hash,
        layer_payload_hash=receipt.layer_payload_hash,
        layered_hash=receipt.layered_hash,
    )

    assert base.base_hash == layered.base_hash


def test_identical_inputs_produce_identical_layered_receipts():
    r1 = build_layered_receipt(_base(), _spec(), {"a": 1, "b": [2, 3]})
    r2 = build_layered_receipt(_base(), _spec(), {"b": [2, 3], "a": 1})

    assert r1.receipt_hash == r2.receipt_hash


def test_payload_mutation_changes_layered_hash() -> None:
    base = _base()
    spec = _spec()
    r1 = build_layered_receipt(base, spec, {"x": 1})
    r2 = build_layered_receipt(base, spec, {"x": 2})
    assert r1.layered_hash != r2.layered_hash


def test_json_safety_rejection() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_layered_receipt(_base(), _spec(), {"x": float("nan")})


def test_receipt_recomputation_and_validation() -> None:
    base = _base()
    spec = _spec()
    payload = {"x": {"y": [1, 2]}}
    receipt = build_layered_receipt(base, spec, payload)
    validate_layered_receipt(receipt, base, spec, payload)


def test_tamper_detection() -> None:
    base = _base()
    spec = _spec()
    payload = {"x": 1}
    receipt = build_layered_receipt(base, spec, payload)
    tampered = LayeredReceipt(
        base_hash=receipt.base_hash,
        layer_spec_hash=receipt.layer_spec_hash,
        layer_payload_hash=receipt.layer_payload_hash,
        layered_hash=receipt.layered_hash,
        receipt_hash="0" * 64,
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_layered_receipt(tampered, base, spec, payload)


def test_immutability_enforcement() -> None:
    base = _base()
    assert isinstance(base.base_metadata, MappingProxyType)
    with pytest.raises(TypeError):
        base.base_metadata["z"] = 2


def test_cross_instance_determinism() -> None:
    base_a = BaseStateReference(base_hash="base-hash-1", base_type="canonical-doc", base_metadata={"v": 1})
    base_b = BaseStateReference(base_hash="base-hash-1", base_type="canonical-doc", base_metadata={"v": 1})
    spec_a = _spec()
    spec_b = _spec()
    payload = {"k": [1, 2], "m": {"n": True}}
    assert build_layered_receipt(base_a, spec_a, payload) == build_layered_receipt(base_b, spec_b, payload)


def test_mixed_key_ordering_is_deterministic() -> None:
    payload_a = {"z": 1, "a": 2, "k": 3}
    payload_b = {"k": 3, "z": 1, "a": 2}
    frozen_a = _canonical_layer_payload(payload_a)
    frozen_b = _canonical_layer_payload(payload_b)
    assert tuple(frozen_a.keys()) == tuple(frozen_b.keys())


def test_canonical_layer_payload_and_frozen_payload_order_agree() -> None:
    payload = {"beta": 2, "alpha": 1, "gamma": 3}
    frozen = _canonical_layer_payload(payload)
    assert tuple(frozen.keys()) == ("alpha", "beta", "gamma")


def test_identical_logical_payloads_produce_identical_layer_payload_hash() -> None:
    base = _base()
    spec = _spec()
    receipt_a = build_layered_receipt(base, spec, {"x": 1, "y": {"b": 2, "a": 1}})
    receipt_b = build_layered_receipt(base, spec, {"y": {"a": 1, "b": 2}, "x": 1})
    assert receipt_a.layer_payload_hash == receipt_b.layer_payload_hash
