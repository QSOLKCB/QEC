from types import MappingProxyType

import pytest

from qec.analysis.layer_spec_contract import (
    LayerCompatibilityConstraint,
    LayerInvariantSet,
    LayerSpec,
    LayerSpecReceipt,
    build_layer_spec_receipt,
    validate_layer_spec_receipt,
)


def _make_spec() -> LayerSpec:
    return LayerSpec(
        layer_id="layer-a",
        layer_version="152.0",
        invariant_set=LayerInvariantSet(invariants=("A", "B")),
        activation_rules={"enabled": True, "router_paths": ["root/a"]},
        removal_rules={"reversible": True},
        compatibility_constraints=(
            LayerCompatibilityConstraint("c2", "readout_shell", {"shell": "x"}),
            LayerCompatibilityConstraint("c1", "router_path", {"path": "root/a"}),
            LayerCompatibilityConstraint("c3", "mask64", {"mask": "0x0f"}),
            LayerCompatibilityConstraint("c4", "hilbert_shift_label", {"label": "shift-1"}),
        ),
    )


def test_deterministic_hash_stability():
    spec = _make_spec()
    assert spec.stable_hash() == spec.stable_hash()


def test_mutation_changes_hash():
    spec_a = _make_spec()
    spec_b = LayerSpec(
        layer_id="layer-a",
        layer_version="152.0",
        invariant_set=LayerInvariantSet(invariants=("A", "B", "C")),
        activation_rules={"enabled": True, "router_paths": ["root/a"]},
        removal_rules={"reversible": True},
        compatibility_constraints=spec_a.compatibility_constraints,
    )
    assert spec_a.stable_hash() != spec_b.stable_hash()


def test_canonical_ordering_equivalence():
    spec_a = _make_spec()
    spec_b = LayerSpec(
        layer_id="layer-a",
        layer_version="152.0",
        invariant_set=LayerInvariantSet(invariants=("B", "A")),
        activation_rules={"router_paths": ["root/a"], "enabled": True},
        removal_rules={"reversible": True},
        compatibility_constraints=tuple(reversed(spec_a.compatibility_constraints)),
    )
    assert spec_a.stable_hash() == spec_b.stable_hash()


def test_duplicate_rejection():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        LayerInvariantSet(invariants=("A", "A"))


def test_empty_id_rejection():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        LayerSpec(
            layer_id="",
            layer_version="152.0",
            invariant_set=LayerInvariantSet(invariants=("A",)),
            activation_rules={"enabled": True},
            removal_rules={"reversible": True},
            compatibility_constraints=(),
        )


def test_json_safety_rejection():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        LayerCompatibilityConstraint("cid", "router_path", {"v": float("inf")})


def test_receipt_recomputation():
    spec = _make_spec()
    receipt = build_layer_spec_receipt(spec)
    validate_layer_spec_receipt(spec, receipt)


def test_tamper_detection():
    spec = _make_spec()
    receipt = build_layer_spec_receipt(spec)
    tampered = LayerSpecReceipt(layer_spec_hash=receipt.layer_spec_hash, receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_layer_spec_receipt(spec, tampered)


def test_immutability_enforcement():
    spec = _make_spec()
    assert isinstance(spec.activation_rules, MappingProxyType)
    with pytest.raises(TypeError):
        spec.activation_rules["x"] = 1


def test_declarative_only_constraints():
    spec = _make_spec()
    kinds = [c.constraint_type for c in spec.compatibility_constraints]
    assert set(kinds) == {"router_path", "readout_shell", "mask64", "hilbert_shift_label"}


def test_no_base_state_requirement():
    spec = _make_spec()
    assert "base" not in spec.to_dict()


def test_no_layered_state_creation():
    spec = _make_spec()
    assert "layered" not in spec.to_canonical_json()


def test_identical_specs_produce_identical_receipts():
    spec1 = _make_spec()
    spec2 = _make_spec()

    assert spec1 is not spec2

    r1 = build_layer_spec_receipt(spec1)
    r2 = build_layer_spec_receipt(spec2)

    assert r1.receipt_hash == r2.receipt_hash
