import pytest

from qec.analysis.hilber_shift_projection import (
    SHIFT_ALGORITHM,
    HILBER_SHIFT_VERSION,
    build_filter_compatibility_receipt,
    build_hilber_shift_spec,
    build_shift_projection_receipt,
    build_shift_stability_receipt,
    validate_shift_stability_receipt,
    ShiftProjectionReceipt,
    FilterCompatibilityReceipt,
    ShiftStabilityReceipt,
)


def _spec(order=("a", "b", "c", "d"), direction="FORWARD", offset=1):
    return build_hilber_shift_spec("s1", "GENERIC", "src", "h" * 64, tuple(order), direction, offset)


def test_determinism_and_order_identity():
    s1 = _spec()
    s2 = _spec()
    assert s1.spec_hash == s2.spec_hash
    r1 = build_shift_projection_receipt(s1)
    r2 = build_shift_projection_receipt(s2)
    assert r1.receipt_hash == r2.receipt_hash

    s3 = _spec(order=("b", "a", "c", "d"))
    assert s3.spec_hash != s1.spec_hash


def test_shift_correctness_forward_reverse_and_offset_bounds():
    f = build_shift_projection_receipt(_spec(direction="FORWARD", offset=1))
    assert f.shifted_item_ids == ("d", "a", "b", "c")
    r = build_shift_projection_receipt(_spec(direction="REVERSE", offset=1))
    assert r.shifted_item_ids == ("b", "c", "d", "a")

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _spec(offset=4)


def test_bijection_enforced():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ShiftProjectionReceipt(
            "s1",
            "x",
            ("a", "b"),
            ("a", "b"),
            ({"item_id": "a", "source_index": 0, "target_index": 0}, {"item_id": "b", "source_index": 1, "target_index": 0}),
            "x",
            "y",
            "",
            "",
        )


def test_hash_integrity_tampering_fails_validation():
    spec = _spec()
    projection = build_shift_projection_receipt(spec)
    comp = build_filter_compatibility_receipt("c1", projection, tuple())
    stability = build_shift_stability_receipt("st1", projection, comp)
    validate_shift_stability_receipt(stability)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        stability.__class__(**{**stability.__dict__, "stability_status": "UNSTABLE", "stability_reason": "ROUNDTRIP_MISMATCH"})


def test_stability_and_immutability_and_json_safety():
    spec = _spec()
    projection = build_shift_projection_receipt(spec)
    comp = build_filter_compatibility_receipt("c1", projection, ("f1",))
    stability = build_shift_stability_receipt("st1", projection, comp)
    assert stability.roundtrip_stable is True

    with pytest.raises(Exception):
        spec.shift_id = "x"

    assert isinstance(spec.to_dict(), dict)
    assert isinstance(projection.to_dict(), dict)


def test_scope_and_version_constants():
    assert HILBER_SHIFT_VERSION == "v153.6"
    assert SHIFT_ALGORITHM == "DISCRETE_CYCLIC_INDEX_SHIFT_V1"
    module_dict = __import__("qec.analysis.hilber_shift_projection", fromlist=["*"]).__dict__
    assert not any(k.startswith("v153.7") for k in module_dict)


def test_incompatible_filter_compatibility_receipt():
    spec = _spec(direction="FORWARD", offset=1)
    projection = build_shift_projection_receipt(spec)
    assert projection.input_order_hash != projection.shifted_order_hash
    comp = build_filter_compatibility_receipt("c1", projection, tuple())
    assert comp.compatibility_status == "INCOMPATIBLE"
    assert comp.compatibility_reason == "IDENTITY_MISMATCH"


def test_compatible_filter_compatibility_receipt_with_zero_offset():
    spec = _spec(direction="FORWARD", offset=0)
    projection = build_shift_projection_receipt(spec)
    assert projection.input_order_hash == projection.shifted_order_hash
    comp = build_filter_compatibility_receipt("c1", projection, tuple())
    assert comp.compatibility_status == "COMPATIBLE"
    assert comp.compatibility_reason == "IDENTITY_PRESERVED"


def test_stability_receipt_stable_with_cyclic_shift():
    spec = _spec(direction="FORWARD", offset=1)
    projection = build_shift_projection_receipt(spec)
    comp = build_filter_compatibility_receipt("c1", projection, tuple())
    stability = build_shift_stability_receipt("st1", projection, comp)
    assert stability.roundtrip_stable is True
    assert stability.stability_status == "STABLE"
    assert stability.stability_reason == "ROUNDTRIP_MATCH"
    validate_shift_stability_receipt(stability)


def test_mismatched_projection_receipt_rejected_in_stability_builder():
    spec1 = _spec(direction="FORWARD", offset=1)
    spec2 = _spec(direction="FORWARD", offset=2)
    projection1 = build_shift_projection_receipt(spec1)
    projection2 = build_shift_projection_receipt(spec2)
    comp = build_filter_compatibility_receipt("c1", projection1, tuple())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_shift_stability_receipt("st1", projection2, comp)


def test_shift_pairs_immutability():
    spec = _spec()
    projection = build_shift_projection_receipt(spec)
    with pytest.raises(TypeError):
        projection.shift_pairs[0]["item_id"] = "hacked"


def test_filter_binding_hashes_validation():
    spec = _spec()
    projection = build_shift_projection_receipt(spec)
    comp = build_filter_compatibility_receipt("c1", projection, ("b_hash", "a_hash"))
    assert comp.filter_binding_hashes == ("a_hash", "b_hash")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_filter_compatibility_receipt("c1", projection, ("a_hash", "a_hash"))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        FilterCompatibilityReceipt(
            "c1", projection.receipt_hash, projection.input_order_hash,
            projection.shifted_order_hash, ("b_hash", "a_hash"),
            "INCOMPATIBLE", "IDENTITY_MISMATCH", "", ""
        )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        FilterCompatibilityReceipt(
            "c1", projection.receipt_hash, projection.input_order_hash,
            projection.shifted_order_hash, ("a_hash", "a_hash"),
            "INCOMPATIBLE", "IDENTITY_MISMATCH", "", ""
        )


def test_malformed_shift_pairs_keyerror():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ShiftProjectionReceipt(
            "s1",
            "x",
            ("a", "b"),
            ("a", "b"),
            ({"item_id": "a", "source_index": 0}, {"item_id": "b", "source_index": 1, "target_index": 1}),
            "x",
            "y",
            "",
            "",
        )
