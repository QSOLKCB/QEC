import json

import pytest

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.search_mask64_contract import (
    MAX_COLLISION_RECORDS,
    MaskCollisionReceipt,
    MaskCompatibilityReceipt,
    SearchMask64,
    SEARCH_MASK64_VERSION,
    UINT64_MAX,
    build_mask_collision_receipt,
    build_mask_compatibility_receipt,
    build_mask_reduction_receipt,
    build_search_mask64,
    validate_mask_compatibility_receipt,
)


def _payload():
    return {"b": 2, "a": {"x": 1, "y": [True, "k"]}}


def test_determinism_and_expected_sha_reduction():
    m1 = build_search_mask64("m", "GENERIC", "s", _payload())
    m2 = build_search_mask64("m", "GENERIC", "s", {"a": {"y": [True, "k"], "x": 1}, "b": 2})
    assert (m1.mask_value, m1.mask_hex, m1.mask_hash) == (m2.mask_value, m2.mask_hex, m2.mask_hash)
    expected = int.from_bytes(__import__("hashlib").sha256(canonical_json(_payload()).encode("utf-8")).digest()[:8], "big", signed=False)
    assert m1.mask_value == expected
    assert len(m1.mask_hex) == 16 and m1.mask_hex == m1.mask_hex.lower()
    assert 0 <= m1.mask_value <= UINT64_MAX


def test_reduction_tampering_failures():
    m = build_search_mask64("m", "GENERIC", "s", _payload())
    r = build_mask_reduction_receipt("r", m, _payload())
    assert r.search_mask_hash == m.mask_hash
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_mask_reduction_receipt("r", SearchMask64(**{**m.__dict__, "canonical_input_hash": "0" * 64}), _payload())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SearchMask64(**{**m.__dict__, "mask_value": -1})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SearchMask64(**{**m.__dict__, "mask_hex": "f" * 16})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        type(r)(**{**r.__dict__, "reduction_hash": "0" * 64})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        type(r)(**{**r.__dict__, "receipt_hash": "0" * 64})


def test_collision_and_compatibility():
    p = _payload()
    m1 = build_search_mask64("m1", "GENERIC", "s1", p)
    m2 = SearchMask64(**{**m1.__dict__, "mask_id": "m2", "source_id": "s2", "mask_hash": ""})
    m2 = SearchMask64(**{**m2.__dict__, "mask_hash": m2.stable_hash()})
    c1 = build_mask_collision_receipt("c1", [m1])
    assert c1.collision_status == "NO_COLLISION"
    c2 = build_mask_collision_receipt("c2", [m1, m2])
    assert c2.collision_status == "INVALID_COLLISION"
    c3 = build_mask_collision_receipt("c3", [m1, m2], equivalent_identity_hash="a" * 64)
    assert c3.collision_status == "KNOWN_EQUIVALENT_COLLISION"
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        MaskCollisionReceipt(**{**c3.__dict__, "participant_mask_hashes": tuple(sorted((m1.mask_hash, m1.mask_hash)))})
    r1 = build_mask_reduction_receipt("r1", m1, p)
    rc = build_mask_compatibility_receipt("ok", [r1], [c1])
    assert rc.compatibility_status == "MASK_COMPATIBLE"
    rw = build_mask_compatibility_receipt("known", [r1], [c3])
    assert rw.compatibility_status == "MASK_COMPATIBLE_WITH_KNOWN_COLLISIONS"
    rb = build_mask_compatibility_receipt("bad", [r1], [c2])
    assert rb.compatibility_status == "MASK_INCOMPATIBLE"
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_mask_compatibility_receipt(type(rb)(**{**rb.__dict__, "compatibility_hash": "0" * 64}), [r1], [c2])


def test_validation_bounds_json_scope():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_search_mask64("", "GENERIC", "s", _payload())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_search_mask64("m", "BAD", "s", _payload())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_search_mask64("m", "GENERIC", "s", {"x": float("nan")})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_search_mask64("m", "GENERIC", "s", {str(i): i for i in range(129)})
    m = build_search_mask64("m", "GENERIC", "s", _payload())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SearchMask64(**{**m.__dict__, "canonical_input_hash": "z" * 64})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SearchMask64(**{**m.__dict__, "mask_value": UINT64_MAX + 1})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SearchMask64(**{**m.__dict__, "mask_hex": "x" * 16})
    base = build_mask_collision_receipt("c", [m])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        MaskCollisionReceipt(**{**base.__dict__, "participant_source_ids": tuple(str(i) for i in range(MAX_COLLISION_RECORDS + 1)), "participant_mask_hashes": tuple("a" * 64 for _ in range(MAX_COLLISION_RECORDS + 1)), "collision_status": "INVALID_COLLISION", "equivalent_identity_hash": None, "collision_hash": "", "receipt_hash": ""})

    d = m.to_dict()
    json.dumps(d, sort_keys=True)
    d["mask_id"] = "changed"
    assert m.mask_id == "m"
    for name in ("HilberShiftSpec", "HilbertShiftSpec", "ShiftProjectionReceipt", "ReadoutShell", "ReadoutShellStack", "ReadoutCombinationMatrix", "MarkovBasisReceipt"):
        from qec.analysis import search_mask64_contract as sm

        assert not hasattr(sm, name)
    for cls in (sm.SearchMask64, sm.MaskReductionReceipt, sm.MaskCollisionReceipt, sm.MaskCompatibilityReceipt):
        for attr in ("apply", "execute", "run", "traverse", "pathfind", "resolve", "project", "readout", "shift", "hilber", "hilbert", "shell", "matrix", "markov"):
            assert not hasattr(cls, attr)


def test_version_mismatch_and_tampered_compatibility():
    p = _payload()
    m = build_search_mask64("m", "GENERIC", "s", p)
    r1 = build_mask_reduction_receipt("r1", m, p)
    c1 = build_mask_collision_receipt("c1", [m])

    # Build a collision receipt with a different mask_version directly
    c_diff_payload = {
        "collision_id": "cx",
        "mask_version": "v_other",
        "mask_value": m.mask_value,
        "mask_hex": m.mask_hex,
        "participant_mask_hashes": [m.mask_hash],
        "participant_source_ids": [m.source_id],
        "collision_status": "NO_COLLISION",
        "collision_reason": "SINGLE_PARTICIPANT",
        "equivalent_identity_hash": None,
    }
    collision_hash = sha256_hex(c_diff_payload)
    c_diff_receipt_payload = dict(c_diff_payload, collision_hash=collision_hash)
    c_diff_receipt_hash = sha256_hex(c_diff_receipt_payload)
    c_diff = MaskCollisionReceipt(**dict(c_diff_receipt_payload, receipt_hash=c_diff_receipt_hash))

    # r1 version == SEARCH_MASK64_VERSION, c_diff version == "v_other" → cross-version mismatch
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_mask_compatibility_receipt("test_ver", [r1], [c_diff])

    # Build a valid compatibility receipt (status = MASK_COMPATIBLE)
    rc = build_mask_compatibility_receipt("ok2", [r1], [c1])
    assert rc.compatibility_status == "MASK_COMPATIBLE"

    # Tamper the derived fields (status + reason) but recompute hashes consistently
    tampered_payload = {
        "compatibility_id": rc.compatibility_id,
        "mask_version": rc.mask_version,
        "mask_reduction_receipt_hashes": list(rc.mask_reduction_receipt_hashes),
        "collision_receipt_hashes": list(rc.collision_receipt_hashes),
        "mask_count": rc.mask_count,
        "collision_count": rc.collision_count,
        "compatibility_status": "MASK_INCOMPATIBLE",
        "compatibility_reason": "INVALID_COLLISIONS_PRESENT",
    }
    tampered_compatibility_hash = sha256_hex(tampered_payload)
    tampered_receipt_payload = dict(tampered_payload, compatibility_hash=tampered_compatibility_hash)
    tampered_receipt_hash = sha256_hex(tampered_receipt_payload)
    tampered_rc = MaskCompatibilityReceipt(**dict(tampered_receipt_payload, receipt_hash=tampered_receipt_hash))
    # validate must reject despite hash chain being self-consistent
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_mask_compatibility_receipt(tampered_rc, [r1], [c1])
