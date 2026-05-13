from dataclasses import FrozenInstanceError, replace
import pytest

from qec.analysis.substrate_constraint_contract import build_substrate_constraint_predicate, build_substrate_contract
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.substrate_state_receipt import build_substrate_state_receipt
from qec.analysis.material_encoding_receipt import build_material_encoding_receipt, build_substrate_drift_receipt
from qec.analysis.per_subsystem_substrate_receipts import (
    build_layer_substrate_compatibility_receipt,
    build_mask_substrate_receipt,
    build_router_substrate_receipt,
    build_readout_substrate_receipt,
    validate_layer_substrate_compatibility_receipt,
    validate_mask_substrate_receipt,
    validate_router_substrate_receipt,
    validate_readout_substrate_receipt,
    validate_layer_substrate_compatibility_receipt_with_artifacts,
    validate_mask_substrate_receipt_with_artifacts,
    validate_router_substrate_receipt_with_artifacts,
    validate_readout_substrate_receipt_with_artifacts,
    _classify_encoding_entry_subsystem,
)


def _mk_state(src: dict):
    cjson = '{"layer":{"ok":1},"mask":{"ok":2},"readout":{"ok":3},"router":{"ok":4}}'
    if src:
        import json
        cjson = json.dumps(src, sort_keys=True, separators=(",", ":"))
    preds = [
        build_substrate_constraint_predicate("LayerToken", "FIELD_PRESENT", ("layer",), {}),
        build_substrate_constraint_predicate("MaskToken", "FIELD_PRESENT", ("mask",), {}),
        build_substrate_constraint_predicate("RouterToken", "FIELD_PRESENT", ("router",), {}),
        build_substrate_constraint_predicate("ReadoutToken", "FIELD_PRESENT", ("readout",), {}),
        build_substrate_constraint_predicate("UnknownToken", "FIELD_PRESENT", ("unknown",), {}),
    ]
    contract = build_substrate_contract("TEST", sha256_hex({"canonical_json": cjson}), "S1", tuple(preds))
    return build_substrate_state_receipt(contract, cjson)


def _mk_state_with_contract(src: dict, contract):
    """Create a substrate state receipt using an existing contract."""
    import json
    cjson = json.dumps(src, sort_keys=True, separators=(",", ":"))
    return build_substrate_state_receipt(contract, cjson)


def _artifacts(observed_same: bool = True):
    ssr = _mk_state({"layer": {"ok": 1}, "mask": {"ok": 2}, "router": {"ok": 4}, "readout": {"ok": 3}})
    mer = build_material_encoding_receipt(ssr)
    obs = mer if observed_same else None
    drift = build_substrate_drift_receipt(mer, obs)
    return ssr, mer, drift


def test_basic_determinism_and_frozen_and_empty():
    ssr, mer, drift = _artifacts()
    a1 = build_layer_substrate_compatibility_receipt(ssr, mer, drift)
    a2 = build_layer_substrate_compatibility_receipt(ssr, mer, drift)
    assert a1.layer_substrate_compatibility_receipt_hash == a2.layer_substrate_compatibility_receipt_hash
    assert a1.to_canonical_json() == a2.to_canonical_json()
    assert a1.to_canonical_bytes() == a2.to_canonical_bytes()
    with pytest.raises(FrozenInstanceError):
        a1.subsystem_label = "X"


def test_all_builders_and_complete_validators():
    ssr, mer, drift = _artifacts()
    l = build_layer_substrate_compatibility_receipt(ssr, mer, drift)
    m = build_mask_substrate_receipt(ssr, mer, drift)
    r = build_router_substrate_receipt(ssr, mer, drift)
    o = build_readout_substrate_receipt(ssr, mer, drift)
    assert validate_layer_substrate_compatibility_receipt_with_artifacts(l, ssr, mer, drift)
    assert validate_mask_substrate_receipt_with_artifacts(m, ssr, mer, drift)
    assert validate_router_substrate_receipt_with_artifacts(r, ssr, mer, drift)
    assert validate_readout_substrate_receipt_with_artifacts(o, ssr, mer, drift)


def test_uppercase_hash_and_wrong_hash_failures():
    ssr, mer, drift = _artifacts()
    l = build_layer_substrate_compatibility_receipt(ssr, mer, drift)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_layer_substrate_compatibility_receipt(replace(l, layer_substrate_compatibility_receipt_hash=l.layer_substrate_compatibility_receipt_hash.upper()))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_layer_substrate_compatibility_receipt(replace(l, layer_substrate_compatibility_receipt_hash="0" * 64))


def test_classification_and_ambiguity():
    ssr, mer, _ = _artifacts()
    e = mer.encoding_entries[0]
    r = ssr.predicate_evaluation_results[0]
    assert _classify_encoding_entry_subsystem(e, r, ssr.substrate_profile_id) == "LAYER"
    # Use a substrate_profile_id that is an exact token for MASK subsystem
    # while the predicate_id "LayerToken" matches LAYER, triggering ambiguity
    with pytest.raises(ValueError, match="SUBSYSTEM_CLASSIFICATION_AMBIGUOUS"):
        _classify_encoding_entry_subsystem(e, r, "Mask")


def test_invalid_counts_and_cross_boundary():
    ssr, mer, drift = _artifacts()
    m = build_mask_substrate_receipt(ssr, mer, drift)
    with pytest.raises(ValueError):
        validate_router_substrate_receipt(m)  # type: ignore[arg-type]
    l = build_layer_substrate_compatibility_receipt(ssr, mer, drift)
    with pytest.raises(ValueError, match="SUBSYSTEM_COUNT_MISMATCH"):
        validate_layer_substrate_compatibility_receipt(replace(l, subsystem_entry_count=True))


def test_drift_classes():
    ssr, mer, drift_clean = _artifacts(True)
    l = build_layer_substrate_compatibility_receipt(ssr, mer, drift_clean)
    assert l.subsystem_substrate_drift_class in {"SUBSYSTEM_SUBSTRATE_DRIFT_CLEAN", "SUBSYSTEM_SUBSTRATE_DRIFT_EMPTY"}
    ssr2, mer2, drift_inc = _artifacts(False)
    l2 = build_layer_substrate_compatibility_receipt(ssr2, mer2, drift_inc)
    assert l2.subsystem_substrate_drift_class in {"SUBSYSTEM_SUBSTRATE_DRIFT_INCOMPLETE", "SUBSYSTEM_SUBSTRATE_DRIFT_EMPTY"}


def test_changed_drift_with_different_observed():
    """Test the 'changed' case where observed differs from expected MaterialEncodingReceipt."""
    from qec.analysis.material_encoding_receipt import EncodingEntry
    
    # Create expected state and encoding receipt
    ssr_expected = _mk_state({"layer": {"ok": 1}, "mask": {"ok": 2}, "router": {"ok": 4}, "readout": {"ok": 3}})
    mer_expected = build_material_encoding_receipt(ssr_expected)
    
    # Create a modified observed MaterialEncodingReceipt by changing an encoding entry
    # This simulates drift where the observed state differs from expected
    e0 = mer_expected.encoding_entries[0]
    changed_e0_payload = {**e0.to_dict(), "encoded_status": "ENCODED_FAIL"}
    changed_e0_payload.pop("encoding_entry_hash")
    changed_e0 = EncodingEntry(**changed_e0_payload, encoding_entry_hash=sha256_hex(changed_e0_payload))
    changed_entries = (changed_e0,) + mer_expected.encoding_entries[1:]
    
    # Recompute counts and class
    passed = sum(1 for e in changed_entries if e.encoded_status == "ENCODED_PASS")
    failed = len(changed_entries) - passed
    mclass = "MATERIAL_ENCODING_COMPATIBLE" if failed == 0 else "MATERIAL_ENCODING_INCOMPATIBLE"
    
    changed_payload = {
        "substrate_state_receipt_hash": mer_expected.substrate_state_receipt_hash,
        "substrate_contract_hash": mer_expected.substrate_contract_hash,
        "substrate_profile_id": mer_expected.substrate_profile_id,
        "encoding_mode": mer_expected.encoding_mode,
        "encoding_entries": [x.to_dict() for x in changed_entries],
        "encoding_entry_count": len(changed_entries),
        "passed_encoding_count": passed,
        "failed_encoding_count": failed,
        "material_encoding_class": mclass,
    }
    from qec.analysis.material_encoding_receipt import MaterialEncodingReceipt
    mer_observed = MaterialEncodingReceipt(
        substrate_state_receipt_hash=mer_expected.substrate_state_receipt_hash,
        substrate_contract_hash=mer_expected.substrate_contract_hash,
        substrate_profile_id=mer_expected.substrate_profile_id,
        encoding_mode=mer_expected.encoding_mode,
        encoding_entries=changed_entries,
        encoding_entry_count=len(changed_entries),
        passed_encoding_count=passed,
        failed_encoding_count=failed,
        material_encoding_class=mclass,
        material_encoding_receipt_hash=sha256_hex(changed_payload),
    )
    
    # Verify they have the same contract/profile but different hashes
    assert mer_expected.substrate_contract_hash == mer_observed.substrate_contract_hash
    assert mer_expected.substrate_profile_id == mer_observed.substrate_profile_id
    assert mer_expected.material_encoding_receipt_hash != mer_observed.material_encoding_receipt_hash
    
    # Build drift receipt from expected and observed
    drift = build_substrate_drift_receipt(mer_expected, mer_observed)
    
    # Verify drift receipt has observed hash and shows drift
    assert drift.observed_material_encoding_receipt_hash == mer_observed.material_encoding_receipt_hash
    assert drift.substrate_drift_class == "SUBSTRATE_DRIFT_CHANGED"
    
    # Build per-subsystem receipt with the observed receipt
    l = build_layer_substrate_compatibility_receipt(ssr_expected, mer_expected, drift, mer_observed)
    
    # Validate the receipt with artifacts including observed
    assert validate_layer_substrate_compatibility_receipt_with_artifacts(l, ssr_expected, mer_expected, drift, mer_observed)
    
    # Verify drift class reflects the change (LAYER entry was changed)
    assert l.subsystem_substrate_drift_class in {"SUBSYSTEM_SUBSTRATE_DRIFT_CHANGED", "SUBSYSTEM_SUBSTRATE_DRIFT_EMPTY"}

