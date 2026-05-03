from __future__ import annotations

import json

import pytest

from qec.analysis.qam_compatibility_profile import (
    MAX_QAM_FEATURES,
    MAX_QAM_LINEAGE_REFERENCES,
    QAMCompatibilityProfile,
    QAMCompatibilityValidationReceipt,
    QAMSpecReceipt,
    build_qam_compatibility_profile,
    build_qam_compatibility_validation_receipt,
    build_qam_spec_receipt,
    validate_qam_compatibility_validation_receipt,
)


def _sample_features():
    return (
        {"feature_id": "f2", "feature_name": "readout shell", "qec_target": "readout", "compatibility_status": "DEFERRED", "rationale": "later"},
        {"feature_id": "f1", "feature_name": "router lineage", "qec_target": "router", "compatibility_status": "SUPPORTED_METADATA_ONLY", "rationale": "metadata"},
    )


def _sample_lineage():
    return (
        {"reference_id": "r2", "reference_uri": "u2"},
        {"reference_id": "r1", "reference_uri": "u1"},
    )


def _sample_notes():
    return ({"note": "b"}, {"note": "a"})


def test_determinism_and_permutations():
    p1 = build_qam_compatibility_profile("p", compatibility_features=_sample_features(), lineage_references=_sample_lineage(), compatibility_notes=_sample_notes())
    p2 = build_qam_compatibility_profile("p", compatibility_features=tuple(reversed(_sample_features())), lineage_references=tuple(reversed(_sample_lineage())), compatibility_notes=tuple(reversed(_sample_notes())))
    assert p1.profile_hash == p2.profile_hash
    s1 = build_qam_spec_receipt(p1)
    s2 = build_qam_spec_receipt(p2)
    assert (s1.spec_hash, s1.receipt_hash) == (s2.spec_hash, s2.receipt_hash)
    v1 = build_qam_compatibility_validation_receipt(p1, s1)
    v2 = build_qam_compatibility_validation_receipt(p2, s2)
    assert (v1.validation_hash, v1.receipt_hash) == (v2.validation_hash, v2.receipt_hash)


def test_validation_failures_and_bounds():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_qam_compatibility_profile("p", compatibility_features=({"feature_id": "x", "compatibility_status": "SUPPORTED_METADATA_ONLY"}, {"feature_id": "x", "compatibility_status": "DEFERRED"}))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_qam_compatibility_profile("p", lineage_references=({"reference_id": "r"}, {"reference_id": "r"}))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_qam_compatibility_profile("p", compatibility_features=({"feature_id": "x", "compatibility_status": "BAD"},))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_qam_compatibility_profile("p", compatibility_features=tuple({"feature_id": f"f{i}", "compatibility_status": "SUPPORTED_METADATA_ONLY"} for i in range(MAX_QAM_FEATURES + 1)))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_qam_compatibility_profile("p", lineage_references=tuple({"reference_id": f"r{i}"} for i in range(MAX_QAM_LINEAGE_REFERENCES + 1)))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_qam_compatibility_profile("p", compatibility_notes=({"x": set([1])},))


def test_tamper_detection():
    p = build_qam_compatibility_profile("p", compatibility_features=_sample_features())
    s = build_qam_spec_receipt(p)
    v = build_qam_compatibility_validation_receipt(p, s)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        badp = QAMCompatibilityProfile(**{**p.__dict__, "profile_hash": "x"})
        build_qam_spec_receipt(badp)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        bads = QAMSpecReceipt(**{**s.__dict__, "spec_hash": "x"})
        build_qam_compatibility_validation_receipt(p, bads)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        badv = QAMCompatibilityValidationReceipt(**{**v.__dict__, "validation_hash": "x"})
        validate_qam_compatibility_validation_receipt(badv, p, s)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        badv = QAMCompatibilityValidationReceipt(**{**v.__dict__, "receipt_hash": "x"})
        validate_qam_compatibility_validation_receipt(badv, p, s)


def test_classification_and_bucket_rules():
    p_ok = build_qam_compatibility_profile("p1", compatibility_features=({"feature_id": "a", "compatibility_status": "SUPPORTED_METADATA_ONLY"},))
    v_ok = build_qam_compatibility_validation_receipt(p_ok, build_qam_spec_receipt(p_ok))
    assert (v_ok.validation_status, v_ok.validation_reason) == ("VALIDATED_METADATA_ONLY", "ALL_FEATURES_METADATA_ONLY")

    p_deferred = build_qam_compatibility_profile("p2", compatibility_features=({"feature_id": "a", "compatibility_status": "DEFERRED"}, {"feature_id": "b", "compatibility_status": "UNSUPPORTED"}))
    v_def = build_qam_compatibility_validation_receipt(p_deferred, build_qam_spec_receipt(p_deferred))
    assert (v_def.validation_status, v_def.validation_reason) == ("VALIDATED_WITH_DEFERRED_FEATURES", "DEFERRED_OR_UNSUPPORTED_FEATURES_PRESENT")

    p_rej = build_qam_compatibility_profile("p3", compatibility_features=({"feature_id": "a", "compatibility_status": "REJECTED"}, {"feature_id": "b", "compatibility_status": "SUPPORTED_METADATA_ONLY"}))
    v_rej = build_qam_compatibility_validation_receipt(p_rej, build_qam_spec_receipt(p_rej))
    assert (v_rej.validation_status, v_rej.validation_reason) == ("REJECTED", "REJECTED_FEATURES_PRESENT")
    buckets = [set(v_rej.accepted_feature_ids), set(v_rej.deferred_feature_ids), set(v_rej.unsupported_feature_ids), set(v_rej.rejected_feature_ids)]
    assert sum(len(x) for x in buckets) == len(set().union(*buckets))


def test_json_immutability_and_to_dict_copy():
    p = build_qam_compatibility_profile("p", compatibility_features=({"feature_id": "f1", "compatibility_status": "SUPPORTED_METADATA_ONLY", "meta": {"k": [1, 2]}},))
    with pytest.raises(TypeError):
        p.compatibility_features[0]["meta"] = {}
    dumped = json.dumps(p.to_dict(), sort_keys=True)
    assert dumped
    out = p.to_dict()
    out["compatibility_features"][0]["meta"]["k"].append(3)
    assert p.to_dict()["compatibility_features"][0]["meta"]["k"] == [1, 2]


def test_scope_guards_and_no_v1535_exports():
    import qec.analysis.qam_compatibility_profile as mod

    for absent in (
        "SearchMask64", "MaskReductionReceipt", "MaskCollisionReceipt", "HilberShiftSpec", "HilbertShiftSpec", "ShiftProjectionReceipt", "ReadoutShell", "ReadoutShellStack", "ReadoutCombinationMatrix", "MarkovBasisReceipt",
    ):
        assert not hasattr(mod, absent)
    forbidden = {"apply", "execute", "run", "traverse", "pathfind", "resolve", "project", "readout", "search", "mask", "reduce", "collide", "shift", "hilber", "hilbert", "shell", "matrix", "markov"}
    for cls in (mod.QAMCompatibilityProfile, mod.QAMSpecReceipt, mod.QAMCompatibilityValidationReceipt):
        for name in forbidden:
            assert not hasattr(cls, name)


def test_notes_mixed_type_sorting_stability():
    # int 1 and str "1" share the same str() output but have distinct canonical_json
    # representations, so ordering must be stable regardless of input permutation.
    p1 = build_qam_compatibility_profile("p", compatibility_notes=(1, "1"))
    p2 = build_qam_compatibility_profile("p", compatibility_notes=("1", 1))
    assert p1.profile_hash == p2.profile_hash


def test_forged_spec_receipt_identity_rejected():
    from qec.analysis.canonical_hashing import sha256_hex as _sha256

    p = build_qam_compatibility_profile(
        "profile_A",
        source_name="Legit Source",
        compatibility_features=({"feature_id": "f1", "compatibility_status": "SUPPORTED_METADATA_ONLY"},),
    )
    s = build_qam_spec_receipt(p)
    # Forge a spec receipt that reuses profile_hash but changes source_name.
    # It is internally self-consistent so QAMSpecReceipt.__post_init__ accepts it.
    forged_spec_payload = {
        "profile_id": s.profile_id,
        "profile_version": s.profile_version,
        "qam_version": s.qam_version,
        "source_name": "Forged Source",
        "source_orcid": s.source_orcid,
        "profile_hash": s.profile_hash,
    }
    forged_spec_hash = _sha256(forged_spec_payload)
    forged_receipt_hash = _sha256({**forged_spec_payload, "spec_hash": forged_spec_hash})
    forged = QAMSpecReceipt(
        profile_id=s.profile_id,
        profile_version=s.profile_version,
        qam_version=s.qam_version,
        source_name="Forged Source",
        source_orcid=s.source_orcid,
        profile_hash=s.profile_hash,
        spec_hash=forged_spec_hash,
        receipt_hash=forged_receipt_hash,
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_qam_compatibility_validation_receipt(p, forged)



def test_metadata_boundary_no_dependency_on_qam_source_orcid():
    p1 = build_qam_compatibility_profile("p", source_orcid="https://orcid.org/0000-0000-0000-0001", compatibility_features=({"feature_id": "a", "compatibility_status": "SUPPORTED_METADATA_ONLY"},))
    p2 = build_qam_compatibility_profile("p", source_orcid="https://orcid.org/0000-0000-0000-0002", compatibility_features=({"feature_id": "a", "compatibility_status": "SUPPORTED_METADATA_ONLY"},))
    assert p1.profile_hash != p2.profile_hash
    for module_name in (
        "qec.analysis.atomic_semantic_lattice_contract",
        "qec.analysis.router_lattice_paths",
        "qec.analysis.readout_projection_receipts",
        "qec.analysis.layered_lattice_projection_receipts",
    ):
        module = __import__(module_name, fromlist=["*"])
        assert not hasattr(module, "source_orcid")
