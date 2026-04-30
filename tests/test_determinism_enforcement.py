import json
from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.determinism_enforcement import (
    ExtractionDeterminismReceipt,
    ExtractionDeterminismSnapshot,
    RESRAGDeterminismReceipt,
    RESRAGDeterminismSnapshot,
    run_extraction_determinism_enforcement,
    run_res_rag_determinism_enforcement,
)


def h(c: str) -> str: return c * 64


def mk_ex(**updates):
    p = {"snapshot_id": "s1", "raw_bytes_hash": h("a"), "extraction_config_hash": h("b"), "schema_hash": h("c"), "query_fields": ("f1", "f2"), "extracted_field_names": ("f1", "f2"), "locale_hash": h("d"), "backend_config_hash": h("e"), "canonicalization_rules_hash": h("f"), "extraction_hash": h("1"), "canonical_hash": h("2")}
    p.update(updates)
    p["snapshot_hash"] = sha256_hex({k: v for k, v in p.items() if k != "snapshot_hash"})
    return ExtractionDeterminismSnapshot(**p)


def mk_rs(**updates):
    p = {"snapshot_id": "r1", "canonical_hash": h("2"), "res_hash": h("3"), "rag_hash": h("4"), "semantic_field_hash": h("5"), "res_rag_mapping_hash": h("6"), "governance_context_hash": h("7"), "resonance_classifier_hash": h("8"), "tolerance_hash": h("9"), "resonance_receipt_hash": h("a")}
    p.update(updates)
    p["snapshot_hash"] = sha256_hex({k: v for k, v in p.items() if k != "snapshot_hash"})
    return RESRAGDeterminismSnapshot(**p)


def test_extraction_validated_and_hash():
    r = run_extraction_determinism_enforcement(mk_ex(), mk_ex())
    assert r.status == "EXTRACTION_DETERMINISM_VALIDATED" and r.result_count == 0 and r.stable_hash == r.computed_stable_hash()


def test_extraction_drifts_all_types():
    b = mk_ex()
    r1 = run_extraction_determinism_enforcement(b, mk_ex(extraction_config_hash=h("0"), schema_hash=h("1"), locale_hash=h("2"), query_fields=("f2", "f1"), extracted_field_names=("f2",), backend_config_hash=h("3"), canonicalization_rules_hash=h("4")))
    r2 = run_extraction_determinism_enforcement(b, mk_ex(extraction_hash=h("5"), canonical_hash=h("6")))
    types = {x.drift_type for x in r1.results} | {x.drift_type for x in r2.results}
    for t in ["CONFIG_DRIFT", "SCHEMA_DRIFT", "LOCALE_DRIFT", "FIELD_DRIFT", "PARTIAL_EXTRACTION", "BACKEND_CONFIG_DRIFT", "CANONICALIZATION_RULE_DRIFT", "BACKEND_INCONSISTENCY", "CANONICAL_OUTPUT_DRIFT"]: assert t in types


def test_resrag_validated_and_hash():
    r = run_res_rag_determinism_enforcement(mk_rs(), mk_rs())
    assert r.status == "RES_RAG_DETERMINISM_VALIDATED" and r.result_count == 0 and r.stable_hash == r.computed_stable_hash()


def test_resrag_drifts_all_types():
    r = run_res_rag_determinism_enforcement(mk_rs(), mk_rs(res_rag_mapping_hash=h("b"), resonance_classifier_hash=h("c"), tolerance_hash=h("d"), semantic_field_hash=h("e"), res_hash=h("f"), rag_hash=h("1"), resonance_receipt_hash=h("2")))
    assert {x.drift_type for x in r.results} == {"RES_RAG_MAPPING_DRIFT", "RESONANCE_CLASSIFIER_DRIFT", "TOLERANCE_DRIFT", "SEMANTIC_FIELD_DRIFT", "RES_STATE_DRIFT", "RAG_STATE_DRIFT", "RESONANCE_OUTPUT_DRIFT"}


def test_invalid_comparison_raises():
    with pytest.raises(ValueError, match="^INVALID_INPUT$"): run_extraction_determinism_enforcement(mk_ex(), mk_ex(raw_bytes_hash=h("b")))
    with pytest.raises(ValueError, match="^INVALID_INPUT$"): run_res_rag_determinism_enforcement(mk_rs(), mk_rs(canonical_hash=h("b")))


def test_determinism_and_receipt_integrity():
    b, o = mk_ex(), mk_ex(schema_hash=h("0"), extraction_config_hash=h("1"))
    r1 = run_extraction_determinism_enforcement(b, o); r2 = run_extraction_determinism_enforcement(b, o)
    assert r1.stable_hash == r2.stable_hash
    assert r1.results == tuple(sorted(r1.results, key=lambda x: (x.drift_type, x.case_id, x.result_hash)))
    with pytest.raises(ValueError, match="^INVALID_INPUT$"): ExtractionDeterminismReceipt(**{**r1.to_dict(), "status": "BAD", "stable_hash": r1.stable_hash})
    with pytest.raises(ValueError, match="^INVALID_INPUT$"): ExtractionDeterminismReceipt(**{**r1.to_dict(), "result_count": 0, "stable_hash": r1.stable_hash})
    with pytest.raises(ValueError, match="^INVALID_INPUT$"): ExtractionDeterminismReceipt(**{**r1.to_dict(), "stable_hash": h("f")})


def test_immutability_and_json_serializable():
    b = mk_ex(); rb = mk_rs()
    with pytest.raises(FrozenInstanceError): b.snapshot_id = "x"
    json.dumps(b.to_dict()); json.dumps(rb.to_dict())


def test_unsorted_results_raise():
    r = run_res_rag_determinism_enforcement(mk_rs(), mk_rs(res_hash=h("f"), rag_hash=h("1")))
    bad = tuple(reversed(r.results))
    with pytest.raises(ValueError, match="^INVALID_INPUT$"): RESRAGDeterminismReceipt(**{**r.to_dict(), "results": bad, "stable_hash": r.stable_hash})
