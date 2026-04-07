from __future__ import annotations

import pytest

from qec.analysis.scientific_certification_kernel import (
    SCIENTIFIC_CERTIFICATION_SCHEMA_VERSION,
    CertificationDecision,
    CertificationFinding,
    build_certification_receipt,
    compile_certification_decision,
    stable_certification_hash,
)


def _base_raw() -> dict[str, object]:
    return {
        "artifact_id": "artifact-1",
        "claim_id": "claim-1",
        "audit_hash": "a" * 64,
        "replay_hash": "b" * 64,
        "proof_report_hash": "c" * 64,
        "rejection_battery_hash": "d" * 64,
        "prior_verdicts": [
            {
                "artifact_id": "artifact-0",
                "certification_verdict": "certified",
                "certification_hash": "e" * 64,
            }
        ],
        "blocking_findings": [],
        "provenance": {"source": "unit"},
        "schema_version": SCIENTIFIC_CERTIFICATION_SCHEMA_VERSION,
    }


def test_certified_verdict() -> None:
    decision, findings, _ = compile_certification_decision(_base_raw())
    assert decision.certification_verdict == "certified"
    assert decision.blocking_findings == 0
    assert all(not f.blocking for f in findings)


def test_conditional_certification() -> None:
    raw = _base_raw()
    raw["prior_verdicts"] = [
        {
            "artifact_id": "artifact-0",
            "certification_verdict": "conditionally_certified",
            "certification_hash": "e" * 64,
        }
    ]
    decision, findings, receipt = compile_certification_decision(raw)
    assert decision.certification_verdict == "conditionally_certified"
    assert decision.blocking_findings == 0
    assert any(f.severity == "warning" for f in findings)
    assert receipt.validation_passed is True


def test_rejected_verdict() -> None:
    raw = _base_raw()
    raw["blocking_findings"] = ["proof_unsatisfied"]
    decision, findings, _ = compile_certification_decision(raw)
    assert decision.certification_verdict == "rejected"
    assert any(f.finding_type == "blocking_failure" and f.blocking for f in findings)


def test_blocking_failure_propagation() -> None:
    raw = _base_raw()
    raw["blocking_findings"] = ["a", "b"]
    decision, _, receipt = compile_certification_decision(raw)
    assert decision.blocking_findings == 1
    assert receipt.blocking_findings == 1


def test_missing_prerequisite_handling() -> None:
    raw = _base_raw()
    raw["audit_hash"] = ""
    with pytest.raises(ValueError, match="audit_hash must be non-empty"):
        compile_certification_decision(raw)


def test_canonical_hash_stability() -> None:
    d1, f1, _ = compile_certification_decision(_base_raw())
    d2, f2, _ = compile_certification_decision(_base_raw())
    assert d1.certification_hash == d2.certification_hash
    assert tuple(x.to_canonical_json() for x in f1) == tuple(x.to_canonical_json() for x in f2)


def test_ordering_independence() -> None:
    raw_a = _base_raw()
    raw_b = _base_raw()
    raw_a["prior_verdicts"] = [
        {"artifact_id": "z", "certification_verdict": "certified", "certification_hash": "1" * 64},
        {"artifact_id": "a", "certification_verdict": "rejected", "certification_hash": "2" * 64},
    ]
    raw_b["prior_verdicts"] = list(reversed(raw_a["prior_verdicts"]))
    d1, f1, r1 = compile_certification_decision(raw_a)
    d2, f2, r2 = compile_certification_decision(raw_b)
    assert d1.to_canonical_json() == d2.to_canonical_json()
    assert tuple(x.to_canonical_json() for x in f1) == tuple(x.to_canonical_json() for x in f2)
    assert r1.to_canonical_json() == r2.to_canonical_json()


def test_receipt_stability() -> None:
    decision, findings, _ = compile_certification_decision(_base_raw())
    r1 = build_certification_receipt(decision, findings)
    r2 = build_certification_receipt(decision, findings)
    assert r1.to_canonical_json() == r2.to_canonical_json()


def test_schema_rejection() -> None:
    raw = _base_raw()
    raw["schema_version"] = "v0"
    with pytest.raises(ValueError, match="unsupported schema version"):
        compile_certification_decision(raw)


def test_duplicate_finding_rejection() -> None:
    finding = CertificationFinding(
        finding_id="dup",
        finding_type="audit_pass",
        message="ok",
        blocking=False,
        severity="info",
    )
    decision = CertificationDecision(
        artifact_id="artifact-1",
        certification_verdict="certified",
        finding_ids=("dup", "dup"),
        blocking_findings=0,
        rationale_summary="certified: all prerequisite kernels passed",
        certification_hash="",
        schema_version=SCIENTIFIC_CERTIFICATION_SCHEMA_VERSION,
    )
    with pytest.raises(ValueError, match="duplicate finding IDs"):
        stable_certification_hash(decision, (finding, finding))
