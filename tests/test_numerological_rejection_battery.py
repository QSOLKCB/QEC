from __future__ import annotations

import dataclasses

import pytest

import qec.analysis.numerological_rejection_battery as _nrb_mod
from qec.analysis.numerological_rejection_battery import (
    NUMEROLOGICAL_REJECTION_BATTERY_SCHEMA_VERSION,
    RejectionBatteryInput,
    RejectionDecision,
    RejectionFinding,
    build_rejection_battery_receipt,
    compile_rejection_battery,
    normalize_rejection_battery_input,
    run_rejection_battery,
    stable_rejection_battery_hash,
    validate_rejection_battery_input,
)


def _base_raw() -> dict[str, object]:
    return {
        "artifact_id": "artifact-1",
        "claim_id": "claim-1",
        "audit_hash": "a" * 64,
        "proof_report_hash": "b" * 64,
        "symbolic_tokens": ["alpha"],
        "numeric_constants": [],
        "cited_measurement_ids": [],
        "cited_evidence_ids": ["e-1"],
        "cited_criterion_ids": ["c-1"],
        "provenance": {"source": "unit"},
        "schema_version": NUMEROLOGICAL_REJECTION_BATTERY_SCHEMA_VERSION,
    }


def test_frozen_dataclasses() -> None:
    inp = normalize_rejection_battery_input(_base_raw())
    with pytest.raises(dataclasses.FrozenInstanceError):
        inp.artifact_id = "other"  # type: ignore[misc]


def test_symbolic_repetition_detection() -> None:
    raw = _base_raw()
    raw["symbolic_tokens"] = ["sigil", "sigil"]
    raw["cited_evidence_ids"] = []
    raw["cited_criterion_ids"] = []
    decision, findings, _ = compile_rejection_battery(raw)
    assert any(f.rejection_type == "unsupported_symbolic_repetition" for f in findings)
    assert decision.rejection_verdict == "flagged"


def test_numeric_motif_and_missing_measurement_grounding() -> None:
    raw = _base_raw()
    raw["numeric_constants"] = ["3", "3"]
    raw["cited_measurement_ids"] = []
    decision, findings, _ = compile_rejection_battery(raw)
    kinds = {f.rejection_type for f in findings}
    assert "unsupported_numeric_motif" in kinds
    assert "ratio_without_measurement_grounding" in kinds
    assert decision.rejection_verdict == "rejected"


def test_symmetry_without_evidence_detection() -> None:
    raw = _base_raw()
    raw["symbolic_tokens"] = ["symmetry"]
    raw["cited_evidence_ids"] = []
    _, findings, _ = compile_rejection_battery(raw)
    assert any(f.rejection_type == "symmetry_without_evidence" for f in findings)


def test_phenomenology_rejection() -> None:
    raw = _base_raw()
    raw["symbolic_tokens"] = ["phenomenology"]
    raw["cited_evidence_ids"] = []
    decision, findings, _ = compile_rejection_battery(raw)
    assert any(f.rejection_type == "phenomenology_without_artifact" for f in findings)
    assert decision.rejection_verdict == "rejected"


def test_evidence_gap_amplification() -> None:
    raw = _base_raw()
    raw["symbolic_tokens"] = ["motif", "motif"]
    raw["cited_evidence_ids"] = ["missing-evidence"]
    _, findings, _ = compile_rejection_battery(raw, available_evidence=["e-1"])
    assert any(f.rejection_type == "evidence_gap_amplification" for f in findings)


def test_accepted_and_flagged_and_rejected_verdicts() -> None:
    accepted, af, _ = compile_rejection_battery(_base_raw(), available_evidence=["e-1"], available_criteria=["c-1"])
    assert accepted.rejection_verdict == "accepted"
    assert af == ()

    flagged_raw = _base_raw()
    flagged_raw["symbolic_tokens"] = ["symmetry"]
    flagged_raw["cited_evidence_ids"] = []
    flagged, _, _ = compile_rejection_battery(flagged_raw)
    assert flagged.rejection_verdict == "flagged"

    rejected_raw = _base_raw()
    rejected_raw["numeric_constants"] = ["2/3"]
    rejected_raw["cited_measurement_ids"] = []
    rejected, _, _ = compile_rejection_battery(rejected_raw)
    assert rejected.rejection_verdict == "rejected"


def test_canonical_hash_stability_and_ordering_independence() -> None:
    raw_a = _base_raw()
    raw_a["symbolic_tokens"] = ["zeta", "alpha", "zeta"]
    raw_a["numeric_constants"] = ["3", "2", "3"]
    raw_a["cited_evidence_ids"] = []
    raw_a["cited_criterion_ids"] = []

    raw_b = _base_raw()
    raw_b["symbolic_tokens"] = ["zeta", "zeta", "alpha"]
    raw_b["numeric_constants"] = ["3", "3", "2"]
    raw_b["cited_evidence_ids"] = []
    raw_b["cited_criterion_ids"] = []

    d1, f1, r1 = compile_rejection_battery(raw_a)
    d2, f2, r2 = compile_rejection_battery(raw_b)
    assert d1.battery_hash == d2.battery_hash
    assert tuple(x.to_canonical_json() for x in f1) == tuple(x.to_canonical_json() for x in f2)
    assert r1.to_canonical_json() == r2.to_canonical_json()


def test_receipt_stability() -> None:
    raw = _base_raw()
    d1, f1, r1 = compile_rejection_battery(raw)
    d2, f2, r2 = compile_rejection_battery(raw)
    assert d1.to_canonical_json() == d2.to_canonical_json()
    assert tuple(x.to_canonical_json() for x in f1) == tuple(x.to_canonical_json() for x in f2)
    assert r1.to_canonical_json() == r2.to_canonical_json()


def test_schema_rejection() -> None:
    raw = _base_raw()
    raw["schema_version"] = "v0"
    with pytest.raises(ValueError, match="unsupported schema version"):
        compile_rejection_battery(raw)


def test_duplicate_finding_rejection() -> None:
    raw = _base_raw()
    decision, findings, _ = compile_rejection_battery(raw)
    dup = RejectionFinding(
        finding_id="dup",
        rejection_type="unsupported_symbolic_repetition",
        message="x",
        related_token="",
        related_numeric_value="",
        severity="warning",
        blocking=False,
    )
    with pytest.raises(ValueError, match="duplicate finding IDs"):
        stable_rejection_battery_hash(
            RejectionDecision(
                artifact_id=decision.artifact_id,
                rejection_verdict="flagged",
                finding_ids=("dup", "dup"),
                blocking_findings=0,
                rationale_summary="flagged: 2 unsupported symbolic repetition",
                battery_hash="",
                schema_version=NUMEROLOGICAL_REJECTION_BATTERY_SCHEMA_VERSION,
            ),
            (dup, dup),
        )


def test_reject_malformed_numeric_constants() -> None:
    raw = _base_raw()
    raw["numeric_constants"] = [object()]
    with pytest.raises(ValueError, match="malformed numeric constants"):
        normalize_rejection_battery_input(raw)


def test_run_rejection_battery_matches_compile() -> None:
    raw = _base_raw()
    inp = normalize_rejection_battery_input(raw)
    d1, f1 = run_rejection_battery(inp)
    d2, f2, _ = compile_rejection_battery(raw)
    assert d1.to_canonical_json() == d2.to_canonical_json()
    assert tuple(x.to_canonical_json() for x in f1) == tuple(x.to_canonical_json() for x in f2)


# --- P1: empty availability catalog means "nothing available", not "skip check" ---

def test_empty_evidence_catalog_triggers_findings() -> None:
    """available_evidence=[] must mean no evidence available; findings should fire."""
    raw = _base_raw()
    raw["cited_evidence_ids"] = ["e-1"]
    raw["cited_criterion_ids"] = []
    raw["symbolic_tokens"] = ["symmetry"]
    # empty list → no available evidence → has_evidence_linkage must be False
    _, findings, _ = compile_rejection_battery(raw, available_evidence=[])
    assert any(f.rejection_type == "symmetry_without_evidence" for f in findings)


def test_empty_criteria_catalog_triggers_symbolic_repetition() -> None:
    """available_criteria=[] must mean no criteria available."""
    raw = _base_raw()
    raw["symbolic_tokens"] = ["alpha", "alpha"]
    raw["cited_evidence_ids"] = []
    raw["cited_criterion_ids"] = ["c-1"]
    # empty lists → no linkage → repetition finding expected
    _, findings, _ = compile_rejection_battery(raw, available_evidence=[], available_criteria=[])
    assert any(f.rejection_type == "unsupported_symbolic_repetition" for f in findings)


def test_none_availability_skips_catalog_check() -> None:
    """available_evidence=None means catalog unknown; intersection check skipped."""
    raw = _base_raw()
    raw["cited_evidence_ids"] = ["e-1"]
    raw["symbolic_tokens"] = ["symmetry"]
    # None → skip intersection → cited evidence is sufficient → no symmetry finding
    _, findings, _ = compile_rejection_battery(raw, available_evidence=None)
    assert not any(f.rejection_type == "symmetry_without_evidence" for f in findings)


# --- P2: validate_rejection_battery_input enforces required IDs and hash format ---

def test_validate_requires_artifact_id() -> None:
    inp = normalize_rejection_battery_input(_base_raw())
    bad = RejectionBatteryInput(
        artifact_id="",
        claim_id=inp.claim_id,
        audit_hash=inp.audit_hash,
        proof_report_hash=inp.proof_report_hash,
        symbolic_tokens=inp.symbolic_tokens,
        numeric_constants=inp.numeric_constants,
        cited_measurement_ids=inp.cited_measurement_ids,
        cited_evidence_ids=inp.cited_evidence_ids,
        cited_criterion_ids=inp.cited_criterion_ids,
        provenance=inp.provenance,
        schema_version=inp.schema_version,
    )
    with pytest.raises(ValueError, match="artifact_id is required"):
        validate_rejection_battery_input(bad)


def test_validate_requires_valid_audit_hash() -> None:
    inp = normalize_rejection_battery_input(_base_raw())
    bad = RejectionBatteryInput(
        artifact_id=inp.artifact_id,
        claim_id=inp.claim_id,
        audit_hash="not-a-hex-hash",
        proof_report_hash=inp.proof_report_hash,
        symbolic_tokens=inp.symbolic_tokens,
        numeric_constants=inp.numeric_constants,
        cited_measurement_ids=inp.cited_measurement_ids,
        cited_evidence_ids=inp.cited_evidence_ids,
        cited_criterion_ids=inp.cited_criterion_ids,
        provenance=inp.provenance,
        schema_version=inp.schema_version,
    )
    with pytest.raises(ValueError, match="audit_hash must be a 64-character lowercase hex string"):
        validate_rejection_battery_input(bad)


def test_validate_requires_valid_proof_report_hash() -> None:
    inp = normalize_rejection_battery_input(_base_raw())
    bad = RejectionBatteryInput(
        artifact_id=inp.artifact_id,
        claim_id=inp.claim_id,
        audit_hash=inp.audit_hash,
        proof_report_hash="tooshort",
        symbolic_tokens=inp.symbolic_tokens,
        numeric_constants=inp.numeric_constants,
        cited_measurement_ids=inp.cited_measurement_ids,
        cited_evidence_ids=inp.cited_evidence_ids,
        cited_criterion_ids=inp.cited_criterion_ids,
        provenance=inp.provenance,
        schema_version=inp.schema_version,
    )
    with pytest.raises(ValueError, match="proof_report_hash must be a 64-character lowercase hex string"):
        validate_rejection_battery_input(bad)


# --- Provenance duplicate key detection ---

def test_duplicate_provenance_key_raises() -> None:
    # Integer key 1 and string key "1" both normalize to "1" via str().strip()
    with pytest.raises(ValueError, match="duplicate provenance key"):
        _nrb_mod._normalize_provenance({1: "val-a", "1": "val-b"})


# --- Related token/value correctness ---

def test_repeated_token_is_the_duplicated_one() -> None:
    """related_token must point to the actually repeated token, not the sorted-first."""
    raw = _base_raw()
    # alpha is sorted first; zeta is the repeated one
    raw["symbolic_tokens"] = ["zeta", "alpha", "zeta"]
    raw["cited_evidence_ids"] = []
    raw["cited_criterion_ids"] = []
    _, findings, _ = compile_rejection_battery(raw)
    rep_finding = next(f for f in findings if f.rejection_type == "unsupported_symbolic_repetition")
    assert rep_finding.related_token == "zeta"


def test_repeated_numeric_is_the_duplicated_one() -> None:
    """related_numeric_value must point to the actually repeated constant, not sorted-first."""
    raw = _base_raw()
    # "2" is sorted first; "3" is repeated
    raw["numeric_constants"] = ["3", "2", "3"]
    raw["cited_measurement_ids"] = []
    _, findings, _ = compile_rejection_battery(raw)
    motif_finding = next(f for f in findings if f.rejection_type == "unsupported_numeric_motif")
    assert motif_finding.related_numeric_value == "3"


# --- stable_rejection_battery_hash ID consistency ---

def test_hash_rejects_mismatched_finding_ids() -> None:
    raw = _base_raw()
    decision, findings, _ = compile_rejection_battery(raw)
    # Build a decision with finding_ids that don't match the (empty) findings
    mismatched = RejectionDecision(
        artifact_id=decision.artifact_id,
        rejection_verdict="flagged",
        finding_ids=("nonexistent-id",),
        blocking_findings=0,
        rationale_summary="flagged: 1 bogus",
        battery_hash="",
        schema_version=NUMEROLOGICAL_REJECTION_BATTERY_SCHEMA_VERSION,
    )
    with pytest.raises(ValueError, match="finding_ids do not match"):
        stable_rejection_battery_hash(mismatched, ())


# --- build_rejection_battery_receipt hash verification ---

def test_receipt_rejects_tampered_battery_hash() -> None:
    raw = _base_raw()
    decision, findings, _ = compile_rejection_battery(raw)
    tampered = RejectionDecision(
        artifact_id=decision.artifact_id,
        rejection_verdict=decision.rejection_verdict,
        finding_ids=decision.finding_ids,
        blocking_findings=decision.blocking_findings,
        rationale_summary=decision.rationale_summary,
        battery_hash="0" * 64,  # wrong hash
        schema_version=decision.schema_version,
    )
    with pytest.raises(ValueError, match="battery_hash mismatch"):
        build_rejection_battery_receipt(tampered, findings)
