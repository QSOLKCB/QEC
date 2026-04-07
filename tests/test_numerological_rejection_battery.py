from __future__ import annotations

import dataclasses

import pytest

from qec.analysis.numerological_rejection_battery import (
    NUMEROLOGICAL_REJECTION_BATTERY_SCHEMA_VERSION,
    RejectionDecision,
    RejectionFinding,
    compile_rejection_battery,
    normalize_rejection_battery_input,
    run_rejection_battery,
    stable_rejection_battery_hash,
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
