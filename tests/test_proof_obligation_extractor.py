from __future__ import annotations

import pytest

from qec.analysis.proof_obligation_extractor import (
    PROOF_OBLIGATION_SCHEMA_VERSION,
    ProofObligation,
    _build_report,
    build_proof_obligation_receipt,
    compile_proof_obligation_report,
    extract_proof_obligations,
    normalize_proof_obligation_input,
    stable_proof_obligation_hash,
    validate_proof_obligation_report,
)


def _base_raw_input() -> dict[str, object]:
    return {
        "claim_id": "claim_1",
        "claim_text": "bounded error holds",
        "experiment_hash": "exp_hash_1",
        "evidence_graph_hash": "graph_hash_1",
        "audit_hash": "audit_hash_1",
        "verdict": "supported",
        "measurement_ids": ["m_obs_2", "m_obs_1"],
        "criterion_ids": ["crit_b", "crit_a"],
        "finding_ids": ["finding_conflict"],
        "expected_relations": {
            "rel_1": {"node_ids": ["n_result", "n_claim"]},
            "rel_2": {"edge": "evidence_edge"},
        },
        "provenance": {"source": "unit-test", "rank": 1},
        "schema_version": PROOF_OBLIGATION_SCHEMA_VERSION,
    }


def test_measurement_and_criterion_obligations_created() -> None:
    normalized = normalize_proof_obligation_input(_base_raw_input())
    obligations = extract_proof_obligations(
        normalized,
        available_measurements=("m_obs_1", "m_obs_2"),
        available_criteria=("crit_a", "crit_b"),
    )
    measurement_types = [o for o in obligations if o.obligation_type == "measurement_availability"]
    criterion_types = [o for o in obligations if o.obligation_type == "criterion_satisfaction"]
    assert len(measurement_types) == 2
    assert len(criterion_types) == 2
    assert all(o.status == "satisfied" for o in measurement_types)
    assert all(o.status == "satisfied" for o in criterion_types)


def test_missing_measurement_is_unsatisfied_and_blocking() -> None:
    normalized = normalize_proof_obligation_input(_base_raw_input())
    obligations = extract_proof_obligations(normalized, available_measurements=("m_obs_1",))
    missing = [o for o in obligations if o.obligation_type == "measurement_availability" and o.related_measurement_id == "m_obs_2"][0]
    assert missing.status == "unsatisfied"
    assert missing.blocking is True


def test_lineage_and_evidence_presence_obligations() -> None:
    normalized = normalize_proof_obligation_input(_base_raw_input())
    obligations = extract_proof_obligations(
        normalized,
        evidence_graph={"nodes": ["n_claim", "n_result"]},
    )
    lineage = [o for o in obligations if o.obligation_type == "lineage_integrity"]
    evidence_presence = [o for o in obligations if o.obligation_type == "evidence_presence"]
    assert len(lineage) == 1
    assert lineage[0].status == "satisfied"
    assert lineage[0].requirement_text == "lineage nodes n_claim,n_result must be linked"
    assert len(evidence_presence) == 1
    # evidence_graph provided but relation has no node_ids to verify → required, not blocking
    assert evidence_presence[0].status == "required"
    assert evidence_presence[0].blocking is False


def test_evidence_presence_unsatisfied_blocking_when_graph_absent() -> None:
    normalized = normalize_proof_obligation_input(_base_raw_input())
    # No evidence_graph provided → evidence_presence must be unsatisfied + blocking
    obligations = extract_proof_obligations(normalized)
    evidence_presence = [o for o in obligations if o.obligation_type == "evidence_presence"]
    assert len(evidence_presence) == 1
    assert evidence_presence[0].status == "unsatisfied"
    assert evidence_presence[0].blocking is True


def test_conflict_and_replay_obligations_created() -> None:
    raw = _base_raw_input()
    raw["verdict"] = "contradicted"
    normalized = normalize_proof_obligation_input(raw)
    obligations = extract_proof_obligations(
        normalized,
        available_findings=("finding_conflict",),
        replay_report={"deterministic_pass": False},
    )
    conflict = [o for o in obligations if o.obligation_type == "conflict_absence"][0]
    replay = [o for o in obligations if o.obligation_type == "replay_consistency"][0]
    assert conflict.status == "unsatisfied"
    assert conflict.blocking is True
    assert replay.status == "unsatisfied"
    assert replay.blocking is True


def test_conflict_absence_satisfied_when_no_conflicts() -> None:
    # verdict = supported and no conflicting findings → conflict_absence always emitted as satisfied
    normalized = normalize_proof_obligation_input(_base_raw_input())
    obligations = extract_proof_obligations(normalized)
    conflict = [o for o in obligations if o.obligation_type == "conflict_absence"]
    assert len(conflict) == 1
    assert conflict[0].status == "satisfied"
    assert conflict[0].blocking is False


def test_deterministic_hash_and_canonical_bytes_stable() -> None:
    raw = _base_raw_input()
    _, report_a, receipt_a = compile_proof_obligation_report(
        raw,
        available_findings=("finding_conflict",),
        available_measurements=("m_obs_1", "m_obs_2"),
        available_criteria=("crit_a", "crit_b"),
        evidence_graph={"nodes": ["n_claim", "n_result"]},
        replay_report={"deterministic_pass": True},
    )
    _, report_b, receipt_b = compile_proof_obligation_report(
        raw,
        available_findings=("finding_conflict",),
        available_measurements=("m_obs_2", "m_obs_1"),
        available_criteria=("crit_b", "crit_a"),
        evidence_graph={"nodes": ["n_result", "n_claim"]},
        replay_report={"deterministic_pass": True},
    )
    assert report_a.to_canonical_bytes() == report_b.to_canonical_bytes()
    assert stable_proof_obligation_hash(report_a) == stable_proof_obligation_hash(report_b)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_defensive_copy_behavior_for_input_dataclass() -> None:
    normalized = normalize_proof_obligation_input(_base_raw_input())
    copied = normalize_proof_obligation_input(normalized)
    assert copied is not normalized
    assert copied.to_canonical_bytes() == normalized.to_canonical_bytes()


def test_unsupported_schema_version_rejected() -> None:
    raw = _base_raw_input()
    raw["schema_version"] = "v0"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        normalize_proof_obligation_input(raw)


def test_unsupported_obligation_type_and_status_rejected() -> None:
    valid_input = normalize_proof_obligation_input(_base_raw_input())
    invalid_obligation = ProofObligation(
        obligation_id="id1",
        obligation_type="invalid",
        claim_id=valid_input.claim_id,
        related_measurement_id="",
        related_criterion_id="",
        related_finding_id="",
        related_node_ids=(),
        requirement_text="x",
        blocking=False,
        status="required",
    )
    with pytest.raises(ValueError, match="unsupported obligation_type"):
        _build_report(valid_input.claim_id, valid_input.verdict, (invalid_obligation,), valid_input.schema_version)

    invalid_status = ProofObligation(
        obligation_id="id2",
        obligation_type="measurement_availability",
        claim_id=valid_input.claim_id,
        related_measurement_id="m1",
        related_criterion_id="",
        related_finding_id="",
        related_node_ids=(),
        requirement_text="measurement m1 must be available",
        blocking=False,
        status="bad",
    )
    with pytest.raises(ValueError, match="unsupported obligation status"):
        _build_report(valid_input.claim_id, valid_input.verdict, (invalid_status,), valid_input.schema_version)


def test_duplicate_obligation_id_rejected() -> None:
    valid_input = normalize_proof_obligation_input(_base_raw_input())
    obligation_a = ProofObligation(
        obligation_id="dup",
        obligation_type="measurement_availability",
        claim_id=valid_input.claim_id,
        related_measurement_id="m_obs_1",
        related_criterion_id="",
        related_finding_id="",
        related_node_ids=(),
        requirement_text="measurement m_obs_1 must be available",
        blocking=False,
        status="satisfied",
    )
    obligation_b = ProofObligation(
        obligation_id="dup",
        obligation_type="criterion_satisfaction",
        claim_id=valid_input.claim_id,
        related_measurement_id="",
        related_criterion_id="crit_a",
        related_finding_id="",
        related_node_ids=(),
        requirement_text="criterion crit_a must be satisfied",
        blocking=False,
        status="satisfied",
    )
    with pytest.raises(ValueError, match="duplicate obligation IDs"):
        _build_report(valid_input.claim_id, valid_input.verdict, (obligation_a, obligation_b), valid_input.schema_version)


def test_receipt_stability() -> None:
    _, report, _ = compile_proof_obligation_report(_base_raw_input())
    receipt_a = build_proof_obligation_receipt(report)
    receipt_b = build_proof_obligation_receipt(report)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_validate_proof_obligation_report_passes_on_valid_report() -> None:
    _, report, _ = compile_proof_obligation_report(
        _base_raw_input(),
        available_measurements=("m_obs_1", "m_obs_2"),
        available_criteria=("crit_a", "crit_b"),
    )
    # Should not raise
    validate_proof_obligation_report(report)


def test_evidence_graph_legacy_node_ids_key_accepted() -> None:
    # Backward-compatible: "node_ids" key must still be accepted
    normalized = normalize_proof_obligation_input(_base_raw_input())
    obligations = extract_proof_obligations(
        normalized,
        evidence_graph={"node_ids": ["n_claim", "n_result"]},
    )
    lineage = [o for o in obligations if o.obligation_type == "lineage_integrity"]
    assert len(lineage) == 1
    assert lineage[0].status == "satisfied"
