from __future__ import annotations

import dataclasses
import json
from types import MappingProxyType

import pytest

from qec.analysis.adversarial_determinism_battery import run_adversarial_determinism_battery
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.counterfactual_replay_kernel import run_counterfactual_replay
from qec.analysis.cross_environment_replay_kernel import (
    EnvironmentReplayArtifact,
    compare_cross_environment_replay,
)
from qec.analysis.failure_ledger import (
    FailureLedger,
    FailureLedgerEntry,
    FailureLedgerReceipt,
    FailureRecord,
    build_failure_ledger,
)
from qec.analysis.fix_proposal_kernel import generate_fix_proposals
from qec.analysis.fix_validation_kernel import validate_fix_proposals
from qec.analysis.issue_normalization_kernel import normalize_review_issues


def _raw_issue(*, summary: str, severity: str = "HIGH", category: str = "VALIDATION") -> dict[str, str]:
    return {
        "summary": summary,
        "body": summary,
        "source": "MANUAL",
        "severity": severity,
        "category": category,
        "target_path": "src/qec/analysis/sample.py",
        "invariant": "FAIL_FAST_VALIDATION",
    }


def _issue_pipeline(raw_issues: tuple[dict[str, str], ...]):
    issue_receipt = normalize_review_issues(raw_issues)
    proposal_receipt = generate_fix_proposals(issue_receipt)
    validation_receipt = validate_fix_proposals(proposal_receipt)
    counterfactual_receipt = run_counterfactual_replay(validation_receipt)
    adversarial_receipt = run_adversarial_determinism_battery({"issues": raw_issues})
    return issue_receipt, proposal_receipt, validation_receipt, counterfactual_receipt, adversarial_receipt


def _cross_env_receipt(*, mismatch: bool):
    workload_id = "workload-1"
    artifact_hash = sha256_hex({"a": 1})
    payload_hash = sha256_hex({"p": 1})
    receipt_hash = sha256_hex({"r": 1})
    metadata_hash = sha256_hex({"m": 1})

    first = EnvironmentReplayArtifact(
        environment_id="env-a",
        workload_id=workload_id,
        artifact_hash=artifact_hash,
        canonical_payload_hash=payload_hash,
        receipt_hash=receipt_hash,
        platform_label="linux",
        python_label="3.11",
        metadata_hash=metadata_hash,
    )
    second = EnvironmentReplayArtifact(
        environment_id="env-b",
        workload_id=workload_id,
        artifact_hash=artifact_hash,
        canonical_payload_hash=payload_hash,
        receipt_hash=sha256_hex({"r": 2}) if mismatch else receipt_hash,
        platform_label="linux",
        python_label="3.11",
        metadata_hash=metadata_hash,
    )
    return compare_cross_environment_replay((first, second))


def test_failure_extraction_and_counts_and_suppression_rate_zero() -> None:
    receipts = _issue_pipeline((_raw_issue(summary="unsafe runtime mutation", severity="CRITICAL"),))
    receipt = build_failure_ledger(*receipts, _cross_env_receipt(mismatch=True))

    assert receipt.failure_count > 0
    assert receipt.suppression_rate == 0
    assert receipt.typed_counts["MISMATCH"] == 1
    assert receipt.typed_counts["CROSS_ENV_FAILURE"] == 1
    assert receipt.typed_counts["DETERMINISM_FAILURE"] >= 1


def test_typed_counts_complete_with_empty_categories_present() -> None:
    receipts = _issue_pipeline((_raw_issue(summary="canonical validation", severity="LOW"),))
    receipt = build_failure_ledger(*receipts, _cross_env_receipt(mismatch=False))

    expected_keys = {
        "VALIDATION_FAILURE",
        "MISMATCH",
        "UNSAFE",
        "INSUFFICIENT",
        "REPLAY_FAILURE",
        "DETERMINISM_FAILURE",
        "ADVERSARIAL_FAILURE",
        "CROSS_ENV_FAILURE",
        "UNKNOWN_FAILURE",
    }
    assert set(receipt.typed_counts) == expected_keys
    assert all(receipt.typed_counts[key] >= 0 for key in expected_keys)


def test_lineage_chain_matches_fields() -> None:
    receipts = _issue_pipeline((_raw_issue(summary="validation gap", severity="MEDIUM"),))
    receipt = build_failure_ledger(*receipts, _cross_env_receipt(mismatch=True))

    for entry in receipt.ledger.entries:
        assert entry.lineage_chain == (
            entry.failure_record.origin_hash,
            entry.issue_hash,
            entry.proposal_hash,
            entry.validation_hash,
            entry.counterfactual_hash,
            entry.adversarial_hash,
            entry.cross_env_hash,
        )


def test_deterministic_ordering_and_replay_stability() -> None:
    raw = (
        _raw_issue(summary="unsafe runtime mutation", severity="CRITICAL"),
        _raw_issue(summary="validation mismatch", severity="HIGH"),
    )
    receipts = _issue_pipeline(raw)
    cross = _cross_env_receipt(mismatch=True)

    first = build_failure_ledger(*receipts, cross)
    second = build_failure_ledger(*receipts, cross)

    assert first.ledger.stable_hash() == second.ledger.stable_hash()
    assert first.stable_hash() == second.stable_hash()
    assert first.to_canonical_bytes() == second.to_canonical_bytes()

    entries = first.ledger.entries
    assert entries == tuple(
        sorted(
            entries,
            key=lambda e: (
                {"CRITICAL": -3, "HIGH": -2, "MEDIUM": -1, "LOW": 0}[e.failure_record.severity],
                e.failure_record.failure_type,
                e.entry_id,
                e.stable_hash(),
            ),
        )
    )


def test_empty_case_returns_empty_status() -> None:
    receipts = _issue_pipeline((_raw_issue(summary="stable validation", severity="LOW"),))
    receipt = build_failure_ledger(*receipts, _cross_env_receipt(mismatch=False))

    assert receipt.ledger_status == "EMPTY"
    assert receipt.failure_count == 0
    assert receipt.suppression_rate == 0


def test_invalid_input_rejection() -> None:
    receipts = _issue_pipeline((_raw_issue(summary="stable validation", severity="LOW"),))
    with pytest.raises(ValueError, match="issue_receipt"):
        build_failure_ledger(  # type: ignore[arg-type]
            {}, receipts[1], receipts[2], receipts[3], receipts[4], _cross_env_receipt(mismatch=False)
        )


def test_frozen_dataclass_immutability() -> None:
    record = FailureRecord(
        failure_id="FAIL-1",
        source_module="x",
        failure_type="UNKNOWN_FAILURE",
        severity="LOW",
        origin_hash="abc",
        description="desc",
    )
    entry = FailureLedgerEntry(
        entry_id="LEDGER-1",
        failure_record=record,
        issue_hash="NONE",
        proposal_hash="NONE",
        validation_hash="NONE",
        counterfactual_hash="NONE",
        adversarial_hash="NONE",
        cross_env_hash="NONE",
        lineage_chain=("abc", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"),
    )
    ledger = FailureLedger(entries=(entry,), entry_count=1, ledger_hash="h")
    receipt = FailureLedgerReceipt(
        schema_version="1.0",
        module_version="v148.7",
        ledger_status="VALID",
        input_hash="in",
        ledger=ledger,
        failure_count=1,
        typed_counts={
            "ADVERSARIAL_FAILURE": 0,
            "CROSS_ENV_FAILURE": 0,
            "DETERMINISM_FAILURE": 0,
            "INSUFFICIENT": 0,
            "MISMATCH": 0,
            "REPLAY_FAILURE": 0,
            "UNSAFE": 0,
            "UNKNOWN_FAILURE": 1,
            "VALIDATION_FAILURE": 0,
        },
        suppression_rate=0,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.failure_id = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.entry_id = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ledger.entry_count = 0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.failure_count = 0  # type: ignore[misc]
    assert isinstance(receipt.typed_counts, MappingProxyType)
    with pytest.raises(TypeError):
        receipt.typed_counts["UNKNOWN_FAILURE"] = 0  # type: ignore[index]


def test_canonical_json_ordering() -> None:
    receipts = _issue_pipeline((_raw_issue(summary="stable validation", severity="LOW"),))
    receipt = build_failure_ledger(*receipts, _cross_env_receipt(mismatch=False))

    parsed = json.loads(receipt.to_canonical_json())
    assert list(parsed.keys()) == sorted(parsed.keys())
