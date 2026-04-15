from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from qec.security.benchmark_vault import (
    BENCHMARK_VAULT_VERSION,
    BenchmarkVaultKernel,
    run_benchmark_vault,
    summarize_benchmark_vault_report,
    validate_benchmark_artifact,
)


def _artifact(*, classification: str = "public") -> dict:
    return {
        "benchmark_id": "bench-001",
        "benchmark_type": "benchmark_corpus",
        "manifest": {
            "benchmark_id": "bench-001",
            "manifest_hash": "mhash-001",
            "lineage_hash": "lhash-001",
            "corpus_classification": classification,
        },
        "metadata": {"lane": "default"},
        "provenance": {
            "origin": "lab-alpha",
            "source_id": "dataset-001",
            "chain_of_custody": "chain-001",
        },
        "declared_contract": "benchmark.formal.v1",
        "source_channel": "benchmark_harness",
        "corpus_classification": classification,
    }


def _policy(**overrides: object) -> dict:
    base = BenchmarkVaultKernel.default_policy().to_dict()
    base.update(overrides)
    return base


def test_happy_path_allow_valid_public_corpus() -> None:
    report, receipt = run_benchmark_vault(benchmark_artifact=_artifact(classification="public"))
    assert report.decision == "allow"
    assert report.public_corpus is True
    assert report.sealed_corpus is False
    assert receipt.version == BENCHMARK_VAULT_VERSION


def test_happy_path_allow_valid_sealed_corpus() -> None:
    report, receipt = run_benchmark_vault(benchmark_artifact=_artifact(classification="sealed"))
    assert report.decision == "allow"
    assert report.sealed_corpus is True
    assert report.public_corpus is False
    assert receipt.vault_admissible is True


def test_advisory_only_warn_path() -> None:
    report, _ = run_benchmark_vault(
        benchmark_artifact=_artifact(),
        intake_firewall_context={"decision": "quarantine"},
    )
    assert report.decision == "warn"
    assert report.warnings == ("intake_firewall_compatibility",)


def test_malformed_artifact_reject() -> None:
    report, receipt = run_benchmark_vault(benchmark_artifact={"benchmark_id": "x"})
    assert report.decision == "reject"
    assert "benchmark_envelope_structure" in report.rejection_reasons
    assert receipt.rejected is True


def test_forbidden_metadata_key_reject() -> None:
    artifact = _artifact()
    artifact["metadata"] = {"benchmark_override": "yes"}
    report, _ = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "reject"
    assert "forbidden_metadata_keys" in report.rejection_reasons


def test_forbidden_manifest_field_reject() -> None:
    artifact = _artifact()
    artifact["manifest"] = dict(artifact["manifest"])
    artifact["manifest"]["sealed_corpus_override"] = True
    report, _ = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "reject"
    assert "forbidden_manifest_field_names" in report.rejection_reasons


def test_incomplete_provenance_quarantine() -> None:
    artifact = _artifact()
    artifact["provenance"] = {"origin": "lab-alpha"}
    report, receipt = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "quarantine"
    assert "required_provenance_fields" in report.quarantine_reasons
    assert receipt.quarantined is True


def test_unsupported_source_channel_reject() -> None:
    artifact = _artifact()
    artifact["source_channel"] = "email"
    report, _ = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "reject"
    assert "source_channel_allowed" in report.rejection_reasons


def test_unsupported_contract_reject() -> None:
    artifact = _artifact()
    artifact["declared_contract"] = "unknown.contract"
    report, _ = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "reject"
    assert "declared_contract_allowed" in report.rejection_reasons


def test_missing_manifest_hash_reject_when_required() -> None:
    artifact = _artifact()
    artifact["manifest"] = dict(artifact["manifest"])
    del artifact["manifest"]["manifest_hash"]
    report, _ = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "reject"
    assert "manifest_hash_presence" in report.rejection_reasons


def test_missing_lineage_hash_reject_when_required_on_sealed() -> None:
    artifact = _artifact(classification="sealed")
    artifact["manifest"] = dict(artifact["manifest"])
    del artifact["manifest"]["lineage_hash"]
    report, _ = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "reject"
    assert "lineage_hash_presence" in report.rejection_reasons


def test_sealed_public_mixing_reject() -> None:
    artifact = _artifact()
    artifact["corpus_classification"] = "sealed|public"
    report, _ = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "reject"
    assert "sealed_public_mixing" in report.rejection_reasons


def test_manifest_identity_conflict_reject() -> None:
    artifact = _artifact()
    artifact["manifest"] = dict(artifact["manifest"])
    artifact["manifest"]["benchmark_id"] = "other-id"
    report, _ = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "reject"
    assert "manifest_identity_conflicts" in report.rejection_reasons


def test_benchmark_override_field_reject() -> None:
    artifact = _artifact()
    artifact["metadata"] = {"custom_override": True}
    report, _ = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "reject"
    assert "undeclared_override_fields" in report.rejection_reasons


def test_non_string_key_rejection() -> None:
    artifact = _artifact()
    artifact["manifest"] = {1: "bad"}  # type: ignore[dict-item]
    report, _ = run_benchmark_vault(benchmark_artifact=artifact)
    assert report.decision == "reject"
    assert "benchmark_envelope_structure" in report.rejection_reasons


def test_canonical_ordering_hash_stability_under_shuffled_mappings() -> None:
    a = _artifact(classification="sealed")
    b = {
        "benchmark_id": "bench-001",
        "benchmark_type": "benchmark_corpus",
        "manifest": {
            "corpus_classification": "sealed",
            "lineage_hash": "lhash-001",
            "benchmark_id": "bench-001",
            "manifest_hash": "mhash-001",
        },
        "metadata": dict(reversed(list(a["metadata"].items()))),
        "provenance": {
            "chain_of_custody": "chain-001",
            "source_id": "dataset-001",
            "origin": "lab-alpha",
        },
        "declared_contract": "benchmark.formal.v1",
        "source_channel": "benchmark_harness",
        "corpus_classification": "sealed",
    }
    report_a, receipt_a = run_benchmark_vault(benchmark_artifact=a)
    report_b, receipt_b = run_benchmark_vault(benchmark_artifact=b)
    assert report_a.to_canonical_json() == report_b.to_canonical_json()
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert report_a.stable_hash() == report_b.stable_hash()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()


def test_receipt_report_immutability() -> None:
    report, receipt = run_benchmark_vault(benchmark_artifact=_artifact())
    with pytest.raises(TypeError):
        report.counts_by_status["passed"] = 0
    with pytest.raises(FrozenInstanceError):
        receipt.decision = "reject"  # type: ignore[misc]
    payload = json.loads(report.to_canonical_json())
    assert "checks" in payload


def test_public_artifact_allowed_when_manifest_and_lineage_hash_disabled() -> None:
    artifact = _artifact(classification="public")
    artifact["manifest"] = {"benchmark_id": "bench-001", "corpus_classification": "public"}
    report, _ = run_benchmark_vault(
        benchmark_artifact=artifact,
        benchmark_vault_policy=_policy(require_manifest_hash=False, require_lineage_hash=False),
    )
    assert report.decision == "allow"


def test_manifest_hash_required_only_when_enabled() -> None:
    artifact = _artifact()
    artifact["manifest"] = dict(artifact["manifest"])
    del artifact["manifest"]["manifest_hash"]

    allow_report, _ = run_benchmark_vault(
        benchmark_artifact=artifact,
        benchmark_vault_policy=_policy(require_manifest_hash=False),
    )
    reject_report, _ = run_benchmark_vault(
        benchmark_artifact=artifact,
        benchmark_vault_policy=_policy(require_manifest_hash=True),
    )

    assert "manifest_hash_presence" not in allow_report.rejection_reasons
    assert "manifest_hash_presence" in reject_report.rejection_reasons


def test_lineage_hash_required_only_when_enabled_for_public_corpus() -> None:
    artifact = _artifact(classification="public")
    artifact["manifest"] = dict(artifact["manifest"])
    del artifact["manifest"]["lineage_hash"]

    allow_report, _ = run_benchmark_vault(
        benchmark_artifact=artifact,
        benchmark_vault_policy=_policy(require_lineage_hash=False),
    )
    reject_report, _ = run_benchmark_vault(
        benchmark_artifact=artifact,
        benchmark_vault_policy=_policy(require_lineage_hash=True),
    )

    assert "lineage_hash_presence" not in allow_report.rejection_reasons
    assert "lineage_hash_presence" in reject_report.rejection_reasons


def test_manifest_required_fields_follow_policy_flags() -> None:
    artifact = _artifact(classification="public")
    artifact["manifest"] = {"benchmark_id": "bench-001", "corpus_classification": "public"}

    report, _ = run_benchmark_vault(
        benchmark_artifact=artifact,
        benchmark_vault_policy=_policy(require_manifest_hash=True, require_lineage_hash=True),
    )

    assert "required_manifest_fields" in report.rejection_reasons
    required_check = next(check for check in report.checks if check.name == "required_manifest_fields")
    assert required_check.policy_value == ("benchmark_id", "lineage_hash", "manifest_hash")


def test_decoder_untouched_guarantee_and_helpers() -> None:
    import qec.security.benchmark_vault as mod

    with open(mod.__file__, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "qec.decoder" not in source
    assert "decoder-untouched" in (mod.__doc__ or "")

    summary = summarize_benchmark_vault_report(run_benchmark_vault(benchmark_artifact=_artifact())[0])
    assert summary["decision"] == "allow"
    validation = validate_benchmark_artifact(_artifact())
    assert validation["valid"] is True


def test_quarantine_vs_reject_distinction_for_provenance_policy() -> None:
    artifact = _artifact()
    artifact["provenance"] = {"origin": "lab-alpha"}
    q_report, _ = run_benchmark_vault(
        benchmark_artifact=artifact,
        benchmark_vault_policy=_policy(quarantine_incomplete_provenance=True),
    )
    r_report, _ = run_benchmark_vault(
        benchmark_artifact=artifact,
        benchmark_vault_policy=_policy(quarantine_incomplete_provenance=False),
    )
    assert q_report.decision == "quarantine"
    assert r_report.decision == "reject"
