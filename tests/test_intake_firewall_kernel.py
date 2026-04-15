from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from qec.security.intake_firewall_kernel import (
    FIREWALL_VERSION,
    IntakeFirewallKernel,
    run_intake_firewall,
    summarize_intake_firewall_report,
    validate_intake_artifact,
)


def _artifact() -> dict:
    return {
        "artifact_id": "artifact-001",
        "artifact_type": "interface_capture",
        "payload": {"syndrome": [0, 1, 1], "shape": [3]},
        "metadata": {"capture_mode": "stable"},
        "provenance": {
            "origin": "lab-alpha",
            "source_id": "sensor-a",
            "chain_of_custody": "chain-1",
        },
        "declared_contract": "interface.normalization.v1",
        "source_channel": "ingest_api",
    }


def _policy(**overrides: object) -> dict:
    base = IntakeFirewallKernel.default_policy().to_dict()
    base.update(overrides)
    return base


def test_happy_path_allow() -> None:
    report, receipt = run_intake_firewall(artifact=_artifact())
    assert report.decision == "allow"
    assert receipt.version == FIREWALL_VERSION
    assert receipt.admitted is True


def test_advisory_only_warn_path() -> None:
    artifact = _artifact()
    artifact["payload"] = {"deprecated_field": "legacy", "ok": 1}
    report, receipt = run_intake_firewall(artifact=artifact)
    assert report.decision == "warn"
    assert "advisory_payload_fields" in report.warnings
    assert receipt.admitted is True


def test_malformed_artifact_reject() -> None:
    report, receipt = run_intake_firewall(artifact={"artifact_id": "x"})
    assert report.decision == "reject"
    assert "artifact_envelope_structure" in report.rejection_reasons
    assert receipt.rejected is True


def test_forbidden_metadata_key_reject() -> None:
    artifact = _artifact()
    artifact["metadata"] = {"benchmark_override": True}
    report, _ = run_intake_firewall(artifact=artifact)
    assert report.decision == "reject"
    assert "forbidden_metadata_keys" in report.rejection_reasons


def test_forbidden_payload_field_reject() -> None:
    artifact = _artifact()
    artifact["payload"] = {"decoder_override": "x"}
    report, _ = run_intake_firewall(artifact=artifact)
    assert report.decision == "reject"
    assert "forbidden_payload_field_names" in report.rejection_reasons


def test_incomplete_provenance_quarantine() -> None:
    artifact = _artifact()
    artifact["provenance"] = {"origin": "lab-alpha"}
    report, receipt = run_intake_firewall(artifact=artifact)
    assert report.decision == "quarantine"
    assert "required_provenance_fields" in report.quarantine_reasons
    assert receipt.quarantined is True


def test_unsupported_source_channel_reject() -> None:
    artifact = _artifact()
    artifact["source_channel"] = "email"
    report, _ = run_intake_firewall(artifact=artifact)
    assert report.decision == "reject"
    assert "source_channel_allowed" in report.rejection_reasons


def test_unsupported_contract_reject() -> None:
    artifact = _artifact()
    artifact["declared_contract"] = "unknown.contract"
    report, _ = run_intake_firewall(artifact=artifact)
    assert report.decision == "reject"
    assert "declared_contract_allowed" in report.rejection_reasons


def test_non_string_key_rejection() -> None:
    artifact = _artifact()
    artifact["payload"] = {1: "bad"}  # type: ignore[dict-item]
    report, _ = run_intake_firewall(artifact=artifact)
    assert report.decision == "reject"
    assert "artifact_envelope_structure" in report.rejection_reasons


def test_nesting_depth_rejection() -> None:
    artifact = _artifact()
    artifact["payload"] = {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}
    report, _ = run_intake_firewall(artifact=artifact, policy=_policy(max_nesting_depth=3))
    assert report.decision == "reject"
    assert "payload_max_nesting_depth" in report.rejection_reasons


def test_sequence_length_rejection() -> None:
    artifact = _artifact()
    artifact["payload"] = {"seq": [0, 1, 2, 3, 4]}
    report, _ = run_intake_firewall(artifact=artifact, policy=_policy(max_sequence_length=2))
    assert report.decision == "reject"
    assert "payload_max_sequence_length" in report.rejection_reasons


def test_mapping_width_rejection() -> None:
    artifact = _artifact()
    artifact["payload"] = {f"k{i}": i for i in range(5)}
    report, _ = run_intake_firewall(artifact=artifact, policy=_policy(max_mapping_width=3))
    assert report.decision == "reject"
    assert "payload_max_mapping_width" in report.rejection_reasons


def test_canonical_hash_stability_under_shuffled_mappings() -> None:
    a = _artifact()
    b = {
        "artifact_id": "artifact-001",
        "artifact_type": "interface_capture",
        "payload": {"shape": [3], "syndrome": [0, 1, 1]},
        "metadata": dict(reversed(list(a["metadata"].items()))),
        "provenance": {
            "chain_of_custody": "chain-1",
            "source_id": "sensor-a",
            "origin": "lab-alpha",
        },
        "declared_contract": "interface.normalization.v1",
        "source_channel": "ingest_api",
    }
    report_a, receipt_a = run_intake_firewall(artifact=a)
    report_b, receipt_b = run_intake_firewall(artifact=b)
    assert report_a.to_canonical_json() == report_b.to_canonical_json()
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert report_a.stable_hash() == report_b.stable_hash()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()


def test_quarantine_vs_reject_distinction() -> None:
    artifact = _artifact()
    artifact["provenance"] = {"origin": "lab-alpha"}
    q_report, _ = run_intake_firewall(artifact=artifact, policy=_policy(quarantine_incomplete_provenance=True))
    r_report, _ = run_intake_firewall(artifact=artifact, policy=_policy(quarantine_incomplete_provenance=False))
    assert q_report.decision == "quarantine"
    assert r_report.decision == "reject"


def test_receipt_report_immutability() -> None:
    report, receipt = run_intake_firewall(artifact=_artifact())
    with pytest.raises(TypeError):
        report.counts_by_status["passed"] = 0
    with pytest.raises(FrozenInstanceError):
        receipt.decision = "reject"  # type: ignore[misc]
    payload = json.loads(report.to_canonical_json())
    assert "checks" in payload


def test_check_snapshot_values_are_frozen() -> None:
    artifact = _artifact()
    artifact["payload"] = {"decoder_override": "x"}
    report, _ = run_intake_firewall(artifact=artifact)
    check = next(item for item in report.checks if item.name == "forbidden_payload_field_names")
    assert isinstance(check.observed_value, tuple)
    before = report.stable_hash()
    with pytest.raises(AttributeError):
        check.observed_value.append("tamper")  # type: ignore[attr-defined]
    assert report.stable_hash() == before


def test_decoder_untouched_guarantee_and_api_helpers() -> None:
    import qec.security.intake_firewall_kernel as mod

    with open(mod.__file__, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "qec.decoder" not in source
    assert "decoder-untouched" in (mod.__doc__ or "")

    summary = summarize_intake_firewall_report(run_intake_firewall(artifact=_artifact())[0])
    assert summary["decision"] == "allow"
    validation = validate_intake_artifact(_artifact())
    assert validation["valid"] is True


def test_unknown_top_level_key_rejection() -> None:
    artifact = _artifact()
    artifact["extra_field"] = "should_be_rejected"
    report, receipt = run_intake_firewall(artifact=artifact)
    assert report.decision == "reject"
    assert "artifact_envelope_structure" in report.rejection_reasons
    assert receipt.rejected is True


def test_envelope_failure_short_circuits_downstream_checks() -> None:
    report, receipt = run_intake_firewall(artifact={"artifact_id": "x"})
    assert report.decision == "reject"
    assert receipt.rejected is True
    check_names = {check.name for check in report.checks}
    assert check_names == {"artifact_envelope_structure"}


def test_envelope_check_decision_effect_always_reject() -> None:
    report_ok, _ = run_intake_firewall(artifact=_artifact())
    envelope_ok = next(c for c in report_ok.checks if c.name == "artifact_envelope_structure")
    assert envelope_ok.passed is True
    assert envelope_ok.decision_effect == "reject"

    report_bad, _ = run_intake_firewall(artifact={"artifact_id": "x"})
    envelope_bad = next(c for c in report_bad.checks if c.name == "artifact_envelope_structure")
    assert envelope_bad.passed is False
    assert envelope_bad.decision_effect == "reject"


def test_build_policy_rejects_string_for_int_fields() -> None:
    with pytest.raises(TypeError, match="max_nesting_depth"):
        IntakeFirewallKernel.build_policy({"max_nesting_depth": "6"})
    with pytest.raises(TypeError, match="max_mapping_width"):
        IntakeFirewallKernel.build_policy({"max_mapping_width": "64"})
    with pytest.raises(TypeError, match="max_sequence_length"):
        IntakeFirewallKernel.build_policy({"max_sequence_length": "512"})


def test_build_policy_rejects_bool_for_int_fields() -> None:
    with pytest.raises(TypeError, match="max_nesting_depth"):
        IntakeFirewallKernel.build_policy({"max_nesting_depth": True})


def test_build_policy_rejects_non_bool_for_bool_fields() -> None:
    with pytest.raises(TypeError, match="strict_string_only_keys"):
        IntakeFirewallKernel.build_policy({"strict_string_only_keys": "true"})
    with pytest.raises(TypeError, match="quarantine_incomplete_provenance"):
        IntakeFirewallKernel.build_policy({"quarantine_incomplete_provenance": 1})
