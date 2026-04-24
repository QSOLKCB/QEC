from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.governance_validation_kernel import (
    GovernanceValidationReceipt,
    validate_governance_recommendation,
)
from qec.analysis.governed_orchestration_layer import GovernancePolicy
from qec.analysis.policy_memory_adaptive_governance import build_policy_memory_governance
from qec.analysis.retro_target_registry import build_retro_target
from qec.analysis.retro_trace_control_kernel import compute_retro_trace_control
from qec.analysis.retro_trace_forecast_kernel import forecast_retro_trace
from qec.analysis.retro_trace_intake_bridge import build_retro_trace
from qec.analysis.retro_trace_policy_sensitivity import analyze_retro_trace_policy_sensitivity


def _target_receipt():
    return build_retro_target(
        target_id="governance-validation-target",
        isa_family="z80",
        word_size=8,
        address_width=16,
        ram_budget=48 * 1024,
        rom_budget=32 * 1024,
        cycle_budget=3_500_000,
        display_budget={"width": 256, "height": 192, "colors": 16},
        audio_budget={"channels": 3, "sample_rate": 44_100},
        input_budget={"buttons": 2, "axes": 0},
        fpu_policy="none",
        provenance="hardware",
    )


def _policy_pair() -> tuple[GovernancePolicy, GovernancePolicy]:
    base = {
        "min_required_score": 0.55,
        "min_required_confidence": 0.95,
        "min_required_margin": 0.45,
        "min_required_convergence": 0.90,
        "allow_tie_break": False,
        "allow_no_improvement": False,
        "require_stable_transition": True,
    }
    alt = {**base, "allow_tie_break": True, "min_required_score": 0.2}
    return (
        GovernancePolicy(**base, stable_hash=sha256_hex(base)),
        GovernancePolicy(**alt, stable_hash=sha256_hex(alt)),
    )


def _retro_trace(case: str, idx: int):
    if case == "stable":
        cycles = tuple(100 + 20 * j for j in range(8))
        cpu = tuple({"pc": 0x1000 + j + idx, "a": 0x10 + (j % 8)} for j in range(8))
        inputs = tuple({"port": 1, "button": "A", "state": j % 2} for j in range(8))
    else:
        cycles = (100 + idx,)
        cpu = ({"pc": 0x3000 + idx, "a": 0x20},)
        inputs = tuple()

    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=cpu,
        memory_trace=({"address": 0x4000 + idx, "op": "read", "value": 0xAB},),
        timing_trace=tuple({"cycle": cycle} for cycle in cycles),
        display_trace=({"scanline": 0, "event": "start"},),
        audio_trace=({"channel": 1, "pattern": "pulse"},),
        input_trace=inputs,
        metadata={"emulator": "retroarch", "rom_hash": f"{case}-{idx}", "version": "1.0.0"},
    )


def _governance_receipt():
    trace = _retro_trace("stable", 4)
    policies = _policy_pair()
    sensitivity = analyze_retro_trace_policy_sensitivity(trace, policies)
    forecast = forecast_retro_trace(trace, horizon=12)
    control = compute_retro_trace_control(trace, sensitivity, forecast)
    return build_policy_memory_governance((control,), sensitivity_receipts=(sensitivity,), forecast_receipts=(forecast,))


def test_valid_governance_recomputation_passes() -> None:
    governance = _governance_receipt()
    receipt = validate_governance_recommendation(governance, governance)
    assert receipt.validation_status == "VALIDATED"
    assert receipt.recommendation_stable is True
    assert receipt.expected_recommendation == receipt.recomputed_recommendation


def test_replay_stability_same_bytes_and_hashes() -> None:
    governance = _governance_receipt()
    left = validate_governance_recommendation(governance, governance)
    right = validate_governance_recommendation(governance, governance)
    assert left.to_dict() == right.to_dict()
    assert left.to_canonical_json() == right.to_canonical_json()
    assert left.stable_hash == right.stable_hash
    assert left.validation_hash == right.validation_hash


def test_mismatch_detection() -> None:
    governance = _governance_receipt()
    expected_payload = governance.to_dict()
    expected_payload["recommendation"] = {
        "label": "RELAX_POLICY",
        "rationale": "deterministic mismatch fixture",
        "rank": 2,
        "stable_hash": sha256_hex(
            {
                "label": "RELAX_POLICY",
                "rationale": "deterministic mismatch fixture",
                "rank": 2,
            }
        ),
    }
    expected_payload["stable_hash"] = sha256_hex({k: v for k, v in expected_payload.items() if k != "stable_hash"})

    receipt = validate_governance_recommendation(governance.to_dict(), expected_payload)
    assert receipt.validation_status == "MISMATCH"
    assert receipt.recommendation_stable is False
    assert receipt.validation_score == 0.0


def test_invalid_memory_rejected() -> None:
    governance = _governance_receipt().to_dict()
    invalid_memory = {k: v for k, v in governance.items() if k != "stable_hash"}
    with pytest.raises(ValueError, match="policy_memory must include stable_hash"):
        validate_governance_recommendation(invalid_memory, governance)


def test_invalid_recommendation_rejected() -> None:
    governance = _governance_receipt()
    invalid_expected = {
        "label": "MAKE_IT_SPICY",
        "rationale": "invalid label",
        "rank": 0,
    }
    invalid_expected["stable_hash"] = sha256_hex({k: v for k, v in invalid_expected.items() if k != "stable_hash"})
    with pytest.raises(ValueError, match="invalid recommendation labels"):
        validate_governance_recommendation(governance.to_dict(), invalid_expected)


def test_receipt_immutability() -> None:
    governance = _governance_receipt()
    receipt = validate_governance_recommendation(governance, governance)
    with pytest.raises(FrozenInstanceError):
        receipt.validation_status = "MISMATCH"


def test_canonical_json_ordering() -> None:
    governance = _governance_receipt().to_dict()
    memory_a = {"signals": governance["signals"], "ledger": governance["ledger"], "recommendation": governance["recommendation"], "summary": governance["summary"], "stable_hash": governance["stable_hash"]}
    memory_b = {"stable_hash": governance["stable_hash"], "summary": governance["summary"], "recommendation": governance["recommendation"], "ledger": governance["ledger"], "signals": governance["signals"]}

    receipt_a = validate_governance_recommendation(memory_a, governance)
    receipt_b = validate_governance_recommendation(memory_b, governance)

    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.stable_hash == receipt_b.stable_hash
    assert receipt_a.validation_status == receipt_b.validation_status


def test_governance_validation_receipt_rejects_mismatched_hash() -> None:
    governance = _governance_receipt()
    receipt = validate_governance_recommendation(governance, governance)
    with pytest.raises(ValueError, match="stable_hash mismatch"):
        GovernanceValidationReceipt(
            schema_version=receipt.schema_version,
            module_version=receipt.module_version,
            validation_status=receipt.validation_status,
            expected_recommendation=receipt.expected_recommendation,
            recomputed_recommendation=receipt.recomputed_recommendation,
            recommendation_stable=receipt.recommendation_stable,
            memory_hash=receipt.memory_hash,
            expected_governance_hash=receipt.expected_governance_hash,
            recomputed_governance_hash=receipt.recomputed_governance_hash,
            validation_hash=receipt.validation_hash,
            stable_hash="0" * 64,
            validation_score=receipt.validation_score,
            hash_match=receipt.hash_match,
        )
