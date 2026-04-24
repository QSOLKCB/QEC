from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.governed_orchestration_layer import GovernancePolicy
from qec.analysis.policy_memory_adaptive_governance import (
    AdaptiveGovernanceRecommendation,
    PolicyMemoryEntry,
    PolicyMemoryGovernanceReceipt,
    PolicyMemoryLedger,
    build_policy_memory_governance,
)
from qec.analysis.retro_target_registry import build_retro_target
from qec.analysis.retro_trace_control_kernel import compute_retro_trace_control
from qec.analysis.retro_trace_forecast_kernel import forecast_retro_trace
from qec.analysis.retro_trace_intake_bridge import build_retro_trace
from qec.analysis.retro_trace_policy_sensitivity import analyze_retro_trace_policy_sensitivity


def _target_receipt():
    return build_retro_target(
        target_id="policy-memory-target",
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
    elif case == "unstable":
        cycles = (10 + idx, 2700 + (idx * 13))
        cpu = tuple({"pc": 0x2000 + j + idx, "a": j % 256} for j in range(24))
        inputs = tuple()
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


def _control_bundle(case: str, idx: int):
    trace = _retro_trace(case, idx)
    policies = _policy_pair()
    sensitivity = analyze_retro_trace_policy_sensitivity(trace, policies)
    forecast = forecast_retro_trace(trace, horizon=12)
    control = compute_retro_trace_control(trace, sensitivity, forecast)
    return control, sensitivity, forecast


def test_deterministic_replay_and_canonical_hash_equality() -> None:
    controls = tuple(_control_bundle("stable", idx)[0] for idx in range(3))
    sensitivities = tuple(_control_bundle("stable", idx)[1] for idx in range(3))
    forecasts = tuple(_control_bundle("stable", idx)[2] for idx in range(3))
    left = build_policy_memory_governance(controls, sensitivity_receipts=sensitivities, forecast_receipts=forecasts)
    right = build_policy_memory_governance(controls, sensitivity_receipts=sensitivities, forecast_receipts=forecasts)
    assert left.to_canonical_json() == right.to_canonical_json()
    assert left.stable_hash() == right.stable_hash()
    assert left.to_canonical_bytes() == right.to_canonical_bytes()


def test_action_and_dominant_factor_count_correctness() -> None:
    bundles = tuple(_control_bundle("unstable", idx) for idx in range(3)) + tuple(_control_bundle("stable", idx + 20) for idx in range(2))
    receipt = build_policy_memory_governance(
        tuple(item[0] for item in bundles),
        sensitivity_receipts=tuple(item[1] for item in bundles),
        forecast_receipts=tuple(item[2] for item in bundles),
    )
    assert sum(count for _, count in receipt.ledger.action_counts) == receipt.ledger.ledger_size
    assert sum(count for _, count in receipt.ledger.dominant_factor_counts) == receipt.ledger.ledger_size


def test_recommendation_classification_escalate_and_maintain_and_review() -> None:
    escalate_bundles = tuple(_control_bundle("unstable", idx) for idx in range(4))
    escalate = build_policy_memory_governance(
        tuple(item[0] for item in escalate_bundles),
        sensitivity_receipts=tuple(item[1] for item in escalate_bundles),
        forecast_receipts=tuple(item[2] for item in escalate_bundles),
    )
    assert escalate.recommendation.label == "ESCALATE_GOVERNANCE"

    hold_bundles = tuple(_control_bundle("stable", idx + 100) for idx in range(5))
    hold = build_policy_memory_governance(
        tuple(item[0] for item in hold_bundles),
        sensitivity_receipts=tuple(item[1] for item in hold_bundles),
        forecast_receipts=tuple(item[2] for item in hold_bundles),
    )
    assert hold.recommendation.label == "MAINTAIN_POLICY"

    sparse = _control_bundle("sparse", 1)
    review = build_policy_memory_governance((sparse[0],), sensitivity_receipts=(sparse[1],), forecast_receipts=(sparse[2],))
    assert review.recommendation.label == "REVIEW_MEMORY"


def test_invalid_empty_control_list_and_duplicate_hash_rejections() -> None:
    with pytest.raises(ValueError, match="empty control_receipts"):
        build_policy_memory_governance(())

    bundle = _control_bundle("stable", 1)
    with pytest.raises(ValueError, match="duplicate or inconsistent hashes"):
        build_policy_memory_governance((bundle[0], bundle[0]))


def test_non_canonical_entry_ordering_rejection() -> None:
    bundles = tuple(_control_bundle("stable", idx + 40) for idx in range(2))
    receipt = build_policy_memory_governance(
        tuple(item[0] for item in bundles),
        sensitivity_receipts=tuple(item[1] for item in bundles),
        forecast_receipts=tuple(item[2] for item in bundles),
    )
    reversed_entries = tuple(reversed(receipt.ledger.entries))
    payload = {
        "entries": tuple(item.to_dict() for item in reversed_entries),
        "ledger_size": len(reversed_entries),
        "action_counts": receipt.ledger.action_counts,
        "dominant_factor_counts": receipt.ledger.dominant_factor_counts,
        "average_control_score": receipt.ledger.average_control_score,
        "average_confidence": receipt.ledger.average_confidence,
        "action_entropy_proxy": receipt.ledger.action_entropy_proxy,
        "replay_stability_score": receipt.ledger.replay_stability_score,
    }
    with pytest.raises(ValueError, match="canonically ordered"):
        PolicyMemoryLedger(
            entries=reversed_entries,
            ledger_size=len(reversed_entries),
            action_counts=receipt.ledger.action_counts,
            dominant_factor_counts=receipt.ledger.dominant_factor_counts,
            average_control_score=receipt.ledger.average_control_score,
            average_confidence=receipt.ledger.average_confidence,
            action_entropy_proxy=receipt.ledger.action_entropy_proxy,
            replay_stability_score=receipt.ledger.replay_stability_score,
            _stable_hash=sha256_hex(payload),
        )


def test_bool_rejection_for_integer_fields_and_bounded_outputs() -> None:
    bundle = _control_bundle("stable", 5)
    receipt = build_policy_memory_governance((bundle[0],), sensitivity_receipts=(bundle[1],), forecast_receipts=(bundle[2],))
    entry = receipt.ledger.entries[0]
    with pytest.raises(ValueError, match="entry_index must be non-negative int"):
        PolicyMemoryEntry(
            entry_index=True,
            control_hash=entry.control_hash,
            decision_action=entry.decision_action,
            control_score=entry.control_score,
            confidence=entry.confidence,
            dominant_factor=entry.dominant_factor,
            sensitivity_hash=entry.sensitivity_hash,
            forecast_hash=entry.forecast_hash,
            memory_weight=entry.memory_weight,
            _stable_hash=entry.stable_hash(),
        )

    for _, count in receipt.ledger.action_counts:
        assert isinstance(count, int)
        assert not isinstance(count, bool)
    for signal in receipt.signals:
        assert 0.0 <= signal.value <= 1.0
    assert 0.0 <= receipt.ledger.average_confidence <= 1.0
    assert 0.0 <= receipt.ledger.average_control_score <= 1.0


def test_direct_receipt_reconstruction_fail_fast() -> None:
    bundle = _control_bundle("stable", 7)
    receipt = build_policy_memory_governance((bundle[0],), sensitivity_receipts=(bundle[1],), forecast_receipts=(bundle[2],))
    tampered_recommendation = replace(
        receipt.recommendation,
        label="RELAX_POLICY",
        rank=2,
        _stable_hash=sha256_hex({"label": "RELAX_POLICY", "rationale": receipt.recommendation.rationale, "rank": 2}),
    )
    tampered_summary = replace(
        receipt.summary,
        recommendation_label="RELAX_POLICY",
        _stable_hash=sha256_hex(
            {
                "recommendation_label": "RELAX_POLICY",
                "governance_confidence": receipt.summary.governance_confidence,
                "replay_stability_score": receipt.summary.replay_stability_score,
                "entry_count": receipt.summary.entry_count,
            }
        ),
    )
    payload = {
        "ledger": receipt.ledger.to_dict(),
        "signals": tuple(item.to_dict() for item in receipt.signals),
        "recommendation": tampered_recommendation.to_dict(),
        "summary": tampered_summary.to_dict(),
    }
    with pytest.raises(ValueError, match="recommendation mismatch"):
        PolicyMemoryGovernanceReceipt(
            ledger=receipt.ledger,
            signals=receipt.signals,
            recommendation=tampered_recommendation,
            summary=tampered_summary,
            _stable_hash=sha256_hex(payload),
        )


def test_snapshot_fixture_lock_canonical_json_and_hash() -> None:
    bundles = tuple(_control_bundle("stable", idx + 200) for idx in range(3))
    receipt = build_policy_memory_governance(
        tuple(item[0] for item in bundles),
        sensitivity_receipts=tuple(item[1] for item in bundles),
        forecast_receipts=tuple(item[2] for item in bundles),
    )
    expected_json = (
        '{"ledger":{"action_counts":[["HOLD",3]],"action_entropy_proxy":0.0,"average_confidence":0.939368295682,'
        '"average_control_score":0.229676107049,"dominant_factor_counts":[["stability",3]],"entries":[{"confidence":0.939368295682,'
        '"control_hash":"4ee1fd4d662f304f68944bfe6b6f47be12f83bb9869e6bda1984083cfcbe2d2c","control_score":0.229676107049,'
        '"decision_action":"HOLD","dominant_factor":"stability","entry_index":1,'
        '"forecast_hash":"b9bf3c036ab12fea70b853b35b14338210d7b5416940efbc4d8e905934f94217","memory_weight":0.655491420229,'
        '"sensitivity_hash":"267ca8f27881471c1981eb0a818f83d867c67aaeab46ba37d172deae4316843b",'
        '"stable_hash":"ff69099ece7fc5bbc32a02cabb48d9294716e17b9bfefd3019e3302f4b7e4781"},{"confidence":0.939368295682,'
        '"control_hash":"7a4f8d77b68a840359f71d50623d57ee0117323a118ede2cd52d5f0e060ad6b7","control_score":0.229676107049,'
        '"decision_action":"HOLD","dominant_factor":"stability","entry_index":0,'
        '"forecast_hash":"dbbce4a651592dd0c8c85a1c976a8c1f90b8801d4aa697de9335dad872ca370c","memory_weight":0.655491420229,'
        '"sensitivity_hash":"11568e2bb5b0faf26f8197a5e9b12c970e95174724784c9d47a0819e4662fc22",'
        '"stable_hash":"352230c65ab51cd1f68a610352650739baa8bd8fd0ffc3e9b62387336c0b4996"},{"confidence":0.939368295682,'
        '"control_hash":"abe08922e29828ab327a456334e939aee58ba60268e18cf48ad9828eefc4b839","control_score":0.229676107049,'
        '"decision_action":"HOLD","dominant_factor":"stability","entry_index":2,'
        '"forecast_hash":"c0007e99fec789d72aa60b5003ecb9077fe6518300335ceac68f03c12128b071","memory_weight":0.655491420229,'
        '"sensitivity_hash":"e6bc4f63a6f1170ed9ef551fa75425c475715d223c002f13ed42630320f1f201",'
        '"stable_hash":"869da51cc1251616b1041d2d64b59c6bb9b8ae0a7fa3573b35865e865d8261a0"}],"ledger_size":3,'
        '"replay_stability_score":1.0,"stable_hash":"12226b99d40aaa2789081590a863f8287fdf120d87bd86fd2acefa510f800c1f"},'
        '"recommendation":{"label":"MAINTAIN_POLICY","rank":0,'
        '"rationale":"recommendation=MAINTAIN_POLICY;escalation=0.0975;adjustment=0.0;hold=0.905491420229;confidence=0.688674381772",'
        '"stable_hash":"16222f8b3543a245233f1e89bb35381c538d188626a254f4b1b6b43192965e34"},"signals":[{"name":"adjustment_pressure",'
        '"stable_hash":"333e2d6464850d278e1f8c8331580f7fcf6ef8e60ac37c608043fa2c28e2362e","value":0.0},{"name":"escalation_pressure",'
        '"stable_hash":"40500f4893a743fd9d68a30d48dccb97e10b9ee431e2d6cab89453fe221eaa19","value":0.0975},{"name":"forecast_instability_memory",'
        '"stable_hash":"22632f58fff33234b7c19d1ab09de77e054dedd607b5cf3c7b854ccc037bb53d","value":0.15},{"name":"governance_confidence",'
        '"stable_hash":"9ac3e8e8c881fd4fdc9f6e217ee83e7e639637b6d9d58c2956d116b531526336","value":0.688674381772},{"name":"hold_stability",'
        '"stable_hash":"d8023edeccfa21d018e38fc887ec1db7ec3ac284756b1e00c12804cf05d05950","value":0.905491420229},{"name":"policy_volatility",'
        '"stable_hash":"fe9fe0c88bba3aef3215e12fe0f112dd8f51d2aee7dfde5317fe753bbe77b24c","value":0.0}],'
        '"stable_hash":"6d02d0a1e9fa0a91218413b2ec5e88d61b31b1d0a4c2c5d69a9138c59e6372a9","summary":{"entry_count":3,'
        '"governance_confidence":0.688674381772,"recommendation_label":"MAINTAIN_POLICY","replay_stability_score":1.0,'
        '"stable_hash":"67bf752a9d857a019a28e4d842d283be0d864379144ccbd01548666d61bcadc0"}}'
    )
    expected_hash = "6d02d0a1e9fa0a91218413b2ec5e88d61b31b1d0a4c2c5d69a9138c59e6372a9"
    assert receipt.to_canonical_json() == expected_json
    assert receipt.stable_hash() == expected_hash


def test_reject_malformed_recommendation_label() -> None:
    with pytest.raises(ValueError, match="invalid recommendation labels"):
        AdaptiveGovernanceRecommendation(
            label="INVALID",
            rationale="bad",
            rank=0,
            _stable_hash=sha256_hex({"label": "INVALID", "rationale": "bad", "rank": 0}),
        )
