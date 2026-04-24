from __future__ import annotations

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.retro_target_registry import build_retro_target
from qec.analysis.retro_trace_control_kernel import (
    RetroTraceControlDecision,
    RetroTraceControlReceipt,
    RetroTraceControlSignal,
    RetroTraceControlSummary,
    _classify_action,
    compute_retro_trace_control,
)
from qec.analysis.retro_trace_forecast_kernel import forecast_retro_trace
from qec.analysis.retro_trace_forecast_lattice_kernel import forecast_retro_trace_lattice
from qec.analysis.retro_trace_intake_bridge import build_retro_trace
from qec.analysis.retro_trace_policy_sensitivity import analyze_retro_trace_policy_sensitivity
from qec.analysis.governed_orchestration_layer import GovernancePolicy


def _target_receipt():
    return build_retro_target(
        target_id="control-target",
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


def _trace_a():
    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple({"pc": 0x1000 + idx, "a": 0x10 + (idx % 16)} for idx in range(8)),
        memory_trace=({"address": 0x4000, "op": "read", "value": 0xAB},),
        timing_trace=tuple({"cycle": 100 + 20 * idx} for idx in range(8)),
        display_trace=({"scanline": 0, "event": "start"},),
        audio_trace=({"channel": 1, "pattern": "pulse"},),
        input_trace=tuple({"port": 1, "button": "A", "state": idx % 2} for idx in range(8)),
        metadata={"emulator": "retroarch", "rom_hash": "ctrl-a", "version": "1.0.0"},
    )


def _trace_a_equivalent():
    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple(reversed(tuple({"pc": 0x1000 + idx, "a": 0x10 + (idx % 16)} for idx in range(8)))),
        memory_trace=({"value": 0xAB, "op": "read", "address": 0x4000},),
        timing_trace=tuple(reversed(tuple({"cycle": 100 + 20 * idx} for idx in range(8)))),
        display_trace=({"event": "start", "scanline": 0},),
        audio_trace=({"pattern": "pulse", "channel": 1},),
        input_trace=tuple(reversed(tuple({"state": idx % 2, "button": "A", "port": 1} for idx in range(8)))),
        metadata={"version": "1.0.0", "rom_hash": "ctrl-a", "emulator": "retroarch"},
    )


def _trace_unstable():
    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple({"pc": 0x2000 + idx, "a": idx % 256} for idx in range(24)),
        memory_trace=tuple(),
        timing_trace=({"cycle": 10}, {"cycle": 2500}),
        display_trace=tuple(),
        audio_trace=tuple(),
        input_trace=tuple(),
        metadata={"emulator": "retroarch", "rom_hash": "ctrl-b", "version": "1.0.0"},
    )


def _bundle(trace, *, with_lattice: bool = False):
    policy_a_payload = {
        "min_required_score": 0.55,
        "min_required_confidence": 0.95,
        "min_required_margin": 0.45,
        "min_required_convergence": 0.90,
        "allow_tie_break": True,
        "allow_no_improvement": False,
        "require_stable_transition": True,
    }
    policy_b_payload = {**policy_a_payload, "allow_tie_break": False}
    policy_a = GovernancePolicy(**policy_a_payload, stable_hash=sha256_hex(policy_a_payload))
    policy_b = GovernancePolicy(**policy_b_payload, stable_hash=sha256_hex(policy_b_payload))
    policies = (policy_a, policy_b)
    sensitivity = analyze_retro_trace_policy_sensitivity(trace, policies)
    forecast = forecast_retro_trace(trace, horizon=12)
    lattice = forecast_retro_trace_lattice(trace, horizon=12, lattice_mode="neutral_atom_5") if with_lattice else None
    return sensitivity, forecast, lattice


def test_replay_determinism_multi_run_identical_hash() -> None:
    trace = _trace_a()
    sensitivity, forecast, lattice = _bundle(trace, with_lattice=True)
    receipts = [compute_retro_trace_control(trace, sensitivity, forecast, lattice) for _ in range(30)]
    assert len({r.stable_hash() for r in receipts}) == 1
    assert len({r.to_canonical_json() for r in receipts}) == 1
    assert len({r.to_canonical_bytes() for r in receipts}) == 1


def test_identical_inputs_identical_decision() -> None:
    trace = _trace_a()
    sensitivity, forecast, lattice = _bundle(trace, with_lattice=True)
    left = compute_retro_trace_control(trace, sensitivity, forecast, lattice)
    right = compute_retro_trace_control(trace, sensitivity, forecast, lattice)
    assert left.decision.action.action == right.decision.action.action
    assert left.to_canonical_json() == right.to_canonical_json()


def test_ordering_invariance_equivalent_trace_same_receipt() -> None:
    trace_a = _trace_a()
    trace_b = _trace_a_equivalent()
    sa, fa, la = _bundle(trace_a, with_lattice=True)
    sb, fb, lb = _bundle(trace_b, with_lattice=True)
    left = compute_retro_trace_control(trace_a, sa, fa, la)
    right = compute_retro_trace_control(trace_b, sb, fb, lb)
    assert left.stable_hash() == right.stable_hash()
    assert left.to_canonical_bytes() == right.to_canonical_bytes()


def test_classification_correctness_hold_adjust_escalate() -> None:
    stable_trace = _trace_a()
    unstable_trace = _trace_unstable()
    s1, f1, _ = _bundle(stable_trace, with_lattice=False)
    s2, f2, _ = _bundle(unstable_trace, with_lattice=False)

    hold_receipt = compute_retro_trace_control(stable_trace, s1, f1)
    escalate_receipt = compute_retro_trace_control(unstable_trace, s2, f2)

    assert hold_receipt.decision.action.action == "HOLD"
    assert escalate_receipt.decision.action.action in {"ADJUST", "ESCALATE"}
    assert _classify_action(0.5) == "ADJUST"


def test_boundary_thresholds_033_066() -> None:
    assert _classify_action(0.329999999999) == "HOLD"
    assert _classify_action(0.33) == "ADJUST"
    assert _classify_action(0.659999999999) == "ADJUST"
    assert _classify_action(0.66) == "ESCALATE"


def test_lattice_vs_non_lattice_behavior() -> None:
    trace = _trace_unstable()
    sensitivity, forecast, _ = _bundle(trace, with_lattice=False)
    _, _, lattice = _bundle(trace, with_lattice=True)
    base = compute_retro_trace_control(trace, sensitivity, forecast)
    with_lattice = compute_retro_trace_control(trace, sensitivity, forecast, lattice)
    assert with_lattice.decision.control_score >= base.decision.control_score
    assert len(with_lattice.decision.signals) == 4
    assert len(base.decision.signals) == 3


def test_invalid_input_rejection_and_mismatched_hash() -> None:
    trace = _trace_a()
    sensitivity, forecast, _ = _bundle(trace, with_lattice=False)
    with pytest.raises(ValueError, match="retro_trace must be RetroTraceReceipt"):
        compute_retro_trace_control(object(), sensitivity, forecast)

    other_trace = _trace_unstable()
    other_sensitivity, _, _ = _bundle(other_trace, with_lattice=False)
    with pytest.raises(ValueError, match="sensitivity retro_trace_hash mismatch"):
        compute_retro_trace_control(trace, other_sensitivity, forecast)


def test_bool_rejection_for_numeric_fields() -> None:
    with pytest.raises(ValueError, match=r"value must be numeric in \[0,1\]"):
        RetroTraceControlSignal(name="stability_pressure", value=True, _stable_hash="a" * 64)


def test_snapshot_fixture_lock_canonical_json_and_hash() -> None:
    trace = _trace_a()
    sensitivity, forecast, lattice = _bundle(trace, with_lattice=True)
    receipt = compute_retro_trace_control(trace, sensitivity, forecast, lattice)
    expected_json = (
        '{"decision":{"action":{"action":"HOLD","action_rank":0,"stable_hash":"7441f2df766591c6d17c530000c29e595cb6d358c9072d83604cc82d33c551d5"},'
        '"confidence":0.929368295682,"control_score":0.273447609467,'
        '"signals":[{"name":"stability_pressure","stable_hash":"1232781a4dc34994875e3c1d387f26cc9f36f1e55beb57eb2ae2b18cffd6b7c5","value":0.317047410481},'
        '{"name":"divergence_pressure","stable_hash":"ce95384aefb66a12995d8cf31b110c44ab09c5d5840949e2954b61e0fdc5c93b","value":0.142857142857},'
        '{"name":"forecast_risk","stable_hash":"c1ce266b6490effae6914a6ac5ac7023ccb5cd828d4d16fc2e6a5cf8731b43c9","value":0.15},'
        '{"name":"locality_pressure","stable_hash":"c8b0713dfaa364072d46751523e0c84db6e7a38b9b7df716dc695caf50ff1ee8","value":0.293857512092}],'
        '"stable_hash":"b89005b9a5095b2897817193b57b4d073dae821ec2ef1af7fde372aaf7c351fa"},'
        '"forecast_hash":"06de442dd7af4c8853dc7e82afef5d4fdaec41057aa4ba1cd9b5a9f8c268c3bd",'
        '"lattice_hash":"b7bb282a0e831e56b3678ec51f1165bc46db1cc5098d45201177855664b17c6c",'
        '"retro_trace_hash":"cf888b36a3e553e742032fdf9a19ab9aaa7850fe6f14df2ff93eced9f06ebd62",'
        '"sensitivity_hash":"e1cc698402f79c353a57996eb64d223d056c287790cc569f2b4f984148462e82",'
        '"stable_hash":"8f0246083b37a80567957b3faf15c891c8ac5628f17c49922d8d77f1a01a226e",'
        '"summary":{"confidence_score":0.929368295682,"decision_rationale":"action=HOLD;dominant=stability;score=0.273447609467;confidence=0.929368295682;region=LLL",'
        '"dominant_factor":"stability","stable_hash":"63af2f8066bd6ab54a984cd617533ebf610e1695dc5dbdb429faf69ac1502cd5"}}'
    )
    expected_hash = "8f0246083b37a80567957b3faf15c891c8ac5628f17c49922d8d77f1a01a226e"
    assert receipt.to_canonical_json() == expected_json
    assert receipt.stable_hash() == expected_hash


def test_direct_receipt_reconstruction_fail_fast() -> None:
    trace = _trace_a()
    sensitivity, forecast, lattice = _bundle(trace, with_lattice=True)
    receipt = compute_retro_trace_control(trace, sensitivity, forecast, lattice)
    summary_payload = {
        "dominant_factor": receipt.summary.dominant_factor,
        "decision_rationale": receipt.summary.decision_rationale,
        "confidence_score": 0.0,
    }
    tampered_summary = RetroTraceControlSummary(
        dominant_factor=summary_payload["dominant_factor"],
        decision_rationale=summary_payload["decision_rationale"],
        confidence_score=summary_payload["confidence_score"],
        _stable_hash=sha256_hex(summary_payload),
    )
    with pytest.raises(ValueError, match="summary confidence_score mismatch"):
        RetroTraceControlReceipt(
            retro_trace_hash=receipt.retro_trace_hash,
            sensitivity_hash=receipt.sensitivity_hash,
            forecast_hash=receipt.forecast_hash,
            lattice_hash=receipt.lattice_hash,
            decision=receipt.decision,
            summary=tampered_summary,
            _stable_hash=receipt.stable_hash(),
        )


def test_reject_non_tuple_signals_structure() -> None:
    trace = _trace_a()
    sensitivity, forecast, _ = _bundle(trace, with_lattice=False)
    receipt = compute_retro_trace_control(trace, sensitivity, forecast)
    with pytest.raises(ValueError, match="signals must be non-empty tuple"):
        RetroTraceControlDecision(
            action=receipt.decision.action,
            control_score=receipt.decision.control_score,
            confidence=receipt.decision.confidence,
            signals=list(receipt.decision.signals),
            _stable_hash=receipt.decision.stable_hash(),
        )
