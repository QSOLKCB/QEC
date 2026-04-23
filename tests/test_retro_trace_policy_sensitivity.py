from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.governed_orchestration_layer import GovernancePolicy
from qec.analysis.retro_target_registry import build_retro_target
from qec.analysis.retro_trace_intake_bridge import build_retro_trace
from qec.analysis.retro_trace_policy_sensitivity import (
    RetroTracePolicyComparison,
    RetroTracePolicyRun,
    RetroTracePolicySensitivityReceipt,
    analyze_retro_trace_policy_sensitivity,
)


def _target_receipt():
    return build_retro_target(
        target_id="z80-home-micro",
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


def _retro_trace():
    return build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=({"pc": 0x1000, "a": 0x10}, {"pc": 0x1001, "a": 0x11}),
        memory_trace=({"address": 0x4000, "op": "read", "value": 0xAB},),
        timing_trace=({"cycle": 100}, {"cycle": 120}),
        display_trace=({"scanline": 0, "event": "start"},),
        audio_trace=({"channel": 1, "pattern": "pulse"},),
        input_trace=({"port": 1, "button": "A", "state": 1},),
        metadata={"emulator": "retroarch", "rom_hash": "abc123", "version": "1.0.0"},
    )


def _policy(
    *,
    min_required_score: float = 0.55,
    min_required_confidence: float = 0.95,
    min_required_margin: float = 0.45,
    min_required_convergence: float = 0.90,
    allow_tie_break: bool = False,
    allow_no_improvement: bool = False,
    require_stable_transition: bool = True,
) -> GovernancePolicy:
    payload = {
        "min_required_score": round(min_required_score, 12),
        "min_required_confidence": round(min_required_confidence, 12),
        "min_required_margin": round(min_required_margin, 12),
        "min_required_convergence": round(min_required_convergence, 12),
        "allow_tie_break": allow_tie_break,
        "allow_no_improvement": allow_no_improvement,
        "require_stable_transition": require_stable_transition,
    }
    return GovernancePolicy(
        min_required_score=min_required_score,
        min_required_confidence=min_required_confidence,
        min_required_margin=min_required_margin,
        min_required_convergence=min_required_convergence,
        allow_tie_break=allow_tie_break,
        allow_no_improvement=allow_no_improvement,
        require_stable_transition=require_stable_transition,
        stable_hash=sha256_hex(payload),
    )


def test_deterministic_replay_identical_hash() -> None:
    retro_trace = _retro_trace()
    policies = (_policy(min_required_score=0.9), _policy(min_required_score=0.2, allow_tie_break=True))
    first = analyze_retro_trace_policy_sensitivity(retro_trace, policies)
    second = analyze_retro_trace_policy_sensitivity(retro_trace, policies)
    assert first.stable_hash() == second.stable_hash()
    assert first.to_canonical_bytes() == second.to_canonical_bytes()


def test_policy_ordering_invariance() -> None:
    retro_trace = _retro_trace()
    p1 = _policy(min_required_score=0.2)
    p2 = _policy(min_required_score=0.8)
    p3 = _policy(min_required_score=0.4, allow_tie_break=True)
    a = analyze_retro_trace_policy_sensitivity(retro_trace, (p1, p2, p3))
    b = analyze_retro_trace_policy_sensitivity(retro_trace, (p3, p1, p2))
    assert a.stable_hash() == b.stable_hash()
    assert a.to_canonical_json() == b.to_canonical_json()


def test_identical_policies_zero_sensitivity() -> None:
    retro_trace = _retro_trace()
    policy = _policy(min_required_score=0.6)
    receipt = analyze_retro_trace_policy_sensitivity(retro_trace, (policy, policy))
    assert receipt.summary.sensitivity_score == 0.0
    assert receipt.summary.classification == "LOW"


def test_high_divergence_high_classification() -> None:
    retro_trace = _retro_trace()
    strict = _policy(
        min_required_score=1.0,
        min_required_confidence=1.0,
        min_required_margin=1.0,
        min_required_convergence=1.0,
        allow_tie_break=False,
        allow_no_improvement=False,
        require_stable_transition=True,
    )
    permissive = _policy(
        min_required_score=0.0,
        min_required_confidence=0.0,
        min_required_margin=0.0,
        min_required_convergence=0.0,
        allow_tie_break=True,
        allow_no_improvement=True,
        require_stable_transition=False,
    )
    receipt = analyze_retro_trace_policy_sensitivity(retro_trace, (strict, permissive))
    assert receipt.summary.classification == "HIGH"


def test_reject_invalid_policy_count() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        analyze_retro_trace_policy_sensitivity(_retro_trace(), (_policy(),))


def test_bounded_outputs() -> None:
    retro_trace = _retro_trace()
    receipt = analyze_retro_trace_policy_sensitivity(
        retro_trace,
        (_policy(min_required_score=0.9), _policy(min_required_score=0.3), _policy(min_required_score=0.1, allow_tie_break=True)),
    )
    assert 0.0 <= receipt.summary.sensitivity_score <= 1.0
    for run in receipt.policy_runs:
        assert 0.0 <= run.strictness <= 1.0
        assert 0.0 <= run.compatibility <= 1.0
        for _, value in run.metrics:
            assert 0.0 <= value <= 1.0
    for comparison in receipt.policy_comparisons:
        assert 0.0 <= comparison.metric_distance <= 1.0
        assert 0.0 <= comparison.sensitivity_score <= 1.0
        assert -1.0 <= comparison.strictness_delta <= 1.0
        assert -1.0 <= comparison.compatibility_delta <= 1.0


def test_reject_non_canonical_hex_hash_fields() -> None:
    receipt = analyze_retro_trace_policy_sensitivity(_retro_trace(), (_policy(min_required_score=0.9), _policy(min_required_score=0.3)))
    run = receipt.policy_runs[0]
    with pytest.raises(ValueError, match="64-char lowercase SHA-256 hex"):
        RetroTracePolicyRun(
            policy_hash="F" * 64,
            strictness=run.strictness,
            compatibility=run.compatibility,
            metrics=run.metrics,
            _stable_hash=run.stable_hash(),
        )
    cmp = receipt.policy_comparisons[0]
    with pytest.raises(ValueError, match="64-char lowercase SHA-256 hex"):
        RetroTracePolicyComparison(
            left_policy_hash=cmp.left_policy_hash,
            right_policy_hash="A" * 64,
            strictness_delta=cmp.strictness_delta,
            compatibility_delta=cmp.compatibility_delta,
            metric_distance=cmp.metric_distance,
            sensitivity_score=cmp.sensitivity_score,
            _stable_hash=cmp.stable_hash(),
        )
    with pytest.raises(ValueError, match="64-char lowercase SHA-256 hex"):
        RetroTracePolicySensitivityReceipt(
            retro_trace_hash="B" * 64,
            policy_runs=receipt.policy_runs,
            policy_comparisons=receipt.policy_comparisons,
            summary=receipt.summary,
            _stable_hash=receipt.stable_hash(),
        )


def test_reject_duplicate_policy_comparison_pairs_with_matching_count() -> None:
    receipt = analyze_retro_trace_policy_sensitivity(
        _retro_trace(),
        (
            _policy(min_required_score=0.9),
            _policy(min_required_score=0.3),
            _policy(min_required_score=0.1, allow_tie_break=True),
        ),
    )
    duplicated = (
        receipt.policy_comparisons[0],
        receipt.policy_comparisons[0],
        receipt.policy_comparisons[2],
    )
    tampered_payload = {
        "retro_trace_hash": receipt.retro_trace_hash,
        "policy_runs": tuple(item.to_dict() for item in receipt.policy_runs),
        "policy_comparisons": tuple(item.to_dict() for item in duplicated),
        "summary": receipt.summary.to_dict(),
    }
    with pytest.raises(ValueError, match="cover each unordered policy pair exactly once"):
        RetroTracePolicySensitivityReceipt(
            retro_trace_hash=receipt.retro_trace_hash,
            policy_runs=receipt.policy_runs,
            policy_comparisons=duplicated,
            summary=receipt.summary,
            _stable_hash=sha256_hex(tampered_payload),
        )


def test_reject_reversed_pair_hash_ordering() -> None:
    receipt = analyze_retro_trace_policy_sensitivity(_retro_trace(), (_policy(min_required_score=0.9), _policy(min_required_score=0.3)))
    comparison = receipt.policy_comparisons[0]
    with pytest.raises(ValueError, match=r"^policy comparison must use canonical hash ordering$"):
        RetroTracePolicyComparison(
            left_policy_hash=comparison.right_policy_hash,
            right_policy_hash=comparison.left_policy_hash,
            strictness_delta=comparison.strictness_delta,
            compatibility_delta=comparison.compatibility_delta,
            metric_distance=comparison.metric_distance,
            sensitivity_score=comparison.sensitivity_score,
            _stable_hash=comparison.stable_hash(),
        )


def test_reject_non_canonical_policy_comparison_ordering() -> None:
    receipt = analyze_retro_trace_policy_sensitivity(
        _retro_trace(),
        (
            _policy(min_required_score=0.9),
            _policy(min_required_score=0.3),
            _policy(min_required_score=0.1, allow_tie_break=True),
        ),
    )
    non_canonical = (
        receipt.policy_comparisons[1],
        receipt.policy_comparisons[0],
        receipt.policy_comparisons[2],
    )
    tampered_payload = {
        "retro_trace_hash": receipt.retro_trace_hash,
        "policy_runs": tuple(item.to_dict() for item in receipt.policy_runs),
        "policy_comparisons": tuple(item.to_dict() for item in non_canonical),
        "summary": receipt.summary.to_dict(),
    }
    with pytest.raises(ValueError, match="policy_comparisons must be canonically ordered"):
        RetroTracePolicySensitivityReceipt(
            retro_trace_hash=receipt.retro_trace_hash,
            policy_runs=receipt.policy_runs,
            policy_comparisons=non_canonical,
            summary=receipt.summary,
            _stable_hash=sha256_hex(tampered_payload),
        )


def test_reject_summary_values_inconsistent_with_comparisons() -> None:
    receipt = analyze_retro_trace_policy_sensitivity(_retro_trace(), (_policy(min_required_score=0.9), _policy(min_required_score=0.3)))
    tampered_summary = replace(
        receipt.summary,
        sensitivity_score=0.0,
        _stable_hash=sha256_hex(
            {
                "policy_count": receipt.summary.policy_count,
                "comparison_count": receipt.summary.comparison_count,
                "sensitivity_score": 0.0,
                "classification": "LOW",
            }
        ),
    )
    tampered_payload = {
        "retro_trace_hash": receipt.retro_trace_hash,
        "policy_runs": tuple(item.to_dict() for item in receipt.policy_runs),
        "policy_comparisons": tuple(item.to_dict() for item in receipt.policy_comparisons),
        "summary": tampered_summary.to_dict(),
    }
    with pytest.raises(ValueError, match="summary sensitivity_score mismatch"):
        RetroTracePolicySensitivityReceipt(
            retro_trace_hash=receipt.retro_trace_hash,
            policy_runs=receipt.policy_runs,
            policy_comparisons=receipt.policy_comparisons,
            summary=tampered_summary,
            _stable_hash=sha256_hex(tampered_payload),
        )
