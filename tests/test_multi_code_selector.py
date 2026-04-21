from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.multi_code_selector import (
    CodeCandidateProfile,
    CodeSelectionPolicy,
    select_runtime_code,
)


def _base_policy() -> CodeSelectionPolicy:
    return CodeSelectionPolicy(
        weight_logical_stability=2.0,
        weight_latency_efficiency=1.0,
        weight_overhead_efficiency=1.0,
        weight_hardware_alignment=2.0,
        weight_noise_fit=2.0,
        weight_convergence_confidence=2.0,
        min_noise_fit=0.6,
        min_hardware_alignment=0.5,
        min_convergence_confidence=0.7,
    )


def test_correct_selection_prefers_higher_weighted_score() -> None:
    policy = _base_policy()
    candidates = [
        CodeCandidateProfile(
            code_id="surface-a",
            code_family="surface",
            logical_stability=0.85,
            latency_efficiency=0.75,
            overhead_efficiency=0.7,
            hardware_alignment=0.8,
            noise_fit=0.82,
            convergence_confidence=0.81,
        ),
        CodeCandidateProfile(
            code_id="qldpc-a",
            code_family="qldpc",
            logical_stability=0.92,
            latency_efficiency=0.88,
            overhead_efficiency=0.84,
            hardware_alignment=0.78,
            noise_fit=0.91,
            convergence_confidence=0.9,
        ),
    ]

    receipt = select_runtime_code(candidates, policy)

    assert receipt.selected_code_id == "qldpc-a"
    assert receipt.selected_code_family == "qldpc"
    assert receipt.ranking_order[0] == "qldpc-a"


def test_tie_break_is_deterministic() -> None:
    policy = _base_policy()
    candidate_a = CodeCandidateProfile(
        code_id="beta",
        code_family="surface",
        logical_stability=0.8,
        latency_efficiency=0.8,
        overhead_efficiency=0.8,
        hardware_alignment=0.7,
        noise_fit=0.8,
        convergence_confidence=0.8,
    )
    candidate_b = CodeCandidateProfile(
        code_id="alpha",
        code_family="qldpc",
        logical_stability=0.8,
        latency_efficiency=0.8,
        overhead_efficiency=0.8,
        hardware_alignment=0.7,
        noise_fit=0.8,
        convergence_confidence=0.8,
    )

    receipt = select_runtime_code([candidate_a, candidate_b], policy)

    assert receipt.selected_code_id == "alpha"
    assert receipt.ranking_order == ("alpha", "beta")


def test_invalid_metric_rejection() -> None:
    with pytest.raises(ValueError, match="logical_stability"):
        CodeCandidateProfile(
            code_id="invalid",
            code_family="surface",
            logical_stability=float("nan"),
            latency_efficiency=0.5,
            overhead_efficiency=0.5,
            hardware_alignment=0.5,
            noise_fit=0.5,
            convergence_confidence=0.5,
        )


def test_invalid_weight_rejection() -> None:
    with pytest.raises(ValueError, match="weight_logical_stability"):
        CodeSelectionPolicy(
            weight_logical_stability=0.0,
            weight_latency_efficiency=1.0,
            weight_overhead_efficiency=1.0,
            weight_hardware_alignment=1.0,
            weight_noise_fit=1.0,
            weight_convergence_confidence=1.0,
            min_noise_fit=0.5,
            min_hardware_alignment=0.5,
            min_convergence_confidence=0.5,
        )


def test_no_admissible_candidate_rejection() -> None:
    policy = _base_policy()
    candidates = [
        CodeCandidateProfile(
            code_id="surface-low-noise",
            code_family="surface",
            logical_stability=0.9,
            latency_efficiency=0.9,
            overhead_efficiency=0.9,
            hardware_alignment=0.9,
            noise_fit=0.2,
            convergence_confidence=0.9,
        )
    ]

    with pytest.raises(ValueError, match="no admissible"):
        select_runtime_code(candidates, policy)


def test_canonical_json_stability() -> None:
    policy = _base_policy()
    candidates = [
        CodeCandidateProfile(
            code_id="surface-a",
            code_family="surface",
            logical_stability=0.9,
            latency_efficiency=0.8,
            overhead_efficiency=0.7,
            hardware_alignment=0.8,
            noise_fit=0.9,
            convergence_confidence=0.9,
        ),
        CodeCandidateProfile(
            code_id="qldpc-b",
            code_family="qldpc",
            logical_stability=0.88,
            latency_efficiency=0.84,
            overhead_efficiency=0.81,
            hardware_alignment=0.79,
            noise_fit=0.89,
            convergence_confidence=0.85,
        ),
    ]

    receipt_a = select_runtime_code(candidates, policy)
    receipt_b = select_runtime_code(list(reversed(candidates)), policy)

    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_stable_hash_determinism() -> None:
    policy = _base_policy()
    candidates = [
        CodeCandidateProfile(
            code_id="surface-a",
            code_family="surface",
            logical_stability=0.7,
            latency_efficiency=0.7,
            overhead_efficiency=0.7,
            hardware_alignment=0.7,
            noise_fit=0.7,
            convergence_confidence=0.7,
        ),
        CodeCandidateProfile(
            code_id="qldpc-a",
            code_family="qldpc",
            logical_stability=0.75,
            latency_efficiency=0.75,
            overhead_efficiency=0.75,
            hardware_alignment=0.75,
            noise_fit=0.75,
            convergence_confidence=0.75,
        ),
    ]

    receipt = select_runtime_code(candidates, policy)

    assert receipt.stable_hash() == receipt.stable_hash()
    assert receipt.replay_identity == select_runtime_code(candidates, policy).replay_identity


def test_frozen_dataclass_immutability() -> None:
    policy = _base_policy()
    candidates = [
        CodeCandidateProfile(
            code_id="surface-a",
            code_family="surface",
            logical_stability=0.7,
            latency_efficiency=0.7,
            overhead_efficiency=0.7,
            hardware_alignment=0.7,
            noise_fit=0.7,
            convergence_confidence=0.7,
        )
    ]

    receipt = select_runtime_code(candidates, policy)

    with pytest.raises(FrozenInstanceError):
        receipt.selected_code_id = "new"  # type: ignore[misc]
