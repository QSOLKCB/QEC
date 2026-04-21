"""Tests for v138.9.1 deterministic multi-code migration engine."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.multi_code_migration_engine import (
    CodeStateProfile,
    MigrationPolicy,
    plan_code_migration,
)


def _profile(code_family: str = "ternary", observables: tuple[str, ...] = ("X", "Z")) -> CodeStateProfile:
    return CodeStateProfile(
        code_id="code-alpha",
        code_family=code_family,
        distance=11,
        logical_error_rate=0.120000000000,
        syndrome_density=0.440000000000,
        check_density=0.380000000000,
        hardware_alignment=0.800000000000,
        stability_score=0.910000000000,
        observables=observables,
        metadata={"operator": "lab-a", "run": "v138"},
    )


def _policy(target_family: str = "surface") -> MigrationPolicy:
    return MigrationPolicy(
        target_family=target_family,
        minimum_compatibility=0.600000000000,
        maximum_projected_loss=0.550000000000,
        require_observable_overlap=True,
        prefer_distance_preservation=True,
        prefer_hardware_alignment=True,
    )


def test_ternary_to_surface_migration_plan_success() -> None:
    receipt = plan_code_migration(_profile("ternary"), _policy("surface"))
    assert receipt.source_code_family == "ternary"
    assert receipt.target_family == "surface"
    assert receipt.assessment.admissible is True
    assert receipt.selected_migration_path == "ternary_to_surface"


def test_qldpc_to_surface_migration_success() -> None:
    receipt = plan_code_migration(_profile("qldpc"), _policy("surface"))
    assert receipt.assessment.admissible is True
    assert receipt.selected_migration_path == "qldpc_to_surface"


def test_unsupported_migration_rejection() -> None:
    with pytest.raises(ValueError, match="unsupported migration pair"):
        plan_code_migration(_profile("surface"), _policy("qldpc"))


def test_invalid_metric_rejection() -> None:
    with pytest.raises(ValueError, match=r"finite float in \[0, 1\]"):
        CodeStateProfile(
            code_id="bad",
            code_family="ternary",
            distance=3,
            logical_error_rate=1.100000000000,
            syndrome_density=0.200000000000,
            check_density=0.200000000000,
            hardware_alignment=0.200000000000,
            stability_score=0.200000000000,
            observables=("X",),
        )


def test_duplicate_observable_rejection() -> None:
    with pytest.raises(ValueError, match="duplicate-free"):
        _profile(observables=("X", "X"))


def test_inadmissible_migration_under_strict_policy() -> None:
    strict = MigrationPolicy(
        target_family="surface",
        minimum_compatibility=0.980000000000,
        maximum_projected_loss=0.050000000000,
        require_observable_overlap=True,
        prefer_distance_preservation=False,
        prefer_hardware_alignment=False,
    )
    receipt = plan_code_migration(_profile("ternary"), strict)
    assert receipt.assessment.admissible is False
    assert "compatibility_below_policy_minimum" in receipt.assessment.reasons


def test_canonical_json_stability() -> None:
    receipt = plan_code_migration(_profile("ternary"), _policy("surface"))
    assert receipt.to_canonical_json() == receipt.to_canonical_json()
    assert receipt.to_canonical_bytes() == receipt.to_canonical_bytes()


def test_stable_hash_and_replay_identity_determinism() -> None:
    receipt_a = plan_code_migration(_profile("ternary"), _policy("surface"))
    receipt_b = plan_code_migration(_profile("ternary"), _policy("surface"))
    assert receipt_a.replay_identity == receipt_b.replay_identity
    assert receipt_a.stable_hash == receipt_b.stable_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_frozen_dataclass_immutability() -> None:
    profile = _profile("ternary")
    with pytest.raises(FrozenInstanceError):
        profile.code_id = "changed"  # type: ignore[misc]


def test_deterministic_step_ordering() -> None:
    receipt = plan_code_migration(_profile("ternary"), _policy("surface"))
    step_types = tuple(step.step_type for step in receipt.migration_steps)
    assert step_types == (
        "normalize_source_profile",
        "map_observable_basis",
        "estimate_distance_retention",
        "project_density_metrics",
        "compute_target_alignment",
        "emit_migration_verdict",
    )
    step_indices = tuple(step.step_index for step in receipt.migration_steps)
    assert step_indices == (0, 1, 2, 3, 4, 5)
