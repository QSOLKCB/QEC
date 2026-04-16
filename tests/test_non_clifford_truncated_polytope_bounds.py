# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.1.3 non-Clifford truncated polytope bounds."""

from __future__ import annotations

import json

import pytest

from qec.simulation.non_clifford_truncated_polytope_bounds import (
    NON_CLIFFORD_TRUNCATED_POLYTOPE_BOUNDS_VERSION,
    GateProfile,
    NonCliffordBoundsValidationError,
    admissibility_projection,
    build_non_clifford_bounds,
    compute_polytope_bounds,
    validate_non_clifford_bounds,
)


def _profile(*, weight: float = 0.6, family: str = "t_gate") -> dict:
    return {
        "gate_family": family,
        "gate_sequence": ("g0", "g1", "g2"),
        "non_clifford_weight": weight,
        "approximation_policy": "deterministic_truncated_polytope",
        "metadata": {"release": NON_CLIFFORD_TRUNCATED_POLYTOPE_BOUNDS_VERSION},
    }


def test_deterministic_same_input_same_bytes() -> None:
    analysis_a = build_non_clifford_bounds(gate_profile=_profile(), truncation_level=3)
    analysis_b = build_non_clifford_bounds(gate_profile=_profile(), truncation_level=3)
    assert analysis_a.to_canonical_json() == analysis_b.to_canonical_json()


def test_lower_bound_upper_bound_invariant() -> None:
    bounds = compute_polytope_bounds(profile=_profile(), truncation_level=2)
    assert all(bound.lower_bound <= bound.upper_bound for bound in bounds)


def test_higher_truncation_tightens_bounds() -> None:
    low = compute_polytope_bounds(profile=_profile(), truncation_level=1)
    high = compute_polytope_bounds(profile=_profile(), truncation_level=5)
    low_width = low[0].upper_bound - low[0].lower_bound
    high_width = high[0].upper_bound - high[0].lower_bound
    assert high_width < low_width


def test_unsupported_gate_family_rejected() -> None:
    with pytest.raises(NonCliffordBoundsValidationError, match="unsupported gate family"):
        build_non_clifford_bounds(gate_profile=_profile(family="unknown"), truncation_level=1)


def test_nan_inf_rejection() -> None:
    with pytest.raises(NonCliffordBoundsValidationError, match="finite"):
        build_non_clifford_bounds(gate_profile=_profile(weight=float("nan")), truncation_level=1)

    with pytest.raises(NonCliffordBoundsValidationError, match="finite"):
        build_non_clifford_bounds(gate_profile=_profile(weight=float("inf")), truncation_level=1)


def test_receipt_hash_stable() -> None:
    analysis = build_non_clifford_bounds(gate_profile=_profile(), truncation_level=4)
    assert analysis.receipt.receipt_hash == analysis.receipt.stable_hash()


def test_receipt_hash_changes_when_threshold_policy_changes() -> None:
    baseline = build_non_clifford_bounds(
        gate_profile=_profile(),
        truncation_level=2,
        policy_flags={"admissibility_threshold": 0.5},
    )
    changed_threshold = build_non_clifford_bounds(
        gate_profile=_profile(),
        truncation_level=2,
        policy_flags={"admissibility_threshold": 0.7},
    )
    assert [bound.admissible for bound in baseline.bounds] == [bound.admissible for bound in changed_threshold.bounds]
    assert baseline.receipt.admissibility_hash != changed_threshold.receipt.admissibility_hash
    assert baseline.receipt.receipt_hash != changed_threshold.receipt.receipt_hash


def test_admissibility_changes_with_truncation() -> None:
    low = build_non_clifford_bounds(gate_profile=_profile(weight=0.6), truncation_level=0)
    high = build_non_clifford_bounds(gate_profile=_profile(weight=0.6), truncation_level=2)
    assert not all(bound.admissible for bound in low.bounds)
    assert all(bound.admissible for bound in high.bounds)


def test_canonical_json_round_trip() -> None:
    analysis = build_non_clifford_bounds(gate_profile=_profile(), truncation_level=3)
    j1 = analysis.to_canonical_json()
    j2 = json.dumps(json.loads(j1), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    assert j1 == j2


def test_orchestration_compatibility_helper_stable() -> None:
    analysis = build_non_clifford_bounds(gate_profile=_profile(), truncation_level=3)
    p1 = admissibility_projection(analysis=analysis, lane_id="lane-01")
    p2 = admissibility_projection(analysis=analysis, lane_id="lane-01")
    assert p1 == p2
    assert p1["projection_hash"] == p2["projection_hash"]


def test_validate_receipt_consistency_failure() -> None:
    analysis = build_non_clifford_bounds(gate_profile=_profile(), truncation_level=3)
    tampered = type(analysis)(
        profile=analysis.profile,
        bounds=analysis.bounds,
        receipt=type(analysis.receipt)(
            profile_hash="0" * 64,
            bound_set_hash=analysis.receipt.bound_set_hash,
            admissibility_hash=analysis.receipt.admissibility_hash,
            admissibility_threshold=analysis.receipt.admissibility_threshold,
            validation_passed=True,
            receipt_hash=analysis.receipt.receipt_hash,
        ),
    )
    valid, errors = validate_non_clifford_bounds(tampered)
    assert not valid
    assert any("profile_hash mismatch" in item for item in errors)


def test_dataclass_profile_input_supported() -> None:
    profile = GateProfile(
        gate_family="toffoli",
        gate_sequence=("toffoli",),
        non_clifford_weight=0.2,
        approximation_policy="deterministic",
        metadata={},
    )
    analysis = build_non_clifford_bounds(gate_profile=profile, truncation_level=1)
    assert analysis.profile.gate_family == "toffoli"
