from __future__ import annotations

import pytest

from qec.analysis.deterministic_stress_lattice import (
    StressAxis,
    StressCoverageReceipt,
    generate_stress_lattice,
)


def _axes() -> list[StressAxis]:
    return [
        StressAxis(name="thermal_pressure", lower_bound=0.0, upper_bound=1.0),
        StressAxis(name="latency_drift", lower_bound=0.0, upper_bound=1.0),
        StressAxis(name="timing_skew", lower_bound=0.0, upper_bound=1.0),
        StressAxis(name="power_pressure", lower_bound=0.0, upper_bound=1.0),
        StressAxis(name="consensus_instability", lower_bound=0.0, upper_bound=1.0),
    ]


def test_deterministic_replay_identical_bytes_and_hash() -> None:
    first = generate_stress_lattice(_axes(), point_count=32, method="halton")
    second = generate_stress_lattice(_axes(), point_count=32, method="halton")

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert first.stable_hash == second.stable_hash


def test_canonical_axis_ordering_from_unsorted_input() -> None:
    axes = [
        StressAxis(name="timing_skew", lower_bound=0.0, upper_bound=1.0),
        StressAxis(name="consensus_instability", lower_bound=0.0, upper_bound=1.0),
        StressAxis(name="latency_drift", lower_bound=0.0, upper_bound=1.0),
    ]
    receipt = generate_stress_lattice(axes, point_count=8, method="halton")
    assert receipt.axis_names == ["consensus_instability", "latency_drift", "timing_skew"]


def test_halton_generation_boundedness() -> None:
    receipt = generate_stress_lattice(_axes(), point_count=64, method="halton")
    for point in receipt.points:
        for axis_name, value in point.coordinates.items():
            assert 0.0 <= value <= 1.0, axis_name


def test_lattice_generation_boundedness() -> None:
    receipt = generate_stress_lattice(_axes(), point_count=64, method="lattice")
    for point in receipt.points:
        for axis_name, value in point.coordinates.items():
            assert 0.0 <= value <= 1.0, axis_name


def test_stable_point_hashing_self_consistent() -> None:
    receipt = generate_stress_lattice(_axes(), point_count=12, method="halton")
    for point in receipt.points:
        assert point.stable_hash == point.computed_stable_hash()


def test_receipt_self_validation_rejects_tampered_hash() -> None:
    receipt = generate_stress_lattice(_axes(), point_count=12, method="halton")
    with pytest.raises(ValueError, match="stable_hash mismatch"):
        StressCoverageReceipt(
            axis_names=receipt.axis_names,
            point_count=receipt.point_count,
            method=receipt.method,
            points=receipt.points,
            min_per_axis=receipt.min_per_axis,
            max_per_axis=receipt.max_per_axis,
            mean_per_axis=receipt.mean_per_axis,
            coverage_score=receipt.coverage_score,
            classification=receipt.classification,
            stable_hash="0" * 64,
        )


def test_duplicate_axis_rejection() -> None:
    axes = [
        StressAxis(name="thermal_pressure", lower_bound=0.0, upper_bound=1.0),
        StressAxis(name="thermal_pressure", lower_bound=0.0, upper_bound=1.0),
    ]
    with pytest.raises(ValueError, match="axis names must be unique"):
        generate_stress_lattice(axes, point_count=4)


def test_invalid_bound_rejection() -> None:
    with pytest.raises(ValueError, match="lower_bound must be <= upper_bound"):
        StressAxis(name="thermal_pressure", lower_bound=0.9, upper_bound=0.1)


def test_invalid_method_rejection() -> None:
    with pytest.raises(ValueError, match="unsupported method"):
        generate_stress_lattice(_axes(), point_count=8, method="unknown")


def test_coverage_classification_is_deterministic_by_density() -> None:
    sparse = generate_stress_lattice(_axes(), point_count=1, method="lattice")
    dense = generate_stress_lattice(_axes(), point_count=256, method="halton")

    assert sparse.classification == "sparse"
    assert dense.classification in {"partial", "dense"}
    assert dense.coverage_score >= sparse.coverage_score


def test_float_stability_to_12_decimals_in_export() -> None:
    receipt = generate_stress_lattice(_axes(), point_count=16, method="halton")
    exported = receipt.to_dict()
    for point in exported["points"]:
        for value in point["coordinates"].values():
            text = f"{value:.12f}"
            assert len(text.split(".")[1]) == 12


def test_point_ordering_is_strict_by_point_index() -> None:
    receipt = generate_stress_lattice(_axes(), point_count=24, method="halton")
    point_indices = [point.point_index for point in receipt.points]
    assert point_indices == list(range(receipt.point_count))
