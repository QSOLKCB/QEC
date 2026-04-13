from __future__ import annotations

import math

import pytest

from qec.analysis.jensen_shannon_signal_divergence_kernel import (
    SignalDistribution,
    build_signal_distribution,
)
from qec.analysis.fisher_rao_geometry_approximation_layer import (
    SCHEMA_VERSION,
    FisherRaoConfig,
    FisherRaoReceipt,
    FisherRaoReport,
    FisherRaoResult,
    build_ascii_fisher_rao_summary,
    compute_fisher_rao_distance,
    compute_geodesic_alignment,
    run_fisher_rao_geometry_layer,
)


def test_same_input_produces_same_bytes() -> None:
    report1, receipt1 = run_fisher_rao_geometry_layer(
        {"a": 4.0, "b": 1.0}, {"a": 2.0, "b": 3.0}
    )
    report2, receipt2 = run_fisher_rao_geometry_layer(
        {"a": 4.0, "b": 1.0}, {"a": 2.0, "b": 3.0}
    )

    assert report1.to_canonical_bytes() == report2.to_canonical_bytes()
    assert report1.to_canonical_json() == report2.to_canonical_json()
    assert receipt1.to_canonical_bytes() == receipt2.to_canonical_bytes()


def test_same_input_produces_same_hash() -> None:
    report1, receipt1 = run_fisher_rao_geometry_layer(
        {"a": 9.0, "b": 1.0}, {"a": 5.0, "b": 5.0}
    )
    report2, receipt2 = run_fisher_rao_geometry_layer(
        {"a": 9.0, "b": 1.0}, {"a": 5.0, "b": 5.0}
    )

    assert report1.report_hash == report2.report_hash
    assert report1.geometry_result.result_hash == report2.geometry_result.result_hash
    assert receipt1.receipt_hash == receipt2.receipt_hash


def test_repeated_run_byte_identity() -> None:
    seen: set[bytes] = set()
    for _ in range(5):
        report, receipt = run_fisher_rao_geometry_layer(
            {"x": 1.0, "y": 2.0, "z": 3.0},
            {"x": 3.0, "y": 2.0, "z": 1.0},
        )
        seen.add(report.to_canonical_bytes())
        seen.add(receipt.to_canonical_bytes())
    # Two distinct artifacts (report + receipt), both stable across runs.
    assert len(seen) == 2


def test_identical_distributions_have_zero_distance() -> None:
    dist = build_signal_distribution({"x": 2.0, "y": 2.0, "z": 6.0})

    distance = compute_fisher_rao_distance(dist, dist)
    alignment = compute_geodesic_alignment(dist, dist)

    assert distance == 0.0
    assert alignment == 0.0

    report, _ = run_fisher_rao_geometry_layer(
        {"x": 2.0, "y": 2.0, "z": 6.0},
        {"x": 2.0, "y": 2.0, "z": 6.0},
    )
    r = report.geometry_result
    assert r.fisher_rao_distance == 0.0
    assert r.normalized_fisher_rao_score == 0.0
    assert r.fisher_rao_distance_score == 0.0
    assert r.geodesic_alignment_score == 0.0
    assert r.manifold_consistency_score == 0.0
    assert r.global_information_geometry_score == 0.0


def test_symmetry_check() -> None:
    a = build_signal_distribution({"a": 3.0, "b": 1.0, "c": 2.0})
    b = build_signal_distribution({"a": 1.0, "b": 3.0, "c": 4.0})

    left_distance = compute_fisher_rao_distance(a, b)
    right_distance = compute_fisher_rao_distance(b, a)
    assert left_distance == right_distance

    left_align = compute_geodesic_alignment(a, b)
    right_align = compute_geodesic_alignment(b, a)
    assert left_align == right_align

    report_lr, _ = run_fisher_rao_geometry_layer(
        {"a": 3.0, "b": 1.0, "c": 2.0},
        {"a": 1.0, "b": 3.0, "c": 4.0},
    )
    report_rl, _ = run_fisher_rao_geometry_layer(
        {"a": 1.0, "b": 3.0, "c": 4.0},
        {"a": 3.0, "b": 1.0, "c": 2.0},
    )

    lr = report_lr.geometry_result
    rl = report_rl.geometry_result
    assert lr.fisher_rao_distance == rl.fisher_rao_distance
    assert lr.normalized_fisher_rao_score == rl.normalized_fisher_rao_score
    assert lr.fisher_rao_distance_score == rl.fisher_rao_distance_score
    assert lr.geodesic_alignment_score == rl.geodesic_alignment_score
    assert lr.manifold_consistency_score == rl.manifold_consistency_score
    assert lr.global_information_geometry_score == rl.global_information_geometry_score


def test_scores_are_bounded_unit_interval() -> None:
    report, _ = run_fisher_rao_geometry_layer(
        {"alpha": 7.0, "beta": 2.0, "gamma": 1.0},
        {"alpha": 1.0, "beta": 4.0, "gamma": 5.0},
    )
    r = report.geometry_result

    assert 0.0 <= r.fisher_rao_distance <= math.pi + 1e-9
    for score in (
        r.normalized_fisher_rao_score,
        r.fisher_rao_distance_score,
        r.geodesic_alignment_score,
        r.manifold_consistency_score,
        r.global_information_geometry_score,
    ):
        assert 0.0 <= score <= 1.0


def test_invalid_distribution_rejected() -> None:
    bad = SignalDistribution(
        labels=("a", "b"), probabilities=(0.8, 0.3), distribution_hash="0" * 64
    )
    good = build_signal_distribution({"a": 1.0, "b": 1.0})

    with pytest.raises(ValueError, match="sum to 1"):
        compute_fisher_rao_distance(bad, good)
    with pytest.raises(ValueError, match="sum to 1"):
        compute_geodesic_alignment(bad, good)

    negative = SignalDistribution(
        labels=("a", "b"), probabilities=(-0.1, 1.1), distribution_hash="0" * 64
    )
    with pytest.raises(ValueError):
        compute_fisher_rao_distance(negative, good)


def test_receipt_integrity_and_lineage() -> None:
    report, receipt = run_fisher_rao_geometry_layer(
        {"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 1.0}
    )

    assert isinstance(report, FisherRaoReport)
    assert isinstance(receipt, FisherRaoReceipt)
    assert isinstance(report.geometry_result, FisherRaoResult)
    assert isinstance(report.config, FisherRaoConfig)

    assert receipt.report_hash == report.report_hash
    assert receipt.byte_length == len(report.to_canonical_bytes())
    assert receipt.validation_passed is True
    assert receipt.config_hash == report.config.config_hash
    assert receipt.source_distribution_hash == report.source_distribution.distribution_hash
    assert receipt.target_distribution_hash == report.target_distribution.distribution_hash
    assert receipt.result_hash == report.geometry_result.result_hash
    assert report.schema_version == SCHEMA_VERSION
    assert len(receipt.receipt_hash) == 64


def test_wrapper_matches_manual_pipeline() -> None:
    source = {"s0": 4.0, "s1": 1.0, "s2": 5.0}
    target = {"s0": 2.0, "s1": 3.0, "s2": 5.0}

    report, receipt = run_fisher_rao_geometry_layer(source, target)

    manual_source = build_signal_distribution(source)
    manual_target = build_signal_distribution(target)
    manual_distance = compute_fisher_rao_distance(manual_source, manual_target)
    manual_alignment = compute_geodesic_alignment(manual_source, manual_target)

    assert report.source_distribution == manual_source
    assert report.target_distribution == manual_target
    assert abs(report.geometry_result.fisher_rao_distance - manual_distance) < 1e-9
    assert abs(report.geometry_result.geodesic_alignment_score - manual_alignment) < 1e-9
    assert receipt.report_hash == report.report_hash


def test_maximal_divergence_bounded() -> None:
    a = build_signal_distribution({"a": 1.0, "b": 0.0})
    b = build_signal_distribution({"a": 0.0, "b": 1.0})

    distance = compute_fisher_rao_distance(a, b)
    assert abs(distance - math.pi) < 1e-12

    report, _ = run_fisher_rao_geometry_layer(
        {"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}
    )
    r = report.geometry_result
    assert abs(r.fisher_rao_distance - math.pi) < 1e-9
    assert r.normalized_fisher_rao_score == 1.0
    assert r.fisher_rao_distance_score == 1.0
    assert r.geodesic_alignment_score == 1.0
    # Chord length sin(distance/2) = sin(pi/2) = 1 at maximal separation.
    assert r.manifold_consistency_score == 1.0
    assert r.global_information_geometry_score == 1.0


def test_ascii_summary_contains_schema_and_hashes() -> None:
    report, _ = run_fisher_rao_geometry_layer(
        {"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 2.0}
    )
    summary = build_ascii_fisher_rao_summary(report)

    assert SCHEMA_VERSION in summary
    assert report.report_hash in summary
    assert report.geometry_result.source_distribution_hash in summary
    assert report.geometry_result.target_distribution_hash in summary
