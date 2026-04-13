from __future__ import annotations

import pytest

from qec.analysis.jensen_shannon_signal_divergence_kernel import (
    SignalDistribution,
    build_signal_distribution,
)
from qec.analysis.optimal_transport_geometry_kernel import (
    SCHEMA_VERSION,
    TransportGeometryConfig,
    TransportGeometryReceipt,
    TransportGeometryReport,
    TransportGeometryResult,
    build_ascii_transport_summary,
    compute_transport_alignment,
    compute_wasserstein_distance,
    run_optimal_transport_geometry_kernel,
)


def test_same_input_produces_same_bytes() -> None:
    report1, receipt1 = run_optimal_transport_geometry_kernel(
        {"a": 4.0, "b": 1.0}, {"a": 2.0, "b": 3.0}
    )
    report2, receipt2 = run_optimal_transport_geometry_kernel(
        {"a": 4.0, "b": 1.0}, {"a": 2.0, "b": 3.0}
    )

    assert report1.to_canonical_bytes() == report2.to_canonical_bytes()
    assert report1.to_canonical_json() == report2.to_canonical_json()
    assert receipt1.to_canonical_bytes() == receipt2.to_canonical_bytes()


def test_same_input_produces_same_hash() -> None:
    report1, receipt1 = run_optimal_transport_geometry_kernel(
        {"a": 9.0, "b": 1.0}, {"a": 5.0, "b": 5.0}
    )
    report2, receipt2 = run_optimal_transport_geometry_kernel(
        {"a": 9.0, "b": 1.0}, {"a": 5.0, "b": 5.0}
    )

    assert report1.report_hash == report2.report_hash
    assert report1.transport_result.result_hash == report2.transport_result.result_hash
    assert receipt1.receipt_hash == receipt2.receipt_hash
    assert report1.config.config_hash == report2.config.config_hash


def test_repeated_run_byte_identity() -> None:
    seen: set[bytes] = set()
    for _ in range(5):
        report, receipt = run_optimal_transport_geometry_kernel(
            {"x": 1.0, "y": 2.0, "z": 3.0},
            {"x": 3.0, "y": 2.0, "z": 1.0},
        )
        seen.add(report.to_canonical_bytes())
        seen.add(receipt.to_canonical_bytes())
    # Two distinct artifacts (report + receipt), both stable across runs.
    assert len(seen) == 2


def test_identical_distributions_have_zero_distance() -> None:
    dist = build_signal_distribution({"x": 2.0, "y": 2.0, "z": 6.0})

    assert compute_wasserstein_distance(dist, dist) == 0.0
    # Raw alignment: 1 means perfectly aligned.
    assert compute_transport_alignment(dist, dist) == 1.0

    report, _ = run_optimal_transport_geometry_kernel(
        {"x": 2.0, "y": 2.0, "z": 6.0},
        {"x": 2.0, "y": 2.0, "z": 6.0},
    )
    r = report.transport_result
    assert r.wasserstein_distance_score == 0.0
    assert r.transport_alignment_score == 0.0
    assert r.cumulative_flow_consistency_score == 0.0
    assert r.global_transport_geometry_score == 0.0


def test_wasserstein_is_symmetric() -> None:
    a = build_signal_distribution({"a": 3.0, "b": 1.0, "c": 2.0})
    b = build_signal_distribution({"a": 1.0, "b": 3.0, "c": 4.0})

    assert compute_wasserstein_distance(a, b) == compute_wasserstein_distance(b, a)
    assert compute_transport_alignment(a, b) == compute_transport_alignment(b, a)

    report_lr, _ = run_optimal_transport_geometry_kernel(
        {"a": 3.0, "b": 1.0, "c": 2.0},
        {"a": 1.0, "b": 3.0, "c": 4.0},
    )
    report_rl, _ = run_optimal_transport_geometry_kernel(
        {"a": 1.0, "b": 3.0, "c": 4.0},
        {"a": 3.0, "b": 1.0, "c": 2.0},
    )
    lr = report_lr.transport_result
    rl = report_rl.transport_result
    assert lr.wasserstein_distance_score == rl.wasserstein_distance_score
    assert lr.transport_alignment_score == rl.transport_alignment_score
    assert lr.cumulative_flow_consistency_score == rl.cumulative_flow_consistency_score
    assert lr.global_transport_geometry_score == rl.global_transport_geometry_score


def test_scores_are_bounded_unit_interval() -> None:
    report, _ = run_optimal_transport_geometry_kernel(
        {"alpha": 7.0, "beta": 2.0, "gamma": 1.0},
        {"alpha": 1.0, "beta": 4.0, "gamma": 5.0},
    )
    r = report.transport_result
    for score in (
        r.wasserstein_distance_score,
        r.transport_alignment_score,
        r.cumulative_flow_consistency_score,
        r.global_transport_geometry_score,
    ):
        assert 0.0 <= score <= 1.0


def test_invalid_distribution_rejected() -> None:
    bad = SignalDistribution(
        labels=("a", "b"), probabilities=(0.8, 0.3), distribution_hash="0" * 64
    )
    good = build_signal_distribution({"a": 1.0, "b": 1.0})

    with pytest.raises(ValueError, match="sum to 1"):
        compute_wasserstein_distance(bad, good)
    with pytest.raises(ValueError, match="sum to 1"):
        compute_transport_alignment(bad, good)

    negative = SignalDistribution(
        labels=("a", "b"), probabilities=(-0.1, 1.1), distribution_hash="0" * 64
    )
    with pytest.raises(ValueError):
        compute_wasserstein_distance(negative, good)

    non_finite = SignalDistribution(
        labels=("a", "b"),
        probabilities=(float("nan"), 0.5),
        distribution_hash="0" * 64,
    )
    with pytest.raises(ValueError):
        compute_wasserstein_distance(non_finite, good)


def test_receipt_integrity_and_lineage() -> None:
    report, receipt = run_optimal_transport_geometry_kernel(
        {"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 1.0}
    )

    assert isinstance(report, TransportGeometryReport)
    assert isinstance(receipt, TransportGeometryReceipt)
    assert isinstance(report.transport_result, TransportGeometryResult)
    assert isinstance(report.config, TransportGeometryConfig)

    assert receipt.report_hash == report.report_hash
    assert receipt.byte_length == len(report.to_canonical_bytes())
    assert receipt.validation_passed is True
    assert receipt.config_hash == report.config.config_hash
    assert receipt.source_distribution_hash == report.source_distribution.distribution_hash
    assert receipt.target_distribution_hash == report.target_distribution.distribution_hash
    assert receipt.result_hash == report.transport_result.result_hash
    assert report.schema_version == SCHEMA_VERSION
    assert len(receipt.receipt_hash) == 64
    assert len(report.report_hash) == 64


def test_wrapper_matches_manual_pipeline() -> None:
    source = {"s0": 4.0, "s1": 1.0, "s2": 5.0}
    target = {"s0": 2.0, "s1": 3.0, "s2": 5.0}

    report, receipt = run_optimal_transport_geometry_kernel(source, target)

    manual_source = build_signal_distribution(source)
    manual_target = build_signal_distribution(target)
    manual_w1 = compute_wasserstein_distance(manual_source, manual_target)
    manual_alignment = compute_transport_alignment(manual_source, manual_target)

    r = report.transport_result
    assert report.source_distribution == manual_source
    assert report.target_distribution == manual_target
    assert abs(r.wasserstein_distance_score - manual_w1) < 1e-9
    # The stored transport_alignment_score is the complement of the raw
    # alignment returned by compute_transport_alignment.
    assert abs(r.transport_alignment_score - (1.0 - manual_alignment)) < 1e-9
    assert receipt.report_hash == report.report_hash


def test_maximal_transport_bounded() -> None:
    a = build_signal_distribution({"a": 1.0, "b": 0.0})
    b = build_signal_distribution({"a": 0.0, "b": 1.0})

    # Disjoint two-bin indicator distributions saturate W1 at exactly 1.
    assert compute_wasserstein_distance(a, b) == 1.0
    # Raw alignment bottoms out at 0.0 at maximal separation.
    assert compute_transport_alignment(a, b) == 0.0

    report, _ = run_optimal_transport_geometry_kernel(
        {"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}
    )
    r = report.transport_result
    assert r.wasserstein_distance_score == 1.0
    assert r.transport_alignment_score == 1.0
    # KS supremum of disjoint indicators is also 1.0.
    assert r.cumulative_flow_consistency_score == 1.0
    assert r.global_transport_geometry_score == 1.0
    # Every exported score remains bounded in [0, 1].
    for score in (
        r.wasserstein_distance_score,
        r.transport_alignment_score,
        r.cumulative_flow_consistency_score,
        r.global_transport_geometry_score,
    ):
        assert 0.0 <= score <= 1.0


def test_wasserstein_closed_form_matches_expected_value() -> None:
    # Three sorted bins, mass transported one step to the right.
    # source: (1, 0, 0) -> CDF (1, 1, 1)
    # target: (0, 1, 0) -> CDF (0, 1, 1)
    # |diff| = (1, 0, 0), sum = 1, normalized / (N-1=2) = 0.5
    a = build_signal_distribution({"b0": 1.0, "b1": 0.0, "b2": 0.0})
    b = build_signal_distribution({"b0": 0.0, "b1": 1.0, "b2": 0.0})
    assert abs(compute_wasserstein_distance(a, b) - 0.5) < 1e-12

    # Mass transported two steps to the right should saturate W1 to 1.0.
    c = build_signal_distribution({"b0": 0.0, "b1": 0.0, "b2": 1.0})
    assert compute_wasserstein_distance(a, c) == 1.0


def test_ascii_summary_contains_schema_and_hashes() -> None:
    report, _ = run_optimal_transport_geometry_kernel(
        {"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 2.0}
    )
    summary = build_ascii_transport_summary(report)

    assert SCHEMA_VERSION in summary
    assert report.report_hash in summary
    assert report.transport_result.source_distribution_hash in summary
    assert report.transport_result.target_distribution_hash in summary


def test_config_canonical_export_and_hash_stable() -> None:
    cfg1 = TransportGeometryConfig()
    cfg2 = TransportGeometryConfig()
    assert cfg1.to_canonical_bytes() == cfg2.to_canonical_bytes()
    assert cfg1.config_hash == cfg2.config_hash
    assert len(cfg1.config_hash) == 64
