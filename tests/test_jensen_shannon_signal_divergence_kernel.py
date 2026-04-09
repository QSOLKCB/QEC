from __future__ import annotations

import pytest

from qec.analysis.jensen_shannon_signal_divergence_kernel import (
    SignalDistribution,
    build_signal_distribution,
    compute_jensen_shannon_divergence,
    run_signal_divergence_kernel,
)


def test_jsd_symmetry() -> None:
    a = build_signal_distribution({"a": 3.0, "b": 1.0})
    b = build_signal_distribution({"a": 1.0, "b": 3.0})

    left = compute_jensen_shannon_divergence(a, b)
    right = compute_jensen_shannon_divergence(b, a)

    assert left.js_divergence_score == right.js_divergence_score
    assert left.distribution_overlap_score == right.distribution_overlap_score
    assert left.entropy_alignment_score == right.entropy_alignment_score
    assert left.global_information_geometry_score == right.global_information_geometry_score


def test_identical_distribution_has_zero_divergence() -> None:
    dist = build_signal_distribution({"x": 2.0, "y": 2.0, "z": 6.0})
    result = compute_jensen_shannon_divergence(dist, dist)
    assert result.js_divergence_score == 0.0


def test_scores_are_bounded_unit_interval() -> None:
    a = build_signal_distribution({"alpha": 10.0, "beta": 0.0})
    b = build_signal_distribution({"alpha": 0.0, "beta": 10.0})
    result = compute_jensen_shannon_divergence(a, b)

    for score in (
        result.js_divergence_score,
        result.distribution_overlap_score,
        result.entropy_alignment_score,
        result.global_information_geometry_score,
    ):
        assert 0.0 <= score <= 1.0


def test_same_input_produces_same_bytes() -> None:
    report1, receipt1 = run_signal_divergence_kernel({"a": 4.0, "b": 1.0}, {"a": 2.0, "b": 3.0})
    report2, receipt2 = run_signal_divergence_kernel({"a": 4.0, "b": 1.0}, {"a": 2.0, "b": 3.0})

    assert report1.to_canonical_bytes() == report2.to_canonical_bytes()
    assert report1.to_canonical_json() == report2.to_canonical_json()
    assert receipt1.to_dict() == receipt2.to_dict()


def test_same_input_produces_same_hash() -> None:
    report1, receipt1 = run_signal_divergence_kernel({"a": 9.0, "b": 1.0}, {"a": 5.0, "b": 5.0})
    report2, receipt2 = run_signal_divergence_kernel({"a": 9.0, "b": 1.0}, {"a": 5.0, "b": 5.0})

    assert report1.report_hash == report2.report_hash
    assert report1.divergence_result.result_hash == report2.divergence_result.result_hash
    assert receipt1.receipt_hash == receipt2.receipt_hash


def test_invalid_distribution_rejected() -> None:
    bad = SignalDistribution(labels=("a", "b"), probabilities=(0.8, 0.3), distribution_hash="0" * 64)
    good = build_signal_distribution({"a": 1.0, "b": 1.0})

    with pytest.raises(ValueError, match="sum to 1"):
        compute_jensen_shannon_divergence(bad, good)


def test_forged_distribution_hash_rejected() -> None:
    valid = build_signal_distribution({"a": 1.0, "b": 1.0})
    forged = SignalDistribution(
        labels=valid.labels,
        probabilities=valid.probabilities,
        distribution_hash="f" * 64,
    )

    with pytest.raises(ValueError, match="distribution_hash does not match"):
        compute_jensen_shannon_divergence(forged, valid)


def test_receipt_integrity() -> None:
    report, receipt = run_signal_divergence_kernel({"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 1.0})
    assert receipt.report_hash == report.report_hash
    assert receipt.byte_length == len(report.to_canonical_bytes())
    assert receipt.validation_passed is True


def test_wrapper_matches_manual_pipeline() -> None:
    source = {"s0": 4.0, "s1": 1.0, "s2": 5.0}
    target = {"s0": 2.0, "s1": 3.0, "s2": 5.0}

    report, receipt = run_signal_divergence_kernel(source, target)

    manual_source = build_signal_distribution(source)
    manual_target = build_signal_distribution(target)
    manual_result = compute_jensen_shannon_divergence(manual_source, manual_target)

    assert report.source_distribution == manual_source
    assert report.target_distribution == manual_target
    assert report.divergence_result == manual_result
    assert receipt.report_hash == report.report_hash
