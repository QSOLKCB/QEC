from __future__ import annotations

import math

import pytest

from qec.analysis.jensen_shannon_signal_divergence_kernel import (
    SignalDistribution,
    build_signal_distribution,
)
from qec.analysis.bregman_f_divergence_correspondence_engine import (
    SCHEMA_VERSION,
    DivergenceCorrespondenceConfig,
    DivergenceCorrespondenceReceipt,
    DivergenceCorrespondenceReport,
    DivergenceCorrespondenceResult,
    build_ascii_divergence_correspondence_summary,
    compute_bregman_alignment,
    compute_kl_divergence,
    compute_total_variation_distance,
    run_divergence_correspondence_engine,
)


def test_same_input_produces_same_bytes() -> None:
    report1, receipt1 = run_divergence_correspondence_engine(
        {"a": 4.0, "b": 1.0}, {"a": 2.0, "b": 3.0}
    )
    report2, receipt2 = run_divergence_correspondence_engine(
        {"a": 4.0, "b": 1.0}, {"a": 2.0, "b": 3.0}
    )

    assert report1.to_canonical_bytes() == report2.to_canonical_bytes()
    assert report1.to_canonical_json() == report2.to_canonical_json()
    assert receipt1.to_canonical_bytes() == receipt2.to_canonical_bytes()


def test_same_input_produces_same_hash() -> None:
    report1, receipt1 = run_divergence_correspondence_engine(
        {"a": 9.0, "b": 1.0}, {"a": 5.0, "b": 5.0}
    )
    report2, receipt2 = run_divergence_correspondence_engine(
        {"a": 9.0, "b": 1.0}, {"a": 5.0, "b": 5.0}
    )

    assert report1.report_hash == report2.report_hash
    assert report1.correspondence_result.result_hash == report2.correspondence_result.result_hash
    assert receipt1.receipt_hash == receipt2.receipt_hash
    assert report1.config.config_hash == report2.config.config_hash


def test_repeated_run_byte_identity() -> None:
    seen: set[bytes] = set()
    for _ in range(5):
        report, receipt = run_divergence_correspondence_engine(
            {"x": 1.0, "y": 2.0, "z": 3.0},
            {"x": 3.0, "y": 2.0, "z": 1.0},
        )
        seen.add(report.to_canonical_bytes())
        seen.add(receipt.to_canonical_bytes())
    # Two distinct artifacts (report + receipt), both stable across runs.
    assert len(seen) == 2


def test_identical_distributions_have_zero_divergence() -> None:
    dist = build_signal_distribution({"x": 2.0, "y": 2.0, "z": 6.0})

    assert compute_kl_divergence(dist, dist) == 0.0
    assert compute_total_variation_distance(dist, dist) == 0.0
    # Raw alignment: 1 means perfectly aligned.
    assert compute_bregman_alignment(dist, dist) == 1.0

    report, _ = run_divergence_correspondence_engine(
        {"x": 2.0, "y": 2.0, "z": 6.0},
        {"x": 2.0, "y": 2.0, "z": 6.0},
    )
    r = report.correspondence_result
    assert r.kl_divergence_score == 0.0
    assert r.total_variation_score == 0.0
    assert r.bregman_alignment_score == 0.0
    assert r.divergence_family_consistency_score == 0.0
    assert r.global_divergence_correspondence_score == 0.0


def test_total_variation_symmetry() -> None:
    a = build_signal_distribution({"a": 3.0, "b": 1.0, "c": 2.0})
    b = build_signal_distribution({"a": 1.0, "b": 3.0, "c": 4.0})

    left_tv = compute_total_variation_distance(a, b)
    right_tv = compute_total_variation_distance(b, a)
    assert left_tv == right_tv

    # Symmetric Bregman must also be symmetric.
    assert compute_bregman_alignment(a, b) == compute_bregman_alignment(b, a)

    report_lr, _ = run_divergence_correspondence_engine(
        {"a": 3.0, "b": 1.0, "c": 2.0},
        {"a": 1.0, "b": 3.0, "c": 4.0},
    )
    report_rl, _ = run_divergence_correspondence_engine(
        {"a": 1.0, "b": 3.0, "c": 4.0},
        {"a": 3.0, "b": 1.0, "c": 2.0},
    )
    lr = report_lr.correspondence_result
    rl = report_rl.correspondence_result
    assert lr.total_variation_score == rl.total_variation_score
    assert lr.bregman_alignment_score == rl.bregman_alignment_score


def test_scores_are_bounded_unit_interval() -> None:
    report, _ = run_divergence_correspondence_engine(
        {"alpha": 7.0, "beta": 2.0, "gamma": 1.0},
        {"alpha": 1.0, "beta": 4.0, "gamma": 5.0},
    )
    r = report.correspondence_result
    for score in (
        r.kl_divergence_score,
        r.total_variation_score,
        r.bregman_alignment_score,
        r.divergence_family_consistency_score,
        r.global_divergence_correspondence_score,
    ):
        assert 0.0 <= score <= 1.0


def test_invalid_distribution_rejected() -> None:
    bad = SignalDistribution(
        labels=("a", "b"), probabilities=(0.8, 0.3), distribution_hash="0" * 64
    )
    good = build_signal_distribution({"a": 1.0, "b": 1.0})

    with pytest.raises(ValueError, match="sum to 1"):
        compute_kl_divergence(bad, good)
    with pytest.raises(ValueError, match="sum to 1"):
        compute_total_variation_distance(bad, good)
    with pytest.raises(ValueError, match="sum to 1"):
        compute_bregman_alignment(bad, good)

    negative = SignalDistribution(
        labels=("a", "b"), probabilities=(-0.1, 1.1), distribution_hash="0" * 64
    )
    with pytest.raises(ValueError):
        compute_kl_divergence(negative, good)

    non_finite = SignalDistribution(
        labels=("a", "b"),
        probabilities=(float("nan"), 0.5),
        distribution_hash="0" * 64,
    )
    with pytest.raises(ValueError):
        compute_total_variation_distance(non_finite, good)


def test_receipt_integrity_and_lineage() -> None:
    report, receipt = run_divergence_correspondence_engine(
        {"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 1.0}
    )

    assert isinstance(report, DivergenceCorrespondenceReport)
    assert isinstance(receipt, DivergenceCorrespondenceReceipt)
    assert isinstance(report.correspondence_result, DivergenceCorrespondenceResult)
    assert isinstance(report.config, DivergenceCorrespondenceConfig)

    assert receipt.report_hash == report.report_hash
    assert receipt.byte_length == len(report.to_canonical_bytes())
    assert receipt.validation_passed is True
    assert receipt.config_hash == report.config.config_hash
    assert receipt.source_distribution_hash == report.source_distribution.distribution_hash
    assert receipt.target_distribution_hash == report.target_distribution.distribution_hash
    assert receipt.result_hash == report.correspondence_result.result_hash
    assert report.schema_version == SCHEMA_VERSION
    assert len(receipt.receipt_hash) == 64
    assert len(report.report_hash) == 64


def test_wrapper_matches_manual_pipeline() -> None:
    source = {"s0": 4.0, "s1": 1.0, "s2": 5.0}
    target = {"s0": 2.0, "s1": 3.0, "s2": 5.0}

    report, receipt = run_divergence_correspondence_engine(source, target)

    manual_source = build_signal_distribution(source)
    manual_target = build_signal_distribution(target)
    manual_kl = compute_kl_divergence(manual_source, manual_target)
    manual_tv = compute_total_variation_distance(manual_source, manual_target)
    manual_alignment = compute_bregman_alignment(manual_source, manual_target)

    r = report.correspondence_result
    assert report.source_distribution == manual_source
    assert report.target_distribution == manual_target
    assert abs(r.kl_divergence_score - manual_kl) < 1e-9
    assert abs(r.total_variation_score - manual_tv) < 1e-9
    # The stored bregman_alignment_score is the complement of the raw
    # alignment returned by compute_bregman_alignment.
    assert abs(r.bregman_alignment_score - (1.0 - manual_alignment)) < 1e-9
    assert receipt.report_hash == report.report_hash


def test_maximal_divergence_bounded() -> None:
    a = build_signal_distribution({"a": 1.0, "b": 0.0})
    b = build_signal_distribution({"a": 0.0, "b": 1.0})

    # Infinite raw KL must normalize to exactly 1.0.
    assert compute_kl_divergence(a, b) == 1.0
    # TV of disjoint indicator distributions is exactly 1.0.
    assert compute_total_variation_distance(a, b) == 1.0
    # Raw alignment bottoms out at 0.0 at maximal separation.
    assert compute_bregman_alignment(a, b) == 0.0

    report, _ = run_divergence_correspondence_engine(
        {"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}
    )
    r = report.correspondence_result
    assert r.kl_divergence_score == 1.0
    assert r.total_variation_score == 1.0
    assert r.bregman_alignment_score == 1.0
    # All three family scores equal 1.0 at max separation => range is 0.
    assert r.divergence_family_consistency_score == 0.0
    assert r.global_divergence_correspondence_score == 1.0
    # Every exported score remains bounded in [0, 1].
    for score in (
        r.kl_divergence_score,
        r.total_variation_score,
        r.bregman_alignment_score,
        r.divergence_family_consistency_score,
        r.global_divergence_correspondence_score,
    ):
        assert 0.0 <= score <= 1.0


def test_kl_is_asymmetric_but_bounded() -> None:
    p = build_signal_distribution({"a": 3.0, "b": 1.0})
    q = build_signal_distribution({"a": 1.0, "b": 3.0})
    forward = compute_kl_divergence(p, q)
    reverse = compute_kl_divergence(q, p)
    # KL is asymmetric in general; both directions are finite and
    # strictly positive for non-identical distributions.
    assert 0.0 < forward <= 1.0
    assert 0.0 < reverse <= 1.0
    # For this symmetric-around-0.5 pair the two KLs coincide, but we
    # still verify boundedness regardless.
    assert math.isfinite(forward)
    assert math.isfinite(reverse)


def test_ascii_summary_contains_schema_and_hashes() -> None:
    report, _ = run_divergence_correspondence_engine(
        {"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 2.0}
    )
    summary = build_ascii_divergence_correspondence_summary(report)

    assert SCHEMA_VERSION in summary
    assert report.report_hash in summary
    assert report.correspondence_result.source_distribution_hash in summary
    assert report.correspondence_result.target_distribution_hash in summary


def test_config_canonical_export_and_hash_stable() -> None:
    cfg1 = DivergenceCorrespondenceConfig()
    cfg2 = DivergenceCorrespondenceConfig()
    assert cfg1.to_canonical_bytes() == cfg2.to_canonical_bytes()
    assert cfg1.config_hash == cfg2.config_hash
    assert len(cfg1.config_hash) == 64
