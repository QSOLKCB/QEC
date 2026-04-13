from __future__ import annotations

from qec.analysis.information_geometry_consensus_kernel import (
    SCHEMA_VERSION,
    InformationGeometryConsensusConfig,
    InformationGeometryConsensusReceipt,
    InformationGeometryConsensusReport,
    InformationGeometryConsensusResult,
    build_ascii_information_geometry_consensus_summary,
    run_information_geometry_consensus_kernel,
)


def _h(ch: str) -> str:
    return ch * 64


def test_same_input_same_bytes() -> None:
    args = dict(
        js_divergence_score=0.2,
        fisher_rao_distance_score=0.3,
        global_divergence_correspondence_score=0.4,
        global_transport_geometry_score=0.5,
        js_divergence_receipt_hash=_h("a"),
        fisher_rao_receipt_hash=_h("b"),
        divergence_correspondence_receipt_hash=_h("c"),
        transport_geometry_receipt_hash=_h("d"),
    )
    report1, receipt1 = run_information_geometry_consensus_kernel(**args)
    report2, receipt2 = run_information_geometry_consensus_kernel(**args)

    assert report1.to_canonical_bytes() == report2.to_canonical_bytes()
    assert receipt1.to_canonical_bytes() == receipt2.to_canonical_bytes()


def test_same_input_same_hash() -> None:
    args = dict(
        js_divergence_score=0.9,
        fisher_rao_distance_score=0.1,
        global_divergence_correspondence_score=0.8,
        global_transport_geometry_score=0.2,
        js_divergence_receipt_hash=_h("1"),
        fisher_rao_receipt_hash=_h("2"),
        divergence_correspondence_receipt_hash=_h("3"),
        transport_geometry_receipt_hash=_h("4"),
    )
    report1, receipt1 = run_information_geometry_consensus_kernel(**args)
    report2, receipt2 = run_information_geometry_consensus_kernel(**args)

    assert report1.report_hash == report2.report_hash
    assert report1.consensus_result.result_hash == report2.consensus_result.result_hash
    assert receipt1.receipt_hash == receipt2.receipt_hash


def test_repeated_run_byte_identity() -> None:
    seen: set[bytes] = set()
    for _ in range(5):
        report, receipt = run_information_geometry_consensus_kernel(
            0.1,
            0.4,
            0.7,
            0.2,
            _h("a"),
            _h("b"),
            _h("c"),
            _h("d"),
        )
        seen.add(report.to_canonical_bytes())
        seen.add(receipt.to_canonical_bytes())
    assert len(seen) == 2


def test_bounded_metrics() -> None:
    report, _ = run_information_geometry_consensus_kernel(
        0.0,
        1.0,
        0.5,
        0.25,
        _h("a"),
        _h("b"),
        _h("c"),
        _h("d"),
    )
    r = report.consensus_result
    for score in (
        r.geometry_consensus_score,
        r.geometry_dispersion_score,
        r.manifold_agreement_score,
        r.geometry_stability_score,
        r.global_information_consensus_score,
    ):
        assert 0.0 <= score <= 1.0


def test_identical_input_zero_dispersion() -> None:
    report, _ = run_information_geometry_consensus_kernel(
        0.42,
        0.42,
        0.42,
        0.42,
        _h("a"),
        _h("b"),
        _h("c"),
        _h("d"),
    )
    r = report.consensus_result
    assert r.geometry_dispersion_score == 0.0
    assert r.manifold_agreement_score == 1.0
    assert r.geometry_stability_score == 1.0


def test_divergent_input_lower_agreement() -> None:
    aligned, _ = run_information_geometry_consensus_kernel(
        0.3,
        0.3,
        0.3,
        0.3,
        _h("a"),
        _h("b"),
        _h("c"),
        _h("d"),
    )
    divergent, _ = run_information_geometry_consensus_kernel(
        0.0,
        0.0,
        1.0,
        1.0,
        _h("a"),
        _h("b"),
        _h("c"),
        _h("d"),
    )
    assert divergent.consensus_result.manifold_agreement_score < aligned.consensus_result.manifold_agreement_score


def test_manual_pipeline_equivalence() -> None:
    report, receipt = run_information_geometry_consensus_kernel(
        0.1,
        0.2,
        0.3,
        0.4,
        _h("a"),
        _h("b"),
        _h("c"),
        _h("d"),
    )
    r = report.consensus_result
    manual_consensus = (0.1 + 0.2 + 0.3 + 0.4) / 4.0
    manual_mad = (abs(0.1 - manual_consensus) + abs(0.2 - manual_consensus) + abs(0.3 - manual_consensus) + abs(0.4 - manual_consensus)) / 4.0
    manual_dispersion = 2.0 * manual_mad
    manual_agreement = 1.0 - manual_dispersion
    manual_stability = 1.0 - (0.4 - 0.1)
    manual_global = (manual_consensus + manual_agreement + manual_stability) / 3.0

    assert abs(r.geometry_consensus_score - manual_consensus) < 1e-12
    assert abs(r.geometry_dispersion_score - manual_dispersion) < 1e-12
    assert abs(r.manifold_agreement_score - manual_agreement) < 1e-12
    assert abs(r.geometry_stability_score - manual_stability) < 1e-12
    assert abs(r.global_information_consensus_score - manual_global) < 1e-12
    assert receipt.report_hash == report.report_hash


def test_receipt_integrity() -> None:
    report, receipt = run_information_geometry_consensus_kernel(
        0.2,
        0.4,
        0.6,
        0.8,
        _h("a"),
        _h("b"),
        _h("c"),
        _h("d"),
    )

    assert isinstance(report, InformationGeometryConsensusReport)
    assert isinstance(receipt, InformationGeometryConsensusReceipt)
    assert isinstance(report.consensus_result, InformationGeometryConsensusResult)
    assert isinstance(report.config, InformationGeometryConsensusConfig)

    assert report.schema_version == SCHEMA_VERSION
    assert receipt.report_hash == report.report_hash
    assert receipt.config_hash == report.config.config_hash
    assert receipt.result_hash == report.consensus_result.result_hash
    assert receipt.js_divergence_receipt_hash == report.js_divergence_receipt_hash
    assert receipt.fisher_rao_receipt_hash == report.fisher_rao_receipt_hash
    assert receipt.divergence_correspondence_receipt_hash == report.divergence_correspondence_receipt_hash
    assert receipt.transport_geometry_receipt_hash == report.transport_geometry_receipt_hash
    assert receipt.byte_length == len(report.to_canonical_bytes())
    assert receipt.validation_passed is True


def test_canonical_export_stability() -> None:
    cfg1 = InformationGeometryConsensusConfig()
    cfg2 = InformationGeometryConsensusConfig()
    assert cfg1.to_canonical_bytes() == cfg2.to_canonical_bytes()
    assert cfg1.config_hash == cfg2.config_hash


def test_ascii_summary_output() -> None:
    report, _ = run_information_geometry_consensus_kernel(
        0.11,
        0.22,
        0.33,
        0.44,
        _h("a"),
        _h("b"),
        _h("c"),
        _h("d"),
    )
    summary = build_ascii_information_geometry_consensus_summary(report)
    assert SCHEMA_VERSION in summary
    assert report.report_hash in summary
    assert report.js_divergence_receipt_hash in summary
    assert report.fisher_rao_receipt_hash in summary
