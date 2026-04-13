from __future__ import annotations

import pytest

from qec.analysis.information_geometric_certification_pack import (
    SCHEMA_VERSION,
    InformationGeometricCertificationConfig,
    InformationGeometricCertificationInput,
    InformationGeometricCertificationReceipt,
    InformationGeometricCertificationReport,
    InformationGeometricCertificationResult,
    build_ascii_information_geometric_certification_summary,
    run_information_geometric_certification_pack,
)


def _h(ch: str) -> str:
    return ch * 64


def _base_input() -> dict[str, float | str]:
    return {
        "js_divergence_score": 0.12,
        "fisher_rao_distance_score": 0.10,
        "global_divergence_correspondence_score": 0.08,
        "global_transport_geometry_score": 0.15,
        "global_information_consensus_score": 0.92,
        "js_divergence_receipt_hash": _h("a"),
        "fisher_rao_receipt_hash": _h("b"),
        "divergence_correspondence_receipt_hash": _h("c"),
        "transport_geometry_receipt_hash": _h("d"),
        "consensus_receipt_hash": _h("e"),
        "geometry_stability_score": 0.9,
        "manifold_agreement_score": 0.91,
        "coverage_ratio": 1.0,
    }


def test_same_input_same_bytes() -> None:
    i = _base_input()
    report1, receipt1 = run_information_geometric_certification_pack(i)
    report2, receipt2 = run_information_geometric_certification_pack(i)

    assert report1.to_canonical_bytes() == report2.to_canonical_bytes()
    assert receipt1.to_canonical_bytes() == receipt2.to_canonical_bytes()


def test_same_input_same_hash() -> None:
    i = _base_input()
    report1, receipt1 = run_information_geometric_certification_pack(i)
    report2, receipt2 = run_information_geometric_certification_pack(i)

    assert report1.report_hash == report2.report_hash
    assert report1.certification_result.result_hash == report2.certification_result.result_hash
    assert receipt1.receipt_hash == receipt2.receipt_hash


def test_repeated_run_byte_identity() -> None:
    seen: set[bytes] = set()
    for _ in range(5):
        report, receipt = run_information_geometric_certification_pack(_base_input())
        seen.add(report.to_canonical_bytes())
        seen.add(receipt.to_canonical_bytes())
    assert len(seen) == 2


def test_bounded_metrics() -> None:
    report, _ = run_information_geometric_certification_pack(_base_input())
    r = report.certification_result
    for score in (
        r.divergence_consistency_score,
        r.manifold_consistency_score,
        r.transport_consistency_score,
        r.consensus_certainty_score,
        r.coverage_completeness_score,
        r.global_information_geometry_certification_score,
    ):
        assert 0.0 <= score <= 1.0


def test_high_consistency_input_produces_strong_certification() -> None:
    report, _ = run_information_geometric_certification_pack(_base_input())
    assert report.certification_result.global_information_geometry_certification_score > 0.8


def test_contradictory_incomplete_degrades_certification() -> None:
    strong, _ = run_information_geometric_certification_pack(_base_input())

    weak_input = _base_input()
    weak_input.update(
        {
            "js_divergence_score": 0.98,
            "fisher_rao_distance_score": 0.95,
            "global_divergence_correspondence_score": 0.97,
            "global_transport_geometry_score": 0.99,
            "global_information_consensus_score": 0.04,
            "geometry_stability_score": 0.05,
            "manifold_agreement_score": 0.03,
            "coverage_ratio": 0.25,
        }
    )
    weak, _ = run_information_geometric_certification_pack(weak_input)

    assert (
        weak.certification_result.global_information_geometry_certification_score
        < strong.certification_result.global_information_geometry_certification_score
    )


def test_out_of_range_value_validation() -> None:
    bad = _base_input()
    bad["global_transport_geometry_score"] = 1.1
    with pytest.raises(ValueError, match="global_transport_geometry_score"):
        run_information_geometric_certification_pack(bad)


def test_missing_required_field_validation() -> None:
    bad = _base_input()
    bad.pop("consensus_receipt_hash")
    with pytest.raises(ValueError, match="missing required fields"):
        run_information_geometric_certification_pack(bad)


def test_canonical_export_stability() -> None:
    cfg1 = InformationGeometricCertificationConfig()
    cfg2 = InformationGeometricCertificationConfig()

    assert cfg1.to_canonical_bytes() == cfg2.to_canonical_bytes()
    assert cfg1.config_hash == cfg2.config_hash


def test_receipt_integrity() -> None:
    report, receipt = run_information_geometric_certification_pack(_base_input())

    assert isinstance(report, InformationGeometricCertificationReport)
    assert isinstance(receipt, InformationGeometricCertificationReceipt)
    assert isinstance(report.certification_result, InformationGeometricCertificationResult)
    assert isinstance(report.config, InformationGeometricCertificationConfig)
    assert isinstance(report.certification_input, InformationGeometricCertificationInput)

    assert report.schema_version == SCHEMA_VERSION
    assert receipt.report_hash == report.report_hash
    assert receipt.config_hash == report.config.config_hash
    assert receipt.result_hash == report.certification_result.result_hash
    assert receipt.byte_length == len(report.to_canonical_bytes())
    assert receipt.validation_passed is True


def test_ascii_text_summary_presence() -> None:
    report, _ = run_information_geometric_certification_pack(_base_input())
    assert report.summary_text
    summary = build_ascii_information_geometric_certification_summary(report)
    assert SCHEMA_VERSION in summary
    assert report.report_hash in summary


def test_dataclass_input_supported() -> None:
    mapping = _base_input()
    inp = InformationGeometricCertificationInput(**mapping)
    report_a, receipt_a = run_information_geometric_certification_pack(mapping)
    report_b, receipt_b = run_information_geometric_certification_pack(inp)

    assert report_a.to_canonical_bytes() == report_b.to_canonical_bytes()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
