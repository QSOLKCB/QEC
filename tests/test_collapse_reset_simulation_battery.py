from __future__ import annotations

import dataclasses

import pytest

from qec.analysis.collapse_reset_simulation_battery import (
    CollapseSimulationReport,
    compute_recovery_regression_score,
    export_simulation_bytes,
    generate_simulation_receipt,
    simulate_state_collapse,
    validate_reset_pathway,
)


def _source_attractor() -> dict[str, object]:
    return {
        "attractor_id": "att-001",
        "coherence": 0.875,
        "state": {"mode": "bounded", "rank": 3},
        "history": ["h2", "h1"],
    }


def _parent_hash() -> str:
    return "ab" * 32


def _reset_pathway() -> tuple[str, ...]:
    return ("rephase", "stabilize", "synchronize")


def test_repeated_run_determinism() -> None:
    report_a = simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash=_parent_hash())
    report_b = simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash=_parent_hash())
    assert report_a == report_b


def test_identical_inputs_produce_identical_bytes() -> None:
    report_a = simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash=_parent_hash())
    report_b = simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash=_parent_hash())
    assert export_simulation_bytes(report_a) == export_simulation_bytes(report_b)


def test_scores_are_bounded() -> None:
    report = simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash=_parent_hash())
    assert 0.0 <= report.collapse_severity_score <= 1.0
    assert 0.0 <= report.reset_success_score <= 1.0
    assert 0.0 <= report.recovery_regression_score <= 1.0


def test_stable_simulation_hash() -> None:
    report_a = simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash=_parent_hash())
    report_b = simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash=_parent_hash())
    assert report_a.stable_simulation_hash == report_b.stable_simulation_hash
    assert len(report_a.stable_simulation_hash) == 64


def test_receipt_stability() -> None:
    report = simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash=_parent_hash())
    receipt_a = generate_simulation_receipt(report)
    receipt_b = generate_simulation_receipt(report)
    assert receipt_a == receipt_b
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_tamper_detection() -> None:
    report = simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash=_parent_hash())
    receipt = generate_simulation_receipt(report)
    tampered_report = dataclasses.replace(report, reset_success_score=min(1.0, report.reset_success_score + 0.01))
    tampered_receipt = generate_simulation_receipt(tampered_report)
    assert tampered_receipt.receipt_hash != receipt.receipt_hash


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError):
        simulate_state_collapse({}, _reset_pathway(), parent_attractor_hash=_parent_hash())
    with pytest.raises(ValueError):
        simulate_state_collapse(_source_attractor(), (), parent_attractor_hash=_parent_hash())
    with pytest.raises(ValueError):
        simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash="bad-hash")
    with pytest.raises(ValueError):
        validate_reset_pathway(("valid", "", "also-valid"))
    with pytest.raises(ValueError):
        validate_reset_pathway(("same", "same"))
    with pytest.raises(ValueError):
        compute_recovery_regression_score(-0.1, 0.2)


def test_export_validates_tampered_report_bounds() -> None:
    valid_report = simulate_state_collapse(_source_attractor(), _reset_pathway(), parent_attractor_hash=_parent_hash())
    invalid_report = CollapseSimulationReport(
        source_attractor_hash=valid_report.source_attractor_hash,
        collapse_severity_score=1.1,
        reset_success_score=valid_report.reset_success_score,
        recovery_regression_score=valid_report.recovery_regression_score,
        reset_state_hash=valid_report.reset_state_hash,
        parent_attractor_hash=valid_report.parent_attractor_hash,
        stable_simulation_hash=valid_report.stable_simulation_hash,
        schema_version=valid_report.schema_version,
    )
    with pytest.raises(ValueError):
        export_simulation_bytes(invalid_report)
