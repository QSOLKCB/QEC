import importlib
import json
import sys

import pytest

from qec.analysis.quantum_noise_fidelity_observatory import (
    NoiseAuditEntry,
    NoiseAuditTrail,
    append_noise_audit_entry,
    build_fidelity_stability_timeline,
    compute_bounded_noise_score,
    compute_perturbation_compensation_metrics,
    derive_noise_observatory_report,
    empty_noise_audit_trail,
    normalize_noise_fidelity_inputs,
    run_quantum_noise_fidelity_observatory,
    validate_noise_audit_trail,
)


def _snapshot(snapshot_id: str, noise: float, fidelity: float, stability: float, drift: float, compensation: float):
    return normalize_noise_fidelity_inputs(
        snapshot_id=snapshot_id,
        noise_level=noise,
        fidelity_score=fidelity,
        stability_score=stability,
        error_drift=drift,
        compensation_factor=compensation,
    )


def test_normalize_noise_fidelity_inputs_is_deterministic():
    a = _snapshot("s1", 0.2, 0.9, 0.8, 0.1, 0.9)
    b = _snapshot("s1", 0.2, 0.9, 0.8, 0.1, 0.9)
    assert a == b
    assert a.to_canonical_json() == b.to_canonical_json()


def test_noise_score_is_bounded():
    score = compute_bounded_noise_score(
        noise_level=2.0,
        error_drift=2.0,
        stability_score=-1.0,
        compensation_factor=-1.0,
    )
    assert 0.0 <= score <= 1.0


def test_noise_score_rejects_nan_inf():
    with pytest.raises(ValueError, match="finite"):
        compute_bounded_noise_score(
            noise_level=float("nan"),
            error_drift=0.2,
            stability_score=0.8,
            compensation_factor=0.9,
        )
    with pytest.raises(ValueError, match="finite"):
        compute_bounded_noise_score(
            noise_level=0.2,
            error_drift=float("inf"),
            stability_score=0.8,
            compensation_factor=0.9,
        )


def test_timeline_is_deterministic():
    snapshots = (
        _snapshot("b", 0.2, 0.7, 0.8, 0.3, 0.6),
        _snapshot("a", 0.1, 0.9, 0.95, 0.05, 0.9),
    )
    t1 = build_fidelity_stability_timeline(snapshots)
    t2 = build_fidelity_stability_timeline(tuple(reversed(snapshots)))
    assert t1 == t2
    assert t1.to_canonical_json() == t2.to_canonical_json()


def test_timeline_average_scores_are_stable():
    t = build_fidelity_stability_timeline(
        (
            _snapshot("a", 0.2, 0.8, 0.7, 0.1, 0.9),
            _snapshot("b", 0.4, 0.6, 0.5, 0.3, 0.7),
        )
    )
    assert t.average_fidelity == 0.7
    assert t.average_stability == 0.6


def test_compensation_metrics_are_bounded():
    metrics = compute_perturbation_compensation_metrics(
        noise_level=10.0,
        error_drift=10.0,
        compensation_factor=-2.0,
    )
    assert 0.0 <= metrics.compensation_effectiveness <= 1.0
    assert 0.0 <= metrics.drift_reduction_score <= 1.0
    assert 0.0 <= metrics.balance_score <= 1.0


def test_observatory_report_health_bands_are_stable():
    metrics = compute_perturbation_compensation_metrics(
        noise_level=0.1,
        error_drift=0.1,
        compensation_factor=0.9,
    )
    strong = derive_noise_observatory_report(
        bounded_noise_score=0.1,
        average_fidelity=0.9,
        average_stability=0.9,
        compensation_metrics=metrics,
    )
    stable = derive_noise_observatory_report(
        bounded_noise_score=0.3,
        average_fidelity=0.7,
        average_stability=0.7,
        compensation_metrics=metrics,
    )
    fragile = derive_noise_observatory_report(
        bounded_noise_score=0.6,
        average_fidelity=0.5,
        average_stability=0.2,
        compensation_metrics=metrics,
    )
    critical = derive_noise_observatory_report(
        bounded_noise_score=0.9,
        average_fidelity=0.3,
        average_stability=0.2,
        compensation_metrics=metrics,
    )
    assert strong.fidelity_health == "strong"
    assert stable.fidelity_health == "stable"
    assert fragile.fidelity_health == "fragile"
    assert critical.fidelity_health == "critical"


def test_noise_audit_trail_chain_is_stable():
    trail = empty_noise_audit_trail()
    s1 = _snapshot("a", 0.1, 0.9, 0.9, 0.05, 0.95)
    s2 = _snapshot("b", 0.2, 0.8, 0.8, 0.1, 0.8)
    trail = append_noise_audit_entry(trail, snapshot_hash=s1.snapshot_hash, noise_score=0.1, fidelity_score=s1.fidelity_score)
    trail = append_noise_audit_entry(trail, snapshot_hash=s2.snapshot_hash, noise_score=0.2, fidelity_score=s2.fidelity_score)
    assert validate_noise_audit_trail(trail) is True


def test_noise_audit_trail_detects_corruption():
    trail = empty_noise_audit_trail()
    s1 = _snapshot("a", 0.1, 0.9, 0.9, 0.05, 0.95)
    trail = append_noise_audit_entry(trail, snapshot_hash=s1.snapshot_hash, noise_score=0.1, fidelity_score=s1.fidelity_score)
    bad_entry = NoiseAuditEntry(
        sequence_id=trail.entries[0].sequence_id,
        snapshot_hash=trail.entries[0].snapshot_hash,
        parent_hash=trail.entries[0].parent_hash,
        noise_score=0.9,
        fidelity_score=trail.entries[0].fidelity_score,
        entry_hash=trail.entries[0].entry_hash,
    )
    bad = NoiseAuditTrail(entries=(bad_entry,), head_hash=trail.head_hash, chain_valid=True)
    assert validate_noise_audit_trail(bad) is False


def test_append_rejects_malformed_noise_audit_trail():
    bad = NoiseAuditTrail(entries=(), head_hash="abc", chain_valid=True)
    with pytest.raises(ValueError, match="malformed"):
        append_noise_audit_entry(bad, snapshot_hash="0" * 64, noise_score=0.1, fidelity_score=0.9)


def test_same_input_same_bytes():
    snapshots = (
        _snapshot("a", 0.1, 0.9, 0.95, 0.05, 0.95),
        _snapshot("b", 0.2, 0.85, 0.9, 0.1, 0.9),
    )
    r1 = run_quantum_noise_fidelity_observatory(snapshots)
    r2 = run_quantum_noise_fidelity_observatory(snapshots)
    b1 = tuple(x.to_canonical_json() for x in r1)
    b2 = tuple(x.to_canonical_json() for x in r2)
    assert b1 == b2


def test_no_decoder_imports():
    module_name = "qec.analysis.quantum_noise_fidelity_observatory"
    for k in list(sys.modules):
        if k == module_name or k.startswith(module_name + "."):
            del sys.modules[k]
    importlib.import_module(module_name)
    assert all(not m.startswith("qec.decoder") for m in sys.modules)


def test_zero_noise_strong_health_path():
    timeline, metrics, report, trail = run_quantum_noise_fidelity_observatory(
        _snapshot("zero", 0.0, 1.0, 1.0, 0.0, 1.0)
    )
    assert timeline.average_fidelity == 1.0
    assert report.fidelity_health == "strong"
    assert metrics.compensation_effectiveness == 1.0
    assert validate_noise_audit_trail(trail) is True


def test_high_drift_critical_health_path():
    timeline, metrics, report, _ = run_quantum_noise_fidelity_observatory(
        _snapshot("critical", 1.0, 0.1, 0.1, 1.0, 0.0)
    )
    assert timeline.average_stability == 0.1
    assert metrics.drift_reduction_score == 0.0
    assert report.fidelity_health == "critical"


def test_hash_format_validation():
    trail = empty_noise_audit_trail()
    with pytest.raises(ValueError, match="64 hex"):
        append_noise_audit_entry(trail, snapshot_hash="abc", noise_score=0.1, fidelity_score=0.9)


def test_empty_audit_trail_valid_baseline():
    trail = empty_noise_audit_trail()
    assert trail.head_hash == "0" * 64
    assert validate_noise_audit_trail(trail) is True


def test_contradictory_chain_valid_flag_returns_false():
    trail = NoiseAuditTrail(entries=(), head_hash="0" * 64, chain_valid=False)
    assert validate_noise_audit_trail(trail) is False


def test_observation_law_same_trajectory_same_observatory_bytes():
    snapshots = (
        _snapshot("law_1", 0.3, 0.8, 0.7, 0.2, 0.7),
        _snapshot("law_2", 0.4, 0.7, 0.6, 0.3, 0.6),
    )
    a = run_quantum_noise_fidelity_observatory(snapshots)
    b = run_quantum_noise_fidelity_observatory(snapshots)
    assert json.dumps([x.to_dict() for x in a], sort_keys=True, separators=(",", ":")) == json.dumps(
        [x.to_dict() for x in b], sort_keys=True, separators=(",", ":")
    )
