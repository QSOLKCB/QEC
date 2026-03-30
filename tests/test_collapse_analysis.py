"""Tests for collapse_analysis — v105.3.1."""

from __future__ import annotations

from qec.analysis.collapse_analysis import (
    compute_collapse_score,
    compute_velocity_acceleration_metrics,
    detect_acceleration_spikes,
    detect_singularity_events,
    predict_basin_switch,
    run_collapse_analysis,
)


# -- flat trajectory (all zeros) ---------------------------------------------

def test_flat_trajectory():
    traj = [1.0, 1.0, 1.0, 1.0, 1.0]
    result = run_collapse_analysis(traj)
    assert result["collapse_score"] == 0.0
    assert result["failure_risk"] == 0.0
    assert result["acceleration_peak"] == 0.0
    assert result["basin_switch_prediction"] is False
    assert result["singularity_events"] == []


# -- smooth trajectory (low risk, no spikes) ---------------------------------

def test_smooth_trajectory():
    traj = [float(i) * 0.1 for i in range(20)]
    result = run_collapse_analysis(traj)
    assert result["collapse_score"] < 0.5
    assert result["singularity_events"] == []


# -- spike detection ---------------------------------------------------------

def test_spike_detection():
    # Mostly constant with a sharp jump
    traj = [0.0] * 10 + [10.0] + [0.0] * 10
    metrics = compute_velocity_acceleration_metrics(traj)
    spikes = detect_acceleration_spikes(metrics["acceleration"])
    assert spikes["spike_count"] > 0


# -- collapse score > 0.5 ---------------------------------------------------

def test_collapse_score_high():
    score = compute_collapse_score(spike_density=0.5, acceleration_peak=2.0)
    assert score > 0.5
    assert 0.0 <= score <= 1.0


def test_collapse_score_clamped():
    score = compute_collapse_score(spike_density=1.5, acceleration_peak=5.0)
    assert score == 1.0

    score_low = compute_collapse_score(spike_density=0.0, acceleration_peak=0.0)
    assert score_low == 0.0


# -- basin switch prediction -------------------------------------------------

def test_basin_switch_prediction_true():
    result = predict_basin_switch(
        spike_density=0.3,
        collapse_score=0.8,
        acceleration_peak=5.0,
        acceleration_mean=1.0,
    )
    assert result["basin_switch_predicted"] is True


def test_basin_switch_prediction_false_low_score():
    result = predict_basin_switch(
        spike_density=0.3,
        collapse_score=0.3,
        acceleration_peak=5.0,
        acceleration_mean=1.0,
    )
    assert result["basin_switch_predicted"] is False


# -- singularity events -----------------------------------------------------

def test_singularity_event_triggered():
    events = detect_singularity_events(spike_density=0.3, collapse_score=0.8)
    assert len(events) == 1
    assert events[0]["type"] == "collapse_event"


def test_singularity_event_not_triggered():
    events = detect_singularity_events(spike_density=0.1, collapse_score=0.5)
    assert events == []


# -- determinism check -------------------------------------------------------

def test_determinism():
    traj = [0.0, 0.1, 0.5, 2.0, 0.3, 0.1, 5.0, 0.2, 0.0, 1.0]
    r1 = run_collapse_analysis(traj)
    r2 = run_collapse_analysis(traj)
    assert r1 == r2


# -- edge cases --------------------------------------------------------------

def test_short_trajectory():
    assert run_collapse_analysis([])["collapse_score"] == 0.0
    assert run_collapse_analysis([1.0])["collapse_score"] == 0.0
    assert run_collapse_analysis([1.0, 2.0])["collapse_score"] == 0.0


def test_empty_acceleration_spikes():
    result = detect_acceleration_spikes([])
    assert result["spike_count"] == 0
    assert result["spike_density"] == 0.0


# -- v105.3.1 hardening tests ------------------------------------------------

def test_zero_std_acceleration_no_spikes():
    """Zero-std acceleration series (all identical values) produces no spikes."""
    series = [1.0, 1.0, 1.0, 1.0, 1.0]
    result = detect_acceleration_spikes(series)
    assert result["spike_count"] == 0
    assert result["spike_indices"] == []
    assert result["spike_density"] == 0.0


def test_spike_density_bounded():
    """spike_density must always be in [0, 1]."""
    # Normal case
    series = [0.0] * 20 + [100.0] + [0.0] * 20
    result = detect_acceleration_spikes(series)
    assert 0.0 <= result["spike_density"] <= 1.0

    # Edge: all zeros
    result2 = detect_acceleration_spikes([0.0] * 10)
    assert 0.0 <= result2["spike_density"] <= 1.0

    # Edge: empty
    result3 = detect_acceleration_spikes([])
    assert result3["spike_density"] == 0.0


def test_event_ordering_stable():
    """Singularity events are sorted by index."""
    events = detect_singularity_events(spike_density=0.5, collapse_score=0.9)
    if len(events) > 1:
        indices = [e["index"] for e in events]
        assert indices == sorted(indices)
    # Single event should have index field
    if events:
        assert "index" in events[0]


def test_repeated_runs_identical_rounded_outputs():
    """Repeated runs return identical rounded outputs (12-decimal reproducibility)."""
    traj = [0.0, 0.1, 0.5, 2.0, 0.3, 0.1, 5.0, 0.2, 0.0, 1.0]
    results = [run_collapse_analysis(traj) for _ in range(5)]
    for r in results[1:]:
        assert r["failure_risk"] == results[0]["failure_risk"]
        assert r["collapse_score"] == results[0]["collapse_score"]
        assert r["acceleration_peak"] == results[0]["acceleration_peak"]
        assert r == results[0]


def test_numeric_outputs_are_rounded():
    """Numeric outputs should be rounded to 12 decimals."""
    traj = [0.0, 0.1, 0.5, 2.0, 0.3, 0.1, 5.0, 0.2, 0.0, 1.0]
    result = run_collapse_analysis(traj)
    # Verify values are finite floats (rounding preserves this)
    for key in ("failure_risk", "collapse_score", "acceleration_peak"):
        val = result[key]
        assert isinstance(val, float)
        assert round(val, 12) == val
