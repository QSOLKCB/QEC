"""Tests for collapse_analysis — v105.3.0."""

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
