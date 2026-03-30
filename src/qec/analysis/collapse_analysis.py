"""Deterministic collapse / singularity analysis — v105.3.1.

Pure analysis layer for detecting acceleration spikes, collapse scores,
singularity events, and basin-switch predictions from decoder metric
trajectories.  All functions are pure, deterministic, and side-effect free.

Dependencies: none (stdlib + math only).
"""

from __future__ import annotations

import math
from typing import Dict, List


# -- finite differences ------------------------------------------------------

def compute_velocity_acceleration_metrics(
    trajectory: list[float],
) -> dict:
    """Compute velocity and acceleration via finite differences.

    Returns dict with keys: velocity, acceleration, acceleration_mean,
    acceleration_peak.  If len(trajectory) < 3 all values are zero/empty.
    """
    if len(trajectory) < 3:
        return {
            "velocity": [],
            "acceleration": [],
            "velocity_peak": 0.0,
            "acceleration_mean": 0.0,
            "acceleration_peak": 0.0,
        }

    velocity = [
        trajectory[i + 1] - trajectory[i]
        for i in range(len(trajectory) - 1)
    ]
    acceleration = [
        velocity[i + 1] - velocity[i]
        for i in range(len(velocity) - 1)
    ]

    abs_vel = [abs(v) for v in velocity]
    vel_peak = max(abs_vel) if abs_vel else 0.0

    abs_acc = [abs(a) for a in acceleration]
    acc_mean = sum(abs_acc) / len(abs_acc) if abs_acc else 0.0
    acc_peak = max(abs_acc) if abs_acc else 0.0

    return {
        "velocity": velocity,
        "acceleration": acceleration,
        "velocity_peak": round(vel_peak, 12),
        "acceleration_mean": round(acc_mean, 12),
        "acceleration_peak": round(acc_peak, 12),
    }


# -- spike detection ---------------------------------------------------------

def detect_acceleration_spikes(
    acceleration_series: list[float],
) -> dict:
    """Detect spikes where |a| > mean + 2*std.

    Returns dict with spike_count, spike_indices, spike_density.
    """
    n = len(acceleration_series)
    if n == 0:
        return {"spike_count": 0, "spike_indices": [], "spike_density": 0.0}

    abs_vals = [abs(a) for a in acceleration_series]
    mean = sum(abs_vals) / n

    variance = sum((v - mean) ** 2 for v in abs_vals) / n
    std = math.sqrt(variance)

    if std <= 0.0:
        return {"spike_count": 0, "spike_indices": [], "spike_density": 0.0}

    threshold = mean + 2.0 * std
    spike_indices: List[int] = [
        i for i, v in enumerate(abs_vals) if v > threshold
    ]
    spike_count = len(spike_indices)
    spike_density = min(1.0, spike_count / n)

    return {
        "spike_count": spike_count,
        "spike_indices": spike_indices,
        "spike_density": spike_density,
    }


# -- collapse score ----------------------------------------------------------

def compute_collapse_score(
    spike_density: float,
    acceleration_peak: float,
) -> float:
    """Weighted collapse score in [0, 1].

    score = 0.7 * spike_density + 0.3 * min(1.0, acceleration_peak)
    Result is clamped to [0, 1].
    """
    raw = 0.7 * spike_density + 0.3 * min(1.0, acceleration_peak)
    return round(max(0.0, min(1.0, raw)), 12)


# -- singularity events -----------------------------------------------------

def detect_singularity_events(
    spike_density: float,
    collapse_score: float,
) -> list:
    """Return list of singularity event dicts.

    A collapse_event is emitted when spike_density > 0.2 and
    collapse_score > 0.6.
    """
    events: List[Dict] = []
    if spike_density > 0.2 and collapse_score > 0.6:
        events.append({
            "type": "collapse_event",
            "index": 0,
            "spike_density": spike_density,
            "collapse_score": collapse_score,
        })
    events = sorted(events, key=lambda e: e["index"])
    return events


# -- basin switch prediction -------------------------------------------------

def predict_basin_switch(
    spike_density: float,
    collapse_score: float,
    acceleration_peak: float,
    acceleration_mean: float,
) -> dict:
    """Predict whether a basin switch is imminent.

    Conditions (all must hold):
      - spike_density > 0
      - collapse_score > 0.6
      - acceleration_peak > acceleration_mean
    """
    prediction = (
        spike_density > 0.0
        and collapse_score > 0.6
        and acceleration_peak > acceleration_mean
    )
    return {
        "basin_switch_predicted": prediction,
        "spike_density": round(spike_density, 12),
        "collapse_score": round(collapse_score, 12),
        "acceleration_peak": round(acceleration_peak, 12),
        "acceleration_mean": round(acceleration_mean, 12),
    }


# -- orchestrator ------------------------------------------------------------

def run_collapse_analysis(trajectory: list[float]) -> dict:
    """Run full collapse analysis pipeline on a trajectory.

    Returns dict with failure_risk, collapse_score, acceleration_peak,
    basin_switch_prediction, and singularity_events.
    """
    metrics = compute_velocity_acceleration_metrics(trajectory)
    spikes = detect_acceleration_spikes(metrics["acceleration"])

    spike_density = spikes["spike_density"]
    acc_peak = metrics["acceleration_peak"]
    acc_mean = metrics["acceleration_mean"]

    score = compute_collapse_score(spike_density, acc_peak)
    events = detect_singularity_events(spike_density, score)
    switch = predict_basin_switch(spike_density, score, acc_peak, acc_mean)

    return {
        "failure_risk": round(score, 12),
        "collapse_score": round(score, 12),
        "acceleration_peak": round(acc_peak, 12),
        "basin_switch_prediction": switch["basin_switch_predicted"],
        "singularity_events": events,
    }
