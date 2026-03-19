"""
v80.5.0 — Neural ODE-Style Trajectory Observer (Deterministic).

Reconstructs continuous-time dynamics from discrete FSM history traces.
Extracts smooth trajectories, velocity estimates, and stability flow metrics
from the state transitions produced by QECFSM.

This is NOT a trained Neural ODE.  It uses deterministic ODE-style
interpolation and finite-difference derivative estimation to analyze
system dynamics from FSM decision history.

Layer 8 — Controller (observational).
Does not modify FSM or decoder internals.  Purely read-only on inputs.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Step 1 — Extract Time Series
# ---------------------------------------------------------------------------

def extract_time_series(
    history: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """Extract numeric time series from FSM history.

    Parameters
    ----------
    history : list[dict]
        FSM history trace (list of transition records).

    Returns
    -------
    dict
        Arrays keyed by ``"time"``, ``"stability_score"``, ``"epsilon"``,
        ``"reject_cycle"``.  Only entries with non-None stability_score
        are included.
    """
    scores: List[float] = []
    epsilons: List[float] = []
    reject_cycles: List[int] = []

    for entry in history:
        s = entry.get("stability_score")
        if s is None:
            continue
        scores.append(float(s))
        epsilons.append(float(entry.get("epsilon", 0.0)))
        reject_cycles.append(int(entry.get("reject_cycle", 0)))

    n = len(scores)
    return {
        "time": np.arange(n, dtype=np.float64),
        "stability_score": np.array(scores, dtype=np.float64),
        "epsilon": np.array(epsilons, dtype=np.float64),
        "reject_cycle": np.array(reject_cycles, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Step 2 — Continuous Trajectory (Cubic Hermite Interpolation)
# ---------------------------------------------------------------------------

def interpolate_trajectory(
    t: np.ndarray,
    y: np.ndarray,
    density: int = 10,
) -> tuple:
    """Interpolate a discrete trajectory onto a denser time grid.

    Uses cubic Hermite interpolation when there are enough points,
    otherwise falls back to linear interpolation.

    Parameters
    ----------
    t : np.ndarray
        Discrete time points.
    y : np.ndarray
        Corresponding values.
    density : int
        Number of interpolated points per original interval.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(t_dense, y_dense)``
    """
    if len(t) < 2:
        return t.copy(), y.copy()

    n_dense = (len(t) - 1) * density + 1
    t_dense = np.linspace(float(t[0]), float(t[-1]), n_dense)

    if len(t) < 3:
        # Linear interpolation for 2 points.
        y_dense = np.interp(t_dense, t, y)
        return t_dense, y_dense

    # Cubic Hermite: estimate slopes via central differences.
    m = np.empty_like(y)
    m[0] = y[1] - y[0]
    m[-1] = y[-1] - y[-2]
    m[1:-1] = (y[2:] - y[:-2]) / 2.0

    y_dense = np.empty(n_dense, dtype=np.float64)
    for i in range(len(t) - 1):
        t0, t1 = float(t[i]), float(t[i + 1])
        h = t1 - t0
        if h == 0:
            continue
        mask = (t_dense >= t0) & (t_dense <= t1)
        if i < len(t) - 2:
            # Avoid double-counting right endpoints except last segment.
            mask = (t_dense >= t0) & (t_dense < t1)
        s = (t_dense[mask] - t0) / h
        s2 = s * s
        s3 = s2 * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        y_dense[mask] = (
            h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]
        )

    return t_dense, y_dense


# ---------------------------------------------------------------------------
# Step 3 — Estimate Dynamics (ODE Approximation)
# ---------------------------------------------------------------------------

def estimate_derivative(
    t: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Estimate dy/dt via numpy gradient.

    Parameters
    ----------
    t : np.ndarray
        Time points.
    y : np.ndarray
        Corresponding values.

    Returns
    -------
    np.ndarray
        Estimated velocity (dy/dt).
    """
    if len(t) < 2:
        return np.zeros_like(y)
    return np.gradient(y, t)


# ---------------------------------------------------------------------------
# Step 4 — Stability Flow Metrics
# ---------------------------------------------------------------------------

def compute_flow_metrics(
    y: np.ndarray,
    dy_dt: np.ndarray,
) -> Dict[str, float]:
    """Compute stability flow metrics from trajectory and its derivative.

    Parameters
    ----------
    y : np.ndarray
        Trajectory values.
    dy_dt : np.ndarray
        Velocity estimates.

    Returns
    -------
    dict
        ``mean_velocity``, ``max_velocity``, ``zero_crossings``,
        ``oscillation_score``, ``convergence_rate``.
    """
    if len(dy_dt) == 0:
        return {
            "mean_velocity": 0.0,
            "max_velocity": 0.0,
            "zero_crossings": 0,
            "oscillation_score": 0.0,
            "convergence_rate": 0.0,
        }

    mean_velocity = float(np.mean(np.abs(dy_dt)))
    max_velocity = float(np.max(np.abs(dy_dt)))

    # Count sign changes in dy/dt.
    signs = np.sign(dy_dt)
    sign_changes = np.diff(signs)
    zero_crossings = int(np.count_nonzero(sign_changes))

    oscillation_score = zero_crossings / len(y) if len(y) > 0 else 0.0
    convergence_rate = float(np.abs(dy_dt[-1]))

    return {
        "mean_velocity": mean_velocity,
        "max_velocity": max_velocity,
        "zero_crossings": zero_crossings,
        "oscillation_score": oscillation_score,
        "convergence_rate": convergence_rate,
    }


# ---------------------------------------------------------------------------
# Step 5 — Phase Classification
# ---------------------------------------------------------------------------

def classify_trajectory(metrics: Dict[str, float]) -> str:
    """Classify the trajectory phase from flow metrics.

    Returns
    -------
    str
        One of ``"convergent"``, ``"oscillatory"``, ``"divergent"``,
        ``"stable_plateau"``.
    """
    osc = metrics.get("oscillation_score", 0.0)
    conv = metrics.get("convergence_rate", 0.0)
    mean_v = metrics.get("mean_velocity", 0.0)

    # Stable plateau: near-zero mean velocity overall.
    if mean_v < 1e-8:
        return "stable_plateau"

    # Oscillatory: high oscillation score.
    if osc > 0.3:
        return "oscillatory"

    # Divergent: final derivative still large (not settling).
    if conv > 0.5 * mean_v and mean_v > 1e-6:
        return "divergent"

    # Default: convergent.
    return "convergent"


# ---------------------------------------------------------------------------
# Step 6 — Main Entry
# ---------------------------------------------------------------------------

def analyze_trajectory(
    history: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    density: int = 10,
) -> Dict[str, Any]:
    """Analyze FSM history as a continuous trajectory.

    Parameters
    ----------
    history : list[dict]
        FSM history trace.
    output_dir : str, optional
        If provided, write ``trajectory_analysis.json`` to this directory.
    density : int
        Interpolation density.

    Returns
    -------
    dict
        Analysis summary with ``n_points``, ``mean_velocity``,
        ``max_velocity``, ``oscillation_score``, ``convergence_rate``,
        and ``classification``.
    """
    series = extract_time_series(history)
    t = series["time"]
    y = series["stability_score"]
    n_points = len(t)

    if n_points < 2:
        result = {
            "n_points": n_points,
            "mean_velocity": 0.0,
            "max_velocity": 0.0,
            "oscillation_score": 0.0,
            "convergence_rate": 0.0,
            "classification": "stable_plateau",
        }
    else:
        t_dense, y_dense = interpolate_trajectory(t, y, density=density)
        dy_dt = estimate_derivative(t_dense, y_dense)
        metrics = compute_flow_metrics(y_dense, dy_dt)
        classification = classify_trajectory(metrics)

        result = {
            "n_points": n_points,
            "mean_velocity": metrics["mean_velocity"],
            "max_velocity": metrics["max_velocity"],
            "oscillation_score": metrics["oscillation_score"],
            "convergence_rate": metrics["convergence_rate"],
            "classification": classification,
        }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "trajectory_analysis.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)

    return result
