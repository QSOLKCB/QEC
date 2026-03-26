"""v104.1.0 — Rotation-aware geometric diagnostics for decoder trajectories.

Treats decoder evolution as a geometric trajectory and extracts
rotation/transport metrics (complex/quaternion-inspired) to predict:
- convergence
- oscillation
- metastability
- basin switching

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only (math module).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

ROUND_PRECISION = 12

# Fixed ordering of state vector components.
_STATE_KEYS = [
    "residual_norm",
    "instability_score",
    "barrier_estimate",
    "boundary_distance",
    "spectral_radius_proxy",
    "convergence_signal",
    "control_signal",
    "basin_switch_score",
]

# Thresholds for event prediction (deterministic).
_ANGULAR_VELOCITY_HIGH = 0.5
_SPIRAL_SCORE_HIGH = 0.6
_CURVATURE_HIGH = 0.5
_INSTABILITY_HIGH = 0.6
_MOVEMENT_LOW = 0.05
_VARIANCE_LOW = 0.05


# ---------------------------------------------------------------------------
# 1. State Embedding
# ---------------------------------------------------------------------------


def build_state_vector(run_step: Dict[str, Any]) -> List[float]:
    """Build a fixed-order state vector from a run step's metrics.

    Parameters
    ----------
    run_step : dict
        A single run step.  Metrics are read from
        ``run_step["metrics"]`` if present, otherwise from *run_step*
        directly.

    Returns
    -------
    list of float
        State vector with fixed ordering.  Missing keys default to
        ``0.0``.  All values rounded to 12 decimals.
    """
    metrics = run_step.get("metrics", run_step)
    return [
        round(float(metrics.get(key, 0.0)), ROUND_PRECISION)
        for key in _STATE_KEYS
    ]


# ---------------------------------------------------------------------------
# 2. Trajectory Construction
# ---------------------------------------------------------------------------


def build_trajectory(runs: List[Dict[str, Any]]) -> List[List[float]]:
    """Build an ordered trajectory of state vectors from run history.

    Parameters
    ----------
    runs : list of dict
        Ordered sequence of run steps.

    Returns
    -------
    list of list of float
        Ordered sequence of state vectors.
    """
    return [build_state_vector(r) for r in runs]


# ---------------------------------------------------------------------------
# 3. Quaternion-Inspired 3D Projection
# ---------------------------------------------------------------------------


def project_to_3d(
    state: List[float],
) -> Tuple[float, float, float]:
    """Project a state vector to 3D using a fixed deterministic mapping.

    Mapping:
        x = residual_norm      (index 0)
        y = instability_score   (index 1)
        z = barrier_estimate    (index 2)

    Parameters
    ----------
    state : list of float
        State vector (length >= 3).

    Returns
    -------
    tuple of float
        (x, y, z) projection.
    """
    x = state[0] if len(state) > 0 else 0.0
    y = state[1] if len(state) > 1 else 0.0
    z = state[2] if len(state) > 2 else 0.0
    return (
        round(x, ROUND_PRECISION),
        round(y, ROUND_PRECISION),
        round(z, ROUND_PRECISION),
    )


# ---------------------------------------------------------------------------
# 4. Rotation / Geometry Metrics
# ---------------------------------------------------------------------------


def _vec_sub(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Subtract two 3-vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_norm(v: Tuple[float, float, float]) -> float:
    """Euclidean norm of a 3-vector."""
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _vec_dot(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    """Dot product of two 3-vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Cross product of two 3-vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def compute_turning_angles(
    traj3d: List[Tuple[float, float, float]],
) -> List[float]:
    """Compute turning angles between successive direction vectors.

    For each triple of consecutive points (p_{i-1}, p_i, p_{i+1}),
    the turning angle is the angle between direction vectors
    d_i = p_i - p_{i-1} and d_{i+1} = p_{i+1} - p_i.

    Parameters
    ----------
    traj3d : list of tuple
        Ordered 3D trajectory points.

    Returns
    -------
    list of float
        Turning angles in radians, length = len(traj3d) - 2.
        Each value in [0, pi].
    """
    if len(traj3d) < 3:
        return []

    angles: List[float] = []
    for i in range(1, len(traj3d) - 1):
        d1 = _vec_sub(traj3d[i], traj3d[i - 1])
        d2 = _vec_sub(traj3d[i + 1], traj3d[i])
        n1 = _vec_norm(d1)
        n2 = _vec_norm(d2)
        if n1 < 1e-15 or n2 < 1e-15:
            angles.append(0.0)
            continue
        cos_theta = _vec_dot(d1, d2) / (n1 * n2)
        # Clamp for numerical safety.
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angles.append(round(math.acos(cos_theta), ROUND_PRECISION))

    return angles


def compute_angular_velocity(angles: List[float]) -> float:
    """Compute angular velocity as mean absolute change in turning angle.

    Parameters
    ----------
    angles : list of float
        Turning angles (output of ``compute_turning_angles``).

    Returns
    -------
    float
        Mean |delta_angle|, or 0.0 if fewer than 2 angles.
    """
    if len(angles) < 2:
        if len(angles) == 1:
            return round(abs(angles[0]), ROUND_PRECISION)
        return 0.0
    deltas = [abs(angles[i] - angles[i - 1]) for i in range(1, len(angles))]
    return round(sum(deltas) / len(deltas), ROUND_PRECISION)


def compute_spiral_score(
    traj3d: List[Tuple[float, float, float]],
) -> float:
    """Compute a spiral score heuristic.

    Combines inward radial trend with nonzero rotation to detect
    spiraling convergence.

    Heuristic:
        - inward_fraction = fraction of steps where distance to
          centroid decreases
        - rotation_fraction = fraction of nonzero turning angles
        - spiral_score = inward_fraction * rotation_fraction

    Parameters
    ----------
    traj3d : list of tuple
        Ordered 3D trajectory points.

    Returns
    -------
    float
        Spiral score in [0, 1].
    """
    if len(traj3d) < 3:
        return 0.0

    # Compute centroid.
    n = len(traj3d)
    cx = sum(p[0] for p in traj3d) / n
    cy = sum(p[1] for p in traj3d) / n
    cz = sum(p[2] for p in traj3d) / n
    centroid = (cx, cy, cz)

    # Compute distances to centroid.
    dists = [_vec_norm(_vec_sub(p, centroid)) for p in traj3d]

    # Inward fraction.
    inward_count = 0
    for i in range(1, len(dists)):
        if dists[i] < dists[i - 1]:
            inward_count += 1
    inward_fraction = inward_count / (len(dists) - 1)

    # Rotation fraction via turning angles.
    angles = compute_turning_angles(traj3d)
    if not angles:
        return 0.0
    nonzero_count = sum(1 for a in angles if abs(a) > 1e-12)
    rotation_fraction = nonzero_count / len(angles)

    return round(inward_fraction * rotation_fraction, ROUND_PRECISION)


def compute_axis_lock(
    traj3d: List[Tuple[float, float, float]],
) -> float:
    """Compute axis-lock score measuring motion alignment to a dominant axis.

    Measures the fraction of total displacement variance concentrated
    along the axis with maximum variance.

    Parameters
    ----------
    traj3d : list of tuple
        Ordered 3D trajectory points.

    Returns
    -------
    float
        Axis-lock score in [0, 1].  1.0 means all motion along one axis.
    """
    if len(traj3d) < 2:
        return 0.0

    # Compute displacements.
    dx = [traj3d[i][0] - traj3d[i - 1][0] for i in range(1, len(traj3d))]
    dy = [traj3d[i][1] - traj3d[i - 1][1] for i in range(1, len(traj3d))]
    dz = [traj3d[i][2] - traj3d[i - 1][2] for i in range(1, len(traj3d))]

    var_x = _population_variance(dx)
    var_y = _population_variance(dy)
    var_z = _population_variance(dz)

    total_var = var_x + var_y + var_z
    if total_var < 1e-15:
        return 0.0

    max_var = max(var_x, var_y, var_z)
    return round(max_var / total_var, ROUND_PRECISION)


def compute_curvature(
    traj3d: List[Tuple[float, float, float]],
) -> float:
    """Compute average discrete curvature using finite differences.

    For each triple (p_{i-1}, p_i, p_{i+1}):
        curvature_i = |cross(d1, d2)| / |d1|^3

    where d1 = p_i - p_{i-1}, d2 = p_{i+1} - 2*p_i + p_{i-1}.

    Parameters
    ----------
    traj3d : list of tuple
        Ordered 3D trajectory points.

    Returns
    -------
    float
        Average curvature, or 0.0 if fewer than 3 points.
    """
    if len(traj3d) < 3:
        return 0.0

    curvatures: List[float] = []
    for i in range(1, len(traj3d) - 1):
        d1 = _vec_sub(traj3d[i], traj3d[i - 1])
        # Second difference: p_{i+1} - 2*p_i + p_{i-1}
        d2 = (
            traj3d[i + 1][0] - 2 * traj3d[i][0] + traj3d[i - 1][0],
            traj3d[i + 1][1] - 2 * traj3d[i][1] + traj3d[i - 1][1],
            traj3d[i + 1][2] - 2 * traj3d[i][2] + traj3d[i - 1][2],
        )
        cross = _vec_cross(d1, d2)
        cross_norm = _vec_norm(cross)
        d1_norm = _vec_norm(d1)
        if d1_norm < 1e-15:
            curvatures.append(0.0)
            continue
        curvatures.append(cross_norm / (d1_norm ** 3))

    if not curvatures:
        return 0.0
    return round(sum(curvatures) / len(curvatures), ROUND_PRECISION)


# ---------------------------------------------------------------------------
# 5. High-Dimensional Coupling (Octonion-Inspired)
# ---------------------------------------------------------------------------


def compute_coupling_metrics(
    traj: List[List[float]],
) -> Dict[str, Any]:
    """Compute high-dimensional coupling metrics (octonion-inspired).

    Metrics:
        - plane_coupling_score: correlation between pairs of dimensions
        - multi_axis_variation: number of axis pairs with significant
          co-variation
        - dimensional_activity: number of dimensions with nonzero variance

    Parameters
    ----------
    traj : list of list of float
        Full-dimensional trajectory.

    Returns
    -------
    dict
        Coupling metrics.
    """
    if len(traj) < 2:
        return {
            "plane_coupling_score": 0.0,
            "multi_axis_variation": 0,
            "dimensional_activity": 0,
        }

    ndim = len(traj[0]) if traj else 0
    if ndim == 0:
        return {
            "plane_coupling_score": 0.0,
            "multi_axis_variation": 0,
            "dimensional_activity": 0,
        }

    # Compute per-dimension deltas.
    deltas: List[List[float]] = []
    for d in range(ndim):
        deltas.append([traj[i + 1][d] - traj[i][d] for i in range(len(traj) - 1)])

    # Dimensional activity: dimensions with nonzero variance.
    dim_variances = [_population_variance(deltas[d]) for d in range(ndim)]
    dimensional_activity = sum(1 for v in dim_variances if v > 1e-12)

    # Plane coupling: average absolute correlation between dimension pairs.
    coupling_scores: List[float] = []
    multi_axis_count = 0
    for i in range(ndim):
        for j in range(i + 1, ndim):
            corr = _correlation(deltas[i], deltas[j])
            coupling_scores.append(abs(corr))
            if abs(corr) > 0.3:
                multi_axis_count += 1

    plane_coupling_score = 0.0
    if coupling_scores:
        plane_coupling_score = sum(coupling_scores) / len(coupling_scores)

    return {
        "plane_coupling_score": round(plane_coupling_score, ROUND_PRECISION),
        "multi_axis_variation": multi_axis_count,
        "dimensional_activity": dimensional_activity,
    }


# ---------------------------------------------------------------------------
# 6. Event Prediction Signals
# ---------------------------------------------------------------------------


def predict_events(geometry_metrics: Dict[str, Any]) -> Dict[str, str]:
    """Predict trajectory events from geometry metrics.

    Deterministic threshold-based rules:
        - high angular_velocity -> oscillation
        - high spiral_score + inward -> convergence likely
        - high curvature + high instability -> basin_switch_risk
        - low movement + low variance -> metastable

    Parameters
    ----------
    geometry_metrics : dict
        Must contain ``angular_velocity``, ``spiral_score``,
        ``curvature``, ``axis_lock``, ``total_displacement``,
        ``displacement_variance``.

    Returns
    -------
    dict
        Event predictions with keys:
        ``convergence``, ``oscillation``, ``basin_switch_risk``,
        ``metastable``.
    """
    angular_vel = geometry_metrics.get("angular_velocity", 0.0)
    spiral = geometry_metrics.get("spiral_score", 0.0)
    curv = geometry_metrics.get("curvature", 0.0)
    displacement = geometry_metrics.get("total_displacement", 0.0)
    disp_var = geometry_metrics.get("displacement_variance", 0.0)
    instability = geometry_metrics.get("mean_instability", 0.0)

    # Convergence.
    if spiral >= _SPIRAL_SCORE_HIGH:
        convergence = "likely"
    elif spiral >= _SPIRAL_SCORE_HIGH * 0.5:
        convergence = "moderate"
    else:
        convergence = "low"

    # Oscillation.
    if angular_vel >= _ANGULAR_VELOCITY_HIGH:
        oscillation = "high"
    elif angular_vel >= _ANGULAR_VELOCITY_HIGH * 0.5:
        oscillation = "moderate"
    else:
        oscillation = "low"

    # Basin switch risk.
    if curv >= _CURVATURE_HIGH and instability >= _INSTABILITY_HIGH:
        basin_switch_risk = "high"
    elif curv >= _CURVATURE_HIGH * 0.5 or instability >= _INSTABILITY_HIGH * 0.5:
        basin_switch_risk = "medium"
    else:
        basin_switch_risk = "low"

    # Metastability.
    if displacement < _MOVEMENT_LOW and disp_var < _VARIANCE_LOW:
        metastable = "likely"
    elif displacement < _MOVEMENT_LOW * 2:
        metastable = "moderate"
    else:
        metastable = "low"

    return {
        "convergence": convergence,
        "oscillation": oscillation,
        "basin_switch_risk": basin_switch_risk,
        "metastable": metastable,
    }


# ---------------------------------------------------------------------------
# 7. Full Pipeline
# ---------------------------------------------------------------------------


def run_trajectory_geometry_analysis(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full trajectory geometry analysis pipeline.

    Pipeline:
        runs -> trajectory -> 3D projection -> rotation metrics
        -> coupling metrics -> event predictions

    Parameters
    ----------
    runs : list of dict
        Ordered sequence of run steps.

    Returns
    -------
    dict
        Complete geometry analysis with keys:
        ``trajectory``, ``traj3d``, ``rotation_metrics``,
        ``coupling_metrics``, ``predictions``.
    """
    # Build trajectory.
    traj = build_trajectory(runs)
    if not traj:
        return _empty_result()

    # Project to 3D.
    traj3d = [project_to_3d(s) for s in traj]

    # Rotation metrics.
    angles = compute_turning_angles(traj3d)
    angular_vel = compute_angular_velocity(angles)
    spiral = compute_spiral_score(traj3d)
    axis_lock = compute_axis_lock(traj3d)
    curv = compute_curvature(traj3d)

    # Displacement metrics for event prediction.
    total_disp = _total_displacement(traj3d)
    disp_var = _displacement_variance(traj3d)

    # Mean instability from state vectors.
    instability_values = [s[1] if len(s) > 1 else 0.0 for s in traj]
    mean_instability = (
        sum(instability_values) / len(instability_values)
        if instability_values
        else 0.0
    )

    rotation_metrics = {
        "turning_angles": angles,
        "angular_velocity": angular_vel,
        "spiral_score": spiral,
        "axis_lock": axis_lock,
        "curvature": curv,
        "total_displacement": round(total_disp, ROUND_PRECISION),
        "displacement_variance": round(disp_var, ROUND_PRECISION),
        "mean_instability": round(mean_instability, ROUND_PRECISION),
    }

    # Coupling metrics.
    coupling = compute_coupling_metrics(traj)

    # Event predictions.
    predictions = predict_events(rotation_metrics)

    return {
        "trajectory_length": len(traj),
        "rotation_metrics": rotation_metrics,
        "coupling_metrics": coupling,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# 8. Formatter
# ---------------------------------------------------------------------------


def format_trajectory_geometry_summary(result: Dict[str, Any]) -> str:
    """Format trajectory geometry analysis as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_trajectory_geometry_analysis``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    rm = result.get("rotation_metrics", {})
    cm = result.get("coupling_metrics", {})
    pred = result.get("predictions", {})

    lines.append("")
    lines.append("=== Trajectory Geometry ===")
    lines.append("")
    lines.append(f"Angular Velocity: {rm.get('angular_velocity', 0.0):.2f}")
    lines.append(f"Spiral Score: {rm.get('spiral_score', 0.0):.2f}")
    lines.append(f"Axis Lock: {rm.get('axis_lock', 0.0):.2f}")
    lines.append(f"Curvature: {rm.get('curvature', 0.0):.2f}")
    lines.append("")
    lines.append("Coupling:")
    lines.append(
        f"  Plane Coupling: {cm.get('plane_coupling_score', 0.0):.2f}"
    )
    lines.append(
        f"  Dimensional Activity: {cm.get('dimensional_activity', 0)}"
    )
    lines.append("")
    lines.append("Predictions:")
    lines.append(f"  Convergence: {pred.get('convergence', 'unknown').title()}")
    lines.append(f"  Oscillation: {pred.get('oscillation', 'unknown').title()}")
    lines.append(
        f"  Basin Switch Risk: {pred.get('basin_switch_risk', 'unknown').title()}"
    )
    lines.append(f"  Metastable: {pred.get('metastable', 'unknown').title()}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _population_variance(values: List[float]) -> float:
    """Compute population variance.  Returns 0.0 for empty list."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _correlation(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation coefficient.  Returns 0.0 on error."""
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx < 1e-15 or sy < 1e-15:
        return 0.0
    return max(-1.0, min(1.0, cov / (sx * sy)))


def _total_displacement(
    traj3d: List[Tuple[float, float, float]],
) -> float:
    """Total path length of a 3D trajectory."""
    if len(traj3d) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(traj3d)):
        total += _vec_norm(_vec_sub(traj3d[i], traj3d[i - 1]))
    return total


def _displacement_variance(
    traj3d: List[Tuple[float, float, float]],
) -> float:
    """Variance of step displacements in a 3D trajectory."""
    if len(traj3d) < 2:
        return 0.0
    steps = [
        _vec_norm(_vec_sub(traj3d[i], traj3d[i - 1]))
        for i in range(1, len(traj3d))
    ]
    return _population_variance(steps)


def _empty_result() -> Dict[str, Any]:
    """Return an empty geometry result."""
    return {
        "trajectory_length": 0,
        "rotation_metrics": {
            "turning_angles": [],
            "angular_velocity": 0.0,
            "spiral_score": 0.0,
            "axis_lock": 0.0,
            "curvature": 0.0,
            "total_displacement": 0.0,
            "displacement_variance": 0.0,
            "mean_instability": 0.0,
        },
        "coupling_metrics": {
            "plane_coupling_score": 0.0,
            "multi_axis_variation": 0,
            "dimensional_activity": 0,
        },
        "predictions": {
            "convergence": "low",
            "oscillation": "low",
            "basin_switch_risk": "low",
            "metastable": "low",
        },
    }
