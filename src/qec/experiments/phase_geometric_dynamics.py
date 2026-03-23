"""v87.0.0 — Geometric Trajectory Dynamics.

Turns ternary syndrome trajectories into geometric objects:
  - Path length (total Euclidean distance traversed)
  - Step vectors (displacement between consecutive points)
  - Curvature (cosine-based angular change between steps)
  - Attractor detection (repeated-state analysis)
  - Confinement / spread (bounding-box extent)

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Step 1 — Path Length
# ---------------------------------------------------------------------------


def compute_trajectory_length(
    series: List[Tuple[int, ...]],
) -> float:
    """Sum of Euclidean distances between consecutive ternary points.

    Parameters
    ----------
    series:
        Ordered list of ternary tuples, e.g. ``[(1, 0, -1, 1), ...]``.

    Returns
    -------
    Total path length (float).  Zero for fewer than 2 points.
    """
    if len(series) < 2:
        return 0.0
    total = 0.0
    for i in range(len(series) - 1):
        a = np.array(series[i], dtype=np.float64)
        b = np.array(series[i + 1], dtype=np.float64)
        total += float(np.linalg.norm(b - a))
    return total


# ---------------------------------------------------------------------------
# Step 2 — Step Vectors
# ---------------------------------------------------------------------------


def compute_step_vectors(
    series: List[Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
    """Compute displacement vectors between consecutive points.

    v_i = p_{i+1} - p_i

    Returns
    -------
    List of displacement tuples.  Empty for fewer than 2 points.
    """
    vectors: List[Tuple[int, ...]] = []
    for i in range(len(series) - 1):
        v = tuple(int(b - a) for a, b in zip(series[i], series[i + 1]))
        vectors.append(v)
    return vectors


# ---------------------------------------------------------------------------
# Step 3 — Curvature
# ---------------------------------------------------------------------------


def compute_curvature(
    step_vectors: List[Tuple[int, ...]],
) -> Dict[str, Optional[float]]:
    """Compute curvature from cosine change between consecutive step vectors.

    For each consecutive pair (v_i, v_{i+1}):
        cos_theta = dot(v_i, v_{i+1}) / (|v_i| * |v_{i+1}|)
        curvature_i = 1 - cos_theta

    Zero-length vectors are skipped (undefined curvature).

    Returns
    -------
    dict with ``mean_curvature`` and ``max_curvature``.
    Both are ``None`` if no valid curvature values exist.
    """
    curvatures: List[float] = []
    for i in range(len(step_vectors) - 1):
        v1 = np.array(step_vectors[i], dtype=np.float64)
        v2 = np.array(step_vectors[i + 1], dtype=np.float64)
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 == 0.0 or norm2 == 0.0:
            continue
        cos_theta = float(np.dot(v1, v2)) / (norm1 * norm2)
        # Clamp to [-1, 1] for numerical safety.
        cos_theta = max(-1.0, min(1.0, cos_theta))
        curvatures.append(1.0 - cos_theta)

    if not curvatures:
        return {"mean_curvature": None, "max_curvature": None}

    return {
        "mean_curvature": float(np.mean(curvatures)),
        "max_curvature": float(np.max(curvatures)),
    }


# ---------------------------------------------------------------------------
# Step 4 — Attractor Detection
# ---------------------------------------------------------------------------


def detect_attractors(
    series: List[Tuple[int, ...]],
) -> Dict[str, Any]:
    """Detect attractor states from repeated-state analysis.

    Rules:
      - A state that appears more than once is an attractor candidate.
      - The state with the longest consecutive run is the strongest attractor.
      - Strength = longest_run / n_steps.

    Returns
    -------
    dict with ``has_attractor`` (bool), ``attractor_state`` (tuple or None),
    ``strength`` (float, 0.0 if no attractor).
    """
    if len(series) == 0:
        return {
            "has_attractor": False,
            "attractor_state": None,
            "strength": 0.0,
        }

    # Count occurrences.
    counts: Dict[Tuple[int, ...], int] = {}
    for s in series:
        counts[s] = counts.get(s, 0) + 1

    # Find longest consecutive run for each state.
    best_state: Optional[Tuple[int, ...]] = None
    best_run = 0
    current_state = series[0]
    current_run = 1

    for i in range(1, len(series)):
        if series[i] == current_state:
            current_run += 1
        else:
            if current_run > best_run:
                best_run = current_run
                best_state = current_state
            current_state = series[i]
            current_run = 1
    # Final run.
    if current_run > best_run:
        best_run = current_run
        best_state = current_state

    # Attractor requires repeated occurrence (run > 1) or being the only state.
    has_attractor = best_run > 1 or len(counts) == 1
    strength = best_run / len(series) if len(series) > 0 else 0.0

    return {
        "has_attractor": has_attractor,
        "attractor_state": best_state,
        "strength": strength,
    }


# ---------------------------------------------------------------------------
# Step 5 — Confinement / Spread
# ---------------------------------------------------------------------------


def compute_spread(
    series: List[Tuple[int, ...]],
) -> float:
    """Compute bounding-box spread of the trajectory.

    Spread = sum of (max - min) per dimension.

    Returns
    -------
    Scalar spread (float).  Zero for empty or single-point series.
    """
    if len(series) == 0:
        return 0.0
    arr = np.array(series, dtype=np.float64)
    per_dim = np.max(arr, axis=0) - np.min(arr, axis=0)
    return float(np.sum(per_dim))


# ---------------------------------------------------------------------------
# Step 6 — Full Analysis
# ---------------------------------------------------------------------------


def run_geometric_dynamics(
    series: List[Tuple[int, ...]],
) -> Dict[str, Any]:
    """Run full geometric trajectory dynamics analysis.

    Parameters
    ----------
    series:
        Ordered list of ternary-encoded tuples.

    Returns
    -------
    dict with ``trajectory_length``, ``mean_curvature``, ``max_curvature``,
    ``spread``, ``attractor``.
    """
    step_vectors = compute_step_vectors(series)
    curvature = compute_curvature(step_vectors)
    attractor = detect_attractors(series)

    return {
        "trajectory_length": compute_trajectory_length(series),
        "mean_curvature": curvature["mean_curvature"],
        "max_curvature": curvature["max_curvature"],
        "spread": compute_spread(series),
        "attractor": attractor,
    }
