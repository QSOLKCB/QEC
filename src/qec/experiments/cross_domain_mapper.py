"""v82.6.0 — Cross-Domain Inverse Design (UFF <-> Sequence Mapping).

Maps between UFF parameter space and sequence space via invariant
similarity.  Given a theta, finds the best-matching sequence; given a
sequence, finds the best-matching theta.

Matching is performed in a shared embedding space derived from
deterministic invariant summaries produced by the existing landscape
engines.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from qec.experiments.sequence_landscape import classify_sequence, run_sequence_landscape
from qec.experiments.uff_landscape import run_uff_landscape


# ---------------------------------------------------------------------------
# Encoding tables — deterministic categorical → integer mappings
# ---------------------------------------------------------------------------

_PHASE_ENCODING: Dict[str, int] = {
    "stable_region": 0,
    "near_boundary": 1,
    "unstable_region": 2,
    "chaotic_transition": 3,
    "unknown": 4,
}

_CLASS_ENCODING: Dict[str, int] = {
    "stable": 0,
    "fragile": 1,
    "boundary_rider": 2,
    "chaotic": 3,
}

_TRAJECTORY_ENCODING: Dict[str, int] = {
    "convergent": 0,
    "oscillating": 1,
    "divergent": 2,
    "chaotic": 3,
    "unknown": 4,
}


# ---------------------------------------------------------------------------
# Step 1 — Shared Embedding
# ---------------------------------------------------------------------------

def embed_summary(summary: Dict[str, Any]) -> np.ndarray:
    """Convert an invariant summary into a numeric embedding vector.

    Works with both UFF point summaries (``input_type`` absent or
    ``"theta"``) and sequence summaries (``input_type == "sequence"``).

    The embedding is a 4-element vector:

    - ``[0]``: stability_score (float)
    - ``[1]``: phase encoded as integer
    - ``[2]``: class encoded as integer
    - ``[3]``: trajectory_class encoded as integer

    For UFF summaries that lack ``class`` or ``trajectory_class``,
    the class is derived via :func:`classify_sequence` (which operates
    on any summary with the right keys), and trajectory defaults to
    ``"unknown"``.

    Parameters
    ----------
    summary : dict
        Invariant summary from either landscape engine.  Not mutated.

    Returns
    -------
    numpy.ndarray
        Shape ``(4,)`` float64 embedding vector.
    """
    stability = float(summary.get("stability_score", 0.0))

    phase_str = str(summary.get("phase", "unknown"))
    phase_enc = _PHASE_ENCODING.get(phase_str, _PHASE_ENCODING["unknown"])

    # Class: use existing key or derive via classify_sequence
    cls_str = summary.get("class", "")
    if not cls_str:
        cls_str = classify_sequence(summary)
    class_enc = _CLASS_ENCODING.get(cls_str, 0)

    traj_str = str(summary.get("trajectory_class", "unknown"))
    traj_enc = _TRAJECTORY_ENCODING.get(traj_str, _TRAJECTORY_ENCODING["unknown"])

    return np.array([stability, phase_enc, class_enc, traj_enc], dtype=np.float64)


# ---------------------------------------------------------------------------
# Step 2 — Distance Metric
# ---------------------------------------------------------------------------

# Weights: stability difference is most important, categorical mismatches
# add fixed penalties.
_W_STABILITY = 1.0
_W_PHASE = 2.0
_W_CLASS = 3.0
_W_TRAJECTORY = 1.0

_WEIGHTS = np.array(
    [_W_STABILITY, _W_PHASE, _W_CLASS, _W_TRAJECTORY], dtype=np.float64,
)


def invariant_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Compute weighted L1 distance between two invariant summaries.

    Parameters
    ----------
    a, b : dict
        Invariant summaries (UFF or sequence).  Not mutated.

    Returns
    -------
    float
        Non-negative distance.  Lower means more similar.
    """
    va = embed_summary(a)
    vb = embed_summary(b)
    return float(np.sum(_WEIGHTS * np.abs(va - vb)))


# ---------------------------------------------------------------------------
# Step 3 — UFF → Sequence Mapping
# ---------------------------------------------------------------------------

def map_theta_to_sequence(
    theta: List[float],
    theta_result: Dict[str, Any],
    sequences: List[Any],
    pipeline_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    """Find the sequence whose invariant profile best matches a UFF theta.

    Parameters
    ----------
    theta : list[float]
        The source UFF parameter vector ``[V0, Rc, beta]``.
    theta_result : dict
        Pre-computed UFF landscape point summary for *theta*.
        Must contain at least ``stability_score`` and ``phase``.
    sequences : list
        Candidate sequences to search over.
    pipeline_fn : callable
        Pipeline function for evaluating sequences (passed to
        :func:`run_sequence_landscape`).

    Returns
    -------
    dict
        Mapping result with keys: ``theta``, ``best_sequence``,
        ``distance``, ``match_summary``.
    """
    if not sequences:
        return {
            "theta": list(theta),
            "best_sequence": None,
            "distance": float("inf"),
            "match_summary": {},
        }

    landscape = run_sequence_landscape(sequences, pipeline_fn=pipeline_fn)
    points = landscape["points"]

    best_point: Optional[Dict[str, Any]] = None
    best_dist = float("inf")
    for point in points:
        d = invariant_distance(theta_result, point)
        if d < best_dist:
            best_dist = d
            best_point = point

    return {
        "theta": list(theta),
        "best_sequence": best_point["input"] if best_point else None,
        "distance": best_dist,
        "match_summary": dict(best_point) if best_point else {},
    }


# ---------------------------------------------------------------------------
# Step 4 — Sequence → UFF Mapping
# ---------------------------------------------------------------------------

def map_sequence_to_theta(
    seq: Any,
    seq_result: Dict[str, Any],
    theta_grid: List[List[float]],
    v_circ_fn: Callable[..., np.ndarray] | None = None,
) -> Dict[str, Any]:
    """Find the UFF theta whose invariant profile best matches a sequence.

    Parameters
    ----------
    seq : Any
        The source sequence identifier.
    seq_result : dict
        Pre-computed sequence summary.  Must contain at least
        ``stability_score``, ``phase``, and ``class``.
    theta_grid : list[list[float]]
        Candidate theta vectors ``[[V0, Rc, beta], ...]``.
    v_circ_fn : callable, optional
        Velocity curve generator passed to :func:`run_uff_landscape`.

    Returns
    -------
    dict
        Mapping result with keys: ``sequence``, ``best_theta``,
        ``distance``, ``match_summary``.
    """
    if not theta_grid:
        return {
            "sequence": seq,
            "best_theta": [],
            "distance": float("inf"),
            "match_summary": {},
        }

    # Decompose grid into per-axis unique values to feed run_uff_landscape.
    V0_vals = sorted(set(t[0] for t in theta_grid))
    Rc_vals = sorted(set(t[1] for t in theta_grid))
    beta_vals = sorted(set(t[2] for t in theta_grid))

    landscape = run_uff_landscape(
        V0_vals, Rc_vals, beta_vals, v_circ_fn=v_circ_fn,
    )
    points = landscape["points"]

    best_point: Optional[Dict[str, Any]] = None
    best_dist = float("inf")
    for point in points:
        d = invariant_distance(seq_result, point)
        if d < best_dist:
            best_dist = d
            best_point = point

    return {
        "sequence": seq,
        "best_theta": list(best_point["theta"]) if best_point else [],
        "distance": best_dist,
        "match_summary": dict(best_point) if best_point else {},
    }


# ---------------------------------------------------------------------------
# Step 5 — Unified API
# ---------------------------------------------------------------------------

def run_cross_domain_mapping(
    *,
    theta: List[float] | None = None,
    theta_result: Dict[str, Any] | None = None,
    sequence: Any | None = None,
    seq_result: Dict[str, Any] | None = None,
    theta_grid: List[List[float]] | None = None,
    sequences: List[Any] | None = None,
    v_circ_fn: Callable[..., np.ndarray] | None = None,
    pipeline_fn: Callable[..., Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Unified cross-domain mapping entry point.

    Exactly one of *theta* or *sequence* must be provided to select the
    mapping direction:

    - **theta provided**: maps UFF → best matching sequence.
      Requires *theta_result*, *sequences*, and *pipeline_fn*.
    - **sequence provided**: maps sequence → best matching theta.
      Requires *seq_result* and *theta_grid*.

    Parameters
    ----------
    theta : list[float], optional
        Source UFF parameter vector.
    theta_result : dict, optional
        Pre-computed UFF summary for *theta*.
    sequence : Any, optional
        Source sequence identifier.
    seq_result : dict, optional
        Pre-computed sequence summary.
    theta_grid : list[list[float]], optional
        Candidate theta vectors for sequence→UFF direction.
    sequences : list, optional
        Candidate sequences for UFF→sequence direction.
    v_circ_fn : callable, optional
        Velocity curve generator for UFF landscape.
    pipeline_fn : callable, optional
        Pipeline function for sequence evaluation.

    Returns
    -------
    dict
        Mapping result (see :func:`map_theta_to_sequence` or
        :func:`map_sequence_to_theta`).

    Raises
    ------
    ValueError
        If both or neither of *theta* and *sequence* are provided,
        or if required companion arguments are missing.
    """
    has_theta = theta is not None
    has_seq = sequence is not None

    if has_theta == has_seq:
        raise ValueError(
            "Exactly one of 'theta' or 'sequence' must be provided."
        )

    if has_theta:
        if theta_result is None:
            raise ValueError(
                "'theta_result' is required when mapping from theta."
            )
        if sequences is None:
            raise ValueError(
                "'sequences' is required when mapping from theta."
            )
        if pipeline_fn is None:
            raise ValueError(
                "'pipeline_fn' is required when mapping from theta."
            )
        return map_theta_to_sequence(
            theta, theta_result, sequences, pipeline_fn,
        )

    # sequence direction
    if seq_result is None:
        raise ValueError(
            "'seq_result' is required when mapping from sequence."
        )
    if theta_grid is None:
        raise ValueError(
            "'theta_grid' is required when mapping from sequence."
        )
    return map_sequence_to_theta(
        sequence, seq_result, theta_grid, v_circ_fn=v_circ_fn,
    )
