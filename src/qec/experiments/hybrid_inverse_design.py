"""v82.8.0 — Hybrid Inverse Design Engine (theta + sequence).

Deterministic bounded search over (theta, sequence) candidate pairs
to find inputs that best produce a target behavior class.

Given a desired behavior (e.g. ``"stable"``, ``"fragile"``, ``"chaotic"``,
``"boundary_rider"``), evaluates all pairs from a bounded candidate space
using the hybrid co-design engine and ranks them by invariant agreement.

This is scan-and-rank, not optimization.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from qec.experiments.hybrid_codesign import (
    run_hybrid_pair,
    score_hybrid_pair,
    summarize_hybrid_pair,
)
from qec.experiments.sequence_landscape import classify_sequence


# ---------------------------------------------------------------------------
# Valid targets
# ---------------------------------------------------------------------------

_VALID_TARGETS = frozenset({"stable", "fragile", "chaotic", "boundary_rider"})

_WORST_SCORE = -1e9  # Scores are higher-is-better; invalid gets worst


# ---------------------------------------------------------------------------
# Step 1 — Target specification
# ---------------------------------------------------------------------------

def build_target_spec(target: str) -> Dict[str, Any]:
    """Build a deterministic scoring specification for a target behavior.

    Parameters
    ----------
    target : str
        One of ``"stable"``, ``"fragile"``, ``"chaotic"``,
        ``"boundary_rider"``.

    Returns
    -------
    dict
        Scoring weights with keys: ``desired_class``,
        ``phase_preference`` (optional), ``min_stability`` (optional).

    Raises
    ------
    ValueError
        If *target* is not a recognized behavior class.
    """
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"Unknown target {target!r}; valid: {sorted(_VALID_TARGETS)}"
        )
    return _TARGET_SPECS[target]


_TARGET_SPECS: Dict[str, Dict[str, Any]] = {
    "stable": {
        "desired_class": "stable",
        "phase_preference": "stable_region",
        "min_stability": None,
    },
    "fragile": {
        "desired_class": "fragile",
        "phase_preference": "near_boundary",
        "min_stability": None,
    },
    "chaotic": {
        "desired_class": "chaotic",
        "phase_preference": "chaotic_transition",
        "min_stability": None,
    },
    "boundary_rider": {
        "desired_class": "boundary_rider",
        "phase_preference": "near_boundary",
        "min_stability": None,
    },
}


# ---------------------------------------------------------------------------
# Step 2 — Candidate space generation
# ---------------------------------------------------------------------------

def generate_hybrid_candidates(
    theta_grid: List[List[float]],
    sequences: List[Any],
) -> List[Dict[str, Any]]:
    """Generate all (theta, sequence) candidate pairs.

    Parameters
    ----------
    theta_grid : list[list[float]]
        UFF parameter vectors.
    sequences : list
        Deterministic sequence inputs.

    Returns
    -------
    list[dict]
        Ordered list of ``{"theta": ..., "sequence": ...}`` pairs.
        Order is preserved: theta outer, sequence inner.
    """
    candidates: List[Dict[str, Any]] = []
    for theta in theta_grid:
        for seq in sequences:
            candidates.append({"theta": list(theta), "sequence": seq})
    return candidates


# ---------------------------------------------------------------------------
# Step 3 — Scoring against target
# ---------------------------------------------------------------------------

def score_against_target(
    pair_summary: Dict[str, Any],
    target_spec: Dict[str, Any],
) -> float:
    """Score a hybrid pair summary against a target specification.

    Higher score is better.  Invalid pairs receive ``_WORST_SCORE``.

    Parameters
    ----------
    pair_summary : dict
        A single pair entry as produced by the hybrid co-design sweep,
        with keys: ``compatibility``, ``alignment``, ``theta_class``,
        ``sequence_class``, ``theta_phase``, ``sequence_phase``.
    target_spec : dict
        Output of :func:`build_target_spec`.

    Returns
    -------
    float
        Scalar score (higher is better).
    """
    # Invalid pairs get worst score
    if pair_summary.get("alignment") == "invalid":
        return _WORST_SCORE

    score = pair_summary.get("compatibility", 0.0)

    desired_class = target_spec["desired_class"]

    # Class match bonus — check both theta and sequence classes
    if pair_summary.get("theta_class") == desired_class:
        score += 1.0
    if pair_summary.get("sequence_class") == desired_class:
        score += 1.0

    # Phase preference bonus
    phase_pref = target_spec.get("phase_preference")
    if phase_pref is not None:
        if pair_summary.get("theta_phase") == phase_pref:
            score += 0.2
        if pair_summary.get("sequence_phase") == phase_pref:
            score += 0.2

    return score


# ---------------------------------------------------------------------------
# Step 4 — Main engine
# ---------------------------------------------------------------------------

def run_hybrid_inverse_design(
    target: str,
    theta_grid: List[List[float]],
    sequences: List[Any],
    *,
    v_circ_fn: Callable[..., np.ndarray] | None = None,
    pipeline_fn: Callable[..., Dict[str, Any]] | None = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run deterministic hybrid inverse design search.

    Scans all (theta, sequence) pairs from a bounded candidate space,
    evaluates each using the hybrid co-design engine, scores against
    the target behavior, and returns the top-k matches.

    Parameters
    ----------
    target : str
        Target behavior class.
    theta_grid : list[list[float]]
        UFF parameter vectors.
    sequences : list
        Deterministic sequence inputs.
    v_circ_fn : callable, optional
        Velocity curve generator for UFF experiments.
    pipeline_fn : callable, optional
        Pipeline function for sequence experiments.
    top_k : int, optional
        Number of best candidates to return (default ``5``).

    Returns
    -------
    dict
        Result with keys: ``target``, ``n_candidates``, ``top_k``,
        ``best_pair``, ``score_distribution``.

    Raises
    ------
    ValueError
        If *target* is not a recognized behavior class or
        *pipeline_fn* is not provided.
    """
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"Unknown target {target!r}; valid: {sorted(_VALID_TARGETS)}"
        )
    if pipeline_fn is None:
        raise ValueError("pipeline_fn is required")

    target_spec = build_target_spec(target)
    candidates = generate_hybrid_candidates(theta_grid, sequences)

    if not candidates:
        return {
            "target": target,
            "n_candidates": 0,
            "top_k": [],
            "best_pair": {},
            "score_distribution": [],
        }

    # Evaluate all pairs through hybrid co-design engine
    scored_entries: List[Dict[str, Any]] = []

    for cand in candidates:
        theta = cand["theta"]
        seq = cand["sequence"]

        # Run hybrid pair evaluation (reuses existing engine)
        raw = run_hybrid_pair(
            theta, seq,
            v_circ_fn=v_circ_fn, pipeline_fn=pipeline_fn,
        )
        summary = summarize_hybrid_pair(
            theta, raw["theta_result"],
            seq, raw["seq_result"],
        )

        # Validity check (same as hybrid_codesign)
        theta_consensus = summary["theta_summary"].get("consensus", False)
        theta_verified = summary["theta_summary"].get("verified", False)
        seq_consensus = summary["sequence_summary"].get("consensus", False)
        seq_verified = summary["sequence_summary"].get("verified", False)

        valid = (
            theta_consensus and theta_verified
            and seq_consensus and seq_verified
        )

        if valid:
            scored = score_hybrid_pair(summary)
        else:
            scored = {"compatibility": 0.0, "alignment": "invalid"}

        theta_phase = summary["theta_summary"].get("phase", "unknown")
        seq_phase = summary["sequence_summary"].get("phase", "unknown")
        theta_class = classify_sequence(summary["theta_summary"])
        seq_class = summary["sequence_summary"].get("class", "unknown")

        pair_info = {
            "theta": list(theta),
            "sequence": seq,
            "compatibility": scored["compatibility"],
            "alignment": scored["alignment"],
            "theta_phase": theta_phase,
            "sequence_phase": seq_phase,
            "theta_class": theta_class,
            "sequence_class": seq_class,
        }

        target_score = score_against_target(pair_info, target_spec)

        entry = {
            "theta": list(theta),
            "sequence": seq,
            "score": target_score,
            "compatibility": scored["compatibility"],
            "alignment": scored["alignment"],
            "class": theta_class,
            "phase": theta_phase,
        }
        scored_entries.append(entry)

    # Sort descending by score (higher is better), stable sort
    scored_entries.sort(key=lambda x: x["score"], reverse=True)

    top_results = scored_entries[:top_k]

    return {
        "target": target,
        "n_candidates": len(candidates),
        "top_k": top_results,
        "best_pair": top_results[0] if top_results else {},
        "score_distribution": [e["score"] for e in scored_entries],
    }
