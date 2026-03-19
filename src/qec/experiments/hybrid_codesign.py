"""v82.7.0 — Hybrid Co-Design Engine (theta + sequence together).

Paired evaluation of UFF parameter vectors and deterministic sequences
in a shared invariant space.  Runs both domains, fuses their invariant
summaries, and scores cross-domain compatibility.

This is co-evaluation, not co-optimization.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from qec.experiments.cross_domain_mapper import embed_summary, invariant_distance
from qec.experiments.sequence_landscape import (
    _extract_sequence_summary,
    classify_sequence,
)
from qec.experiments.uff_landscape import _extract_point_summary


# ---------------------------------------------------------------------------
# Alignment thresholds
# ---------------------------------------------------------------------------

_ALIGNED_THRESHOLD = 2.0
_NEAR_ALIGNED_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# Step 1 — Run both domains
# ---------------------------------------------------------------------------

def run_hybrid_pair(
    theta: List[float],
    sequence: Any,
    *,
    v_circ_fn: Callable[..., np.ndarray] | None = None,
    pipeline_fn: Callable[..., Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Run UFF and sequence experiments for a single (theta, sequence) pair.

    Parameters
    ----------
    theta : list[float]
        UFF parameter vector ``[V0, Rc, beta]``.
    sequence : Any
        Deterministic sequence input (e.g. MIDI path, integer index).
    v_circ_fn : callable, optional
        Velocity curve generator for UFF experiment.
    pipeline_fn : callable, optional
        Pipeline function for sequence experiment (e.g.
        ``run_midi_cube_experiment``).

    Returns
    -------
    dict
        ``{"theta_result": ..., "seq_result": ...}``

    Raises
    ------
    ValueError
        If *pipeline_fn* is not provided.
    """
    if pipeline_fn is None:
        raise ValueError("pipeline_fn is required")

    from qec.experiments.uff_bridge import run_uff_experiment

    theta_result = run_uff_experiment(theta, v_circ_fn=v_circ_fn)
    seq_result = pipeline_fn(sequence)

    return {"theta_result": theta_result, "seq_result": seq_result}


# ---------------------------------------------------------------------------
# Step 2 — Extract shared summaries
# ---------------------------------------------------------------------------

def summarize_hybrid_pair(
    theta: List[float],
    theta_result: Dict[str, Any],
    sequence: Any,
    seq_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce fused invariant summaries for a (theta, sequence) pair.

    Parameters
    ----------
    theta : list[float]
        UFF parameter vector.
    theta_result : dict
        Raw output of ``run_uff_experiment``.
    sequence : Any
        Sequence input.
    seq_result : dict
        Raw output of a sequence pipeline.

    Returns
    -------
    dict
        Fused summary with keys: ``theta_summary``, ``sequence_summary``,
        ``theta_embedding``, ``sequence_embedding``, ``distance``.
    """
    theta_summary = _extract_point_summary(theta, theta_result)
    seq_summary = _extract_sequence_summary(sequence, seq_result)
    seq_summary["class"] = classify_sequence(seq_summary)

    theta_emb = embed_summary(theta_summary)
    seq_emb = embed_summary(seq_summary)
    distance = invariant_distance(theta_summary, seq_summary)

    return {
        "theta_summary": theta_summary,
        "sequence_summary": seq_summary,
        "theta_embedding": theta_emb.tolist(),
        "sequence_embedding": seq_emb.tolist(),
        "distance": distance,
    }


# ---------------------------------------------------------------------------
# Step 3 — Compatibility score
# ---------------------------------------------------------------------------

def score_hybrid_pair(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Score a fused hybrid summary for cross-domain compatibility.

    Parameters
    ----------
    summary : dict
        Output of :func:`summarize_hybrid_pair`.

    Returns
    -------
    dict
        ``{"compatibility": float, "alignment": str}``
    """
    distance = float(summary["distance"])
    compatibility = 1.0 / (1.0 + distance)

    if distance <= _ALIGNED_THRESHOLD:
        alignment = "aligned"
    elif distance <= _NEAR_ALIGNED_THRESHOLD:
        alignment = "near_aligned"
    else:
        alignment = "mismatched"

    return {"compatibility": compatibility, "alignment": alignment}


# ---------------------------------------------------------------------------
# Step 4 — Co-design sweep
# ---------------------------------------------------------------------------

def run_hybrid_codesign(
    theta_grid: List[List[float]],
    sequences: List[Any],
    *,
    v_circ_fn: Callable[..., np.ndarray] | None = None,
    pipeline_fn: Callable[..., Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Evaluate all (theta, sequence) pairs and return a compatibility map.

    Parameters
    ----------
    theta_grid : list[list[float]]
        UFF parameter vectors.
    sequences : list
        Deterministic sequence inputs.
    v_circ_fn : callable, optional
        Velocity curve generator for UFF experiments.
    pipeline_fn : callable, optional
        Pipeline function for sequence experiments.

    Returns
    -------
    dict
        Landscape with keys: ``n_pairs``, ``best_pair``, ``worst_pair``,
        ``alignment_counts``, ``pairs``.
    """
    if pipeline_fn is None:
        raise ValueError("pipeline_fn is required")

    pairs: List[Dict[str, Any]] = []
    alignment_counts = {"aligned": 0, "near_aligned": 0, "mismatched": 0, "invalid": 0}

    best_pair: Optional[Dict[str, Any]] = None
    worst_pair: Optional[Dict[str, Any]] = None
    best_compat = -1.0
    worst_compat = 2.0

    for theta in theta_grid:
        for seq in sequences:
            raw = run_hybrid_pair(
                theta, seq,
                v_circ_fn=v_circ_fn, pipeline_fn=pipeline_fn,
            )
            summary = summarize_hybrid_pair(
                theta, raw["theta_result"],
                seq, raw["seq_result"],
            )

            # --- Validity check ---
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

            entry = {
                "theta": list(theta),
                "sequence": seq,
                "distance": summary["distance"],
                "compatibility": scored["compatibility"],
                "alignment": scored["alignment"],
                "theta_phase": summary["theta_summary"].get("phase", "unknown"),
                "sequence_phase": summary["sequence_summary"].get("phase", "unknown"),
                "theta_class": classify_sequence(summary["theta_summary"]),
                "sequence_class": summary["sequence_summary"].get("class", "unknown"),
            }
            pairs.append(entry)
            alignment_counts[scored["alignment"]] += 1

            if scored["compatibility"] > best_compat:
                best_compat = scored["compatibility"]
                best_pair = entry
            if scored["compatibility"] < worst_compat:
                worst_compat = scored["compatibility"]
                worst_pair = entry

    return {
        "n_pairs": len(pairs),
        "best_pair": best_pair if best_pair is not None else {},
        "worst_pair": worst_pair if worst_pair is not None else {},
        "alignment_counts": alignment_counts,
        "pairs": pairs,
    }
