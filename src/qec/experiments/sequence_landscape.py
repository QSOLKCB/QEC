"""v82.4.0 — Sequence Intelligence Engine.

Unified landscape mapping for sequence-driven inputs (MIDI/Cube/DNA)
and parameter-driven inputs (UFF theta vectors).

Maps: sequence → invariant behavior, producing the same summary format
as the UFF landscape so both systems share a common analysis space.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Step 1 — Extract summary from a single sequence result
# ---------------------------------------------------------------------------

def _extract_sequence_summary(
    seq: Any,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract a compact invariant/stability summary from pipeline output.

    Parameters
    ----------
    seq : Any
        The input sequence identifier (path, index, or descriptor).
    result : dict
        Output of a pipeline function (e.g. ``run_midi_cube_experiment``).
        Not mutated.

    Returns
    -------
    dict
        Summary with keys: ``input``, ``input_type``, ``stability_score``,
        ``phase``, ``invariant_strength``, ``consensus``, ``verified``,
        ``trajectory_class``.
    """
    # --- Stability score from probe/invariants ---
    probe = result.get("probe", {})
    history = probe.get("history", [])
    # Find the last step with a stability_score
    stability_score = 0.0
    phase = "unknown"
    for entry in reversed(history):
        if entry.get("stability_score") is not None:
            stability_score = float(entry["stability_score"])
            break
    for entry in reversed(history):
        if entry.get("phase") is not None:
            phase = str(entry["phase"])
            break

    # --- Invariant strength: count of strong invariants ---
    invariants_data = result.get("invariants", {})
    inv_history = invariants_data.get("history", [])
    # Look through invariant history for invariant analysis entries
    strong_count = 0
    for entry in reversed(inv_history):
        inv_info = entry.get("invariants")
        if isinstance(inv_info, dict):
            strong_count = len(inv_info.get("strong_invariants", []))
            break

    # --- Consensus and verification ---
    consensus_data = result.get("consensus", {})
    consensus = bool(consensus_data.get("consensus", False))

    proof_data = result.get("proof", {})
    verified = bool(proof_data.get("verified", False))

    # --- Trajectory class ---
    trajectory_data = result.get("trajectory", {})
    trajectory_class = str(trajectory_data.get("classification", "unknown"))

    return {
        "input": seq,
        "input_type": "sequence",
        "stability_score": stability_score,
        "phase": phase,
        "invariant_strength": strong_count,
        "consensus": consensus,
        "verified": verified,
        "trajectory_class": trajectory_class,
    }


# ---------------------------------------------------------------------------
# Step 2 — Classify sequence behavior
# ---------------------------------------------------------------------------

def classify_sequence(summary: Dict[str, Any]) -> str:
    """Classify a sequence based on its invariant summary.

    Classification rules (evaluated in priority order):

    - **chaotic**: stability_score >= 2.0 or phase is ``chaotic_transition``
    - **boundary_rider**: phase is ``near_boundary`` or ``unstable_region``
    - **fragile**: stability_score >= 0.5 and invariant_strength < 2
    - **stable**: default

    Parameters
    ----------
    summary : dict
        Output of ``_extract_sequence_summary``.  Not mutated.

    Returns
    -------
    str
        One of ``"stable"``, ``"fragile"``, ``"chaotic"``,
        ``"boundary_rider"``.
    """
    score = summary.get("stability_score", 0.0)
    phase = summary.get("phase", "")
    strength = summary.get("invariant_strength", 0)

    if score >= 2.0 or phase == "chaotic_transition":
        return "chaotic"

    if phase in ("near_boundary", "unstable_region"):
        return "boundary_rider"

    if score >= 0.5 and strength < 2:
        return "fragile"

    return "stable"


# ---------------------------------------------------------------------------
# Step 3 — Run sequence landscape
# ---------------------------------------------------------------------------

def run_sequence_landscape(
    sequences: List[Any],
    *,
    pipeline_fn: Callable[..., Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a deterministic landscape sweep over sequences.

    For each sequence, calls ``pipeline_fn(seq)`` and extracts a compact
    invariant/stability summary.  Aggregates phase counts and class counts,
    and identifies best (lowest stability_score) and worst (highest)
    sequences.

    Parameters
    ----------
    sequences : list
        Deterministic sequence inputs.  Each element is passed as the
        first positional argument to *pipeline_fn*.
    pipeline_fn : callable
        Pipeline function, e.g. ``run_midi_cube_experiment``.
        Must accept a single positional argument and return a dict
        with keys ``probe``, ``invariants``, ``trajectory``,
        ``consensus``, ``proof``.
    output_dir : str, optional
        If provided, writes ``sequence_landscape.json`` to this directory.

    Returns
    -------
    dict
        Landscape map with keys: ``n_sequences``, ``phase_counts``,
        ``class_counts``, ``best_sequences``, ``worst_sequences``,
        ``points``.
    """
    if not sequences:
        return {
            "n_sequences": 0,
            "phase_counts": {},
            "class_counts": {},
            "best_sequences": [],
            "worst_sequences": [],
            "points": [],
        }

    points: List[Dict[str, Any]] = []
    for seq in sequences:
        result = pipeline_fn(seq)
        summary = _extract_sequence_summary(seq, result)
        summary["class"] = classify_sequence(summary)
        points.append(summary)

    # --- Aggregate phase counts ---
    phase_counts: Dict[str, int] = {}
    for p in points:
        phase = p["phase"]
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    # --- Aggregate class counts ---
    class_counts: Dict[str, int] = {}
    for p in points:
        cls = p["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # --- Best and worst ---
    sorted_points = sorted(points, key=lambda p: p["stability_score"])
    best = sorted_points[:1]
    worst = sorted_points[-1:]

    landscape: Dict[str, Any] = {
        "n_sequences": len(points),
        "phase_counts": phase_counts,
        "class_counts": class_counts,
        "best_sequences": [p["input"] for p in best],
        "worst_sequences": [p["input"] for p in worst],
        "points": points,
    }

    # --- Optional JSON output ---
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "sequence_landscape.json")
        with open(out_path, "w") as f:
            json.dump(landscape, f, indent=2, sort_keys=True)

    return landscape
