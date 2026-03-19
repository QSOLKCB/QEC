"""
v74.5.0 — Invariant Engine (Stability, Lyapunov Proxy, Phase Classification).

Interprets perturbation probe results to derive:
- stability scores (Lyapunov-like proxy)
- invariant candidates (conserved quantities)
- phase region classification
- feature robustness ranking

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRIFT_FEATURES = ("energy", "centroid", "spread", "zcr")

_BOUNDARY_CROSSING_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Step 1 — Stability Score (Lyapunov Proxy)
# ---------------------------------------------------------------------------

def compute_stability_score(probe: dict) -> float:
    """Compute a Lyapunov-like stability score from perturbation probe output.

    Lower values indicate more stability; higher values indicate instability.
    The score is in [0, +inf).

    Parameters
    ----------
    probe : dict
        Output from ``run_perturbation_probe``.  Not mutated.

    Returns
    -------
    float
        Non-negative stability score.
    """
    stable_ratio = probe["stable_ratio"]
    boundary_crossings = probe["boundary_crossings"]
    mean_drift = probe["mean_drift"]

    base = 1.0 - stable_ratio
    crossing_penalty = _BOUNDARY_CROSSING_WEIGHT * boundary_crossings
    drift_sum = sum(mean_drift.get(k, 0.0) for k in _DRIFT_FEATURES)

    score = base + crossing_penalty + drift_sum
    return float(max(score, 0.0))


# ---------------------------------------------------------------------------
# Step 2 — Invariant Candidates
# ---------------------------------------------------------------------------

def identify_invariants(
    probe: dict,
    drift_threshold: float = 1e-4,
) -> dict:
    """Classify features as strong, weak, or non-invariants based on drift.

    Parameters
    ----------
    probe : dict
        Output from ``run_perturbation_probe``.  Not mutated.
    drift_threshold : float
        Drift below this value → strong invariant.
        Drift below 10× this value → weak invariant.
        Everything else → non-invariant.

    Returns
    -------
    dict
        Keys: ``strong_invariants``, ``weak_invariants``, ``non_invariants``.
        Each maps to a sorted list of feature names.
    """
    mean_drift = probe["mean_drift"]
    weak_limit = drift_threshold * 10.0

    strong: List[str] = []
    weak: List[str] = []
    non: List[str] = []

    for feature in _DRIFT_FEATURES:
        drift = abs(mean_drift.get(feature, 0.0))
        if drift < drift_threshold:
            strong.append(feature)
        elif drift < weak_limit:
            weak.append(feature)
        else:
            non.append(feature)

    return {
        "strong_invariants": sorted(strong),
        "weak_invariants": sorted(weak),
        "non_invariants": sorted(non),
    }


# ---------------------------------------------------------------------------
# Step 3 — Phase Classification
# ---------------------------------------------------------------------------

_CHAOTIC_CROSSING_THRESHOLD = 3
_CHAOTIC_DRIFT_THRESHOLD = 1.0


def classify_phase(probe: dict) -> str:
    """Classify the phase region from perturbation probe output.

    Parameters
    ----------
    probe : dict
        Output from ``run_perturbation_probe``.  Not mutated.

    Returns
    -------
    str
        One of ``"stable_region"``, ``"near_boundary"``,
        ``"unstable_region"``, ``"chaotic_transition"``.
    """
    stable_ratio = probe["stable_ratio"]
    boundary_crossings = probe["boundary_crossings"]
    mean_drift = probe["mean_drift"]

    drift_sum = sum(mean_drift.get(k, 0.0) for k in _DRIFT_FEATURES)

    # Check chaotic first (highest priority).
    if (boundary_crossings >= _CHAOTIC_CROSSING_THRESHOLD
            or drift_sum >= _CHAOTIC_DRIFT_THRESHOLD):
        return "chaotic_transition"

    if stable_ratio > 0.9 and boundary_crossings == 0:
        return "stable_region"

    if stable_ratio > 0.7 and boundary_crossings > 0:
        return "near_boundary"

    if stable_ratio < 0.5 and boundary_crossings > 0:
        return "unstable_region"

    # Fallback: use drift-based heuristic for intermediate cases.
    if boundary_crossings > 0:
        return "near_boundary"

    return "stable_region"


# ---------------------------------------------------------------------------
# Step 4 — Sensitivity Ranking
# ---------------------------------------------------------------------------

def rank_features(probe: dict) -> List[Tuple[str, float]]:
    """Rank features by drift magnitude (ascending = most stable first).

    Parameters
    ----------
    probe : dict
        Output from ``run_perturbation_probe``.  Not mutated.

    Returns
    -------
    list[tuple[str, float]]
        Features sorted by drift ascending.
    """
    mean_drift = probe["mean_drift"]
    ranked = [(k, float(mean_drift.get(k, 0.0))) for k in _DRIFT_FEATURES]
    ranked.sort(key=lambda pair: pair[1])
    return ranked


# ---------------------------------------------------------------------------
# Step 5 — Main Entry Point
# ---------------------------------------------------------------------------

def run_invariant_analysis(
    probe: dict,
    *,
    drift_threshold: float = 1e-4,
    output_dir: Optional[str] = None,
) -> dict:
    """Run full invariant analysis on perturbation probe output.

    Parameters
    ----------
    probe : dict
        Output from ``run_perturbation_probe``.  Not mutated.
    drift_threshold : float
        Threshold for invariant classification (default ``1e-4``).
    output_dir : str, optional
        If provided, writes ``invariant_analysis.json`` to this directory.

    Returns
    -------
    dict
        Analysis summary with keys: ``stability_score``, ``phase``,
        ``invariants``, ``feature_ranking``, ``most_stable``,
        ``most_sensitive``.
    """
    # Deep-copy to guarantee no mutation of caller data.
    probe = copy.deepcopy(probe)

    stability_score = compute_stability_score(probe)
    phase = classify_phase(probe)
    invariants = identify_invariants(probe, drift_threshold=drift_threshold)
    feature_ranking = rank_features(probe)

    result = {
        "stability_score": stability_score,
        "phase": phase,
        "invariants": invariants,
        "feature_ranking": feature_ranking,
        "most_stable": probe.get("most_stable", ""),
        "most_sensitive": probe.get("most_sensitive", ""),
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "invariant_analysis.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)

    return result
