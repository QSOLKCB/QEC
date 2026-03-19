"""v82.2.0 — Invariant Landscape Mapping.

Deterministic parameter sweep over UFF model parameters.
Evaluates many theta = [V0, Rc, beta] values via run_uff_experiment,
extracts compact invariant/stability summaries, and returns a
structured landscape map.

This is parameter-space cartography, not optimization.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from qec.experiments.invariant_engine import run_invariant_analysis
from qec.experiments.perturbation_probe import run_perturbation_probe
from qec.experiments.uff_bridge import build_sample, run_uff_experiment


# ---------------------------------------------------------------------------
# Step 1 — Deterministic Theta Grid
# ---------------------------------------------------------------------------

def generate_theta_grid(
    V0_values: List[float],
    Rc_values: List[float],
    beta_values: List[float],
) -> List[List[float]]:
    """Generate a deterministic parameter grid.

    Parameters
    ----------
    V0_values : list[float]
        Amplitude values to sweep.
    Rc_values : list[float]
        Characteristic radius values to sweep.
    beta_values : list[float]
        Shape parameter values to sweep.

    Returns
    -------
    list[list[float]]
        Ordered list of [V0, Rc, beta] parameter vectors.
        Order is preserved: V0 outer, Rc middle, beta inner.
    """
    grid: List[List[float]] = []
    for V0 in V0_values:
        for Rc in Rc_values:
            for beta in beta_values:
                grid.append([float(V0), float(Rc), float(beta)])
    return grid


# ---------------------------------------------------------------------------
# Step 2 — Extract compact summary from a single experiment result
# ---------------------------------------------------------------------------

def _extract_point_summary(
    theta: List[float],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract a compact invariant/stability summary from an experiment result.

    Re-runs the perturbation probe and invariant analysis on the extracted
    features to obtain a clean, single-pass stability assessment.

    Parameters
    ----------
    theta : list[float]
        The [V0, Rc, beta] parameter vector.
    result : dict
        Output of ``run_uff_experiment``.

    Returns
    -------
    dict
        Compact summary with keys: ``theta``, ``stability_score``,
        ``phase``, ``most_stable``, ``most_sensitive``, ``consensus``,
        ``verified``.
    """
    sample = build_sample(result["features"])
    probe = run_perturbation_probe(sample)
    analysis = run_invariant_analysis(probe)

    return {
        "theta": list(theta),
        "stability_score": analysis["stability_score"],
        "phase": analysis["phase"],
        "most_stable": analysis["most_stable"],
        "most_sensitive": analysis["most_sensitive"],
        "consensus": result["consensus"]["consensus"],
        "verified": result["proof"]["verified"],
    }


# ---------------------------------------------------------------------------
# Step 3 — Run landscape sweep
# ---------------------------------------------------------------------------

def run_uff_landscape(
    V0_values: List[float],
    Rc_values: List[float],
    beta_values: List[float],
    *,
    v_circ_fn: Callable[..., np.ndarray] | None = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a deterministic parameter sweep and return a landscape map.

    For each theta in the grid, runs ``run_uff_experiment`` and extracts
    a compact stability summary.  Aggregates phase counts and identifies
    the best (lowest stability score) and worst (highest) parameter vectors.

    Parameters
    ----------
    V0_values : list[float]
        Amplitude values to sweep.
    Rc_values : list[float]
        Characteristic radius values to sweep.
    beta_values : list[float]
        Shape parameter values to sweep.
    v_circ_fn : callable, optional
        Velocity curve generator passed to ``run_uff_experiment``.
    output_dir : str, optional
        If provided, writes ``uff_landscape.json`` to this directory.

    Returns
    -------
    dict
        Landscape map with keys: ``n_points``, ``best_theta``,
        ``worst_theta``, ``phase_counts``, ``points``.
    """
    V0_values = list(V0_values)
    Rc_values = list(Rc_values)
    beta_values = list(beta_values)

    grid = generate_theta_grid(V0_values, Rc_values, beta_values)

    if not grid:
        return {
            "n_points": 0,
            "best_theta": [],
            "worst_theta": [],
            "phase_counts": {},
            "points": [],
        }

    points: List[Dict[str, Any]] = []
    for theta in grid:
        result = run_uff_experiment(theta, v_circ_fn=v_circ_fn)
        summary = _extract_point_summary(theta, result)
        points.append(summary)

    # --- Aggregate landscape metrics ---
    phase_counts: Dict[str, int] = {}
    for p in points:
        phase = p["phase"]
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    best = min(points, key=lambda p: p["stability_score"])
    worst = max(points, key=lambda p: p["stability_score"])

    landscape: Dict[str, Any] = {
        "n_points": len(points),
        "best_theta": best["theta"],
        "worst_theta": worst["theta"],
        "phase_counts": phase_counts,
        "points": points,
    }

    # --- Optional JSON output ---
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "uff_landscape.json")
        with open(out_path, "w") as f:
            json.dump(landscape, f, indent=2, sort_keys=True)

    return landscape
