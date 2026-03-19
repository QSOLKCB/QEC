"""
v74.4.0 — Perturbation Probe Layer (Invariant Sensitivity Engine).

Applies small deterministic perturbations to sonic analysis result dicts,
re-runs comparison and classification, and measures how outputs change.

Enables:
- invariant detection (features that do not affect classification)
- boundary sensitivity analysis (perturbations that flip classification)
- stability scoring (fraction of perturbations preserving classification)

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional

from qec.experiments.sonic_comparison import (
    classify_comparison,
    compare_sonic_features,
)


# ---------------------------------------------------------------------------
# Feature keys eligible for perturbation
# ---------------------------------------------------------------------------

_PERTURBABLE_KEYS = ("rms_energy", "spectral_centroid_hz",
                     "spectral_spread_hz", "zero_crossing_rate")

# Short aliases used in output dicts
_KEY_ALIASES = {
    "rms_energy": "energy",
    "spectral_centroid_hz": "centroid",
    "spectral_spread_hz": "spread",
    "zero_crossing_rate": "zcr",
}


# ---------------------------------------------------------------------------
# Step 1 — Perturbation Generator
# ---------------------------------------------------------------------------

def generate_perturbations(
    result: dict,
    epsilon: float,
    n: int,
) -> List[dict]:
    """Generate *n* deterministic perturbations of *result*.

    Each perturbation modifies numeric fields (energy, centroid, spread, zcr)
    by ``epsilon * ((i % 3) - 1)`` where *i* is the perturbation index.

    Parameters
    ----------
    result : dict
        A sonic analysis result dict.  Not mutated.
    epsilon : float
        Perturbation magnitude.
    n : int
        Number of perturbations to generate.  Must be >= 0.

    Returns
    -------
    list[dict]
        Deep-copied and perturbed result dicts.
    """
    perturbations: List[dict] = []
    for i in range(n):
        p = copy.deepcopy(result)
        delta = epsilon * ((i % 3) - 1)
        for key in _PERTURBABLE_KEYS:
            if key in p and isinstance(p[key], (int, float)):
                p[key] = float(p[key]) + delta
        perturbations.append(p)
    return perturbations


# ---------------------------------------------------------------------------
# Step 2 — Re-run comparison and classification on perturbed samples
# ---------------------------------------------------------------------------

def _compare_and_classify(original: dict, perturbed: dict) -> dict:
    """Compare *original* to *perturbed* and classify the comparison.

    Returns a dict with comparison metrics and the classification label.
    """
    metrics = compare_sonic_features(original, perturbed)
    label = classify_comparison(metrics)
    return {"metrics": metrics, "classification": label}


# ---------------------------------------------------------------------------
# Step 3 — Compute Stability Metrics
# ---------------------------------------------------------------------------

def compute_perturbation_metrics(
    original_classification: str,
    perturbed_results: List[dict],
) -> dict:
    """Compute stability metrics from perturbation probe results.

    Parameters
    ----------
    original_classification : str
        Classification label for the unperturbed state (e.g. ``"stable"``).
    perturbed_results : list[dict]
        Each entry has ``"metrics"`` and ``"classification"`` keys
        (as returned by ``_compare_and_classify``).

    Returns
    -------
    dict
        Stability summary with keys: ``stable_ratio``,
        ``boundary_crossings``, ``mean_drift``, ``most_sensitive``,
        ``most_stable``.
    """
    n = len(perturbed_results)
    if n == 0:
        return {
            "stable_ratio": 1.0,
            "boundary_crossings": 0,
            "mean_drift": {alias: 0.0 for alias in _KEY_ALIASES.values()},
            "most_sensitive": "none",
            "most_stable": "none",
        }

    # Classification stability ------------------------------------------
    same_count = sum(
        1 for r in perturbed_results
        if r["classification"] == original_classification
    )
    stable_ratio = same_count / n

    # Boundary crossings ------------------------------------------------
    boundary_crossings = n - same_count

    # Metric drift ------------------------------------------------------
    drift_key_map = {
        "energy_delta": "energy",
        "centroid_delta": "centroid",
        "spread_delta": "spread",
        "zcr_delta": "zcr",
    }
    drift_accum: Dict[str, float] = {alias: 0.0 for alias in drift_key_map.values()}
    for r in perturbed_results:
        m = r["metrics"]
        for metric_key, alias in drift_key_map.items():
            drift_accum[alias] += abs(m[metric_key])

    mean_drift = {alias: drift_accum[alias] / n for alias in drift_accum}

    # Sensitivity ranking -----------------------------------------------
    most_sensitive = max(mean_drift, key=mean_drift.get)  # type: ignore[arg-type]
    most_stable = min(mean_drift, key=mean_drift.get)  # type: ignore[arg-type]

    return {
        "stable_ratio": float(stable_ratio),
        "boundary_crossings": int(boundary_crossings),
        "mean_drift": {k: float(v) for k, v in mean_drift.items()},
        "most_sensitive": most_sensitive,
        "most_stable": most_stable,
    }


# ---------------------------------------------------------------------------
# Step 4 — Main Entry Point
# ---------------------------------------------------------------------------

def run_perturbation_probe(
    result: dict,
    *,
    epsilon: float = 1e-3,
    n: int = 9,
    reference: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Run a full perturbation probe on a sonic analysis result.

    Parameters
    ----------
    result : dict
        Sonic analysis result dict.  Not mutated.
    epsilon : float
        Perturbation magnitude (default ``1e-3``).
    n : int
        Number of perturbations (default ``9``).
    reference : dict, optional
        A second result dict to compare against.  If ``None``, the original
        *result* is used as both baseline and reference for self-comparison.
    output_dir : str, optional
        If provided, writes ``perturbation_summary.json`` to this directory.

    Returns
    -------
    dict
        Probe summary with keys: ``stable_ratio``, ``boundary_crossings``,
        ``mean_drift``, ``most_sensitive``, ``most_stable``.
    """
    # Deep-copy to guarantee no mutation of caller data.
    original = copy.deepcopy(result)
    ref = copy.deepcopy(reference) if reference is not None else original

    # Classify the unperturbed comparison.
    baseline_metrics = compare_sonic_features(ref, original)
    original_classification = classify_comparison(baseline_metrics)

    # Generate perturbations and run comparison pipeline.
    perturbations = generate_perturbations(result, epsilon, n)
    perturbed_results = [
        _compare_and_classify(ref, p) for p in perturbations
    ]

    # Compute stability metrics.
    summary = compute_perturbation_metrics(
        original_classification, perturbed_results,
    )

    # Step 5 — Optional JSON output.
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "perturbation_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    return summary
