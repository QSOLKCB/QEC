"""Deterministic metrics & topology experiment probe (v98.7).

Observation-only harness that probes field metrics, multiscale metrics,
and strategy topology using fixed deterministic inputs.  Does not modify
any core system behaviour.

Includes calibration summary helpers and trajectory experiment generators
for threshold review.

Dependencies: numpy only (via analysis layer).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from qec.analysis.attractor_analysis import (
    analyze_attractors,
    analyze_trajectory,
)
from qec.analysis.field_metrics import compute_field_metrics
from qec.analysis.multiscale_metrics import compute_multiscale_summary
from qec.analysis.strategy_topology import compute_strategy_topology


# ---------------------------------------------------------------------------
# 1. Deterministic test inputs
# ---------------------------------------------------------------------------

def generate_test_inputs() -> List[Dict[str, Any]]:
    """Return deterministic test-case signals for metric probing.

    Covers stable, transitional, oscillatory, unstable, and mixed regimes
    with a broad set of fixed, human-readable patterns.
    """
    n = 12
    return [
        # --- original six ---
        {"name": "constant", "values": [1.0] * n},
        {"name": "linear_ramp", "values": [float(i) for i in range(n)]},
        {"name": "oscillating", "values": [1.0 if i % 2 == 0 else -1.0 for i in range(n)]},
        {"name": "noisy_deterministic", "values": [math.sin(float(i)) for i in range(n)]},
        {"name": "step_change", "values": [0.0] * (n // 2) + [5.0] * (n // 2)},
        {
            "name": "mixed_pattern",
            "values": [math.sin(float(i)) + 0.5 * (i % 3) for i in range(n)],
        },
        # --- expanded calibration patterns ---
        {
            "name": "low_amplitude_osc",
            "values": [0.1 * (1.0 if i % 2 == 0 else -1.0) for i in range(n)],
        },
        {
            "name": "high_amplitude_osc",
            "values": [5.0 * (1.0 if i % 2 == 0 else -1.0) for i in range(n)],
        },
        {
            "name": "slow_ramp",
            "values": [0.1 * i for i in range(n)],
        },
        {
            "name": "sharp_ramp",
            "values": [3.0 * i for i in range(n)],
        },
        {
            "name": "double_step",
            "values": [0.0] * 4 + [3.0] * 4 + [6.0] * 4,
        },
        {
            "name": "plateau_then_spike",
            "values": [1.0] * 10 + [10.0, 1.0],
        },
        {
            "name": "alternating_blocks",
            "values": [0.0] * 3 + [2.0] * 3 + [0.0] * 3 + [2.0] * 3,
        },
        {
            "name": "damped_oscillation",
            "values": [math.cos(float(i)) * math.exp(-0.3 * i) for i in range(n)],
        },
        {
            "name": "symmetric_bell",
            "values": [math.exp(-0.5 * (i - 5.5) ** 2) for i in range(n)],
        },
        {
            "name": "asymmetric_mixed",
            "values": [0.5 * i + 0.3 * math.sin(2.0 * i) for i in range(n)],
        },
    ]


# ---------------------------------------------------------------------------
# 2. Evaluate metrics
# ---------------------------------------------------------------------------

def evaluate_metrics(values: List[float]) -> Dict[str, Any]:
    """Compute field and multiscale metrics for *values*."""
    return {
        "field": compute_field_metrics(values),
        "multiscale": compute_multiscale_summary(values),
    }


# ---------------------------------------------------------------------------
# 3. Classify state
# ---------------------------------------------------------------------------

def classify_state(metrics: Dict[str, Any]) -> str:
    """Deterministic state classification from computed metrics."""
    field = metrics["field"]
    multi = metrics["multiscale"]

    phi = float(field["phi_alignment"])
    consistency = float(multi["scale_consistency"])
    divergence = float(multi["scale_divergence"])
    curvature_info = field["curvature"]
    abs_curv = float(curvature_info["abs_curvature"])
    nonlinear = float(field["nonlinear_response"])

    if phi > 0.8 and consistency > 0.8:
        return "stable"
    if divergence > 0.5:
        return "transitional"
    if abs_curv > 1.0 or nonlinear > 1.0:
        return "unstable"
    return "mixed"


# ---------------------------------------------------------------------------
# 4. Mock strategy generation
# ---------------------------------------------------------------------------

class _MockStrategy:
    """Lightweight strategy object with action_type and params."""

    def __init__(self, action_type: str, params: Dict[str, Any]) -> None:
        self.action_type = action_type
        self.params = dict(params)


def generate_mock_strategies() -> Dict[str, Any]:
    """Return a deterministic set of mock strategies for topology probing."""
    return {
        "s1": _MockStrategy("damping", {"alpha": 0.1, "beta": 0.5}),
        "s2": _MockStrategy("damping", {"alpha": 0.15, "beta": 0.5}),
        "s3": _MockStrategy("damping", {"alpha": 0.1, "beta": 0.9}),
        "s4": _MockStrategy("scaling", {"alpha": 0.1, "beta": 0.5}),
        "s5": _MockStrategy("scaling", {"alpha": 0.8, "beta": 0.2}),
        "s6": _MockStrategy("rotation", {"theta": 1.0, "phi": 0.0}),
    }


# ---------------------------------------------------------------------------
# 5. Topology analysis
# ---------------------------------------------------------------------------

def analyze_topology(strategies: Dict[str, Any]) -> Dict[str, Any]:
    """Compute strategy topology summary."""
    topo = compute_strategy_topology(strategies)
    # Compute average distance from the topology matrix
    distances = list(topo["topology"].values())
    avg_distance = float(np.mean(distances)) if distances else 0.0
    return {
        "clusters": topo["clusters"],
        "dominant": topo["dominant"],
        "average_distance": avg_distance,
    }


# ---------------------------------------------------------------------------
# 6. Full experiment runner
# ---------------------------------------------------------------------------

def run_experiments() -> Dict[str, Any]:
    """Run the full deterministic metrics & topology probe."""
    inputs = generate_test_inputs()

    input_results = []
    for case in inputs:
        metrics = evaluate_metrics(case["values"])
        classification = classify_state(metrics)
        attractor = analyze_attractors(metrics)
        input_results.append({
            "name": case["name"],
            "classification": classification,
            "metrics": metrics,
            "attractor": attractor,
        })

    strategies = generate_mock_strategies()
    topology = analyze_topology(strategies)

    return {
        "inputs": input_results,
        "topology": topology,
    }


# ---------------------------------------------------------------------------
# 7. Report printer
# ---------------------------------------------------------------------------

def summarize_experiment_patterns(
    results: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Compute grouped summaries by regime from experiment results.

    Returns
    -------
    dict
        Keys are regime names; values are dicts with count and averaged
        basin_score, summary_score, phi, divergence, curvature, resonance.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for entry in results["inputs"]:
        att = entry.get("attractor", {})
        regime = att.get("regime", "unknown")
        sig = att.get("signals", {})
        field = entry["metrics"]["field"]
        multi = entry["metrics"]["multiscale"]
        summary_score = (
            0.4 * float(field["phi_alignment"])
            + 0.3 * float(multi["scale_consistency"])
            - 0.3 * float(field["curvature"]["abs_curvature"])
        )
        if regime not in groups:
            groups[regime] = {
                "count": 0,
                "basin_scores": [],
                "summary_scores": [],
                "phis": [],
                "divergences": [],
                "curvatures": [],
                "resonances": [],
            }
        g = groups[regime]
        g["count"] += 1
        g["basin_scores"].append(att.get("basin_score", 0.0))
        g["summary_scores"].append(summary_score)
        g["phis"].append(sig.get("phi", 0.0))
        g["divergences"].append(sig.get("divergence", 0.0))
        g["curvatures"].append(sig.get("curvature", 0.0))
        g["resonances"].append(sig.get("resonance", 0.0))

    summary: Dict[str, Dict[str, Any]] = {}
    for regime, g in sorted(groups.items()):
        n = g["count"]
        summary[regime] = {
            "count": n,
            "avg_basin_score": sum(g["basin_scores"]) / n,
            "avg_summary_score": sum(g["summary_scores"]) / n,
            "avg_phi": sum(g["phis"]) / n,
            "avg_divergence": sum(g["divergences"]) / n,
            "avg_curvature": sum(g["curvatures"]) / n,
            "avg_resonance": sum(g["resonances"]) / n,
        }
    return summary


def print_experiment_report(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of experiment results."""
    print("=" * 60)
    print("Deterministic Metrics Probe — Experiment Report")
    print("=" * 60)

    for entry in results["inputs"]:
        m = entry["metrics"]
        field = m["field"]
        multi = m["multiscale"]
        curv_info = field["curvature"]
        att = entry.get("attractor", {})
        sig = att.get("signals", {})

        summary_score = (
            0.4 * float(field["phi_alignment"])
            + 0.3 * float(multi["scale_consistency"])
            - 0.3 * float(curv_info["abs_curvature"])
        )

        print(f"\n--- {entry['name']} ---")
        print(f"  classification:      {entry['classification']}")
        print(f"  regime:              {att.get('regime', 'n/a')}")
        print(f"  basin_score:         {att.get('basin_score', 0.0):.6f}")
        print(f"  summary_score:       {summary_score:.6f}")
        print(f"  phi_alignment:       {field['phi_alignment']:.6f}")
        print(f"  scale_consistency:   {multi['scale_consistency']:.6f}")
        print(f"  scale_divergence:    {multi['scale_divergence']:.6f}")
        print(f"  abs_curvature:       {curv_info['abs_curvature']:.6f}")
        print(f"  curvature_variation: {curv_info['curvature_variation']:.6f}")
        print(f"  resonance:           {field['resonance']:.6f}")
        print(f"  complexity:          {field['complexity']:.6f}")
        # compact summary line
        regime = att.get("regime", "n/a").upper()
        basin = att.get("basin_score", 0.0)
        phi_v = sig.get("phi", 0.0)
        cons_v = sig.get("consistency", 0.0)
        curv_v = sig.get("curvature", 0.0)
        print(
            f"  >> {regime} | basin={basin:.2f} | phi={phi_v:.2f}"
            f" | consistency={cons_v:.2f} | curvature={curv_v:.2f}"
        )

    topo = results["topology"]
    print("\n--- Strategy Topology ---")
    print(f"  clusters:          {topo['clusters']}")
    print(f"  dominant strategy: {topo['dominant']}")
    print(f"  average distance:  {topo['average_distance']:.6f}")

    # --- calibration summary ---
    pattern_summary = summarize_experiment_patterns(results)
    print("\n--- Calibration Summary by Regime ---")
    for regime, info in pattern_summary.items():
        print(f"\n  [{regime}] (n={info['count']})")
        print(f"    avg_basin_score:   {info['avg_basin_score']:.4f}")
        print(f"    avg_summary_score: {info['avg_summary_score']:.4f}")
        print(f"    avg_phi:           {info['avg_phi']:.4f}")
        print(f"    avg_divergence:    {info['avg_divergence']:.4f}")
        print(f"    avg_curvature:     {info['avg_curvature']:.4f}")
        print(f"    avg_resonance:     {info['avg_resonance']:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 8. Trajectory experiment sequences
# ---------------------------------------------------------------------------

def _make_probe_metrics(
    phi: float = 0.5,
    consistency: float = 0.5,
    divergence: float = 0.1,
    abs_curvature: float = 0.1,
    curvature_variation: float = 0.1,
    resonance: float = 0.1,
    complexity: float = 0.1,
) -> Dict[str, Any]:
    """Build a synthetic metrics dict for trajectory probing."""
    return {
        "field": {
            "phi_alignment": phi,
            "curvature": {
                "abs_curvature": abs_curvature,
                "curvature_variation": curvature_variation,
            },
            "resonance": resonance,
            "complexity": complexity,
        },
        "multiscale": {
            "scale_consistency": consistency,
            "scale_divergence": divergence,
        },
    }


def generate_metric_sequences() -> List[Dict[str, Any]]:
    """Return fixed metric sequences representing known trajectory types.

    Each entry has a 'name' and a 'sequence' of synthetic metrics dicts
    suitable for ``analyze_trajectory()``.
    """
    return [
        {
            "name": "stable_convergence",
            "sequence": [
                _make_probe_metrics(phi=0.85, consistency=0.85, abs_curvature=0.1),
                _make_probe_metrics(phi=0.88, consistency=0.87, abs_curvature=0.08),
                _make_probe_metrics(phi=0.90, consistency=0.90, abs_curvature=0.05),
                _make_probe_metrics(phi=0.92, consistency=0.92, abs_curvature=0.04),
            ],
        },
        {
            "name": "regime_transition",
            "sequence": [
                _make_probe_metrics(phi=0.9, consistency=0.9, abs_curvature=0.1),
                _make_probe_metrics(phi=0.6, consistency=0.6, divergence=0.5),
                _make_probe_metrics(phi=0.3, consistency=0.3, abs_curvature=0.6, complexity=0.7),
            ],
        },
        {
            "name": "oscillatory_switching",
            "sequence": [
                _make_probe_metrics(resonance=0.7, curvature_variation=0.4),
                _make_probe_metrics(phi=0.5, consistency=0.3, resonance=0.1),
                _make_probe_metrics(resonance=0.8, curvature_variation=0.5),
                _make_probe_metrics(phi=0.5, consistency=0.3, resonance=0.1),
            ],
        },
        {
            "name": "unstable_escalation",
            "sequence": [
                _make_probe_metrics(abs_curvature=0.3, complexity=0.3),
                _make_probe_metrics(abs_curvature=0.5, complexity=0.5),
                _make_probe_metrics(abs_curvature=0.8, complexity=0.7),
                _make_probe_metrics(abs_curvature=1.2, complexity=0.9),
            ],
        },
        {
            "name": "mixed_plateau",
            "sequence": [
                _make_probe_metrics(phi=0.5, consistency=0.3),
                _make_probe_metrics(phi=0.5, consistency=0.3),
                _make_probe_metrics(phi=0.5, consistency=0.3),
            ],
        },
    ]


def run_trajectory_experiments() -> List[Dict[str, Any]]:
    """Run trajectory analysis on all fixed metric sequences.

    Returns a list of dicts with name, trajectory result, and regimes.
    """
    sequences = generate_metric_sequences()
    results = []
    for seq in sequences:
        traj = analyze_trajectory(seq["sequence"])
        results.append({
            "name": seq["name"],
            "trajectory": traj,
        })
    return results
