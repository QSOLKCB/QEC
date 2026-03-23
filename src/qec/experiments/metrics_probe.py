"""Deterministic metrics & topology experiment probe (v98.6).

Observation-only harness that probes field metrics, multiscale metrics,
and strategy topology using fixed deterministic inputs.  Does not modify
any core system behaviour.

Dependencies: numpy only (via analysis layer).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from qec.analysis.field_metrics import compute_field_metrics
from qec.analysis.multiscale_metrics import compute_multiscale_summary
from qec.analysis.strategy_topology import compute_strategy_topology


# ---------------------------------------------------------------------------
# 1. Deterministic test inputs
# ---------------------------------------------------------------------------

def generate_test_inputs() -> List[Dict[str, Any]]:
    """Return deterministic test-case signals for metric probing."""
    n = 12
    return [
        {"name": "constant", "values": [1.0] * n},
        {"name": "linear_ramp", "values": [float(i) for i in range(n)]},
        {"name": "oscillating", "values": [1.0 if i % 2 == 0 else -1.0 for i in range(n)]},
        {"name": "noisy_deterministic", "values": [math.sin(float(i)) for i in range(n)]},
        {"name": "step_change", "values": [0.0] * (n // 2) + [5.0] * (n // 2)},
        {
            "name": "mixed_pattern",
            "values": [math.sin(float(i)) + 0.5 * (i % 3) for i in range(n)],
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
        input_results.append({
            "name": case["name"],
            "classification": classification,
            "metrics": metrics,
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

def print_experiment_report(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of experiment results."""
    print("=" * 60)
    print("Deterministic Metrics Probe — Experiment Report")
    print("=" * 60)

    for entry in results["inputs"]:
        m = entry["metrics"]
        field = m["field"]
        multi = m["multiscale"]
        print(f"\n--- {entry['name']} ---")
        print(f"  classification:    {entry['classification']}")
        print(f"  phi_alignment:     {field['phi_alignment']:.6f}")
        print(f"  scale_consistency: {multi['scale_consistency']:.6f}")
        print(f"  scale_divergence:  {multi['scale_divergence']:.6f}")
        print(f"  curvature:         {field['curvature']['abs_curvature']:.6f}")
        print(f"  complexity:        {field['complexity']:.6f}")

    topo = results["topology"]
    print("\n--- Strategy Topology ---")
    print(f"  clusters:          {topo['clusters']}")
    print(f"  dominant strategy: {topo['dominant']}")
    print(f"  average distance:  {topo['average_distance']:.6f}")
    print("=" * 60)
