#!/usr/bin/env python
"""v82.6.0 — Cross-Domain Inverse Design demo script.

Demonstrates both mapping directions:
  1. theta → best matching sequence
  2. sequence → best matching theta

Uses deterministic fake pipelines so the script runs without external
dependencies.  Replace the pipeline functions with real implementations
(e.g. ``run_midi_cube_experiment``, ``run_uff_experiment``) for
production use.
"""

from __future__ import annotations

import json
import sys


# ---------------------------------------------------------------------------
# Deterministic fake pipelines (same as test helpers)
# ---------------------------------------------------------------------------

def _make_fake_result(
    *,
    stability_score: float = 0.1,
    phase: str = "stable_region",
    classification: str = "convergent",
    consensus: bool = True,
    verified: bool = True,
    strong_invariants: list | None = None,
) -> dict:
    if strong_invariants is None:
        strong_invariants = ["energy", "centroid"]
    return {
        "probe": {
            "final_state": "ACCEPT",
            "steps": 5,
            "history": [{
                "from_state": "INVARIANT",
                "to_state": "EVALUATE",
                "stability_score": stability_score,
                "phase": phase,
                "epsilon": 1e-3,
                "reject_cycle": 0,
                "decision": "ACCEPT",
                "thresholds": None,
                "reason": None,
            }],
        },
        "invariants": {
            "history": [{
                "from_state": "INVARIANT",
                "to_state": "EVALUATE",
                "stability_score": stability_score,
                "phase": phase,
                "invariants": {
                    "strong_invariants": strong_invariants,
                    "weak_invariants": [],
                    "non_invariants": [],
                },
            }],
            "final_state": "ACCEPT",
        },
        "trajectory": {
            "n_points": 5,
            "mean_velocity": 0.01,
            "max_velocity": 0.05,
            "oscillation_score": 0.0,
            "convergence_rate": 0.001,
            "classification": classification,
        },
        "verification": {"match": True, "final_hash": "abc123", "steps": 5},
        "proof": {"payload": {}, "signature": "sig", "verified": verified},
        "consensus": {
            "n_agents": 3,
            "consensus": consensus,
            "agreement_ratio": 1.0,
            "consensus_hash": "abc123",
        },
    }


def _fake_pipeline(seq: int) -> dict:
    """Deterministic pipeline that varies by integer index."""
    if seq % 3 == 0:
        return _make_fake_result(
            stability_score=0.05, phase="stable_region",
            classification="convergent",
            strong_invariants=["energy", "centroid", "spread"],
        )
    elif seq % 3 == 1:
        return _make_fake_result(
            stability_score=0.8, phase="near_boundary",
            classification="oscillating",
            strong_invariants=["energy"],
        )
    else:
        return _make_fake_result(
            stability_score=2.5, phase="chaotic_transition",
            classification="divergent",
            strong_invariants=[],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from qec.experiments.cross_domain_mapper import (
        invariant_distance,
        map_theta_to_sequence,
        run_cross_domain_mapping,
    )

    print("=" * 60)
    print("Cross-Domain Inverse Design — v82.6.0")
    print("=" * 60)

    # --- Direction A: theta → sequence ---
    print("\nMODE: theta -> sequence")
    theta = [1.5, 2.0, 0.8]
    theta_result = {
        "stability_score": 0.1,
        "phase": "stable_region",
        "class": "stable",
        "trajectory_class": "convergent",
    }
    sequences = list(range(6))

    result_a = map_theta_to_sequence(
        theta, theta_result, sequences, _fake_pipeline,
    )

    print(f"TARGET THETA: {result_a['theta']}")
    print(f"BEST MATCH SEQUENCE: {result_a['best_sequence']}")
    print(f"DISTANCE: {result_a['distance']:.4f}")
    print(f"MATCH CLASS: {result_a['match_summary'].get('class', 'N/A')}")
    print(f"CONSENSUS: {result_a['match_summary'].get('consensus', 'N/A')}")
    print(f"VERIFIED: {result_a['match_summary'].get('verified', 'N/A')}")

    # --- Direction B: sequence → theta (via unified API) ---
    print("\n" + "-" * 60)
    print("\nMODE: sequence -> theta")

    # Build a small theta grid
    theta_grid = [
        [1.0, 2.0, 0.5],
        [2.0, 3.0, 1.0],
        [5.0, 1.0, 2.0],
    ]

    # Pre-computed sequence summary (chaotic)
    seq_result = {
        "stability_score": 2.5,
        "phase": "chaotic_transition",
        "class": "chaotic",
        "trajectory_class": "divergent",
    }

    # For this demo we use the unified API but monkeypatch the UFF landscape
    # to avoid needing real UFF infrastructure.
    from qec.experiments import cross_domain_mapper as cdm
    _original = cdm.run_uff_landscape

    def _fake_uff_landscape(V0, Rc, beta, *, v_circ_fn=None, output_dir=None):
        return {
            "n_points": 3,
            "best_theta": [1.0, 2.0, 0.5],
            "worst_theta": [5.0, 1.0, 2.0],
            "phase_counts": {"stable_region": 1, "near_boundary": 1, "chaotic_transition": 1},
            "points": [
                {"theta": [1.0, 2.0, 0.5], "stability_score": 0.1, "phase": "stable_region",
                 "most_stable": "energy", "most_sensitive": "zcr", "consensus": True, "verified": True},
                {"theta": [2.0, 3.0, 1.0], "stability_score": 1.2, "phase": "near_boundary",
                 "most_stable": "centroid", "most_sensitive": "spread", "consensus": True, "verified": True},
                {"theta": [5.0, 1.0, 2.0], "stability_score": 3.5, "phase": "chaotic_transition",
                 "most_stable": "energy", "most_sensitive": "zcr", "consensus": True, "verified": True},
            ],
        }

    cdm.run_uff_landscape = _fake_uff_landscape
    try:
        result_b = run_cross_domain_mapping(
            sequence="chaotic_seq",
            seq_result=seq_result,
            theta_grid=theta_grid,
        )
    finally:
        cdm.run_uff_landscape = _original

    print(f"TARGET SEQUENCE: chaotic_seq")
    print(f"BEST MATCH THETA: {result_b['best_theta']}")
    print(f"DISTANCE: {result_b['distance']:.4f}")
    print(f"MATCH CLASS: {result_b['match_summary'].get('phase', 'N/A')}")
    print(f"CONSENSUS: {result_b['match_summary'].get('consensus', 'N/A')}")
    print(f"VERIFIED: {result_b['match_summary'].get('verified', 'N/A')}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
