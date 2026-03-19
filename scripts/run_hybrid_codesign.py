#!/usr/bin/env python
"""v82.7.0 — Hybrid Co-Design Engine demo script.

Evaluates a small grid of (theta, sequence) pairs and prints
the compatibility landscape.

Uses deterministic fake pipelines so the script runs without
external dependencies.
"""

from __future__ import annotations

import sys


# ---------------------------------------------------------------------------
# Deterministic fake pipelines
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
        "features": {
            "energy": 1.0,
            "spread": 0.5,
            "zcr": 0.2,
            "centroid": 3.0,
            "gradient_energy": 0.1,
            "curvature": 0.05,
        },
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


def _fake_uff(theta, *, v_circ_fn=None, **kwargs):
    """Deterministic UFF mock keyed on theta[0]."""
    s = float(theta[0]) * 0.1
    return _make_fake_result(stability_score=s, phase="stable_region")


def _fake_pipeline(seq: int) -> dict:
    """Deterministic pipeline keyed on integer index."""
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
    import qec.experiments.hybrid_codesign as hc

    # Monkeypatch UFF to use deterministic mock
    import qec.experiments.uff_bridge as _uff_mod
    _uff_mod.run_uff_experiment = _fake_uff  # type: ignore[attr-defined]

    print("=" * 60)
    print("Hybrid Co-Design Engine — v82.7.0")
    print("=" * 60)

    theta_grid = [
        [1.0, 2.0, 0.5],
        [2.0, 3.0, 1.0],
        [5.0, 1.0, 2.0],
    ]
    sequences = [0, 1, 2]

    result = hc.run_hybrid_codesign(
        theta_grid, sequences, pipeline_fn=_fake_pipeline,
    )

    print(f"\nPAIRS: {result['n_pairs']}")

    best = result["best_pair"]
    print(f"BEST THETA: {best.get('theta')}")
    print(f"BEST SEQUENCE: {best.get('sequence')}")
    print(f"BEST DISTANCE: {best.get('distance', 0):.4f}")
    print(f"BEST ALIGNMENT: {best.get('alignment')}")

    worst = result["worst_pair"]
    print(f"\nWORST THETA: {worst.get('theta')}")
    print(f"WORST SEQUENCE: {worst.get('sequence')}")
    print(f"WORST DISTANCE: {worst.get('distance', 0):.4f}")
    print(f"WORST ALIGNMENT: {worst.get('alignment')}")

    print(f"\nALIGNMENT_COUNTS: {result['alignment_counts']}")

    print("\n" + "-" * 60)
    print("All pairs:")
    for p in result["pairs"]:
        print(
            f"  theta={p['theta']}  seq={p['sequence']}  "
            f"dist={p['distance']:.3f}  compat={p['compatibility']:.3f}  "
            f"align={p['alignment']}"
        )

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
