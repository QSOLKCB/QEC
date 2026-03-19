#!/usr/bin/env python
"""v82.8.0 — Hybrid Inverse Design Engine demo script.

Scans a small (theta, sequence) candidate space for pairs that best
match each target behavior class.

Uses deterministic fake pipelines so the script runs without
external dependencies.
"""

from __future__ import annotations


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
    from qec.experiments.hybrid_inverse_design import run_hybrid_inverse_design

    # Monkeypatch UFF to use deterministic mock
    import qec.experiments.uff_bridge as _uff_mod
    _uff_mod.run_uff_experiment = _fake_uff  # type: ignore[attr-defined]

    print("=" * 60)
    print("Hybrid Inverse Design Engine — v82.8.0")
    print("=" * 60)

    theta_grid = [
        [1.0, 2.0, 0.5],
        [2.0, 3.0, 1.0],
        [5.0, 1.0, 2.0],
    ]
    sequences = [0, 1, 2]

    for target in ("stable", "fragile", "chaotic", "boundary_rider"):
        result = run_hybrid_inverse_design(
            target,
            theta_grid,
            sequences,
            pipeline_fn=_fake_pipeline,
            top_k=3,
        )

        best = result["best_pair"]
        print(f"\nTARGET: {target}")
        print(f"  CANDIDATES: {result['n_candidates']}")
        print(f"  BEST SCORE: {best.get('score', 'N/A'):.4f}")
        print(f"  BEST THETA: {best.get('theta')}")
        print(f"  BEST SEQUENCE: {best.get('sequence')}")
        print(f"  BEST ALIGNMENT: {best.get('alignment')}")
        print(f"  TOP {len(result['top_k'])}:")
        for i, entry in enumerate(result["top_k"]):
            print(
                f"    [{i}] theta={entry['theta']}  seq={entry['sequence']}  "
                f"score={entry['score']:.4f}  align={entry['alignment']}"
            )

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
