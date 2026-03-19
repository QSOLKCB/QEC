#!/usr/bin/env python3
"""v82.4.0 — Sequence Intelligence Engine Runner Script.

Runs a small deterministic sequence landscape sweep using synthetic
cube sequences and prints a summary.

Usage
-----
    python scripts/run_sequence_landscape.py
"""

from __future__ import annotations

import numpy as np

from qec.experiments.midi_cube_bridge import (
    build_sample,
    extract_features,
    init_cube,
)
from qec.controller.execution_proof import create_execution_proof
from qec.controller.multi_agent_verifier import verify_multi_agent
from qec.controller.qec_fsm import QECFSM
from qec.controller.replay_engine import verify_run
from qec.controller.trajectory_observer import analyze_trajectory
from qec.experiments.sequence_landscape import run_sequence_landscape


# ---------------------------------------------------------------------------
# Demo keys (same as midi_cube_bridge)
# ---------------------------------------------------------------------------

_DEMO_PRIVATE_KEY_PEM = (
    b"-----BEGIN PRIVATE KEY-----\n"
    b"MC4CAQAwBQYDK2VwBCIEIO4Nngc2zhyTpxaDALLMVmUQ6OOjMk0eOgLjGnLLY2nN\n"
    b"-----END PRIVATE KEY-----\n"
)

_DEMO_PUBLIC_KEY_PEM = (
    b"-----BEGIN PUBLIC KEY-----\n"
    b"MCowBQYDK2VwAyEAk5fIB0Cvc5fb2v0wizvCJjQFro2sald9OS1eyUO1soM=\n"
    b"-----END PUBLIC KEY-----\n"
)

_DEFAULT_CONFIG = {
    "stability_threshold": 0.5,
    "boundary_crossing_threshold": 2,
    "max_reject_cycles": 3,
    "epsilon": 1e-3,
    "n_perturbations": 9,
    "drift_threshold": 1e-4,
}


# ---------------------------------------------------------------------------
# Synthetic cube pipeline (no MIDI file needed)
# ---------------------------------------------------------------------------

def _synthetic_cube_pipeline(seed: int) -> dict:
    """Run QEC pipeline on a deterministic synthetic cube.

    Parameters
    ----------
    seed : int
        Seed that determines the cube pattern.

    Returns
    -------
    dict
        Full pipeline result compatible with sequence_landscape.
    """
    cube = init_cube()
    rng = np.random.RandomState(seed)
    # Fill deterministic blocks
    for _ in range(8):
        x = int(rng.randint(0, 5))
        y = int(rng.randint(0, 5))
        z = int(rng.randint(0, 5))
        s = int(rng.randint(2, 4))
        color = rng.randint(0, 256, size=3).astype(np.uint8)
        cube[x:x + s, y:y + s, z:z + s] = color

    features = extract_features(cube)
    sample_input = build_sample(features)

    config = dict(_DEFAULT_CONFIG)
    fsm = QECFSM(config=dict(config))
    fsm_result = fsm.run(sample_input, max_steps=20)
    history = fsm_result["history"]

    trajectory = analyze_trajectory(history)
    verification = verify_run(sample_input, history, config, max_steps=20)
    proof = create_execution_proof(
        verify_result=verification,
        signer_id="sequence-landscape",
        private_key_pem=_DEMO_PRIVATE_KEY_PEM,
        public_key_pem=_DEMO_PUBLIC_KEY_PEM,
        metadata={"seed": seed},
    )
    consensus = verify_multi_agent(
        initial_input=sample_input,
        history=history,
        config=config,
        proof=proof,
    )

    return {
        "events": 8,
        "features": features,
        "probe": fsm_result,
        "invariants": {
            "history": history,
            "final_state": fsm_result["final_state"],
        },
        "trajectory": trajectory,
        "verification": verification,
        "proof": proof,
        "consensus": consensus,
    }


def main() -> None:
    """Run a small sequence landscape and print summary."""
    sequences = list(range(8))

    landscape = run_sequence_landscape(
        sequences,
        pipeline_fn=_synthetic_cube_pipeline,
    )

    print("=== Sequence Intelligence Engine ===")
    print(f"SEQUENCES   : {landscape['n_sequences']}")
    print(f"PHASE_COUNTS: {landscape['phase_counts']}")
    print(f"CLASS_COUNTS: {landscape['class_counts']}")
    print(f"BEST        : {landscape['best_sequences']}")
    print(f"WORST       : {landscape['worst_sequences']}")

    # Check consensus across all points
    all_consensus = all(p["consensus"] for p in landscape["points"])
    print(f"CONSENSUS   : {all_consensus}")


if __name__ == "__main__":
    main()
