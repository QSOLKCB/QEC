#!/usr/bin/env python3
"""v82.5.0 — Inverse Design Engine Runner Script.

Runs a small bounded deterministic search for sequences that produce
a target behavior class.

Usage
-----
    python scripts/run_inverse_design.py
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
from qec.experiments.inverse_design import run_inverse_design


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
# Synthetic cube pipeline (note-sequence → cube → QEC FSM)
# ---------------------------------------------------------------------------

def _note_sequence_pipeline(seq: list) -> dict:
    """Run QEC pipeline on a deterministic cube derived from a note sequence.

    Maps note events to cube voxel writes, then runs the full FSM +
    verification + consensus pipeline.
    """
    cube = init_cube()

    # Deterministic cube fill from note events
    for i, event in enumerate(seq):
        note = event["note"]
        vel = event["velocity"]
        # Map note/velocity/index to cube coordinates deterministically
        x = note % 6
        y = (vel + i) % 6
        z = (note + i * 7) % 6
        s = 2
        r = (note * 3) % 256
        g = (vel * 2) % 256
        b = ((note + vel) * 5) % 256
        cube[x:x + s, y:y + s, z:z + s] = np.array(
            [r, g, b], dtype=np.uint8,
        )

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
        signer_id="inverse-design",
        private_key_pem=_DEMO_PRIVATE_KEY_PEM,
        public_key_pem=_DEMO_PUBLIC_KEY_PEM,
        metadata={"n_events": len(seq)},
    )
    consensus = verify_multi_agent(
        initial_input=sample_input,
        history=history,
        config=config,
        proof=proof,
    )

    return {
        "events": len(seq),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run inverse design search and print results."""
    notes = [60, 62, 64, 67]
    lengths = [2, 3]
    target = "stable"

    result = run_inverse_design(
        target=target,
        notes=notes,
        lengths=lengths,
        pipeline_fn=_note_sequence_pipeline,
        top_k=5,
    )

    print("=== Inverse Design Engine ===")
    print(f"TARGET     : {result['target']}")
    print(f"CANDIDATES : {result['n_candidates']}")

    if result["best"]:
        best = result["best"][0]
        print(f"BEST SCORE : {best['score']}")
        print(f"BEST SEQ   : {best['sequence']}")
        print(f"BEST CLASS : {best['summary'].get('sequence_class', 'unknown')}")
        print(f"CONSENSUS  : {best['summary'].get('consensus', False)}")
        print(f"VERIFIED   : {best['summary'].get('verified', False)}")
    else:
        print("NO CANDIDATES FOUND")


if __name__ == "__main__":
    main()
