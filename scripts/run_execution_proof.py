#!/usr/bin/env python3
"""
v81.1.0 — Signed Execution Proof Runner Script.

Runs the QEC FSM, replays for verification, then constructs and signs
a cryptographic proof of execution using Ed25519.

Usage
-----
    python scripts/run_execution_proof.py
"""

from __future__ import annotations

from qec.controller.qec_fsm import QECFSM
from qec.controller.replay_engine import verify_run
from qec.controller.execution_proof import create_execution_proof

# ---------------------------------------------------------------------------
# Demo Ed25519 key pair (pre-generated, for demonstration only)
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


def main() -> None:
    """Run FSM, verify replay, sign execution proof, print summary."""
    sample_input = {
        "rms_energy": 0.012,
        "spectral_centroid_hz": 480.0,
        "spectral_spread_hz": 210.0,
        "zero_crossing_rate": 0.048,
        "fft_top_peaks": [
            {"frequency_hz": 100.0, "magnitude": 0.5},
            {"frequency_hz": 200.0, "magnitude": 0.3},
        ],
    }

    config = {
        "stability_threshold": 0.5,
        "boundary_crossing_threshold": 2,
        "max_reject_cycles": 3,
        "epsilon": 1e-3,
        "n_perturbations": 9,
        "drift_threshold": 1e-4,
    }

    # --- Original FSM run ---
    fsm = QECFSM(config=dict(config))
    original = fsm.run(sample_input, max_steps=20)

    print("=== FSM Run ===")
    print(f"Final state : {original['final_state']}")
    print(f"Steps       : {original['steps']}")

    # --- Replay verification ---
    verify_result = verify_run(
        sample_input, original["history"], config, max_steps=20,
    )

    print(f"\n=== Replay Verification ===")
    print(f"MATCH       : {verify_result['match']}")
    print(f"FINAL HASH  : {verify_result['final_hash']}")
    print(f"STEPS       : {verify_result['steps']}")

    # --- Signed execution proof ---
    proof = create_execution_proof(
        verify_result=verify_result,
        signer_id="demo-runner",
        private_key_pem=_DEMO_PRIVATE_KEY_PEM,
        public_key_pem=_DEMO_PUBLIC_KEY_PEM,
        metadata={"version": "81.1.0"},
    )

    print(f"\n=== Execution Proof ===")
    print(f"MATCH       : {proof['payload']['match']}")
    print(f"FINAL HASH  : {proof['payload']['final_hash']}")
    print(f"SIGNED      : True")
    print(f"VERIFIED    : {proof['verified']}")
    print(f"SIGNATURE   : {proof['signature'][:32]}...")


if __name__ == "__main__":
    main()
