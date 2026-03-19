#!/usr/bin/env python3
"""
v81.0.0 — Replay Engine Runner Script.

Runs the QEC FSM, replays the execution, and verifies deterministic
reproducibility via hash chain verification.

Usage
-----
    python scripts/run_replay_engine.py
"""

from __future__ import annotations

import json

from qec.controller.qec_fsm import QECFSM
from qec.controller.replay_engine import (
    build_hash_chain,
    compare_histories,
    replay_fsm,
    verify_run,
)


def main() -> None:
    """Run FSM, replay, and print verification summary."""
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

    # --- Original run ---
    fsm = QECFSM(config=dict(config))
    original = fsm.run(sample_input, max_steps=20)

    print("=== Original FSM Run ===")
    print(f"Final state : {original['final_state']}")
    print(f"Steps       : {original['steps']}")
    print(f"History len : {len(original['history'])}")

    # --- Build hash chain for original ---
    chain = build_hash_chain(original["history"])
    print(f"\n=== Hash Chain (original) ===")
    print(f"Final hash  : {chain['final_hash']}")
    print(f"Chain len   : {len(chain['step_hashes'])}")

    # --- Replay and verify ---
    result = verify_run(
        sample_input, original["history"], config, max_steps=20,
    )

    print(f"\n=== Replay Verification ===")
    print(f"MATCH       : {result['match']}")
    print(f"FINAL HASH  : {result['final_hash']}")
    print(f"STEPS       : {result['steps']}")
    if result["mismatch_index"] is not None:
        print(f"MISMATCH AT : step {result['mismatch_index']}")

    return None


if __name__ == "__main__":
    main()
