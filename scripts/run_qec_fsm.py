#!/usr/bin/env python3
"""
v80.0.0 — QEC FSM Runner Script.

Loads sample input, runs the deterministic FSM controller, and prints
a structured summary.

Usage
-----
    python scripts/run_qec_fsm.py
"""

from __future__ import annotations

import json
import sys

from qec.controller.qec_fsm import QECFSM


def main() -> None:
    """Run the QEC FSM on sample input and print summary."""
    # Sample sonic-analysis-like input data.
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

    fsm = QECFSM(config=config)
    result = fsm.run(sample_input, max_steps=20)

    print("=== QEC FSM Result ===")
    print(json.dumps(result, indent=2, default=str))

    print(f"\nFinal state : {result['final_state']}")
    print(f"Total steps : {result['steps']}")
    print(f"History len : {len(result['history'])}")

    return None


if __name__ == "__main__":
    main()
