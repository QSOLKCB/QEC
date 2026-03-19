#!/usr/bin/env python3
"""
v80.5.0 — Trajectory Observer Runner Script.

Runs the QEC FSM, feeds history into the trajectory observer,
prints a summary, and optionally writes JSON output.

Usage
-----
    python scripts/run_trajectory_observer.py
"""

from __future__ import annotations

import json
import sys

from qec.controller.qec_fsm import QECFSM
from qec.controller.trajectory_observer import analyze_trajectory


def main() -> None:
    """Run FSM then analyze the trajectory."""
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

    # Step 1: Run FSM.
    fsm = QECFSM(config=config)
    result = fsm.run(sample_input, max_steps=20)

    print("=== QEC FSM Result ===")
    print(f"Final state : {result['final_state']}")
    print(f"Total steps : {result['steps']}")
    print(f"History len : {len(result['history'])}")

    # Step 2: Analyze trajectory.
    analysis = analyze_trajectory(
        result["history"],
        output_dir="output",
    )

    print("\n=== Trajectory Analysis ===")
    print(json.dumps(analysis, indent=2, sort_keys=True))

    print(f"\nClassification: {analysis['classification']}")
    print(f"Points        : {analysis['n_points']}")
    print(f"Mean velocity : {analysis['mean_velocity']:.6f}")
    print(f"Max velocity  : {analysis['max_velocity']:.6f}")
    print(f"Oscillation   : {analysis['oscillation_score']:.4f}")
    print(f"Convergence   : {analysis['convergence_rate']:.6f}")

    return None


if __name__ == "__main__":
    main()
