#!/usr/bin/env python3
"""
v74.6.0 — Run invariant sonification on perturbation probe output.

Loads perturbation probe results, runs invariant analysis, then generates
deterministic audio signals and writes WAV output to artifacts/sonic/.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the project root is on the path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, os.path.join(_project_root, "src"))

from qec.experiments.invariant_engine import run_invariant_analysis
from qec.experiments.invariant_sonification import (
    generate_invariant_signal,
    generate_sequence_sound,
    write_wav,
)


def _default_probe() -> dict:
    """Return a representative probe dict for demonstration purposes."""
    return {
        "stable_ratio": 0.8,
        "boundary_crossings": 1,
        "mean_drift": {
            "energy": 0.002,
            "centroid": 0.05,
            "spread": 0.0001,
            "zcr": 0.3,
        },
        "most_sensitive": "zcr",
        "most_stable": "spread",
    }


def main() -> None:
    """Entry point for invariant sonification."""
    output_dir = os.path.join(_project_root, "artifacts", "sonic")

    # Check for existing probe output; fall back to demo probe.
    probe_path = os.path.join(output_dir, "perturbation_probe.json")
    if os.path.exists(probe_path):
        print(f"Loading probe from: {probe_path}")
        with open(probe_path, "r") as f:
            probe = json.load(f)
    else:
        print("No probe file found; using default demonstration probe.")
        probe = _default_probe()

    # Run invariant analysis.
    analysis = run_invariant_analysis(probe, output_dir=output_dir)

    # Inject mean_drift into analysis for sonification layer.
    analysis["mean_drift"] = probe.get("mean_drift", {})

    print()
    print("=== Invariant Analysis ===")
    print(f"  Stability score : {analysis['stability_score']:.4f}")
    print(f"  Phase           : {analysis['phase']}")
    print(f"  Strong invariants: {analysis['invariants']['strong_invariants']}")
    print(f"  Weak invariants  : {analysis['invariants']['weak_invariants']}")
    print(f"  Non-invariants   : {analysis['invariants']['non_invariants']}")
    print(f"  Most stable      : {analysis['most_stable']}")
    print(f"  Most sensitive   : {analysis['most_sensitive']}")

    # Generate single-frame sonification.
    print()
    print("Generating invariant sonification signal...")
    signal = generate_invariant_signal(analysis, duration=2.0, sr=44100)
    wav_path = os.path.join(output_dir, "invariant_sound.wav")
    write_wav(signal, wav_path, sr=44100)
    print(f"  Single-frame WAV: {wav_path}")
    print(f"  Samples: {len(signal)}")

    # Generate multi-frame sequence (3 variation frames).
    probes = [
        _default_probe(),
        {
            "stable_ratio": 0.5,
            "boundary_crossings": 2,
            "mean_drift": {"energy": 0.1, "centroid": 0.2, "spread": 0.05, "zcr": 0.5},
            "most_sensitive": "zcr",
            "most_stable": "energy",
        },
        {
            "stable_ratio": 0.95,
            "boundary_crossings": 0,
            "mean_drift": {"energy": 0.0001, "centroid": 0.0002, "spread": 0.00005, "zcr": 0.0001},
            "most_sensitive": "centroid",
            "most_stable": "spread",
        },
    ]

    analyses = []
    for p in probes:
        a = run_invariant_analysis(p)
        a["mean_drift"] = p.get("mean_drift", {})
        analyses.append(a)

    seq_signal = generate_sequence_sound(analyses, duration_per_frame=1.5, sr=44100)
    seq_path = os.path.join(output_dir, "invariant_sequence.wav")
    write_wav(seq_signal, seq_path, sr=44100)
    print(f"  Sequence WAV: {seq_path}")
    print(f"  Sequence samples: {len(seq_signal)}")

    print()
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
