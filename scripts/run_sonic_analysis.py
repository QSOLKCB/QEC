#!/usr/bin/env python3
"""
v74.0.0 — Run batch sonic artifact analysis on repository MP3 files.

Loads the three fixed MP3 inputs from the repository root,
runs deterministic spectral analysis, and writes results to
artifacts/sonic/.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the project root is on the path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, os.path.join(_project_root, "src"))

from qec.experiments.sonic_batch_analysis import run_sonic_batch_analysis

# Fixed deterministic inputs
_AUDIO_FILES = [
    os.path.join(_project_root, "QEC Fault Lines Sonification.mp3"),
    os.path.join(_project_root, "QSOL Triplet Polymeter - Producer Bounce.mp3"),
    os.path.join(
        _project_root,
        "Spectral Algebraics Live \u2013 Quantum Nostalgia Ambient.mp3",
    ),
]


def main() -> None:
    """Entry point for batch sonic analysis."""
    output_root = os.path.join(_project_root, "artifacts", "sonic")

    # Validate inputs exist
    missing = [p for p in _AUDIO_FILES if not os.path.exists(p)]
    if missing:
        print("WARNING: Missing audio files:")
        for m in missing:
            print(f"  - {m}")

    available = [p for p in _AUDIO_FILES if os.path.exists(p)]
    if not available:
        print("ERROR: No audio files found.  Nothing to analyse.")
        sys.exit(1)

    print(f"Analysing {len(available)} audio file(s)...")
    summary = run_sonic_batch_analysis(available, output_root=output_root)

    print()
    print("=== Batch Summary ===")
    print(f"  Files analysed : {summary['n_files']}")
    print(f"  Mean duration  : {summary['mean_duration']:.2f} s")
    print(f"  Mean RMS energy: {summary['mean_energy']:.6f}")
    print(f"  Mean centroid  : {summary['mean_centroid']:.2f} Hz")
    print(f"  Var centroid   : {summary['variance_centroid']:.2f}")
    print()
    print(f"Artifacts written to: {output_root}")
    print(f"Batch summary: {os.path.join(output_root, 'batch_summary.json')}")


if __name__ == "__main__":
    main()
