#!/usr/bin/env python
"""
v74.3.0 — Run dual-lattice visualiser on pre-computed sonic artifacts.

Usage:
    python scripts/run_sonic_visualizer.py [--sonic-dir artifacts/sonic] \
                                           [--output-dir artifacts/sonic_visualization]

Reads:
    - artifacts/sonic/*/analysis.json   (v74.0 per-file analyses)
    - artifacts/sonic/sequence_analysis.json  (v74.2 sequence analysis)

Writes:
    - artifacts/sonic_visualization/rubik_frame_*.png
    - artifacts/sonic_visualization/sierpinski_state.png
    - artifacts/sonic_visualization/combined_view.png
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-lattice invariant visualiser (v74.3.0)",
    )
    parser.add_argument(
        "--sonic-dir",
        default="artifacts/sonic",
        help="Root directory containing v74.0/v74.2 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/sonic_visualization",
        help="Output directory for visualisation PNGs.",
    )
    args = parser.parse_args()

    sonic_dir = args.sonic_dir
    output_dir = args.output_dir

    # Locate per-file analysis.json files.
    pattern = os.path.join(sonic_dir, "*", "analysis.json")
    analysis_paths = sorted(glob.glob(pattern))
    if not analysis_paths:
        print(f"No analysis.json files found in {pattern}", file=sys.stderr)
        sys.exit(1)

    # Locate sequence_analysis.json.
    seq_path = os.path.join(sonic_dir, "sequence_analysis.json")
    if not os.path.isfile(seq_path):
        print(f"sequence_analysis.json not found at {seq_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(analysis_paths)} analysis file(s).")
    print(f"Sequence analysis: {seq_path}")
    print(f"Output directory:  {output_dir}")

    from qec.experiments.sonic_visualizer import visualize

    outputs = visualize(analysis_paths, seq_path, output_dir=output_dir)

    print("\nGenerated artifacts:")
    for name, path in sorted(outputs.items()):
        print(f"  {name}: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
