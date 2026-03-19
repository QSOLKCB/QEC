#!/usr/bin/env python3
"""
v82.0.0 — UFF-to-QEC Bridge Runner Script.

Runs a UFF velocity curve through the full QEC pipeline and prints
a summary of the unified experiment artifact.

Usage
-----
    python scripts/run_uff_bridge.py
"""

from __future__ import annotations

from qec.experiments.uff_bridge import run_uff_experiment


def main() -> None:
    """Run UFF→QEC bridge and print summary."""
    theta = [220.0, 5.0, 2.0]

    result = run_uff_experiment(theta)

    print("=== UFF → QEC Bridge ===")
    print(f"THETA       : {result['theta']}")
    print(f"ENERGY      : {result['features']['energy']:.6f}")
    print(f"PHASE       : {result['invariants']['final_state']}")
    print(f"INVARIANTS  : {result['invariants']['final_state']}")
    print(f"CONSENSUS   : {result['consensus']['consensus']}")
    print(f"HASH        : {result['verification']['final_hash']}")
    print(f"VERIFIED    : {result['proof']['verified']}")


if __name__ == "__main__":
    main()
