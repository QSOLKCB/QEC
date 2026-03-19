#!/usr/bin/env python3
"""v82.2.0 — Invariant Landscape Mapping Runner Script.

Runs a small deterministic parameter sweep and prints a summary
of the landscape map.

Usage
-----
    python scripts/run_uff_landscape.py
"""

from __future__ import annotations

from qec.experiments.uff_landscape import run_uff_landscape


def main() -> None:
    """Run a tiny landscape sweep and print summary."""
    V0_values = [100.0, 150.0]
    Rc_values = [1.0, 2.0]
    beta_values = [1.0, 2.0]

    landscape = run_uff_landscape(
        V0_values,
        Rc_values,
        beta_values,
    )

    print("=== UFF Invariant Landscape ===")
    print(f"POINTS      : {landscape['n_points']}")
    print(f"BEST THETA  : {landscape['best_theta']}")
    print(f"WORST THETA : {landscape['worst_theta']}")
    print(f"PHASE COUNTS: {landscape['phase_counts']}")


if __name__ == "__main__":
    main()
