#!/usr/bin/env python3
"""Run the deterministic benchmark + stress framework.

Usage:
    python scripts/run_benchmark_stress.py [--n-vars N] [--n-iters N] [--output PATH]
"""

import argparse
import sys

from src.qec.experiments.benchmark_stress import (
    results_to_json,
    run_benchmark_stress,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark stress framework for compute_bp_dynamics_metrics"
    )
    parser.add_argument("--n-vars", type=int, default=50, help="Number of LLR variables")
    parser.add_argument("--n-iters", type=int, default=30, help="Base iterations per scenario")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    results = run_benchmark_stress(
        n_vars=args.n_vars,
        n_iters=args.n_iters,
    )

    json_str = results_to_json(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
            f.write("\n")
        print(f"Results written to {args.output}")
    else:
        print(json_str)

    # Summary
    print(f"\n--- Summary ---", file=sys.stderr)
    print(f"Scenarios: {results['n_scenarios']}", file=sys.stderr)
    for s in results["scenarios"]:
        print(
            f"  {s['scenario']:25s} -> {s['regime']:25s} "
            f"(cosine={s['fidelity']['cosine']:.4f})",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
