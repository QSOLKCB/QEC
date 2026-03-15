#!/usr/bin/env python
"""qec-exp CLI."""

from __future__ import annotations

import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.discovery.threshold_search import SpectralSearchConfig, run_spectral_threshold_search
from src.qec.generation.deterministic_construction import construct_deterministic_tanner_graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qec-exp")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("spectral-search", help="Run deterministic spectral threshold search")
    search.add_argument("--iterations", type=int, default=10)
    search.add_argument("--population", type=int, default=3)
    search.add_argument("--max-phase-diagram-size", type=int, default=4)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--output-dir", type=str, default="experiments/threshold_search")
    search.add_argument("--enable-bp-diagnostics", action="store_true")
    search.add_argument("--min-predicted-threshold", type=float, default=0.0)
    search.add_argument("--rank-by-prediction", action="store_true")
    search.add_argument("--max-bp-candidates", type=int, default=5)

    args = parser.parse_args(argv)
    if args.command == "spectral-search":
        spec = {
            "num_variables": 12,
            "num_checks": 6,
            "variable_degree": 3,
            "check_degree": 6,
        }
        H0 = construct_deterministic_tanner_graph(spec)
        cfg = SpectralSearchConfig(
            iterations=args.iterations,
            population=args.population,
            max_phase_diagram_size=args.max_phase_diagram_size,
            seed=args.seed,
            output_dir=args.output_dir,
            enable_bp_diagnostics=bool(args.enable_bp_diagnostics),
            min_predicted_threshold=float(args.min_predicted_threshold),
            rank_by_prediction=bool(args.rank_by_prediction),
            max_bp_candidates=int(args.max_bp_candidates),
        )
        print("Starting spectral threshold search")
        result = run_spectral_threshold_search(H0, config=cfg)
        for item in result["history"]:
            print(f"iteration {item['iteration'] + 1}")
            print(f"threshold = {item['threshold']}")
            metrics = item.get("spectral_metrics") or {}
            if "bp_iterations" in metrics:
                print(f"bp iterations = {metrics['bp_iterations']}")
                print(f"final residual = {metrics['bp_final_residual']}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
