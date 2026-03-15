"""qec-exp CLI entrypoint."""

from __future__ import annotations

import argparse
import json

from src.qec.discovery.threshold_search import run_threshold_search


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qec-exp", description="QEC deterministic experiment CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search-threshold", help="Run deterministic Tanner-graph threshold search")
    search.add_argument("--iterations", type=int, default=20)
    search.add_argument("--population", type=int, default=6)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--max-graphs", type=int, default=200)
    search.add_argument("--max-phase-diagram-size", type=int, default=6)
    search.add_argument("--artifacts-root", default="experiments/threshold_search")
    search.add_argument("--num-variables", type=int, default=24)
    search.add_argument("--num-checks", type=int, default=12)
    search.add_argument("--variable-degree", type=int, default=3)
    search.add_argument("--check-degree", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "search-threshold":
        print("Starting Tanner graph search")
        spec = {
            "num_variables": args.num_variables,
            "num_checks": args.num_checks,
            "variable_degree": args.variable_degree,
            "check_degree": args.check_degree,
        }
        result = run_threshold_search(
            spec,
            iterations=args.iterations,
            population=args.population,
            seed=args.seed,
            max_graphs_evaluated=args.max_graphs,
            max_phase_diagram_size=args.max_phase_diagram_size,
            artifacts_root=args.artifacts_root,
        )
        for item in result["history"]:
            print(f"iteration {item['iteration']}")
            print(f"candidate threshold = {item['threshold']:.3f}")

        best = result.get("best")
        if best is not None:
            print(f"best threshold = {best['threshold']:.3f}")
        print(json.dumps({"artifact_dir": result["artifact_dir"]}, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
