"""
CLI entrypoint for the benchmarking framework.

Usage::

    python -m src.bench --config path/to/config.json --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import BenchmarkConfig
from .runner import run_benchmark
from .schema import dumps_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.bench",
        description="Run a QEC benchmark sweep.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a JSON benchmark config file.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Path for the JSON result file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON output (non-canonical but readable).",
    )
    parser.add_argument(
        "--stability-phase-diagram", action="store_true",
        help="Run the v8.0 stability phase diagram experiment.",
    )
    parser.add_argument(
        "--grid-resolution", type=int, default=20,
        help="Grid resolution for phase diagram (default: 20).",
    )
    parser.add_argument(
        "--perturbations-per-cell", type=int, default=10,
        help="Perturbations per grid cell (default: 10).",
    )

    args = parser.parse_args(argv)

    if args.stability_phase_diagram:
        return _run_phase_diagram(args)

    config = BenchmarkConfig.load(args.config)
    result = run_benchmark(config)

    if args.pretty:
        text = json.dumps(result, sort_keys=True, indent=2)
    else:
        text = dumps_result(result)

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
            f.write("\n")
        print(f"Results written to {args.out}", file=sys.stderr)
    else:
        print(text)

    return 0


def _run_phase_diagram(args) -> int:
    """Run the stability phase diagram experiment."""
    import numpy as np

    from qec.experiments.stability_phase_diagram import (
        run_stability_phase_diagram_experiment,
        serialize_phase_diagram_artifact,
    )

    config = BenchmarkConfig.load(args.config)

    # Build H from config (use first code if available)
    from qec_qldpc_codes import create_code
    code = create_code(
        config.code_name if hasattr(config, "code_name") else "steane",
        config.lifting_size if hasattr(config, "lifting_size") else 7,
        seed=0,
    )
    H = code.H_X if hasattr(code, "H_X") else np.array([[1, 1, 0], [0, 1, 1]])

    result = run_stability_phase_diagram_experiment(
        H,
        grid_resolution=args.grid_resolution,
        perturbations_per_cell=args.perturbations_per_cell,
    )

    out_path = args.out or "artifacts/stability_phase_diagram.json"
    serialize_phase_diagram_artifact(result, out_path)

    # Print ASCII diagram
    print(result["ascii_phase_diagram"], file=sys.stderr)
    print(f"Results written to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
