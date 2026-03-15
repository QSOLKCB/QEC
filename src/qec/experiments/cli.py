"""Command-line interface for deterministic experiment orchestration."""

from __future__ import annotations

import argparse
from typing import Any

from .phase_diagram_orchestrator import (
    PhaseDiagramOrchestrator,
    render_ascii_heatmap,
)


def _run_bp_threshold(config: dict[str, Any]) -> dict[str, Any]:
    """Deterministic baseline BP-threshold proxy metrics."""
    p = float(config.get("parameters", {}).get("physical_error_rate", 0.0))
    iters = float(config.get("parameters", {}).get("decoder_iterations", 0.0))

    spectral_radius = round(0.78 + 6.5 * p - 0.003 * iters, 12)
    mean_llr = round((1.0 - p) * (1.0 + iters / 100.0), 12)
    decoder_success_rate = max(0.0, min(1.0, round(1.0 - 3.5 * p + 0.002 * iters, 12)))

    return {
        "bp_converged": spectral_radius < 1.0,
        "spectral_radius": spectral_radius,
        "mean_llr": mean_llr,
        "decoder_success_rate": decoder_success_rate,
    }


EXPERIMENT_REGISTRY = {
    "bp-threshold": _run_bp_threshold,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qec-exp", description="QEC experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase = subparsers.add_parser("phase-diagram", help="Run deterministic phase-diagram sweep")
    phase.add_argument("experiment", help="Experiment name (e.g. bp-threshold)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "phase-diagram":
        orchestrator = PhaseDiagramOrchestrator(experiment_registry=EXPERIMENT_REGISTRY)
        config = {
            "experiment": args.experiment,
            "x_axis": {
                "name": "physical_error_rate",
                "values": [0.01, 0.02, 0.03, 0.04],
            },
            "y_axis": {
                "name": "decoder_iterations",
                "values": [10, 20, 30, 40],
            },
        }
        print(f"Running phase diagram for {args.experiment}")
        print("Grid size: 4 × 4")

        result = orchestrator.run(config)
        total = len(result["statuses"])
        for index, status in enumerate(result["statuses"], start=1):
            print(f"[{index}/{total}] {status}")

        payload = result["phase_diagram"]
        heatmap = payload["heatmap"]

        print("Phase diagram complete")
        print(f"artifact: {result['artifact']}")
        print()
        print(render_ascii_heatmap(payload, heatmap))


if __name__ == "__main__":
    main()
