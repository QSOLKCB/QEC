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
import itertools
import json
from pathlib import Path
from typing import Any

from .experiment_hash import ExperimentHash, ExperimentRunner
from .registry import DEFAULT_REGISTRY, discover_experiments


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_base_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    config["experiment_name"] = args.experiment
    return config


def _run_one(config: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    experiment_name = str(config["experiment_name"])
    experiment_fn = DEFAULT_REGISTRY.get(experiment_name)
    experiment_hash = ExperimentHash.compute(config)
    artifact_dir = Path("experiments") / experiment_hash

    print(f"Running experiment: {experiment_name}")
    print(f"hash: {experiment_hash}")
    print(f"artifact: {artifact_dir.as_posix()}/")

    if dry_run:
        return {"experiment_hash": experiment_hash, "artifact_dir": str(artifact_dir)}

    runner = ExperimentRunner(artifacts_root="experiments")
    results = runner.run(config, experiment_fn)
    return {"experiment_hash": experiment_hash, "artifact_dir": str(artifact_dir), "results": results}


def _run_sweep(base_config: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
    sweep = base_config.get("sweep", {})
    if not isinstance(sweep, dict) or not sweep:
        raise ValueError("sweep config must provide a non-empty 'sweep' dictionary")

    keys = sorted(sweep.keys())
    value_lists = [list(sweep[k]) for k in keys]

    outputs: list[dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        config = dict(base_config)
        config.pop("sweep", None)
        for key, value in zip(keys, values):
            config[key] = value
        outputs.append(_run_one(config=config, dry_run=dry_run))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qec-exp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("experiment")
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--config")
    run_parser.add_argument("--dry-run", action="store_true")

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("experiment")
    sweep_parser.add_argument("--seed", type=int)
    sweep_parser.add_argument("--config")
    sweep_parser.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    discover_experiments()
    args = build_parser().parse_args(argv)

    if args.command == "list":
        print("Available experiments:")
        print()
        for name in DEFAULT_REGISTRY.list():
            print(name)
        return 0

    config = _build_base_config(args)
    if args.command == "run":
        _run_one(config=config, dry_run=bool(args.dry_run))
        return 0

    if args.command == "sweep":
        _run_sweep(base_config=config, dry_run=bool(args.dry_run))
        return 0

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
