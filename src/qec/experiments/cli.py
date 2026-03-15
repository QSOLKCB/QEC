"""CLI for deterministic experiment registry execution."""

from __future__ import annotations

import argparse
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
