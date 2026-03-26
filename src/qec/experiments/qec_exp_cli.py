"""qec-exp CLI for deterministic experiment orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bp_threshold_estimator import BPThresholdEstimator
from .experiment_hash import ExperimentHash, ExperimentRunner


_DEFAULT_PHASE_DIAGRAM = {
    "experiment": "bp-threshold",
    "x_axis": "physical_error_rate",
    "x_values": [0.02, 0.03, 0.04, 0.05],
    "y_axis": "decoder_iterations",
    "grid": [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, -1],
    ],
}


def _experiment_config(experiment: str) -> dict[str, Any]:
    return {"experiment": experiment}


def _ensure_bp_phase_diagram(experiment: str, artifacts_root: str) -> Path:
    if experiment != "bp-threshold":
        raise ValueError(f"Unknown experiment: {experiment}")

    config = _experiment_config(experiment)
    runner = ExperimentRunner(artifacts_root=artifacts_root)
    runner.run(config, lambda spec: {"experiment": spec["experiment"], "status": "ok"})

    exp_hash = ExperimentHash.compute(config)
    experiment_dir = Path(artifacts_root) / exp_hash
    experiment_dir.mkdir(parents=True, exist_ok=True)
    phase_diagram_path = experiment_dir / "phase_diagram.json"
    if not phase_diagram_path.exists():
        serialized = json.dumps(_DEFAULT_PHASE_DIAGRAM, sort_keys=True, indent=2)
        phase_diagram_path.write_text(f"{serialized}\n", encoding="utf-8")
    return experiment_dir


def _cmd_phase_diagram(args: argparse.Namespace) -> int:
    experiment_dir = _ensure_bp_phase_diagram(args.experiment, args.artifacts_root)
    print(f"artifact: {experiment_dir / 'phase_diagram.json'}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    return _cmd_phase_diagram(args)


def _cmd_estimate_threshold(args: argparse.Namespace) -> int:
    print("Loading phase diagram")
    experiment_dir = _ensure_bp_phase_diagram(args.experiment, args.artifacts_root)
    phase_diagram_path = experiment_dir / "phase_diagram.json"

    estimator = BPThresholdEstimator(smooth=args.smooth)
    phase_diagram = estimator.load_phase_diagram(phase_diagram_path)
    success_rates = estimator.compute_success_rates(phase_diagram)

    print("Estimating BP threshold...")
    estimate = estimator.estimate_threshold(success_rates)

    artifact_path = experiment_dir / "threshold_estimate.json"
    estimator.write_threshold_artifact(
        artifact_path,
        experiment=args.experiment,
        success_rates=success_rates,
        threshold=estimate,
    )

    threshold_value = estimate.get("threshold")
    if threshold_value is None:
        print("Estimated threshold: unavailable")
    else:
        print(f"Estimated threshold: {float(threshold_value):.6f}")
    print(f"artifact: {artifact_path}")
    print()
    print(estimator.build_ascii_report(success_rates, estimate))
    return 0


def _cmd_policy_memory(args: argparse.Namespace) -> int:
    from qec.analysis.policy_memory import (
        format_policy_memory_summary,
        import_policy_memory,
        init_policy_memory,
    )

    memory = init_policy_memory()
    if args.memory_file is not None:
        memory_path = Path(args.memory_file)
        if memory_path.exists():
            data = json.loads(memory_path.read_text(encoding="utf-8"))
            memory = import_policy_memory(data)

    if args.show_policy_memory:
        print(format_policy_memory_summary(memory))
        return 0

    if getattr(args, "show_policy_clusters", False) or getattr(
        args, "show_archetypes", False
    ):
        from qec.analysis.policy_clustering import (
            cluster_policies,
            extract_policy_archetypes,
            format_policy_clusters_summary,
            rank_archetypes,
        )
        from qec.analysis.policy import Policy

        policies_data = memory.get("policies", {})
        policies = []
        for name in sorted(policies_data.keys()):
            entry = policies_data[name]
            policies.append(Policy.from_dict(name, entry["policy_dict"]))

        clusters = cluster_policies(policies)
        archetypes = extract_policy_archetypes(memory)
        ranked = rank_archetypes(archetypes, memory)
        print(format_policy_clusters_summary(clusters, ranked))
        return 0

    if args.replay_policies or args.use_policy_memory:
        print("Policy memory loaded.")
        print(format_policy_memory_summary(memory))
        return 0

    # Default: show memory.
    print(format_policy_memory_summary(memory))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qec-exp", description="Deterministic experiment CLI")
    parser.add_argument(
        "--artifacts-root",
        default="experiments",
        help="Root directory for deterministic experiment artifacts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run named experiment")
    run_parser.add_argument("experiment", help="Experiment name")
    run_parser.set_defaults(func=_cmd_run)

    phase_parser = sub.add_parser("phase-diagram", help="Generate phase diagram artifact")
    phase_parser.add_argument("experiment", help="Experiment name")
    phase_parser.set_defaults(func=_cmd_phase_diagram)

    threshold_parser = sub.add_parser("estimate-threshold", help="Estimate BP threshold")
    threshold_parser.add_argument("experiment", help="Experiment name")
    threshold_parser.add_argument("--smooth", action="store_true", help="Apply moving-average smoothing")
    threshold_parser.set_defaults(func=_cmd_estimate_threshold)

    memory_parser = sub.add_parser("policy-memory", help="Show or replay policy memory")
    memory_parser.add_argument(
        "--show-policy-memory", action="store_true",
        help="Display stored policy memory",
    )
    memory_parser.add_argument(
        "--use-policy-memory", action="store_true",
        help="Use policy memory in meta-control",
    )
    memory_parser.add_argument(
        "--replay-policies", action="store_true",
        help="Replay top policies from memory",
    )
    memory_parser.add_argument(
        "--memory-file", default=None,
        help="Path to JSON file with exported policy memory",
    )
    memory_parser.add_argument(
        "--show-policy-clusters", action="store_true",
        help="Display policy clusters from memory",
    )
    memory_parser.add_argument(
        "--show-archetypes", action="store_true",
        help="Display archetype policies extracted from clusters",
    )
    memory_parser.add_argument(
        "--use-archetypes", action="store_true",
        help="Include archetypes in meta-control policy candidates",
    )
    memory_parser.set_defaults(func=_cmd_policy_memory)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
