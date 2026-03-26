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
        "--ternary-bosonic", action="store_true",
        help="Run the ternary bosonic decoder experiment (print states + score).",
    )
    parser.add_argument(
        "--strategy-selection", action="store_true",
        help="Enable trust-aware strategy selection (use with --ternary-bosonic).",
    )
    parser.add_argument(
        "--strategy-generate", action="store_true",
        help="Generate 27 deterministic strategies, evaluate, score, and select "
             "(use with --ternary-bosonic).",
    )
    parser.add_argument(
        "--compare-state-systems", action="store_true",
        help="Compare ternary vs quaternary state systems with 54 total strategies "
             "(use with --ternary-bosonic --strategy-generate).",
    )
    parser.add_argument(
        "--strategy-prune", action="store_true",
        help="Apply dominance pruning (Pareto frontier) to strategy set "
             "(use with --ternary-bosonic --strategy-generate --compare-state-systems).",
    )
    parser.add_argument(
        "--structure-aware", action="store_true",
        help="Apply structure-aware dominance with consistency gap, revival "
             "detection, and redundancy pruning "
             "(use with --ternary-bosonic --strategy-generate "
             "--compare-state-systems --strategy-prune).",
    )
    parser.add_argument(
        "--analyze-strategies", action="store_true",
        help="Run full analysis pipeline: cluster, embed, Pareto, compare "
             "(use with --ternary-bosonic --strategy-generate "
             "--compare-state-systems --strategy-prune --structure-aware).",
    )
    parser.add_argument(
        "--show-pareto", action="store_true",
        help="Show Pareto front in analysis output.",
    )
    parser.add_argument(
        "--show-clusters", action="store_true",
        help="Show strategy clusters in analysis output.",
    )
    parser.add_argument(
        "--show-map", action="store_true",
        help="Show ASCII strategy map visualization.",
    )
    parser.add_argument(
        "--track-strategies", action="store_true",
        help="Track strategy behavior across runs for trajectory analysis "
             "(use with --ternary-bosonic --strategy-generate).",
    )
    parser.add_argument(
        "--show-trajectory", action="store_true",
        help="Show trajectory metrics (mean, variance, stability, trend) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-regimes", action="store_true",
        help="Show regime classification (stable/oscillatory/transitional) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-taxonomy", action="store_true",
        help="Show strategy taxonomy (behavioral type classification) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-evolution", action="store_true",
        help="Show strategy evolution analysis (transition patterns) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-phase-space", action="store_true",
        help="Show phase space analysis (attractors, basins, escape dynamics) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-geometry", action="store_true",
        help="Show flow geometry embedding (coordinates, distances, clusters) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-multistate", action="store_true",
        help="Show multi-state analysis (ternary classification, phase membership) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-coupling", action="store_true",
        help="Show coupled dynamics analysis (interactions, synchronization) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-control", action="store_true",
        help="Show control analysis (interventions, response metrics, optimizer) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-feedback", action="store_true",
        help="Show feedback control analysis (iterative closed-loop adaptation) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-global-control", action="store_true",
        help="Show global multi-strategy feedback control analysis "
             "(coordinated interventions, conflict resolution, convergence) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-hierarchical-control", action="store_true",
        help="Show hierarchical control analysis "
             "(policy routing, local/global merge, convergence) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-meta-control", action="store_true",
        help="Show meta-control analysis "
             "(dynamic policy selection, switching detection, convergence) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--policy", type=str, default=None,
        help="Comma-separated list of policies for meta-control "
             "(e.g. stability_first,sync_first,balanced).",
    )
    parser.add_argument(
        "--show-policy-refinement", action="store_true",
        help="Show policy refinement analysis "
             "(deterministic threshold optimization) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--refine-policy", type=str, default=None,
        help="Refine a specific policy by name "
             "(e.g. stability_first). "
             "Use with --show-policy-refinement.",
    )
    parser.add_argument(
        "--show-strategy-graph", action="store_true",
        help="Show strategy graph analysis "
             "(policy transitions, node metrics, stability) "
             "(use with --track-strategies).",
    )
    parser.add_argument(
        "--show-policy-topology", action="store_true",
        help="Show policy topology classification "
             "(stable, converging, diverging, cyclic, mixed) "
             "(use with --track-strategies).",
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

    if args.ternary_bosonic:
        return _run_ternary_bosonic(args)

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


def _run_ternary_bosonic(args) -> int:
    """Run the ternary bosonic decoder experiment."""
    import numpy as np

    from qec.experiments.concatenated_bosonic_decoder import (
        format_summary,
        run_concatenated_bosonic_experiment,
    )

    # Deterministic demo signals
    raw = np.array([0.1, -0.8, 0.5, 0.0, -0.3, 0.9, -0.1, 0.4], dtype=np.float64)

    result = run_concatenated_bosonic_experiment(raw, threshold=0.3, rounds=3)
    print(format_summary(result), file=sys.stderr)

    if getattr(args, "strategy_selection", False):
        from qec.analysis.strategy_adapter import (
            format_selection_summary,
            run_strategy_selection,
        )

        strategy_configs = [
            {"name": "conservative", "threshold": 0.4, "rounds": 3},
            {"name": "balanced", "threshold": 0.3, "rounds": 3},
            {"name": "aggressive", "threshold": 0.2, "rounds": 3},
        ]
        sel = run_strategy_selection(
            result,
            trust_signals={"stability": 0.8, "global_trust": 0.6},
            strategy_configs=strategy_configs,
        )
        print(format_selection_summary(sel), file=sys.stderr)

    if getattr(args, "strategy_generate", False):
        from qec.analysis.strategy_adapter import (
            format_generation_summary,
            run_generation_selection_pipeline,
        )

        base_strategy = {
            "config": {
                "threshold": result.get("threshold", 0.3),
                "rounds": result.get("rounds", 3),
                "confidence_scale": 1.0,
            },
            "metrics": dict(result.get("metrics", {})),
        }

        if getattr(args, "compare_state_systems", False):
            if getattr(args, "strategy_prune", False):
                if getattr(args, "structure_aware", False):
                    if getattr(args, "analyze_strategies", False):
                        from qec.analysis.strategy_adapter import (
                            format_analysis_summary,
                            run_analysis_pipeline,
                        )

                        analysis_result = run_analysis_pipeline(
                            base_strategy,
                            raw_signals=raw,
                            trust_signals={"stability": 0.8, "global_trust": 0.6},
                        )
                        print(
                            format_analysis_summary(
                                analysis_result,
                                show_pareto=getattr(args, "show_pareto", False),
                                show_clusters=getattr(args, "show_clusters", False),
                                show_map=getattr(args, "show_map", False),
                            ),
                            file=sys.stderr,
                        )
                    else:
                        from qec.analysis.strategy_adapter import (
                            format_structure_aware_summary,
                            run_structure_aware_pipeline,
                        )

                        sa_result = run_structure_aware_pipeline(
                            base_strategy,
                            raw_signals=raw,
                            trust_signals={"stability": 0.8, "global_trust": 0.6},
                        )
                        print(
                            format_structure_aware_summary(sa_result),
                            file=sys.stderr,
                        )
                else:
                    from qec.analysis.strategy_adapter import (
                        format_pruning_summary,
                        run_pruned_pipeline,
                    )

                    prune_result = run_pruned_pipeline(
                        base_strategy,
                        raw_signals=raw,
                        trust_signals={"stability": 0.8, "global_trust": 0.6},
                    )
                    print(format_pruning_summary(prune_result), file=sys.stderr)
            else:
                from qec.analysis.strategy_adapter import (
                    format_comparison_summary,
                    run_dual_generation_pipeline,
                )

                dual_result = run_dual_generation_pipeline(
                    base_strategy,
                    raw_signals=raw,
                    trust_signals={"stability": 0.8, "global_trust": 0.6},
                )
                print(format_comparison_summary(dual_result), file=sys.stderr)
        else:
            gen_result = run_generation_selection_pipeline(
                base_strategy,
                trust_signals={"stability": 0.8, "global_trust": 0.6},
            )
            print(format_generation_summary(gen_result), file=sys.stderr)

    if getattr(args, "track_strategies", False):
        from qec.analysis.strategy_adapter import (
            format_taxonomy_summary,
            format_trajectory_summary,
            run_taxonomy_analysis,
        )

        # Simulate multiple runs with slight signal perturbations.
        # Each run produces a full strategy set for trajectory tracking.
        run_results = []
        perturbations = [0.0, 0.02, -0.01]
        for delta in perturbations:
            perturbed = raw + delta
            run_result = run_concatenated_bosonic_experiment(
                perturbed, threshold=0.3, rounds=3,
            )
            base = {
                "config": {
                    "threshold": run_result.get("threshold", 0.3),
                    "rounds": run_result.get("rounds", 3),
                    "confidence_scale": 1.0,
                },
                "metrics": dict(run_result.get("metrics", {})),
            }
            from qec.analysis.strategy_generation import generate_strategies
            from qec.analysis.strategy_selection import rank_strategies

            strategies = generate_strategies(base)
            ranked = rank_strategies(
                strategies,
                trust_signals={"stability": 0.8, "global_trust": 0.6},
            )
            run_results.append({"strategies": ranked})

        traj_result = run_taxonomy_analysis(run_results)

        if getattr(args, "show_trajectory", False) or getattr(args, "show_regimes", False):
            print(format_trajectory_summary(traj_result), file=sys.stderr)

        if getattr(args, "show_taxonomy", False):
            print(format_taxonomy_summary(traj_result), file=sys.stderr)

        if getattr(args, "show_evolution", False):
            from qec.analysis.strategy_adapter import (
                format_evolution_summary,
                run_evolution_analysis,
            )

            evo_result = run_evolution_analysis(run_results)
            print(format_evolution_summary(evo_result), file=sys.stderr)

        phase_result = None
        if getattr(args, "show_phase_space", False) or getattr(args, "show_geometry", False):
            from qec.analysis.strategy_adapter import run_phase_space_analysis

            phase_result = run_phase_space_analysis(run_results)

        if getattr(args, "show_phase_space", False):
            from qec.analysis.strategy_adapter import format_phase_space_summary

            print(format_phase_space_summary(phase_result), file=sys.stderr)

        if getattr(args, "show_geometry", False):
            from qec.analysis.strategy_adapter import (
                format_flow_geometry_summary,
                run_flow_geometry_analysis,
            )

            geo_result = run_flow_geometry_analysis(
                run_results,
                phase_space_result=phase_result,
            )
            print(
                format_flow_geometry_summary(
                    geo_result,
                    show_map=True,
                ),
                file=sys.stderr,
            )

        multistate_result = None
        coupled_result = None

        if getattr(args, "show_multistate", False):
            from qec.analysis.strategy_adapter import (
                format_multistate_summary,
                run_multistate_analysis,
            )

            multistate_result = run_multistate_analysis(
                run_results,
                phase_space_result=phase_result,
            )
            print(format_multistate_summary(multistate_result), file=sys.stderr)

        if getattr(args, "show_coupling", False):
            from qec.analysis.strategy_adapter import (
                format_coupled_dynamics_summary,
                run_coupled_dynamics_analysis,
            )

            coupled_result = run_coupled_dynamics_analysis(
                run_results,
                multistate_result=multistate_result,
            )
            print(
                format_coupled_dynamics_summary(coupled_result),
                file=sys.stderr,
            )

        if getattr(args, "show_control", False):
            from qec.analysis.strategy_adapter import (
                format_control_summary,
                run_control_analysis,
            )

            control_result = run_control_analysis(
                run_results,
                multistate_result=multistate_result,
                coupled_result=coupled_result,
            )
            print(format_control_summary(control_result), file=sys.stderr)

        if getattr(args, "show_feedback", False):
            from qec.analysis.strategy_adapter import (
                format_feedback_summary,
                run_feedback_analysis,
            )

            feedback_result = run_feedback_analysis(
                run_results,
                multistate_result=multistate_result,
            )
            print(format_feedback_summary(feedback_result), file=sys.stderr)

        if getattr(args, "show_global_control", False):
            from qec.analysis.strategy_adapter import (
                format_global_control_summary,
                run_global_control_analysis,
            )

            global_result = run_global_control_analysis(
                run_results,
                multistate_result=multistate_result,
                coupled_result=coupled_result,
            )
            print(
                format_global_control_summary(global_result),
                file=sys.stderr,
            )

        if getattr(args, "show_hierarchical_control", False):
            from qec.analysis.strategy_adapter import (
                format_hierarchical_control_summary,
                run_hierarchical_control_analysis,
            )

            hierarchical_result = run_hierarchical_control_analysis(
                run_results,
                multistate_result=multistate_result,
                coupled_result=coupled_result,
            )
            print(
                format_hierarchical_control_summary(hierarchical_result),
                file=sys.stderr,
            )

        if getattr(args, "show_meta_control", False):
            from qec.analysis.strategy_adapter import (
                format_meta_control_summary,
                run_meta_control_analysis,
            )

            meta_policies = None
            if getattr(args, "policy", None):
                from qec.analysis.policy import get_policy

                policy_names = [
                    n.strip() for n in args.policy.split(",") if n.strip()
                ]
                meta_policies = [get_policy(n) for n in policy_names]

            meta_result = run_meta_control_analysis(
                run_results,
                policies=meta_policies,
                multistate_result=multistate_result,
                coupled_result=coupled_result,
            )
            print(
                format_meta_control_summary(meta_result),
                file=sys.stderr,
            )

        if getattr(args, "show_policy_refinement", False):
            from qec.analysis.strategy_adapter import (
                format_policy_refinement_adapter_summary,
                run_policy_refinement_analysis,
            )

            refine_policies = None
            if getattr(args, "refine_policy", None):
                from qec.analysis.policy import get_policy

                policy_names = [
                    n.strip()
                    for n in args.refine_policy.split(",")
                    if n.strip()
                ]
                refine_policies = [get_policy(n) for n in policy_names]

            refinement_result = run_policy_refinement_analysis(
                run_results,
                policies=refine_policies,
                multistate_result=multistate_result,
                coupled_result=coupled_result,
            )
            print(
                format_policy_refinement_adapter_summary(refinement_result),
                file=sys.stderr,
            )

        if getattr(args, "show_strategy_graph", False) or getattr(
            args, "show_policy_topology", False
        ):
            from qec.analysis.strategy_adapter import (
                format_strategy_graph_adapter_summary,
                run_strategy_graph_analysis,
            )

            graph_policies = None
            if getattr(args, "policy", None):
                from qec.analysis.policy import get_policy

                policy_names = [
                    n.strip() for n in args.policy.split(",") if n.strip()
                ]
                graph_policies = [get_policy(n) for n in policy_names]

            graph_result = run_strategy_graph_analysis(
                run_results,
                policies=graph_policies,
                multistate_result=multistate_result,
                coupled_result=coupled_result,
            )
            print(
                format_strategy_graph_adapter_summary(graph_result),
                file=sys.stderr,
            )

    if args.out:
        text = json.dumps(result, sort_keys=True, indent=2)
        with open(args.out, "w") as f:
            f.write(text)
            f.write("\n")
        print(f"\nResults written to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
