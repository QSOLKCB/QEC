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
    search.add_argument("--enable-pareto", action="store_true")
    search.add_argument("--enable-nb-predictor", action="store_true")
    search.add_argument("--enable-learning", action="store_true")
    search.add_argument("--enable-nb-flow-mutation", action="store_true")
    search.add_argument("--enable-multi-mode-nb-mutation", action="store_true")
    search.add_argument("--multi-mode-k", type=int, default=3)
    search.add_argument("--enable-spectral-mutation-memory", action="store_true")
    search.add_argument("--memory-max-records", type=int, default=1000)
    search.add_argument("--nb-mutation-modes", type=int, default=3)
    search.add_argument("--enable-ipr-localized-nb-flow", action="store_true")
    search.add_argument("--enable-nb-spectral-annealing", action="store_true")
    search.add_argument("--annealing-base-mutation-size", type=int, default=4)
    search.add_argument("--ipr-localization-fraction", type=float, default=0.1)
    search.add_argument("--min-predicted-threshold", type=float, default=0.0)
    search.add_argument("--rank-by-prediction", action="store_true")
    search.add_argument("--max-bp-candidates", type=int, default=5)
    search.add_argument("--enable-predictor-recalibration", action="store_true")
    search.add_argument("--recalibration-interval", type=int, default=20)
    search.add_argument("--enable-spectral-trapping-repair", action="store_true")
    search.add_argument("--trapping-localization-fraction", type=float, default=0.2)

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
            enable_pareto=bool(args.enable_pareto),
            enable_nb_predictor=bool(args.enable_nb_predictor),
            enable_learning=bool(args.enable_learning),
            enable_nb_flow_mutation=bool(args.enable_nb_flow_mutation),
            enable_multi_mode_nb_mutation=bool(args.enable_multi_mode_nb_mutation),
            multi_mode_k=int(args.multi_mode_k),
            enable_spectral_mutation_memory=bool(args.enable_spectral_mutation_memory),
            memory_max_records=int(args.memory_max_records),
            nb_mutation_modes=int(args.nb_mutation_modes),
            enable_ipr_localized_nb_flow=bool(args.enable_ipr_localized_nb_flow),
            enable_nb_spectral_annealing=bool(args.enable_nb_spectral_annealing),
            annealing_base_mutation_size=int(args.annealing_base_mutation_size),
            ipr_localization_fraction=float(args.ipr_localization_fraction),
            min_predicted_threshold=float(args.min_predicted_threshold),
            rank_by_prediction=bool(args.rank_by_prediction),
            max_bp_candidates=int(args.max_bp_candidates),
            enable_predictor_recalibration=bool(args.enable_predictor_recalibration),
            recalibration_interval=int(args.recalibration_interval),
            enable_spectral_trapping_repair=bool(args.enable_spectral_trapping_repair),
            trapping_localization_fraction=float(args.trapping_localization_fraction),
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
