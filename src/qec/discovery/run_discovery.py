"""
v10.0.0 — Discovery CLI Entry Point.

Run the QLDPC discovery engine from the command line:

    python -m qec.discovery.run_discovery \\
        --population 50 --generations 100 --seed 42 \\
        --archive-path ./discovery_archive.db

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Ensure repo root is on path
_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.discovery.population_engine import DiscoveryEngine
from src.qec.archive.storage import DiscoveryArchive
from src.qec.utils.artifact_metadata import generate_run_metadata


def main() -> None:
    """CLI entry point for the discovery engine."""
    parser = argparse.ArgumentParser(
        description="QLDPC Deterministic Structure Discovery Engine v10.0.0",
    )
    parser.add_argument(
        "--population", type=int, default=50,
        help="Population size per generation (default: 50)",
    )
    parser.add_argument(
        "--generations", type=int, default=500,
        help="Number of evolutionary generations (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed for deterministic execution (default: 42)",
    )
    parser.add_argument(
        "--archive-path", type=str, default="./discovery_archive.db",
        help="Path for persistent SQLite archive (default: ./discovery_archive.db)",
    )
    parser.add_argument(
        "--num-variables", type=int, default=12,
        help="Number of variable nodes (default: 12)",
    )
    parser.add_argument(
        "--num-checks", type=int, default=6,
        help="Number of check nodes (default: 6)",
    )
    parser.add_argument(
        "--variable-degree", type=int, default=3,
        help="Target variable node degree (default: 3)",
    )
    parser.add_argument(
        "--check-degree", type=int, default=6,
        help="Target check node degree (default: 6)",
    )
    parser.add_argument(
        "--init-strategy", type=str, default="peg",
        choices=["random", "peg"],
        help="Population initialization strategy (default: peg)",
    )
    parser.add_argument(
        "--no-decoder-aware",
        action="store_false",
        dest="decoder_aware",
        default=True,
        help="Disable decoder-aware fitness evaluation (enabled by default)",
    )
    parser.add_argument(
        "--bp-trials", type=int, default=50,
        help="Number of BP probe trials for decoder-aware mode (default: 50)",
    )
    parser.add_argument(
        "--bp-iterations", type=int, default=10,
        help="Max BP iterations per probe trial (default: 10)",
    )
    parser.add_argument(
        "--absorbing-set-max-size", type=int, default=6,
        help="Maximum absorbing-set candidate size (default: 6)",
    )
    parser.add_argument(
        "--disable-absorbing-set-risk",
        action="store_true",
        default=False,
        help="Disable absorbing-set risk in fitness evaluation",
    )

    args = parser.parse_args()

    spec = {
        "num_variables": args.num_variables,
        "num_checks": args.num_checks,
        "variable_degree": args.variable_degree,
        "check_degree": args.check_degree,
        "init_strategy": args.init_strategy,
    }

    print(f"QLDPC Discovery Engine v11.0.0")
    print(f"  Spec: {spec}")
    print(f"  Population: {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Seed: {args.seed}")
    print(f"  Init strategy: {args.init_strategy}")
    print(f"  Decoder-aware: {args.decoder_aware}")
    if args.decoder_aware:
        print(f"  BP trials: {args.bp_trials}")
        print(f"  BP iterations: {args.bp_iterations}")
    print(f"  Archive: {args.archive_path}")
    print()

    # Generate metadata
    metadata = generate_run_metadata(args.seed)

    # Run discovery
    engine = DiscoveryEngine(
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
        archive_path=args.archive_path,
        decoder_aware=args.decoder_aware,
        bp_trials=args.bp_trials,
        bp_iterations=args.bp_iterations,
    )

    result = engine.run(spec)

    # Store results in persistent archive
    archive = DiscoveryArchive(db_path=args.archive_path)
    try:
        if result["best_H"] is not None:
            code_id = archive.add_code(
                result["best_H"],
                fitness=result["best"]["fitness"] if result["best"] else None,
                metrics=result["best"]["metrics"] if result["best"] else None,
                generation=args.generations,
                seed=args.seed,
                run_metadata=metadata,
            )
            print(f"Best code stored: {code_id}")

        # Store archive entries
        for entry in result.get("archive", []):
            if entry is not None:
                # Retrieve the H from elite_history is not available in
                # serialized form; archive entries are already stored
                pass

        print(f"Archive contains {archive.count()} codes")
    finally:
        archive.close()

    # Print summary
    print()
    print("=== Discovery Complete ===")
    if result["best"]:
        print(f"  Best fitness: {result['best']['fitness']}")
        print(f"  Best code ID: {result['best']['code_id']}")
        metrics = result["best"].get("metrics", {})
        if metrics:
            print(f"  Girth: {metrics.get('girth', 'N/A')}")
            print(f"  NBT spectral radius: {metrics.get('nbt_spectral_radius', 'N/A')}")
            print(f"  Expansion: {metrics.get('expansion', 'N/A')}")
    print(f"  Archive location: {args.archive_path}")
    print(f"  Generations run: {args.generations}")

    # Write JSON summary
    summary_path = args.archive_path.replace(".db", "_summary.json")
    summary = {
        "metadata": metadata,
        "spec": spec,
        "config": {
            "population": args.population,
            "generations": args.generations,
            "seed": args.seed,
        },
        "best": result["best"],
        "elite_history": result["elite_history"],
        "generation_summaries": result["generation_summaries"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
    print(f"  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
