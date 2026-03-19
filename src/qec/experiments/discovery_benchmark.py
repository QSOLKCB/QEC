"""
v9.2.0 — Discovery Benchmark.

Runs the discovery engine across multiple graph specifications
and records comparative results.  Includes reproducibility metadata.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Any

from qec.discovery.discovery_engine import run_structure_discovery
from qec.utils.reproducibility import collect_environment_metadata
from utils.canonicalize import canonicalize


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def run_discovery_benchmark(
    specs: list[dict[str, Any]],
    *,
    num_generations: int = 10,
    population_size: int = 8,
    base_seed: int = 0,
    archive_top_k: int = 5,
    output_path: str = "artifacts/discovery_benchmark.json",
) -> dict[str, Any]:
    """Run discovery across multiple specifications.

    Parameters
    ----------
    specs : list[dict[str, Any]]
        List of generation specifications.
    num_generations : int
        Generations per spec.
    population_size : int
        Population size per generation.
    base_seed : int
        Deterministic base seed.
    archive_top_k : int
        Archive elites per category.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    dict[str, Any]
        Benchmark results.
    """
    results: list[dict[str, Any]] = []

    for spec_idx, spec in enumerate(specs):
        spec_seed = _derive_seed(base_seed, f"bench_spec_{spec_idx}")

        discovery = run_structure_discovery(
            spec,
            num_generations=num_generations,
            population_size=population_size,
            base_seed=spec_seed,
            archive_top_k=archive_top_k,
        )

        best = discovery["best_candidate"]
        results.append({
            "spec_index": spec_idx,
            "num_variables": spec["num_variables"],
            "num_checks": spec["num_checks"],
            "variable_degree": spec["variable_degree"],
            "check_degree": spec["check_degree"],
            "best_candidate_id": best["candidate_id"] if best else "",
            "best_composite_score": (
                best["objectives"].get("composite_score", 0.0) if best else 0.0
            ),
            "best_instability_score": (
                best["objectives"].get("instability_score", 0.0) if best else 0.0
            ),
            "best_spectral_radius": (
                best["objectives"].get("spectral_radius", 0.0) if best else 0.0
            ),
            "best_bethe_margin": (
                best["objectives"].get("bethe_margin", 0.0) if best else 0.0
            ),
            "archive_summary": discovery["archive_summary"],
            "num_generations_run": len(discovery["generation_summaries"]),
        })

    metadata = collect_environment_metadata(
        generation_count=num_generations,
        population_size=population_size,
    )

    artifact = {
        "metadata": metadata,
        "benchmark_results": {
            "num_specs": len(specs),
            "config": {
                "num_generations": num_generations,
                "population_size": population_size,
                "base_seed": base_seed,
                "archive_top_k": archive_top_k,
            },
            "results": results,
        },
    }

    artifact = canonicalize(artifact)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return artifact
