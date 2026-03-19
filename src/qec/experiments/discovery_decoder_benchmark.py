"""
v9.4.0 — Discovery Decoder Benchmark Experiment.

Loads discovered Tanner graphs, runs decoder benchmarks, builds a
performance table, and saves results as a reproducible artifact.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Any

import numpy as np

from qec.benchmark.benchmark_table import build_benchmark_table
from qec.generation.tanner_graph_generator import (
    generate_tanner_graph_candidates,
)
from qec.utils.reproducibility import collect_environment_metadata
from utils.canonicalize import canonicalize


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def run_discovery_decoder_benchmark(
    specs: list[dict[str, Any]],
    *,
    candidates_per_spec: int = 4,
    trials: int = 100,
    error_rate: float = 0.05,
    max_iters: int = 100,
    base_seed: int = 0,
    output_path: str = "artifacts/decoder_benchmark.json",
) -> dict[str, Any]:
    """Run decoder benchmark across discovered Tanner graphs.

    Workflow:
    1. Generate candidate graphs from specifications.
    2. Run decoder benchmark on each graph.
    3. Build performance table.
    4. Save reproducible artifact.

    Parameters
    ----------
    specs : list[dict[str, Any]]
        List of generation specifications.
    candidates_per_spec : int
        Number of candidate graphs per specification.
    trials : int
        Number of decoding trials per graph.
    error_rate : float
        Bit-flip probability for error generation.
    max_iters : int
        Maximum BP iterations per trial.
    base_seed : int
        Deterministic base seed.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    dict[str, Any]
        Benchmark artifact with metadata and results.
    """
    graphs: list[dict[str, Any]] = []

    for spec_idx, spec in enumerate(specs):
        spec_seed = _derive_seed(base_seed, f"spec_{spec_idx}")
        candidates = generate_tanner_graph_candidates(
            spec, candidates_per_spec, base_seed=spec_seed,
        )

        for cand_idx, candidate in enumerate(candidates):
            graph_id = (
                f"spec{spec_idx}_cand{cand_idx}"
            )
            graphs.append({
                "graph_id": graph_id,
                "H": candidate["H"],
            })

    # Build benchmark table
    table_seed = _derive_seed(base_seed, "benchmark_table")
    table = build_benchmark_table(
        graphs,
        trials=trials,
        error_rate=error_rate,
        max_iters=max_iters,
        base_seed=table_seed,
    )

    metadata = collect_environment_metadata()
    metadata["benchmark_config"] = {
        "num_specs": len(specs),
        "candidates_per_spec": candidates_per_spec,
        "trials": trials,
        "error_rate": error_rate,
        "max_iters": max_iters,
        "base_seed": base_seed,
    }

    artifact = canonicalize({
        "metadata": metadata,
        "benchmark_results": table,
    })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return artifact
