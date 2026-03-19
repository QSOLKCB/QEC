"""
v9.4.0 — Benchmark Table Builder.

Evaluates a collection of discovered Tanner graphs and produces a
structured benchmark table with spectral and decoder performance metrics.

Layer 6 — Benchmark.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no hidden randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

from qec.benchmark.discovery_benchmark import run_decoder_benchmark
from qec.diagnostics.non_backtracking_spectrum import (
    compute_non_backtracking_spectrum,
)


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def build_benchmark_table(
    graphs: list[dict[str, Any]],
    *,
    trials: int = 100,
    error_rate: float = 0.05,
    max_iters: int = 100,
    base_seed: int = 0,
) -> list[dict[str, Any]]:
    """Build a benchmark table for a set of discovered graphs.

    Each graph entry must have keys:
    - ``graph_id`` : str
    - ``H`` : np.ndarray, parity-check matrix

    Parameters
    ----------
    graphs : list[dict[str, Any]]
        List of graph entries with ``graph_id`` and ``H``.
    trials : int
        Number of decoding trials per graph.
    error_rate : float
        Bit-flip probability for error generation.
    max_iters : int
        Maximum BP iterations per trial.
    base_seed : int
        Deterministic base seed.

    Returns
    -------
    list[dict[str, Any]]
        List of benchmark result dictionaries, each with:
        - ``graph_id`` : str
        - ``spectral_radius`` : float
        - ``bp_success_rate`` : float
        - ``avg_iterations`` : float
    """
    table: list[dict[str, Any]] = []

    for graph_idx, graph_entry in enumerate(graphs):
        graph_id = graph_entry["graph_id"]
        H = np.asarray(graph_entry["H"], dtype=np.float64)

        # Compute spectral radius
        spectrum = compute_non_backtracking_spectrum(H)
        spectral_radius = spectrum["spectral_radius"]

        # Run decoder benchmark
        graph_seed = _derive_seed(base_seed, f"graph_{graph_idx}")
        bench = run_decoder_benchmark(
            H,
            trials=trials,
            error_rate=error_rate,
            max_iters=max_iters,
            base_seed=graph_seed,
        )

        table.append({
            "graph_id": graph_id,
            "spectral_radius": round(spectral_radius, 6),
            "bp_success_rate": round(bench["bp_success_rate"], 6),
            "avg_iterations": round(bench["avg_iterations"], 6),
        })

    return table
