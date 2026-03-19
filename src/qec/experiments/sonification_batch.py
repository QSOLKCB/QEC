"""Batch sonification experiment runner (v72.5.0).

Runs comparison + interpretation over multiple inputs and aggregates
results into summary statistics.

Builds on:
    v72.3.0 — run_sonification_comparison
    v72.4.0 — interpret_sonification_comparison
"""

from __future__ import annotations

import copy
import json
import os
from typing import Optional

import numpy as np

from qec.experiments.sonification_comparison import run_sonification_comparison
from qec.experiments.sonification_interpretation import (
    interpret_sonification_comparison,
)


def run_sonification_batch(
    results: list[dict],
    output_dir: Optional[str] = None,
) -> dict:
    """Run comparison + interpretation over multiple inputs.

    Parameters
    ----------
    results : list[dict]
        Each element is a result dict suitable for
        ``run_sonification_comparison``.
    output_dir : str or None
        If provided, write ``batch_summary.json`` to this directory.

    Returns
    -------
    dict
        Contains per-sample outputs and aggregated statistics.
    """
    # Deep-copy inputs to guarantee no mutation.
    inputs = copy.deepcopy(results)

    n_samples = len(inputs)

    if n_samples == 0:
        summary = {
            "n_samples": 0,
            "mean_score": 0.0,
            "verdict_counts": {
                "invalid": 0,
                "multidim_improves_structure": 0,
                "channels_redundant": 0,
                "baseline_more_stable": 0,
                "tradeoff": 0,
            },
            "best_index": -1,
            "worst_index": -1,
            "samples": [],
        }
        if output_dir is not None:
            _write_summary(summary, output_dir)
        return summary

    # Step 1 — Per-sample processing.
    samples: list[dict] = []
    for item in inputs:
        comparison = run_sonification_comparison(item)
        interpretation = interpret_sonification_comparison(comparison)
        samples.append({
            "comparison": comparison,
            "interpretation": interpretation,
        })

    # Step 2 — Aggregation.
    scores = np.array(
        [s["interpretation"]["composite_score"] for s in samples],
        dtype=np.float64,
    )

    mean_score = float(np.mean(scores))

    verdict_counts = {
        "invalid": 0,
        "multidim_improves_structure": 0,
        "channels_redundant": 0,
        "baseline_more_stable": 0,
        "tradeoff": 0,
    }
    for s in samples:
        verdict = s["interpretation"]["verdict"]
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

    best_index = int(np.argmax(scores))
    worst_index = int(np.argmin(scores))

    # Step 3 — Summary structure.
    summary = {
        "n_samples": n_samples,
        "mean_score": mean_score,
        "verdict_counts": verdict_counts,
        "best_index": best_index,
        "worst_index": worst_index,
        "samples": samples,
    }

    if output_dir is not None:
        _write_summary(summary, output_dir)

    return summary


def _write_summary(summary: dict, output_dir: str) -> None:
    """Write batch_summary.json to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "batch_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)


def _json_default(obj):
    """Handle numpy types during JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
