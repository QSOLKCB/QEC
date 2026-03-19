"""Multi-run comparative sonification experiment engine (v73.0.0).

Runs batch experiments across multiple datasets, compares them
statistically, and produces cross-run insights.

Builds on:
    v72.5.0 — run_sonification_batch
    v72.4.0 — interpret_sonification_comparison
"""

from __future__ import annotations

import copy
import json
import os
from typing import Optional

import numpy as np

from qec.experiments.sonification_batch import run_sonification_batch


def run_sonification_multirun(
    datasets: list[list[dict]],
    output_dir: Optional[str] = None,
) -> dict:
    """Run batch experiments across multiple datasets.

    Parameters
    ----------
    datasets : list[list[dict]]
        Each element is a list of result dicts suitable for
        ``run_sonification_batch``.
    output_dir : str or None
        If provided, write ``multirun_summary.json`` to this directory.

    Returns
    -------
    dict
        Contains per-run summaries and cross-run statistics.
    """
    # Deep-copy inputs to guarantee no mutation.
    inputs = copy.deepcopy(datasets)

    n_runs = len(inputs)

    if n_runs == 0:
        summary = {
            "n_runs": 0,
            "global_mean_score": 0.0,
            "global_variance": 0.0,
            "stability_score": 1.0,
            "verdict_totals": {
                "invalid": 0,
                "multidim_improves_structure": 0,
                "channels_redundant": 0,
                "baseline_more_stable": 0,
                "tradeoff": 0,
            },
            "best_run_index": -1,
            "runs": [],
        }
        if output_dir is not None:
            _write_summary(summary, output_dir)
        return summary

    # Step 1 — Run batches.
    runs: list[dict] = []
    for dataset in inputs:
        batch_summary = run_sonification_batch(dataset)
        runs.append(batch_summary)

    # Step 2 — Cross-run metrics.
    mean_scores = np.array(
        [r["mean_score"] for r in runs],
        dtype=np.float64,
    )

    global_mean = float(np.mean(mean_scores))
    global_variance = float(np.var(mean_scores))
    stability_score = 1.0 / (1.0 + global_variance)

    # Verdict aggregation.
    verdict_totals = {
        "invalid": 0,
        "multidim_improves_structure": 0,
        "channels_redundant": 0,
        "baseline_more_stable": 0,
        "tradeoff": 0,
    }
    for r in runs:
        for key in verdict_totals:
            verdict_totals[key] += r["verdict_counts"].get(key, 0)

    # Step 3 — Best run.
    best_run_index = int(np.argmax(mean_scores))

    # Step 4 — Output.
    summary = {
        "n_runs": n_runs,
        "global_mean_score": global_mean,
        "global_variance": global_variance,
        "stability_score": stability_score,
        "verdict_totals": verdict_totals,
        "best_run_index": best_run_index,
        "runs": runs,
    }

    if output_dir is not None:
        _write_summary(summary, output_dir)

    return summary


def _write_summary(summary: dict, output_dir: str) -> None:
    """Write multirun_summary.json to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "multirun_summary.json")
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
