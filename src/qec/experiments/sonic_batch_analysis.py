"""
v74.0.0 — Deterministic batch sonic artifact analysis.

Aggregates per-file spectral analysis results across multiple audio files,
producing a batch summary with statistical metrics.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from qec.experiments.sonic_analysis import analyse_file


def run_sonic_batch_analysis(
    paths: List[str],
    output_root: str = "artifacts/sonic",
) -> Dict[str, Any]:
    """Run analysis on multiple audio files and aggregate results.

    Parameters
    ----------
    paths : list[str]
        Paths to audio files.
    output_root : str
        Root directory for output artifacts.

    Returns
    -------
    dict
        Batch summary containing per-file results and aggregate metrics.
    """
    if not paths:
        return {
            "n_files": 0,
            "mean_duration": 0.0,
            "mean_energy": 0.0,
            "mean_centroid": 0.0,
            "variance_centroid": 0.0,
            "files": [],
        }

    file_results: List[Dict[str, Any]] = []

    for path in paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        # Sanitise directory name
        safe_name = basename.replace(" ", "_").replace("–", "-")
        file_output_dir = os.path.join(output_root, safe_name)
        result = analyse_file(path, file_output_dir)
        file_results.append(result)

    durations = [r["duration_seconds"] for r in file_results]
    energies = [r["rms_energy"] for r in file_results]
    centroids = [r["spectral_centroid_hz"] for r in file_results]

    summary: Dict[str, Any] = {
        "n_files": len(file_results),
        "mean_duration": float(np.mean(durations)),
        "mean_energy": float(np.mean(energies)),
        "mean_centroid": float(np.mean(centroids)),
        "variance_centroid": float(np.var(centroids)),
        "files": file_results,
    }

    # Write batch summary JSON
    os.makedirs(output_root, exist_ok=True)
    summary_path = os.path.join(output_root, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary
