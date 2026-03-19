"""
v74.1.0 — Deterministic sequential sonic state-transition analysis.

Processes an ordered sequence of audio files, runs v74.0 analysis on each,
compares consecutive pairs, and produces a structured transition report.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from qec.experiments.sonic_analysis import analyse_file
from qec.experiments.sonic_comparison import (
    classify_comparison,
    compare_sonic_features,
)


# ---------------------------------------------------------------------------
# Sequential Analysis
# ---------------------------------------------------------------------------

def analyze_sonic_sequence(
    paths: List[str],
    *,
    output_dir: str = "artifacts/sonic",
) -> dict:
    """Analyse an ordered sequence of audio files as a state trajectory.

    For each consecutive pair *(i, i+1)* the function:

    1. Runs ``analyse_file`` from v74.0 on both files.
    2. Computes pairwise deltas via ``compare_sonic_features``.
    3. Classifies the transition via ``classify_comparison``.

    Parameters
    ----------
    paths : list[str]
        Ordered file paths representing system states.
    output_dir : str
        Root directory for per-file analysis artifacts.

    Returns
    -------
    dict
        ``{"n_states": int, "transitions": [...]}``
    """
    if not paths:
        return {"n_states": 0, "transitions": []}

    # Run per-file analysis, collecting results in order.
    analyses: List[Dict[str, Any]] = []
    for p in paths:
        fname = Path(p).stem
        file_out = os.path.join(output_dir, fname)
        result = analyse_file(p, file_out)
        analyses.append(result)

    if len(analyses) < 2:
        return {"n_states": len(analyses), "transitions": []}

    # Compare consecutive pairs.
    transitions: List[Dict[str, Any]] = []
    for i in range(len(analyses) - 1):
        metrics = compare_sonic_features(analyses[i], analyses[i + 1])
        classification = classify_comparison(metrics)
        transitions.append({
            "from": analyses[i].get("source_file", f"state_{i}"),
            "to": analyses[i + 1].get("source_file", f"state_{i + 1}"),
            "from_index": i,
            "to_index": i + 1,
            "metrics": metrics,
            "classification": classification,
        })

    return {
        "n_states": len(analyses),
        "transitions": transitions,
    }


# ---------------------------------------------------------------------------
# Artifact Writer
# ---------------------------------------------------------------------------

def run_sequence_analysis(
    paths: List[str],
    *,
    output_dir: str = "artifacts/sonic",
) -> dict:
    """Run full sequence analysis and write ``sequence_analysis.json``.

    Parameters
    ----------
    paths : list[str]
        Ordered audio file paths.
    output_dir : str
        Root artifact directory.

    Returns
    -------
    dict
        The sequence analysis result.
    """
    result = analyze_sonic_sequence(paths, output_dir=output_dir)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sequence_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    return result
