"""Concatenated bosonic decoder experiment — experimental design layer.

Runs the full ternary bosonic pipeline: analog normalization →
ternary quantization → message passing (3–5 rounds) → metrics.
Returns diagnostics for comparison with baselines.

Does not modify decoder internals. Fully deterministic. Opt-in only.

Dependencies: numpy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from qec.analysis.bosonic_interface import normalize_to_bipolar
from qec.analysis.decoder_design_baselines import run_baselines
from qec.analysis.decoder_design_metrics import compute_all_metrics
from qec.analysis.ternary_message_passing import run_message_passing_round
from qec.analysis.ternary_quantization import quantize_ternary, ternary_stats


def _build_chain_adjacency(n: int) -> np.ndarray:
    """Build a simple chain adjacency matrix for n nodes."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def run_concatenated_bosonic_experiment(
    raw_signals: np.ndarray,
    *,
    threshold: float = 0.3,
    rounds: int = 3,
    adjacency: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run the concatenated ternary bosonic decoder experiment.

    Parameters
    ----------
    raw_signals : np.ndarray
        Raw analog input signals (1-D).
    threshold : float
        Ternary quantization threshold in (0, 1].
    rounds : int
        Number of message-passing rounds (3–5 recommended).
    adjacency : np.ndarray, optional
        N×N adjacency matrix. If None, uses a simple chain graph.

    Returns
    -------
    dict
        Full diagnostics including per-round states, metrics,
        baseline comparisons, and final design score.
    """
    if not (3 <= rounds <= 5):
        raise ValueError(f"rounds must be 3–5, got {rounds}")

    raw = np.asarray(raw_signals, dtype=np.float64)
    n = raw.size

    # Step 1: Normalize
    normalized = normalize_to_bipolar(raw)

    # Step 2: Ternary quantization
    ternary = quantize_ternary(normalized, threshold=threshold)
    initial_stats = ternary_stats(ternary)

    # Step 3: Build adjacency if not provided
    if adjacency is None:
        adjacency = _build_chain_adjacency(n)

    # Step 4: Message passing rounds
    states = ternary.copy()
    confidences = np.where(ternary != 0, 0.8, 0.2).astype(np.float64)
    round_diagnostics = []

    for r in range(rounds):
        states, confidences = run_message_passing_round(
            states, confidences, adjacency,
        )
        round_diagnostics.append({
            "round": r + 1,
            "states": states.tolist(),
            "confidences": confidences.tolist(),
            "stats": ternary_stats(states),
        })

    # Step 5: Baselines
    baselines = run_baselines(normalized)

    # Step 6: Metrics
    metrics = compute_all_metrics(
        states, confidences, baselines["hard_threshold"],
    )

    return {
        "n_signals": n,
        "threshold": threshold,
        "rounds": rounds,
        "initial_ternary": ternary.tolist(),
        "initial_stats": initial_stats,
        "round_diagnostics": round_diagnostics,
        "final_states": states.tolist(),
        "final_confidences": confidences.tolist(),
        "baselines": {k: v.tolist() for k, v in baselines.items()},
        "metrics": metrics,
    }


def format_summary(result: Dict[str, Any]) -> str:
    """Format a human-readable summary of the experiment result.

    Parameters
    ----------
    result : dict
        Output of :func:`run_concatenated_bosonic_experiment`.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines = [
        "=== Ternary Bosonic Decoder ===",
        f"Signals: {result['n_signals']}",
        f"Threshold: {result['threshold']}",
        f"Rounds: {result['rounds']}",
        "",
        "Final states: " + str(result["final_states"]),
        "",
        "Metrics:",
    ]
    for k, v in result["metrics"].items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append(f"\nDesign score: {result['metrics']['design_score']:.4f}")
    return "\n".join(lines)
