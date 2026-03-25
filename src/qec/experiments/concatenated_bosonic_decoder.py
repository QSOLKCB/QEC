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
    confidence_scale: float = 1.0,
    neutral_bias: float = 0.0,
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
    confidence_scale : float
        Multiplicative scaling applied at initialization and after each
        message-passing round (compounding). Default 1.0.
    neutral_bias : float
        Additive bias applied to normalized signals before quantization
        and to neutral-state initial confidence. Default 0.0.

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

    # Step 2: Ternary quantization (apply neutral_bias before snapping)
    biased = np.clip(normalized + neutral_bias, -1.0, 1.0)
    ternary = quantize_ternary(biased, threshold=threshold)
    initial_stats = ternary_stats(ternary)

    # Step 3: Build adjacency if not provided
    if adjacency is None:
        adjacency = _build_chain_adjacency(n)

    # Step 4: Message passing rounds
    states = ternary.copy()
    neutral_conf = float(np.clip(0.2 + neutral_bias, 0.0, 1.0))
    confidences = np.where(ternary != 0, 0.8, neutral_conf).astype(np.float64)
    confidences = np.clip(confidences * confidence_scale, 0.0, 1.0)
    round_diagnostics = []

    for r in range(rounds):
        states, confidences = run_message_passing_round(
            states, confidences, adjacency,
        )
        # Apply confidence scaling after each round so it feeds back
        confidences = np.clip(confidences * confidence_scale, 0.0, 1.0)
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
