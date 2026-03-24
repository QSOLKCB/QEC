"""Quaternary bosonic decoder experiment — experimental design layer.

Runs the full quaternary bosonic pipeline: analog normalization →
quaternary quantization → message passing (3–5 rounds) → metrics.
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
from qec.analysis.quaternary_message_passing import run_message_passing_round
from qec.analysis.quaternary_quantization import (
    quantize_quaternary,
    quaternary_stats,
)


def _build_chain_adjacency(n: int) -> np.ndarray:
    """Build a simple chain adjacency matrix for n nodes."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def _quaternary_to_ternary_for_metrics(states: np.ndarray) -> np.ndarray:
    """Convert quaternary states to ternary sign for metric computation.

    Maps: -1.0, -0.5 → -1; 0.5, 1.0 → +1.
    Quaternary has no neutral state, so result is in {-1, +1}.
    """
    result = np.ones(states.shape, dtype=np.int8)
    result[states < 0] = -1
    return result


def run_quaternary_bosonic_experiment(
    raw_signals: np.ndarray,
    *,
    rounds: int = 3,
    adjacency: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run the quaternary bosonic decoder experiment.

    Parameters
    ----------
    raw_signals : np.ndarray
        Raw analog input signals (1-D).
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

    # Step 2: Quaternary quantization
    quaternary = quantize_quaternary(normalized)
    initial_stats = quaternary_stats(quaternary)

    # Step 3: Build adjacency if not provided
    if adjacency is None:
        adjacency = _build_chain_adjacency(n)

    # Step 4: Message passing rounds
    states = quaternary.copy()
    # Strong states (|s| == 1.0) get high confidence, soft states get lower
    confidences = np.where(
        np.abs(states) == 1.0, 0.8, 0.5,
    ).astype(np.float64)
    round_diagnostics = []

    for r in range(rounds):
        states, confidences = run_message_passing_round(
            states, confidences, adjacency,
        )
        round_diagnostics.append({
            "round": r + 1,
            "states": states.tolist(),
            "confidences": confidences.tolist(),
            "stats": quaternary_stats(states),
        })

    # Step 5: Baselines
    baselines = run_baselines(normalized)

    # Step 6: Metrics — use ternary sign mapping for compatibility
    ternary_states = _quaternary_to_ternary_for_metrics(states)
    metrics = compute_all_metrics(
        ternary_states, confidences, baselines["hard_threshold"],
    )

    return {
        "n_signals": n,
        "state_system": "quaternary",
        "rounds": rounds,
        "initial_quaternary": quaternary.tolist(),
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
        Output of :func:`run_quaternary_bosonic_experiment`.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines = [
        "=== Quaternary Bosonic Decoder ===",
        f"Signals: {result['n_signals']}",
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
