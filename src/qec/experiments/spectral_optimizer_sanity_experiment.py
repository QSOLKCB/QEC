"""
v7.5.0 — Spectral Optimizer Sanity Experiment + Predictor Stability Probe.

Deterministic validation: baseline vs optimized graph decode,
with mini stability probe for the spectral instability predictor.

Layer 5 — Experiment.  Does NOT import or modify the decoder (Layer 1).
Fully deterministic.  All floats rounded to 12 decimal places.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from qec.experiments.spectral_graph_optimizer import (
    _compute_graph_instability_score,
    run_spectral_graph_optimization,
)
from qec.experiments.tanner_graph_repair import (
    _extract_edges, _apply_swap, _edges_to_H,
)

_INSTABILITY_THRESHOLD = 0.5
_ROUND = 12


def _predict_instability(H: np.ndarray) -> dict[str, Any]:
    """Compute instability prediction for a parity-check matrix."""
    score = _compute_graph_instability_score(H)["instability_score"]
    return {
        "predicted_instability_score": round(score, _ROUND),
        "predicted_failure": score > _INSTABILITY_THRESHOLD,
    }


def _decode_trials(
    H: np.ndarray,
    instances: list[dict[str, Any]],
    decode_fn: Callable[..., Any],
    syndrome_fn: Callable[..., Any],
    *,
    max_iters: int = 50,
    bp_mode: str = "min_sum",
) -> dict[str, Any]:
    """Run decode over instances, return FER/WER/avg_iterations/actual_failure."""
    n_trials = len(instances)
    if n_trials == 0:
        return {"fer": 0.0, "wer": 0.0, "avg_iterations": 0.0, "actual_failure": False}

    frame_errors = 0
    total_bit_errors = 0
    total_iterations = 0
    total_bits = 0
    any_failure = False

    for inst in instances:
        result = decode_fn(
            H, inst["llr"],
            max_iters=max_iters, mode=bp_mode,
            schedule="flooding", syndrome_vec=inst["s"],
        )
        correction, iters = result[0], result[1]
        s_corr = syndrome_fn(H, correction)
        if not bool(np.array_equal(s_corr, inst["s"])):
            frame_errors += 1
            any_failure = True
        total_bit_errors += int(np.sum(correction != inst["e"]))
        total_iterations += int(iters)
        total_bits += len(inst["e"])

    fer = round(frame_errors / n_trials, _ROUND)
    wer = round(total_bit_errors / total_bits, _ROUND) if total_bits > 0 else 0.0
    return {
        "fer": fer,
        "wer": wer,
        "avg_iterations": round(total_iterations / n_trials, _ROUND),
        "actual_failure": any_failure,
    }


def run_spectral_optimizer_sanity_experiment(
    H: np.ndarray,
    instances: list[dict[str, Any]],
    decode_fn: Callable[..., Any],
    syndrome_fn: Callable[..., Any],
    *,
    max_iters: int = 50,
    bp_mode: str = "min_sum",
    instability_threshold: float = _INSTABILITY_THRESHOLD,
    optimizer_max_iterations: int = 10,
) -> dict[str, Any]:
    """Run spectral optimizer sanity experiment with predictor probe.

    Returns JSON-serializable report with baseline, optimized, optimizer,
    structure, and predictor_probe sections.
    """
    H_original = np.array(H, dtype=np.float64)

    # Predictor evaluation (original graph).
    pred_original = _predict_instability(H_original)

    # Structural instability (original).
    initial_score = _compute_graph_instability_score(H_original)["instability_score"]

    # Baseline decode.
    baseline = _decode_trials(
        H_original, instances, decode_fn, syndrome_fn,
        max_iters=max_iters, bp_mode=bp_mode,
    )

    # Optimize graph.
    opt_result = run_spectral_graph_optimization(
        H_original, max_iterations=optimizer_max_iterations,
    )
    repair_count = len(opt_result["swaps_applied"])

    # Reconstruct H_optimized by replaying swaps.
    H_optimized = np.array(H_original, dtype=np.float64)
    m, n = H_optimized.shape
    for swap_record in opt_result["swaps_applied"]:
        edges = _extract_edges(H_optimized)
        H_optimized = _edges_to_H(_apply_swap(edges, swap_record["swap"]), m, n)

    # Predictor evaluation (optimized graph).
    pred_optimized = _predict_instability(H_optimized)

    # Structural instability after optimization.
    final_score = _compute_graph_instability_score(H_optimized)["instability_score"]
    delta = round(final_score - initial_score, _ROUND)

    # Decode optimized graph.
    optimized = _decode_trials(
        H_optimized, instances, decode_fn, syndrome_fn,
        max_iters=max_iters, bp_mode=bp_mode,
    )

    return {
        "baseline": {
            "fer": baseline["fer"], "wer": baseline["wer"],
            "avg_iterations": baseline["avg_iterations"],
            "actual_failure": baseline["actual_failure"],
        },
        "optimized": {
            "fer": optimized["fer"], "wer": optimized["wer"],
            "avg_iterations": optimized["avg_iterations"],
            "actual_failure": optimized["actual_failure"],
        },
        "optimizer": {"repair_count": repair_count},
        "structure": {
            "initial_instability_score": round(initial_score, _ROUND),
            "final_instability_score": round(final_score, _ROUND),
            "instability_delta": delta,
        },
        "predictor_probe": {
            "predicted_failure": pred_original["predicted_failure"],
            "actual_failure": baseline["actual_failure"],
            "predicted_failure_optimized": pred_optimized["predicted_failure"],
            "actual_failure_optimized": optimized["actual_failure"],
        },
    }


def print_sanity_report(report: dict[str, Any]) -> None:
    """Print human-readable sanity experiment report."""
    b, o, s = report["baseline"], report["optimized"], report["structure"]
    pp = report["predictor_probe"]
    print()
    print("Spectral Optimizer Sanity + Predictor Probe")
    print("=" * 46)
    print(f"  Baseline")
    print(f"    FER:               {b['fer']}")
    print(f"    WER:               {b['wer']}")
    print(f"    Avg Iterations:    {b['avg_iterations']}")
    print(f"    Predicted Failure: {pp['predicted_failure']}")
    print(f"    Actual Failure:    {b['actual_failure']}")
    print(f"  Optimized")
    print(f"    FER:               {o['fer']}")
    print(f"    WER:               {o['wer']}")
    print(f"    Avg Iterations:    {o['avg_iterations']}")
    print(f"    Predicted Failure: {pp['predicted_failure_optimized']}")
    print(f"    Actual Failure:    {o['actual_failure']}")
    print(f"  Optimizer")
    print(f"    Repairs Applied:   {report['optimizer']['repair_count']}")
    print(f"  Structural Change")
    print(f"    Instability Before: {s['initial_instability_score']}")
    print(f"    Instability After:  {s['final_instability_score']}")
    print(f"    Instability Delta:  {s['instability_delta']}")
    print()
