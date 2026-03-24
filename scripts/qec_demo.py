#!/usr/bin/env python
"""QEC Deterministic Adaptive Pipeline Demo.

Runs the full adaptive control loop on fixed deterministic inputs:

    metrics -> attractor -> strategy -> evaluation -> adaptation -> memory

Demonstrates regime classification, strategy selection, evaluation feedback,
transition learning (v99.3), and multi-step lookahead (v99.4).

All outputs are deterministic — repeated runs produce identical results.

Dependencies: qec package (numpy, scipy).
"""

from __future__ import annotations

import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from typing import Any, Dict, List

from qec.analysis.attractor_analysis import analyze_attractors
from qec.analysis.strategy_evaluation import evaluate_strategy
from qec.analysis.strategy_memory import (
    compute_attractor_id,
    compute_regime_aware_score,
    compute_regime_key,
    update_regime_memory,
)
from qec.analysis.strategy_transition import select_next_strategy
from qec.analysis.strategy_transition_learning import record_transition_outcome
from qec.experiments.metrics_probe import (
    evaluate_metrics,
    generate_mock_strategies,
    generate_test_inputs,
)


def run_demo() -> Dict[str, Any]:
    """Run the full deterministic adaptive pipeline demo.

    Returns structured results for verification.
    """
    inputs = generate_test_inputs()

    # Convert mock strategies to dicts for the transition layer
    raw_strategies = generate_mock_strategies()
    strategies = {
        sid: {
            "action_type": s.action_type,
            "params": dict(s.params),
            "confidence": getattr(s, "confidence", 0.0),
        }
        for sid, s in raw_strategies.items()
    }

    # Pipeline state
    prev_strategy = None
    prev_state = None
    prev_full_metrics = None
    prev_regime = None
    prev_attractor_id = None
    eval_history: List[Dict[str, Any]] = []
    strategy_memory: Dict[Any, List[Dict[str, Any]]] = {}
    transition_memory: Dict[Any, Dict[str, Any]] = {}
    step_counter = 0
    results: List[Dict[str, Any]] = []

    print("=" * 70)
    print("QEC Deterministic Adaptive Pipeline Demo")
    print("=" * 70)
    print()
    print("Pipeline: metrics -> attractor -> strategy -> evaluation"
          " -> adaptation -> memory")
    print()

    for case in inputs:
        name = case["name"]

        # --- Stage 1: Metrics ---
        metrics = evaluate_metrics(case["values"])

        # --- Stage 2: Attractor analysis ---
        attractor = analyze_attractors(metrics)
        regime = attractor["regime"]
        basin_score = attractor["basin_score"]
        attractor_id = compute_attractor_id(basin_score)

        # --- Stage 3: Strategy selection (with adaptation + memory) ---
        full_metrics = {**metrics, "attractor": attractor}
        decision = select_next_strategy(
            full_metrics,
            strategies,
            prev_strategy,
            prev_state,
            history=eval_history if eval_history else None,
            memory=strategy_memory if strategy_memory else None,
            transition_memory=transition_memory if transition_memory else None,
        )
        selected = decision["strategy"]
        selected_id = selected.get("id", "")
        selected_score = selected.get("score", 0.0)

        # --- Stage 4: Evaluation ---
        evaluation = None
        outcome = None
        eval_score = None
        if prev_full_metrics is not None:
            eval_result = evaluate_strategy(
                prev_full_metrics, full_metrics, history=eval_history,
            )
            eval_history = eval_result.get("history", eval_history)
            evaluation = eval_result
            outcome = eval_result["outcome"]
            eval_score = eval_result["evaluation"]["score"]

            # --- Stage 5+6: Transition learning ---
            if prev_regime is not None and prev_attractor_id is not None:
                transition_memory = record_transition_outcome(
                    transition_memory,
                    prev_regime,
                    prev_attractor_id,
                    selected_id,
                    regime,
                    attractor_id,
                    eval_score,
                )

        # --- Stage 6: Memory update ---
        if selected_id and evaluation:
            ev = evaluation.get("evaluation", {})
            rk = compute_regime_key(regime, attractor_id)
            strategy_memory = update_regime_memory(
                strategy_memory,
                rk,
                selected_id,
                {
                    "step": step_counter,
                    "score": ev.get("score", 0.0),
                    "metrics": {"basin_score": basin_score},
                },
            )

        # --- Gather adaptation info ---
        adapt = decision.get("adaptation")
        global_bias = adapt["bias"] if adapt else 0.0
        transition_bias = adapt.get("transition_bias", 1.0) if adapt else 1.0
        multi_step_factor = adapt.get("multi_step_factor", 1.0) if adapt else 1.0

        # --- Regime-aware score for reporting ---
        rk = compute_regime_key(regime, attractor_id)
        regime_score_info = compute_regime_aware_score(
            strategy_memory, rk, selected_id,
        )
        local_bias = max(-0.2, min(0.2, 0.2 * regime_score_info["final_score"]))

        # --- Print step ---
        print(f"[Step {step_counter:2d}] {name}")
        print(f"  regime: {regime:<14s}  attractor: {attractor_id:<8s}"
              f"  basin_score: {basin_score:.4f}")
        print(f"  strategy: {selected_id:<4s}"
              f"  score: {selected_score:.4f}"
              f"  bias: {global_bias:+.4f} (global)"
              f"  {local_bias:+.4f} (local)")
        print(f"  transition_bias: {transition_bias:.4f}"
              f"  multi_step_factor: {multi_step_factor:.4f}")

        if evaluation:
            ev_info = evaluation["evaluation"]
            direction = ev_info["direction"].upper()
            print(f"  eval: {direction:<9s}"
                  f"  score: {eval_score:+.4f}"
                  f"  outcome: {outcome}")

        transition = decision.get("transition")
        if transition:
            print(f"  transition: {transition['from']} -> {transition['to']}"
                  f"  ({transition['change']})")

        print()

        # Store for next iteration
        prev_strategy = selected
        prev_state = decision["state"]
        prev_full_metrics = full_metrics
        prev_regime = regime
        prev_attractor_id = attractor_id
        step_counter += 1

        results.append({
            "name": name,
            "regime": regime,
            "attractor_id": attractor_id,
            "basin_score": basin_score,
            "strategy_id": selected_id,
            "strategy_score": selected_score,
            "eval_score": eval_score,
            "outcome": outcome,
            "transition_bias": transition_bias,
            "multi_step_factor": multi_step_factor,
        })

    # --- Summary ---
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    regime_counts: Dict[str, int] = {}
    for r in results:
        regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1
    print(f"\nRegime distribution ({len(results)} steps):")
    for regime_name in sorted(regime_counts.keys()):
        print(f"  {regime_name:<14s} {regime_counts[regime_name]}")

    print(f"\nMemory keys: {len(strategy_memory)}")
    print(f"Transition memory entries: {len(transition_memory)}")
    print(f"Evaluation history length: {len(eval_history)}")
    print()

    return {
        "steps": results,
        "regime_counts": regime_counts,
        "memory_keys": len(strategy_memory),
        "transition_entries": len(transition_memory),
        "history_length": len(eval_history),
    }


def main() -> int:
    """Entry point."""
    run_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
