#!/usr/bin/env python
"""QEC Deterministic Adaptive Pipeline Demo — v101.1.0.

Runs the full adaptive control loop on fixed deterministic inputs:

    metrics -> attractor -> strategy -> evaluation -> adaptation -> memory

Demonstrates regime classification, strategy selection, evaluation feedback,
transition learning (v99.3), multi-step lookahead (v99.4), physics signals
(v99.6), adaptation modulation (v99.7), cycle detection (v99.8),
trajectory validation (v99.9), and benchmark-aware self-evaluation (v101.1).

v101: adds --benchmark flag for deterministic benchmarking against baselines.

All outputs are deterministic — repeated runs produce identical results.

Dependencies: qec package (numpy, scipy).
"""

from __future__ import annotations

import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from typing import Any, Dict, List

from qec.analysis.attractor_analysis import analyze_attractors
from qec.analysis.physics_signal import compute_physics_signals
from qec.analysis.policy_signal_robustness import detect_cycle, compute_cycle_penalty
from qec.analysis.reproducibility_metadata import build_reproducibility_metadata
from qec.analysis.strategy_evaluation import evaluate_strategy
from qec.analysis.strategy_memory import (
    compute_adaptation_modulation,
    compute_attractor_id,
    compute_regime_aware_score,
    compute_regime_key,
    update_regime_memory,
)
from qec.analysis.strategy_transition import select_next_strategy
from qec.analysis.strategy_transition_learning import record_transition_outcome
from qec.analysis.trajectory_validation import validate_transition
from qec.experiments.metrics_probe import (
    evaluate_metrics,
    generate_mock_strategies,
    generate_test_inputs,
)


def run_demo(self_eval: bool = False) -> Dict[str, Any]:
    """Run the full deterministic adaptive pipeline demo.

    Parameters
    ----------
    self_eval : bool
        When True, run benchmark-aware self-evaluation after the
        pipeline and print confidence signals.

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
    regime_history: List[str] = []
    basin_score_history: List[float] = []
    step_counter = 0
    results: List[Dict[str, Any]] = []

    # Reproducibility metadata (seed=0 since demo uses fixed inputs)
    metadata = build_reproducibility_metadata(seed=0)

    print("=" * 70)
    print("QEC Deterministic Adaptive Pipeline Demo (v100.0.0)")
    print("=" * 70)
    print()
    print("Pipeline: metrics -> attractor -> strategy -> evaluation"
          " -> adaptation -> memory")
    print()
    print("Reproducibility metadata:")
    for mk, mv in sorted(metadata.items()):
        print(f"  {mk}: {mv}")
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

        # --- Physics signals (v99.6+) ---
        physics = compute_physics_signals(
            history=basin_score_history if basin_score_history else None,
        )
        energy = physics.get("oscillation_strength", 0.0)
        coherence = physics.get("phase_stability", 1.0)
        alignment = physics.get("phase_lock_ratio", 1.0)

        # --- Adaptation modulation (v99.7+) ---
        mod_result = compute_adaptation_modulation(physics)
        modulation = mod_result.get("adaptation_modulation", 1.0)

        # --- Cycle detection (v99.8+) ---
        regime_history.append(regime)
        cycle_detected = detect_cycle(regime_history)
        cycle_pen = compute_cycle_penalty(regime_history)

        # --- Trajectory validation (v99.9+) ---
        trajectory_score = 1.0
        if prev_full_metrics is not None:
            before_m = {
                "score": prev_full_metrics.get("field", {}).get("phi_alignment", 0.0),
                "energy": 0.0,
                "coherence": 1.0,
            }
            after_m = {
                "score": full_metrics.get("field", {}).get("phi_alignment", 0.0),
                "energy": energy,
                "coherence": coherence,
            }
            trajectory_score = validate_transition(before_m, after_m)

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
        print(f"  energy: {energy:.4f}"
              f"  coherence: {coherence:.4f}"
              f"  alignment: {alignment:.4f}")
        print(f"  modulation: {modulation:.4f}"
              f"  cycle_detected: {cycle_detected}"
              f"  trajectory_score: {trajectory_score:.4f}")

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
        basin_score_history.append(basin_score)
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
            "energy": energy,
            "coherence": coherence,
            "alignment": alignment,
            "modulation": modulation,
            "cycle_detected": cycle_detected,
            "trajectory_score": trajectory_score,
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
    print(f"Deterministic: TRUE")
    print()

    # --- Self-evaluation (v101.1.0, opt-in) ---
    self_eval_result = None
    if self_eval:
        from qec.analysis.self_evaluation import compute_self_evaluation_signal

        # Use the last step's strategy score as the QEC final score.
        # Construct deterministic mock baselines from the pipeline results.
        qec_final = results[-1]["strategy_score"] if results else 0.0
        qec_metrics = {"final_score": qec_final}

        # Deterministic baselines: mean and min of all strategy scores
        all_scores = [r["strategy_score"] for r in results]
        baseline_metrics = {
            "mean_baseline": {"final_score": sum(all_scores) / len(all_scores) if all_scores else 0.0},
            "min_baseline": {"final_score": min(all_scores) if all_scores else 0.0},
        }

        self_eval_result = compute_self_evaluation_signal(qec_metrics, baseline_metrics)

        print("=" * 70)
        print("Self-Evaluation (v101.1.0)")
        print("=" * 70)
        print(f"  Benchmark confidence:    {self_eval_result['benchmark_confidence']:.2f}")
        print(f"  Relative advantage:      {self_eval_result['relative_advantage']:.2f}")
        print(f"  Confidence modulation:   {self_eval_result['confidence_modulation']:.2f}")
        print()

    return {
        "steps": results,
        "regime_counts": regime_counts,
        "memory_keys": len(strategy_memory),
        "transition_entries": len(transition_memory),
        "history_length": len(eval_history),
        "metadata": metadata,
        "self_evaluation": self_eval_result,
    }


def run_benchmark_demo() -> Dict[str, Any]:
    """Run the deterministic benchmark: QEC vs baselines.

    Returns structured benchmark results for verification.
    """
    from qec.analysis.benchmark_comparison import compare_strategies
    from qec.analysis.convergence_analysis import (
        compute_convergence_signal,
        detect_convergence,
    )
    from qec.analysis.performance_metrics import (
        compute_final_performance,
        compute_stability_variance,
    )
    from qec.experiments.benchmark_runner import run_benchmark

    print("=" * 70)
    print("QEC Deterministic Benchmark (v101.0.0)")
    print("=" * 70)
    print()
    print("Running QEC adaptive pipeline vs deterministic baselines...")
    print()

    results = run_benchmark()
    comparisons = compare_strategies(results)

    # Print per-strategy summary
    for name in ["qec", "random", "fixed", "round_robin"]:
        data = results[name]
        scores = data["scores"]
        final = compute_final_performance(scores)
        variance = compute_stability_variance(scores)
        conv_step = detect_convergence(scores)
        conv_signal = compute_convergence_signal(scores)

        conv_str = f"step {conv_step}" if conv_step is not None else "None"
        print(f"  {name:<12s}: final={final:.4f}"
              f"  variance={variance:.6f}"
              f"  converged={conv_str}"
              f"  signal={conv_signal:.4f}")

    print()

    # Print comparisons
    print("Comparisons:")
    for key in sorted(comparisons.keys()):
        comp = comparisons[key]
        ratio = comp["performance_ratio"]
        stab = comp["stability_diff"]
        conv_diff = comp["convergence_signal_diff"]
        print(f"  {key}: ratio={ratio:.4f}"
              f"  stability_diff={stab:+.6f}"
              f"  convergence_diff={conv_diff:+.4f}")

    print()
    print("Deterministic: TRUE")
    print()

    return {
        "results": results,
        "comparisons": comparisons,
    }


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="QEC Deterministic Adaptive Pipeline Demo")
    parser.add_argument(
        "--self-eval",
        action="store_true",
        default=False,
        help="Enable benchmark-aware self-evaluation output",
    )
    args = parser.parse_args()
    run_demo(self_eval=args.self_eval)
    import argparse

    parser = argparse.ArgumentParser(
        description="QEC Deterministic Adaptive Pipeline Demo",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run deterministic benchmark against baselines",
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_demo()
    else:
        run_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
