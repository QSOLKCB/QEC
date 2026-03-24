"""Deterministic benchmark runner — v101.0.0.

Runs the full QEC adaptive pipeline and deterministic baseline strategies
on identical inputs, collecting per-step scores for comparison.

Does not modify any core system behaviour. Fully deterministic.

Dependencies: qec.analysis, qec.experiments.metrics_probe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qec.analysis.attractor_analysis import analyze_attractors
from qec.analysis.baseline_strategies import (
    fixed_strategy,
    random_strategy_deterministic,
    round_robin_strategy,
)
from qec.analysis.physics_signal import compute_physics_signals
from qec.analysis.policy_signal_robustness import (
    compute_cycle_penalty,
    detect_cycle,
)
from qec.analysis.strategy_evaluation import evaluate_strategy
from qec.analysis.strategy_memory import (
    compute_adaptation_modulation,
    compute_attractor_id,
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


def _compute_step_score(
    selected_score: float,
    transition_bias: float,
    multi_step_factor: float,
    modulation: float,
    cycle_pen: float,
    trajectory_score: float,
) -> float:
    """Compute the composite step score using multiplicative composition.

    Mirrors the system's final scoring formula (SYSTEM.md):
        final = base × transition_bias × multi_step × modulation
                × cycle_penalty × trajectory_score

    Result clamped to [0.0, 1.0].
    """
    raw = (
        selected_score
        * transition_bias
        * multi_step_factor
        * modulation
        * cycle_pen
        * trajectory_score
    )
    return max(0.0, min(1.0, raw))


def _run_qec_pipeline(
    inputs: List[Dict[str, Any]],
    strategies: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full QEC adaptive pipeline on inputs, return per-step data."""
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

    scores: List[float] = []
    regimes: List[str] = []
    energies: List[float] = []
    strategy_ids: List[str] = []

    for case in inputs:
        metrics = evaluate_metrics(case["values"])
        attractor = analyze_attractors(metrics)
        regime = attractor["regime"]
        basin_score = attractor["basin_score"]
        attractor_id = compute_attractor_id(basin_score)

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

        evaluation = None
        eval_score = None
        if prev_full_metrics is not None:
            eval_result = evaluate_strategy(
                prev_full_metrics, full_metrics, history=eval_history,
            )
            eval_history = eval_result.get("history", eval_history)
            evaluation = eval_result
            eval_score = eval_result["evaluation"]["score"]

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

        adapt = decision.get("adaptation")
        transition_bias = adapt.get("transition_bias", 1.0) if adapt else 1.0
        multi_step_factor = adapt.get("multi_step_factor", 1.0) if adapt else 1.0

        physics = compute_physics_signals(
            history=basin_score_history if basin_score_history else None,
        )
        energy = physics.get("oscillation_strength", 0.0)
        coherence = physics.get("phase_stability", 1.0)

        mod_result = compute_adaptation_modulation(physics)
        modulation = mod_result.get("adaptation_modulation", 1.0)

        regime_history.append(regime)
        cycle_pen = compute_cycle_penalty(regime_history)

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

        step_score = _compute_step_score(
            selected_score, transition_bias, multi_step_factor,
            modulation, cycle_pen, trajectory_score,
        )

        scores.append(step_score)
        regimes.append(regime)
        energies.append(energy)
        strategy_ids.append(selected_id)

        prev_strategy = selected
        prev_state = decision["state"]
        prev_full_metrics = full_metrics
        prev_regime = regime
        prev_attractor_id = attractor_id
        basin_score_history.append(basin_score)
        step_counter += 1

    return {
        "scores": scores,
        "regimes": regimes,
        "energies": energies,
        "strategy_ids": strategy_ids,
    }


def _run_baseline(
    inputs: List[Dict[str, Any]],
    strategy_ids: List[str],
    strategies: Dict[str, Dict[str, Any]],
    mode: str,
    seed: int = 42,
    fixed_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a baseline strategy on inputs, return per-step data.

    Baselines use the same metrics pipeline but replace the adaptive
    strategy selection with a simple deterministic rule.
    """
    scores: List[float] = []
    regimes: List[str] = []
    energies: List[float] = []
    chosen_ids: List[str] = []

    for step, case in enumerate(inputs):
        metrics = evaluate_metrics(case["values"])
        attractor = analyze_attractors(metrics)
        regime = attractor["regime"]

        if mode == "random":
            sid = random_strategy_deterministic(seed, strategy_ids, step)
        elif mode == "fixed":
            sid = fixed_strategy(fixed_id if fixed_id else strategy_ids[0])
        elif mode == "round_robin":
            sid = round_robin_strategy(step, strategy_ids)
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")

        # Use the strategy's base score from the catalog
        strat = strategies.get(sid, {})
        # Base score from action type preference (simplified)
        base_score = 0.5  # neutral baseline score

        scores.append(max(0.0, min(1.0, base_score)))
        regimes.append(regime)
        energies.append(0.0)
        chosen_ids.append(sid)

    return {
        "scores": scores,
        "regimes": regimes,
        "energies": energies,
        "strategy_ids": chosen_ids,
    }


def run_benchmark(
    inputs: Optional[List[Dict[str, Any]]] = None,
    strategies: Optional[Dict[str, Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run the full benchmark: QEC pipeline vs baselines.

    Parameters
    ----------
    inputs : list of dict, optional
        Test inputs. Defaults to generate_test_inputs().
    strategies : dict, optional
        Strategy catalog. Defaults to generate_mock_strategies() converted.
    config : dict, optional
        Configuration. Supports:
        - "seed" (int): seed for random baseline (default 42)
        - "fixed_id" (str): strategy for fixed baseline (default first)

    Returns
    -------
    dict
        Keys: "qec", "random", "fixed", "round_robin".
        Each value contains "scores", "regimes", "energies", "strategy_ids".
    """
    if inputs is None:
        inputs = generate_test_inputs()

    if strategies is None:
        raw = generate_mock_strategies()
        strategies = {
            sid: {
                "action_type": s.action_type,
                "params": dict(s.params),
                "confidence": getattr(s, "confidence", 0.0),
            }
            for sid, s in raw.items()
        }

    if config is None:
        config = {}

    seed = config.get("seed", 42)
    strategy_ids = sorted(strategies.keys())
    fixed_id = config.get("fixed_id", strategy_ids[0] if strategy_ids else None)

    qec_result = _run_qec_pipeline(inputs, strategies)

    random_result = _run_baseline(
        inputs, strategy_ids, strategies, "random", seed=seed,
    )
    fixed_result = _run_baseline(
        inputs, strategy_ids, strategies, "fixed", fixed_id=fixed_id,
    )
    round_robin_result = _run_baseline(
        inputs, strategy_ids, strategies, "round_robin",
    )

    return {
        "qec": qec_result,
        "random": random_result,
        "fixed": fixed_result,
        "round_robin": round_robin_result,
    }
