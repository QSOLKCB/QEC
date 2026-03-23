"""Deterministic Multi-Strategy Consensus Engine (v97.9.0).

Evaluates multiple competing strategies derived from laws, assigns
structured perspectives (De Bono thinking-hats style), resolves
conflicts using Byzantine-inspired consensus logic, and produces
a unified, stable strategy.

Pipeline: laws -> strategies -> hat scores -> consensus matrix ->
          Byzantine filter -> consensus selection -> unified strategy

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qec.analysis.strategy_topology import compute_strategy_topology


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Byzantine filter: strategy must win at least this fraction of comparisons
BYZANTINE_WIN_RATIO = 0.5
# Minimum confidence for a strategy to survive filtering
BYZANTINE_MIN_CONFIDENCE = 0.1
# Hat weights for blue-hat meta-score
HAT_WEIGHTS: Dict[str, float] = {
    "white": 0.20,
    "red": 0.15,
    "black": 0.20,
    "yellow": 0.20,
    "green": 0.10,
    "blue": 0.15,
}
HAT_NAMES: List[str] = ["white", "red", "black", "yellow", "green", "blue"]


# ---------------------------------------------------------------------------
# STEP 1 — STRATEGY EXTRACTION
# ---------------------------------------------------------------------------

# Reuse: maps law.action -> (action_type, default_params)
_ACTION_MAP: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "reduce_oscillation": ("adjust_damping", {"alpha": 0.5}),
    "stabilize": ("schedule_update", {"mode": "sequential"}),
    "increase_damping": ("adjust_damping", {"alpha": 0.3}),
    "decrease_damping": ("adjust_damping", {"alpha": 0.8}),
    "reweight_high": ("reweight_messages", {"weight": 1.5}),
    "reweight_low": ("reweight_messages", {"weight": 0.5}),
    "freeze_unstable": ("freeze_nodes", {"threshold": 0.1}),
    "sequential_schedule": ("schedule_update", {"mode": "sequential"}),
    "parallel_schedule": ("schedule_update", {"mode": "parallel"}),
    "hard_correction": ("correction_mode", {"mode": "hard"}),
    "soft_correction": ("correction_mode", {"mode": "soft"}),
    "clamp_correction": ("correction_mode", {"mode": "clamp"}),
}


def _law_confidence(law: Any) -> float:
    """Extract confidence from law scores."""
    return float(law.scores.get("confidence", law.scores.get("law_score", 0.0)))


def _law_specificity(law: Any) -> int:
    """Number of conditions on a law."""
    return law.condition_count()


def _map_law_to_action(law: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Map a law action string to (action_type, params)."""
    if law.action not in _ACTION_MAP:
        return None
    action_type, params = _ACTION_MAP[law.action]
    return (action_type, dict(params))


class Strategy:
    """A single strategy derived from one law."""

    __slots__ = ("law_id", "law", "action_type", "params")

    def __init__(
        self,
        law_id: str,
        law: Any,
        action_type: str,
        params: Dict[str, Any],
    ) -> None:
        self.law_id = law_id
        self.law = law
        self.action_type = action_type
        self.params = dict(params)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law_id": self.law_id,
            "action_type": self.action_type,
            "params": dict(self.params),
        }


def extract_strategies(
    laws: List[Any], metrics: Dict[str, float]
) -> Dict[str, Strategy]:
    """Extract applicable strategies from laws given current metrics.

    Returns {law_id: Strategy} for laws whose conditions are satisfied
    and whose actions map to known action types.
    """
    strategies: Dict[str, Strategy] = {}
    for law in laws:
        if not law.evaluate(metrics):
            continue
        mapping = _map_law_to_action(law)
        if mapping is None:
            continue
        action_type, params = mapping
        strategies[law.id] = Strategy(
            law_id=law.id,
            law=law,
            action_type=action_type,
            params=params,
        )
    return strategies


# ---------------------------------------------------------------------------
# STEP 2 — THINKING HATS (PERSPECTIVES)
# ---------------------------------------------------------------------------


def _clamp(x: float) -> float:
    """Clamp to [0, 1]."""
    return max(0.0, min(1.0, float(x)))


def white_hat(strategy: Strategy, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Data perspective: confidence and coverage."""
    confidence = _law_confidence(strategy.law)
    coverage = float(strategy.law.scores.get("coverage", 0.0))
    score = _clamp(0.5 * confidence + 0.5 * coverage)
    return {"score": score, "reason": f"confidence={confidence:.3f}, coverage={coverage:.3f}"}


def red_hat(strategy: Strategy, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Instability perspective: variance and oscillation risk.

    High variance or high delta -> low score (more instability).
    Score is inverted: 1.0 = stable, 0.0 = highly unstable.
    """
    variance = metrics.get("variance", 0.0)
    delta = metrics.get("delta", 0.0)
    # Use sigmoid-like transform: 1 / (1 + x) to map [0, inf) -> (0, 1]
    var_score = 1.0 / (1.0 + variance)
    delta_score = 1.0 / (1.0 + delta)
    score = _clamp(0.5 * var_score + 0.5 * delta_score)
    return {"score": score, "reason": f"var_stability={var_score:.3f}, delta_stability={delta_score:.3f}"}


def black_hat(strategy: Strategy, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Risk perspective: conflict potential and low confidence.

    Low confidence or few conditions -> higher risk -> lower score.
    """
    confidence = _law_confidence(strategy.law)
    specificity = _law_specificity(strategy.law)
    # More conditions = more specific = lower risk of false positive
    spec_score = _clamp(1.0 - 1.0 / (1.0 + specificity))
    score = _clamp(0.5 * confidence + 0.5 * spec_score)
    return {"score": score, "reason": f"confidence={confidence:.3f}, specificity_score={spec_score:.3f}"}


def yellow_hat(strategy: Strategy, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Benefit perspective: stability gain potential.

    High variance state + stabilizing action = high benefit.
    """
    variance = metrics.get("variance", 0.0)
    # Strategies acting on high-variance states have more benefit potential
    benefit_potential = _clamp(1.0 - 1.0 / (1.0 + variance))
    law_score = float(strategy.law.scores.get("law_score", 0.0))
    score = _clamp(0.5 * benefit_potential + 0.5 * law_score)
    return {"score": score, "reason": f"benefit_potential={benefit_potential:.3f}, law_score={law_score:.3f}"}


def green_hat(strategy: Strategy, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Novelty perspective: how unexplored this action region is.

    Proxied by inverse specificity — fewer conditions means broader
    applicability, which correlates with less-explored territory.
    """
    specificity = _law_specificity(strategy.law)
    # Fewer conditions = broader scope = potentially more novel
    novelty = _clamp(1.0 / (1.0 + specificity))
    coverage = float(strategy.law.scores.get("coverage", 0.0))
    # Low coverage also suggests novelty (less frequently triggered)
    coverage_novelty = _clamp(1.0 - coverage)
    score = _clamp(0.5 * novelty + 0.5 * coverage_novelty)
    return {"score": score, "reason": f"novelty={novelty:.3f}, coverage_novelty={coverage_novelty:.3f}"}


def blue_hat(
    strategy: Strategy,
    metrics: Dict[str, float],
    other_scores: Dict[str, float],
) -> Dict[str, Any]:
    """Control perspective: weighted meta-score across other hats."""
    total = 0.0
    for hat_name in ["white", "red", "black", "yellow", "green"]:
        total += HAT_WEIGHTS[hat_name] * other_scores.get(hat_name, 0.0)
    # Normalize by sum of non-blue weights
    non_blue_weight = sum(HAT_WEIGHTS[h] for h in ["white", "red", "black", "yellow", "green"])
    score = _clamp(total / non_blue_weight) if non_blue_weight > 0 else 0.0
    return {"score": score, "reason": f"meta_score={score:.3f}"}


_HAT_EVALUATORS = {
    "white": white_hat,
    "red": red_hat,
    "black": black_hat,
    "yellow": yellow_hat,
    "green": green_hat,
}


# ---------------------------------------------------------------------------
# STEP 3 — STRATEGY SCORING
# ---------------------------------------------------------------------------


def score_strategy(
    strategy: Strategy, metrics: Dict[str, float]
) -> Dict[str, float]:
    """Score a strategy across all six thinking hats.

    Returns {hat_name: score} with all scores in [0, 1].
    """
    scores: Dict[str, float] = {}
    for hat_name, evaluator in sorted(_HAT_EVALUATORS.items()):
        result = evaluator(strategy, metrics)
        scores[hat_name] = float(result["score"])
    # Blue hat uses the other scores
    blue_result = blue_hat(strategy, metrics, scores)
    scores["blue"] = float(blue_result["score"])
    return scores


def score_strategy_detailed(
    strategy: Strategy, metrics: Dict[str, float]
) -> Dict[str, Dict[str, Any]]:
    """Score a strategy with full reasoning details.

    Returns {hat_name: {"score": float, "reason": str}}.
    """
    details: Dict[str, Dict[str, Any]] = {}
    partial_scores: Dict[str, float] = {}
    for hat_name, evaluator in sorted(_HAT_EVALUATORS.items()):
        result = evaluator(strategy, metrics)
        details[hat_name] = result
        partial_scores[hat_name] = float(result["score"])
    blue_result = blue_hat(strategy, metrics, partial_scores)
    details["blue"] = blue_result
    return details


# ---------------------------------------------------------------------------
# STEP 4 — CONSENSUS MATRIX
# ---------------------------------------------------------------------------


def build_consensus_matrix(
    strategy_ids: List[str],
    strategy_scores: Dict[str, Dict[str, float]],
    strategies: Dict[str, Strategy],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build pairwise comparison matrix between strategies.

    For each pair (i, j), compute:
    - agreement: 1.0 if same action_type, 0.0 otherwise
    - conflict: 1.0 if different action_type, 0.0 otherwise
    - dominance: fraction of hats where i scores higher than j

    Returns {id_i: {id_j: {"agreement": f, "conflict": f, "dominance": f}}}.
    """
    matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
    for id_i in strategy_ids:
        matrix[id_i] = {}
        scores_i = strategy_scores[id_i]
        strat_i = strategies[id_i]
        for id_j in strategy_ids:
            if id_i == id_j:
                matrix[id_i][id_j] = {
                    "agreement": 1.0,
                    "conflict": 0.0,
                    "dominance": 0.5,
                }
                continue
            scores_j = strategy_scores[id_j]
            strat_j = strategies[id_j]
            # Agreement / conflict based on action type
            same_action = 1.0 if strat_i.action_type == strat_j.action_type else 0.0
            # Dominance: fraction of hats where i > j
            wins = 0
            total_hats = len(HAT_NAMES)
            for hat in HAT_NAMES:
                si = scores_i.get(hat, 0.0)
                sj = scores_j.get(hat, 0.0)
                if si > sj:
                    wins += 1
            dominance = float(wins) / float(total_hats) if total_hats > 0 else 0.5
            matrix[id_i][id_j] = {
                "agreement": same_action,
                "conflict": 1.0 - same_action,
                "dominance": dominance,
            }
    return matrix


# ---------------------------------------------------------------------------
# STEP 5 — BYZANTINE FILTER
# ---------------------------------------------------------------------------


def filter_strategies(
    strategy_ids: List[str],
    strategy_scores: Dict[str, Dict[str, float]],
    consensus_matrix: Dict[str, Dict[str, Dict[str, float]]],
    strategies: Dict[str, Strategy],
) -> List[str]:
    """Remove strategies that are Byzantine-faulty.

    A strategy is removed if:
    1. It conflicts with the majority (more conflict than agreement), OR
    2. Its average confidence is below BYZANTINE_MIN_CONFIDENCE, OR
    3. It loses more comparisons than it wins (dominance < 0.5 majority).

    Returns surviving strategy IDs in deterministic order.
    """
    if not strategy_ids:
        return []

    survivors: List[str] = []
    n = len(strategy_ids)

    for sid in strategy_ids:
        # Check 1: majority conflict
        if n > 1:
            total_conflict = 0.0
            total_agreement = 0.0
            for other_id in strategy_ids:
                if other_id == sid:
                    continue
                cell = consensus_matrix[sid][other_id]
                total_conflict += cell["conflict"]
                total_agreement += cell["agreement"]
            # If more conflict than agreement, mark as faulty
            if total_conflict > total_agreement and total_agreement + total_conflict > 0:
                continue

        # Check 2: minimum confidence
        confidence = _law_confidence(strategies[sid].law)
        if confidence < BYZANTINE_MIN_CONFIDENCE:
            continue

        # Check 3: win ratio
        if n > 1:
            total_dominance = 0.0
            comparisons = 0
            for other_id in strategy_ids:
                if other_id == sid:
                    continue
                total_dominance += consensus_matrix[sid][other_id]["dominance"]
                comparisons += 1
            avg_dominance = total_dominance / comparisons if comparisons > 0 else 0.5
            if avg_dominance < BYZANTINE_WIN_RATIO:
                continue

        survivors.append(sid)

    # If all filtered out, keep the single best by average hat score
    if not survivors and strategy_ids:
        best_id = max(
            strategy_ids,
            key=lambda sid: (
                sum(strategy_scores[sid].get(h, 0.0) for h in HAT_NAMES) / len(HAT_NAMES),
                sid,  # deterministic tiebreaker
            ),
        )
        survivors = [best_id]

    return survivors


# ---------------------------------------------------------------------------
# STEP 6 — CONSENSUS SELECTION
# ---------------------------------------------------------------------------


def _average_hat_score(scores: Dict[str, float]) -> float:
    """Compute average score across all hats."""
    total = sum(scores.get(h, 0.0) for h in HAT_NAMES)
    return total / len(HAT_NAMES)


def _agreement_score(
    sid: str,
    strategy_ids: List[str],
    consensus_matrix: Dict[str, Dict[str, Dict[str, float]]],
) -> float:
    """Total agreement of sid with other strategies."""
    total = 0.0
    for other_id in strategy_ids:
        if other_id == sid:
            continue
        total += consensus_matrix[sid][other_id]["agreement"]
    return total


def select_consensus(
    survivor_ids: List[str],
    strategy_scores: Dict[str, Dict[str, float]],
    consensus_matrix: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    """Select the consensus strategy from survivors.

    Maximize: average score across all hats.
    Tiebreak 1: agreement with others.
    Tiebreak 2: lexicographic ID.

    Returns the selected strategy ID.
    """
    if not survivor_ids:
        raise ValueError("No strategies to select from")
    if len(survivor_ids) == 1:
        return survivor_ids[0]

    def sort_key(sid: str) -> Tuple[float, float, str]:
        avg = _average_hat_score(strategy_scores[sid])
        agreement = _agreement_score(sid, survivor_ids, consensus_matrix)
        # Negate for descending, use id for ascending tiebreak
        return (-avg, -agreement, sid)

    ranked = sorted(survivor_ids, key=sort_key)
    return ranked[0]


# ---------------------------------------------------------------------------
# STEP 7 — CONSENSUS STRATEGY OBJECT
# ---------------------------------------------------------------------------


class ConsensusStrategy:
    """A consensus-selected strategy with full reasoning trace."""

    def __init__(
        self,
        strategy: Strategy,
        scores: Dict[str, float],
        supporters: List[str],
    ) -> None:
        self.strategy = strategy
        self.scores = dict(scores)
        self.supporters = list(supporters)

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the selected strategy action to state.

        Returns a new state dict (no mutation of input).
        """
        values = np.array(state["values"], dtype=np.float64)
        action = self.strategy.action_type
        params = self.strategy.params

        if action == "adjust_damping":
            new_values = values * float(params.get("alpha", 1.0))
        elif action == "reweight_messages":
            new_values = values * float(params.get("weight", 1.0))
        elif action == "freeze_nodes":
            threshold = float(params.get("threshold", 0.1))
            mask = np.abs(values) < threshold
            new_values = values.copy()
            new_values[mask] = 0.0
        elif action == "schedule_update":
            mode = params.get("mode", "parallel")
            if mode == "sequential":
                new_values = np.cumsum(values) / np.arange(1, len(values) + 1)
            else:
                new_values = values.copy()
        elif action == "correction_mode":
            mode = params.get("mode", "soft")
            if mode == "hard":
                new_values = np.sign(values)
            elif mode == "soft":
                new_values = np.tanh(values)
            elif mode == "clamp":
                new_values = np.clip(values, -1.0, 1.0)
            else:
                new_values = values.copy()
        else:
            new_values = values.copy()

        return {"values": new_values, "step": state.get("step", 0)}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for inspection."""
        return {
            "selected_law_id": self.strategy.law_id,
            "action_type": self.strategy.action_type,
            "params": dict(self.strategy.params),
            "scores": dict(self.scores),
            "supporters": list(self.supporters),
        }

    def __repr__(self) -> str:
        return (
            f"ConsensusStrategy(law_id={self.strategy.law_id!r}, "
            f"action={self.strategy.action_type!r})"
        )


# ---------------------------------------------------------------------------
# STEP 8 — INTEGRATION HOOK
# ---------------------------------------------------------------------------


def extract_metrics(state: Dict[str, Any]) -> Dict[str, float]:
    """Compute deterministic metrics from state."""
    values = np.array(state["values"], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "variance": float(np.var(values)),
        "delta": 0.0,
    }


def run_consensus(
    laws: List[Any], state: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the full deterministic consensus pipeline.

    Pipeline:
    1. Extract metrics from state
    2. Extract strategies from laws
    3. Score each strategy across all hats
    4. Build pairwise consensus matrix
    5. Byzantine-filter faulty strategies
    6. Select consensus winner
    7. Build ConsensusStrategy object

    Returns:
    - selected: ConsensusStrategy (or None if no applicable strategies)
    - all_scores: {law_id: {hat: score}}
    - reasoning: list of trace entries
    """
    reasoning: List[Dict[str, Any]] = []

    # Step 1: metrics
    metrics = extract_metrics(state)
    reasoning.append({"step": "extract_metrics", "metrics": dict(metrics)})

    # Step 2: strategies
    strategies = extract_strategies(laws, metrics)
    if not strategies:
        reasoning.append({"step": "extract_strategies", "count": 0})
        return {"selected": None, "all_scores": {}, "reasoning": reasoning}
    strategy_ids = sorted(strategies.keys())
    reasoning.append({"step": "extract_strategies", "count": len(strategies), "ids": list(strategy_ids)})

    # Step 3: score
    all_scores: Dict[str, Dict[str, float]] = {}
    for sid in strategy_ids:
        all_scores[sid] = score_strategy(strategies[sid], metrics)
    reasoning.append({"step": "score_strategies", "scores": {k: dict(v) for k, v in all_scores.items()}})

    # Step 4: consensus matrix
    matrix = build_consensus_matrix(strategy_ids, all_scores, strategies)
    reasoning.append({"step": "build_matrix", "size": len(strategy_ids)})

    # Step 5: Byzantine filter
    survivors = filter_strategies(strategy_ids, all_scores, matrix, strategies)
    reasoning.append({"step": "byzantine_filter", "survivors": list(survivors), "removed": len(strategy_ids) - len(survivors)})

    # Step 6: select
    winner_id = select_consensus(survivors, all_scores, matrix)
    reasoning.append({"step": "select_consensus", "winner": winner_id})

    # Step 7: build object
    winner_strategy = strategies[winner_id]
    winner_scores = all_scores[winner_id]
    # Supporters: strategies with same action type
    supporters = [
        sid for sid in survivors
        if strategies[sid].action_type == winner_strategy.action_type
    ]
    consensus = ConsensusStrategy(
        strategy=winner_strategy,
        scores=winner_scores,
        supporters=sorted(supporters),
    )

    # Step 8: strategy topology
    topology = compute_strategy_topology(strategies)
    reasoning.append({"step": "strategy_topology", "dominant": topology.get("dominant", "")})

    return {
        "selected": consensus,
        "all_scores": all_scores,
        "reasoning": reasoning,
        "topology": topology,
    }
