"""Deterministic Autonomous Loop Controller (v98.0.0).

Implements a closed-loop system:
    EXPLORE -> META_ANALYZE -> UPDATE -> EXECUTE

Integrates consensus engine, strategy composer, meta-law miner,
and law evolution into a unified deterministic control loop.

Features:
- State machine with deterministic transitions
- Stagnation detection (no improvement over N steps)
- Oscillation detection (repeated variance pattern)
- Fallback to previous stable strategy

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Optional, Tuple

import copy

import numpy as np

from qec.analysis.meta_law_miner import extract_meta_laws
from qec.analysis.strategy_composer import compose_strategies

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

_NORM_DIGITS = 12

# State machine states
STATE_EXPLORE = "EXPLORE"
STATE_META_ANALYZE = "META_ANALYZE"
STATE_UPDATE = "UPDATE"
STATE_EXECUTE = "EXECUTE"

# Valid transitions (deterministic)
_TRANSITIONS = {
    STATE_EXPLORE: STATE_META_ANALYZE,
    STATE_META_ANALYZE: STATE_UPDATE,
    STATE_UPDATE: STATE_EXECUTE,
    STATE_EXECUTE: STATE_EXPLORE,
}

# Detection thresholds
STAGNATION_WINDOW = 3
OSCILLATION_WINDOW = 4
OSCILLATION_THRESHOLD = 0.01


def _norm(x: float) -> float:
    """Normalize a float to fixed precision to avoid drift."""
    return round(float(x), _NORM_DIGITS)


# ---------------------------------------------------------------------------
# STEP 1 — STATE MACHINE
# ---------------------------------------------------------------------------


def next_state(current: str) -> str:
    """Deterministic state transition."""
    if current not in _TRANSITIONS:
        return STATE_EXPLORE
    return _TRANSITIONS[current]


# ---------------------------------------------------------------------------
# STEP 2 — STAGNATION DETECTION
# ---------------------------------------------------------------------------


def detect_stagnation(
    metric_history: List[float],
    window: int = STAGNATION_WINDOW,
) -> bool:
    """Detect if the metric has not improved over the last `window` steps.

    Returns True if the last `window` values show no improvement
    (i.e., max - min < epsilon).
    """
    if len(metric_history) < window:
        return False
    recent = metric_history[-window:]
    spread = max(recent) - min(recent)
    return spread < 1e-10


# ---------------------------------------------------------------------------
# STEP 3 — OSCILLATION DETECTION
# ---------------------------------------------------------------------------


def detect_oscillation(
    metric_history: List[float],
    window: int = OSCILLATION_WINDOW,
    threshold: float = OSCILLATION_THRESHOLD,
) -> bool:
    """Detect oscillation in metric history.

    Oscillation is detected when the sign of consecutive differences
    alternates for the last `window` steps, indicating the metric
    bounces up and down.
    """
    if len(metric_history) < window:
        return False
    recent = metric_history[-window:]
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    if any(abs(d) < threshold for d in diffs):
        return False
    # Check sign alternation
    signs = [1 if d > 0 else -1 for d in diffs]
    for i in range(len(signs) - 1):
        if signs[i] == signs[i + 1]:
            return False
    return True


# ---------------------------------------------------------------------------
# STEP 4 — EVALUATE RESULTS
# ---------------------------------------------------------------------------


def evaluate_results(
    state: Dict[str, Any],
) -> Dict[str, float]:
    """Compute evaluation metrics from state.

    Returns dict with: mean, variance, improvement.
    """
    values = np.array(state.get("values", [0.0]), dtype=np.float64)
    mean = _norm(float(np.mean(values)))
    variance = _norm(float(np.var(values)))
    prev_mean = float(state.get("prev_mean", mean))
    improvement = _norm(mean - prev_mean)
    return {
        "mean": mean,
        "variance": variance,
        "improvement": improvement,
    }


# ---------------------------------------------------------------------------
# STEP 5 — STRATEGY TO DICT CONVERSION
# ---------------------------------------------------------------------------


def _consensus_to_strategy_dicts(
    consensus_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert consensus result to strategy dicts for composer.

    Extracts all scored strategies from the consensus result.
    """
    selected = consensus_result.get("selected")
    if selected is None:
        return []
    all_scores = consensus_result.get("all_scores", {})
    strategy = selected.strategy
    strategies = []
    # Build a strategy dict for the selected strategy
    blue_score = float(selected.scores.get("blue", 0.0))
    strategies.append({
        "action_type": strategy.action_type,
        "params": dict(strategy.params),
        "confidence": blue_score,
        "law_id": strategy.law_id,
    })
    return strategies


# ---------------------------------------------------------------------------
# STEP 6 — UPDATE LAWS FROM META-LAWS
# ---------------------------------------------------------------------------


def update_laws_from_meta(
    laws: List[Any],
    meta_result: Dict[str, Any],
) -> List[Any]:
    """Update law set based on meta-law analysis.

    Currently: filter out laws identified as redundant (keep the first
    in each redundant pair by lexicographic ID ordering).
    Laws are not mutated; a new list is returned.
    """
    redundant_ids: set = set()
    for id_a, id_b in meta_result.get("redundant_pairs", []):
        # Keep the lexicographically smaller ID, mark the other
        if id_a < id_b:
            redundant_ids.add(id_b)
        else:
            redundant_ids.add(id_a)

    return [law for law in laws if law.id not in redundant_ids]


# ---------------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------------


class AutonomousLoop:
    """Deterministic closed-loop controller.

    State machine: EXPLORE -> META_ANALYZE -> UPDATE -> EXECUTE -> EXPLORE

    Each step produces deterministic outputs given identical inputs.
    """

    def __init__(self) -> None:
        self.phase = STATE_EXPLORE
        self.metric_history: List[float] = []
        self.step_count = 0
        self.last_stable_strategy: Optional[List[Dict[str, Any]]] = None

    def step(
        self,
        laws: List[Any],
        state: Dict[str, Any],
        consensus_fn: Any,
    ) -> Dict[str, Any]:
        """Execute one loop iteration.

        Args:
            laws: current law set
            state: current system state (must have 'values' key)
            consensus_fn: callable(laws, state) -> consensus result dict

        Returns dict with:
            - laws: updated law list
            - state: updated state dict
            - metrics: evaluation metrics
            - phase: current phase after step
            - composed_strategy: composite strategy (or None)
            - stagnation: bool
            - oscillation: bool
            - fallback_used: bool
        """
        self.step_count += 1
        fallback_used = False

        # Phase 1: EXPLORE — run consensus to get strategy
        consensus_result = consensus_fn(laws, state)

        # Phase 2: META_ANALYZE — compose strategies
        raw_strategies = _consensus_to_strategy_dicts(consensus_result)
        composed = compose_strategies(raw_strategies) if raw_strategies else []

        # Phase 3: evaluate current state
        eval_metrics = evaluate_results(state)
        self.metric_history.append(eval_metrics["mean"])

        # Phase 4: extract meta-laws and update
        meta_result = extract_meta_laws(laws)
        updated_laws = update_laws_from_meta(laws, meta_result)

        # Detection
        stagnation = detect_stagnation(self.metric_history)
        oscillation = detect_oscillation(self.metric_history)

        # Fallback logic
        if (stagnation or oscillation) and self.last_stable_strategy:
            composed = list(self.last_stable_strategy)
            fallback_used = True
        elif composed and not stagnation and not oscillation:
            self.last_stable_strategy = list(composed)

        # Update state with prev_mean for next iteration
        new_state = dict(state)
        new_state["prev_mean"] = eval_metrics["mean"]

        # Advance phase
        self.phase = next_state(self.phase)

        return {
            "laws": updated_laws,
            "state": new_state,
            "metrics": eval_metrics,
            "phase": self.phase,
            "composed_strategy": composed,
            "stagnation": stagnation,
            "oscillation": oscillation,
            "fallback_used": fallback_used,
            "step_count": self.step_count,
            "meta_laws": meta_result.get("meta_laws", []),
        }
