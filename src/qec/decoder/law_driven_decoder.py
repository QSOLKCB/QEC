"""Deterministic Law-Driven Decoder Engine (v97.8.0).

Converts laws into executable control actions, composes multiple laws
into a coherent strategy, and applies actions to state iteratively.

Pipeline: laws -> actions -> strategy -> behavior

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS = 50
CONVERGENCE_VARIANCE_THRESHOLD = 1e-6
OSCILLATION_WINDOW = 5

# ---------------------------------------------------------------------------
# ACTION MAPPING TABLE — explicit, no inference
# ---------------------------------------------------------------------------

ACTION_MAP: Dict[str, Tuple[str, Dict[str, Any]]] = {
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


# ---------------------------------------------------------------------------
# STEP 1 — ACTION PRIMITIVES (pure, return new state)
# ---------------------------------------------------------------------------


def adjust_damping(state: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    """Scale values by damping factor alpha. Returns new state."""
    values = np.array(state["values"], dtype=np.float64)
    new_values = values * float(alpha)
    return {"values": new_values, "step": state["step"]}


def reweight_messages(state: Dict[str, Any], weight: float) -> Dict[str, Any]:
    """Multiply values by weight factor. Returns new state."""
    values = np.array(state["values"], dtype=np.float64)
    new_values = values * float(weight)
    return {"values": new_values, "step": state["step"]}


def freeze_nodes(state: Dict[str, Any], mask: Any) -> Dict[str, Any]:
    """Zero out values at masked indices. Returns new state."""
    values = np.array(state["values"], dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    new_values = values.copy()
    new_values[mask_arr] = 0.0
    return {"values": new_values, "step": state["step"]}


def schedule_update(state: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Apply scheduling effect to values. Returns new state."""
    values = np.array(state["values"], dtype=np.float64)
    if mode == "sequential":
        # Sequential: apply cumulative mean smoothing
        new_values = np.cumsum(values) / np.arange(1, len(values) + 1)
    else:
        # Parallel (default): no reordering effect
        new_values = values.copy()
    return {"values": new_values, "step": state["step"]}


def correction_mode(state: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Apply correction transform to values. Returns new state."""
    values = np.array(state["values"], dtype=np.float64)
    if mode == "hard":
        new_values = np.sign(values)
    elif mode == "soft":
        new_values = np.tanh(values)
    elif mode == "clamp":
        new_values = np.clip(values, -1.0, 1.0)
    else:
        new_values = values.copy()
    return {"values": new_values, "step": state["step"]}


# Action dispatch table
_ACTION_DISPATCH = {
    "adjust_damping": lambda state, params: adjust_damping(state, params["alpha"]),
    "reweight_messages": lambda state, params: reweight_messages(state, params["weight"]),
    "freeze_nodes": lambda state, params: freeze_nodes(state, params["mask"]),
    "schedule_update": lambda state, params: schedule_update(state, params["mode"]),
    "correction_mode": lambda state, params: correction_mode(state, params["mode"]),
}


# ---------------------------------------------------------------------------
# STEP 2 — LAW -> ACTION MAPPING
# ---------------------------------------------------------------------------


def map_law_to_action(law: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Map a law's action string to (action_type, parameters).

    Uses explicit ACTION_MAP table. Returns None if action not mapped.
    """
    action_str = law.action
    if action_str not in ACTION_MAP:
        return None
    action_type, params = ACTION_MAP[action_str]
    return (action_type, dict(params))


# ---------------------------------------------------------------------------
# STEP 3 — APPLICABLE LAWS
# ---------------------------------------------------------------------------


def get_applicable_laws(laws: List[Any], metrics: Dict[str, float]) -> List[Any]:
    """Return laws where all conditions are satisfied.

    Uses law.evaluate(metrics) which checks all Condition objects.
    Returns laws in original order (stable).
    """
    return [law for law in laws if law.evaluate(metrics)]


# ---------------------------------------------------------------------------
# STEP 4 — ACTION AGGREGATION
# ---------------------------------------------------------------------------


def aggregate_actions(
    laws: List[Any],
) -> Dict[str, List[Tuple[Any, Dict[str, Any]]]]:
    """Group mapped actions by action_type.

    Returns {action_type: [(law, params), ...]}.
    Laws with unmapped actions are skipped.
    """
    groups: Dict[str, List[Tuple[Any, Dict[str, Any]]]] = {}
    for law in laws:
        mapping = map_law_to_action(law)
        if mapping is None:
            continue
        action_type, params = mapping
        if action_type not in groups:
            groups[action_type] = []
        groups[action_type].append((law, params))
    return groups


# ---------------------------------------------------------------------------
# STEP 5 — CONFLICT RESOLUTION
# ---------------------------------------------------------------------------


def _law_confidence(law: Any) -> float:
    """Extract confidence score from law, defaulting to 0.0."""
    return float(law.scores.get("confidence", law.scores.get("law_score", 0.0)))


def _law_specificity(law: Any) -> int:
    """Number of conditions (higher = more specific)."""
    return law.condition_count()


def resolve_conflicts(
    groups: Dict[str, List[Tuple[Any, Dict[str, Any]]]],
) -> Dict[str, Dict[str, Any]]:
    """For each action_type, select ONE action deterministically.

    Resolution order:
    1. Highest confidence
    2. Highest specificity (condition count)
    3. Lexicographic law.id (tiebreaker)

    Returns {action_type: params}.
    """
    resolved: Dict[str, Dict[str, Any]] = {}
    for action_type, candidates in sorted(groups.items()):
        if not candidates:
            continue
        # Sort by (-confidence, -specificity, law.id) for determinism
        ranked = sorted(
            candidates,
            key=lambda lp: (-_law_confidence(lp[0]), -_law_specificity(lp[0]), lp[0].id),
        )
        _winner_law, winner_params = ranked[0]
        resolved[action_type] = dict(winner_params)
    return resolved


# ---------------------------------------------------------------------------
# STEP 6 — STRATEGY OBJECT
# ---------------------------------------------------------------------------


class DecoderStrategy:
    """An inspectable, serializable collection of resolved actions."""

    def __init__(self, actions: Dict[str, Dict[str, Any]]) -> None:
        self.actions = dict(actions)

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all actions to state deterministically.

        Actions are applied in sorted action_type order for determinism.
        Returns a new state dict (no mutation of input).
        """
        current = {"values": np.array(state["values"], dtype=np.float64), "step": state["step"]}
        for action_type in sorted(self.actions.keys()):
            params = self.actions[action_type]
            if action_type in _ACTION_DISPATCH:
                current = _ACTION_DISPATCH[action_type](current, params)
        return current

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for inspection."""
        return {"actions": {k: dict(v) for k, v in sorted(self.actions.items())}}

    def __repr__(self) -> str:
        return f"DecoderStrategy(actions={self.actions!r})"


# ---------------------------------------------------------------------------
# STEP 7 — SIMPLE STATE MODEL (helpers)
# ---------------------------------------------------------------------------


def make_state(values: Any, step: int = 0) -> Dict[str, Any]:
    """Create a canonical state dict."""
    return {"values": np.array(values, dtype=np.float64), "step": int(step)}


# ---------------------------------------------------------------------------
# STEP 9 — METRIC EXTRACTION
# ---------------------------------------------------------------------------


def extract_metrics(
    state: Dict[str, Any], prev_state: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """Compute simple deterministic metrics from state.

    Metrics:
    - mean: mean of values
    - variance: variance of values
    - delta: L2 norm of change from previous state (0.0 if no prev)
    """
    values = np.array(state["values"], dtype=np.float64)
    metrics: Dict[str, float] = {
        "mean": float(np.mean(values)),
        "variance": float(np.var(values)),
    }
    if prev_state is not None:
        prev_values = np.array(prev_state["values"], dtype=np.float64)
        diff = values - prev_values
        metrics["delta"] = float(np.sqrt(np.sum(diff * diff)))
    else:
        metrics["delta"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# STEP 10 — EVALUATION
# ---------------------------------------------------------------------------


def detect_oscillation(trajectory: List[Dict[str, Any]], window: int = OSCILLATION_WINDOW) -> bool:
    """Detect oscillation by checking if variance alternates up/down.

    Looks at the last `window` states. Returns True if variance
    direction alternates for all consecutive pairs in the window.
    """
    if len(trajectory) < window:
        return False
    recent = trajectory[-window:]
    variances = [float(np.var(s["values"])) for s in recent]
    directions = []
    for i in range(1, len(variances)):
        diff = variances[i] - variances[i - 1]
        if diff > 0:
            directions.append(1)
        elif diff < 0:
            directions.append(-1)
        else:
            directions.append(0)
    # Oscillation: all consecutive pairs alternate sign
    if len(directions) < 2:
        return False
    for i in range(1, len(directions)):
        if directions[i] == 0 or directions[i - 1] == 0:
            return False
        if directions[i] == directions[i - 1]:
            return False
    return True


def evaluate_run(
    trajectory: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate a decoder run for convergence and stability.

    Returns:
    - converged: bool (final variance below threshold)
    - steps_to_convergence: int or None
    - final_variance: float
    - oscillating: bool
    """
    if not trajectory:
        return {
            "converged": False,
            "steps_to_convergence": None,
            "final_variance": float("inf"),
            "oscillating": False,
        }

    variances = [float(np.var(s["values"])) for s in trajectory]
    final_variance = variances[-1]
    converged = final_variance < CONVERGENCE_VARIANCE_THRESHOLD

    steps_to_convergence = None
    for i, v in enumerate(variances):
        if v < CONVERGENCE_VARIANCE_THRESHOLD:
            steps_to_convergence = i
            break

    oscillating = detect_oscillation(trajectory)

    return {
        "converged": converged,
        "steps_to_convergence": steps_to_convergence,
        "final_variance": final_variance,
        "oscillating": oscillating,
    }


# ---------------------------------------------------------------------------
# STEP 8 — DECODER LOOP
# ---------------------------------------------------------------------------


def build_strategy(laws: List[Any], metrics: Dict[str, float]) -> DecoderStrategy:
    """Build a strategy from laws and current metrics.

    Pipeline: applicable laws -> map to actions -> aggregate -> resolve -> strategy
    """
    applicable = get_applicable_laws(laws, metrics)
    groups = aggregate_actions(applicable)
    resolved = resolve_conflicts(groups)
    return DecoderStrategy(resolved)


def run_decoder(
    laws: List[Any],
    initial_state: Dict[str, Any],
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Dict[str, Any]:
    """Run the full deterministic decoder loop.

    Loop:
    1. Extract metrics from current state
    2. Get applicable laws
    3. Map laws to actions
    4. Resolve conflicts
    5. Build strategy
    6. Apply strategy to state

    Returns:
    - final_state: last state
    - trajectory: list of all states
    - strategies: list of strategies applied per step
    - evaluation: convergence/stability evaluation
    """
    current = make_state(initial_state["values"], initial_state.get("step", 0))
    trajectory: List[Dict[str, Any]] = [current]
    strategies: List[Dict[str, Any]] = []
    prev_state: Optional[Dict[str, Any]] = None

    for step in range(max_steps):
        metrics = extract_metrics(current, prev_state)
        strategy = build_strategy(laws, metrics)
        strategies.append(strategy.to_dict())

        if not strategy.actions:
            # No applicable actions — stop early
            break

        prev_state = current
        new_state = strategy.apply(current)
        new_state["step"] = step + 1
        current = new_state
        trajectory.append(current)

        # Early convergence check
        variance = float(np.var(current["values"]))
        if variance < CONVERGENCE_VARIANCE_THRESHOLD:
            break

    evaluation = evaluate_run(trajectory)

    return {
        "final_state": current,
        "trajectory": trajectory,
        "strategies": strategies,
        "evaluation": evaluation,
    }
