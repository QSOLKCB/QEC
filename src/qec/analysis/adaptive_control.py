"""Adaptive trend-aware damping controller.

Consumes outputs from run_control_memory() and dynamically adjusts
damping based on trend state.  All functions are pure and deterministic.
"""


def compute_adaptive_damping(base_damping: float, trend_state: str) -> float:
    """Adjust damping based on trend state.

    rising  -> stronger damping (decrease toward 0.4)
    falling -> relax damping   (increase toward 1.0)
    stable  -> hold
    """
    if trend_state == "rising":
        return round(max(0.4, base_damping - 0.1), 12)
    if trend_state == "falling":
        return round(min(1.0, base_damping + 0.1), 12)
    return round(base_damping, 12)


def compute_response_mode(trend_state: str) -> str:
    """Map trend state to response mode."""
    mapping = {
        "rising": "stabilize",
        "falling": "recover",
        "stable": "hold",
    }
    return mapping.get(trend_state, "hold")


def run_adaptive_control(base_damping: float, memory_result: dict) -> dict:
    """Compute adaptive control outputs from a control-memory result.

    Parameters
    ----------
    base_damping : float
        Nominal damping value.
    memory_result : dict
        Output of ``run_control_memory`` with keys
        ``smoothed_signal``, ``trend_delta``, ``trend_state``.

    Returns
    -------
    dict with ``adaptive_damping``, ``response_mode``, ``control_gain``.
    """
    trend_state = memory_result["trend_state"]
    adaptive_damping = compute_adaptive_damping(base_damping, trend_state)
    response_mode = compute_response_mode(trend_state)
    control_gain = round(adaptive_damping / max(base_damping, 1e-12), 12)
    return {
        "adaptive_damping": adaptive_damping,
        "response_mode": response_mode,
        "control_gain": control_gain,
    }
