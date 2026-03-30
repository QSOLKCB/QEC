"""Self-healing control layer for oscillation suppression and transition stabilization.

Consumes outputs from run_adaptive_control() and run_regime_jump_analysis()
to suppress oscillatory behavior and stabilize regime transitions.
"""


def compute_healing_damping(adaptive_damping: float, regime_behavior: str) -> float:
    """Compute healing damping based on regime behavior."""
    if regime_behavior == "oscillatory":
        return round(max(0.3, adaptive_damping - 0.1), 12)
    if regime_behavior == "transition":
        return round(max(0.3, adaptive_damping - 0.05), 12)
    return round(adaptive_damping, 12)


def compute_healing_mode(regime_behavior: str) -> str:
    """Return healing mode for given regime behavior."""
    if regime_behavior == "oscillatory":
        return "suppress"
    if regime_behavior == "transition":
        return "stabilize"
    return "hold"


def compute_escalation_freeze(regime_behavior: str) -> bool:
    """Return whether escalation should be frozen."""
    return regime_behavior == "oscillatory"


def run_self_healing_control(adaptive_result: dict, regime_result: dict) -> dict:
    """Run self-healing control combining adaptive and regime analysis results.

    Parameters
    ----------
    adaptive_result : dict
        Output from run_adaptive_control(). Expected keys:
        adaptive_damping, response_mode, control_gain.
    regime_result : dict
        Output from run_regime_jump_analysis(). Expected keys:
        jump_detected, jump_type, coherence_length, regime_behavior.

    Returns
    -------
    dict
        healing_damping, healing_mode, escalation_frozen, healing_gain.
    """
    adaptive_damping = adaptive_result.get("adaptive_damping", 0.0)
    regime_behavior = regime_result.get("regime_behavior", "locked")

    healing_damping = compute_healing_damping(adaptive_damping, regime_behavior)
    healing_mode = compute_healing_mode(regime_behavior)
    escalation_frozen = compute_escalation_freeze(regime_behavior)
    healing_gain = round(
        healing_damping / max(adaptive_result.get("adaptive_damping", 1e-12), 1e-12),
        12,
    )

    return {
        "healing_damping": healing_damping,
        "healing_mode": healing_mode,
        "escalation_frozen": escalation_frozen,
        "healing_gain": healing_gain,
    }
