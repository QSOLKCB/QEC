"""Self-healing control layer for oscillation suppression and transition stabilization.

Consumes outputs from run_adaptive_control() and run_regime_jump_analysis()
to suppress oscillatory behavior and stabilize regime transitions.
"""


def compute_suppression_intensity(
    regime_behavior: str,
    coherence_length: int,
) -> float:
    """Compute suppression intensity scaled by coherence length.

    Longer oscillation persistence produces stronger suppression.
    Transitions scale mildly. Locked regimes produce no suppression.
    """
    if regime_behavior == "oscillatory":
        return round(min(0.2, 0.05 * coherence_length), 12)
    if regime_behavior == "transition":
        return round(min(0.1, 0.02 * coherence_length), 12)
    return 0.0


def compute_healing_damping(
    adaptive_damping: float,
    regime_behavior: str,
    coherence_length: int = 1,
) -> float:
    """Compute healing damping based on regime behavior and coherence length."""
    suppression = compute_suppression_intensity(regime_behavior, coherence_length)
    return round(max(0.3, adaptive_damping - suppression), 12)


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
        healing_damping, healing_mode, escalation_frozen, healing_gain,
        suppression_intensity.
    """
    adaptive_damping = adaptive_result.get("adaptive_damping", 0.0)
    regime_behavior = regime_result.get("regime_behavior", "locked")
    coherence_length = regime_result.get("coherence_length", 1)

    healing_damping = compute_healing_damping(
        adaptive_damping, regime_behavior, coherence_length
    )
    suppression_intensity = compute_suppression_intensity(
        regime_behavior, coherence_length
    )
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
        "suppression_intensity": suppression_intensity,
    }
