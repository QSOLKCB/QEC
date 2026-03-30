"""Regime history window analysis — persistence and oscillation diagnostics."""


def compute_state_counts(history: list[str]) -> dict[str, int]:
    """Return count of each regime state, keys sorted alphabetically."""
    counts: dict[str, int] = {}
    for state in history:
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def compute_persistence_ratio(history: list[str], target_state: str) -> float:
    """Return fraction of history occupied by target_state."""
    if not history:
        return 0.0
    count = sum(1 for s in history if s == target_state)
    return round(count / len(history), 12)


def compute_transition_count(history: list[str]) -> int:
    """Count state-to-state transitions in history."""
    if len(history) <= 1:
        return 0
    return sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])


def classify_history_behavior(transition_count: int, oscillation_ratio: float) -> str:
    """Classify window behavior from transition count and oscillation ratio."""
    if transition_count >= 3:
        return "unstable"
    if oscillation_ratio >= 0.5:
        return "persistent_oscillation"
    return "stable_window"


def run_regime_history_analysis(history: list[str]) -> dict:
    """Run full regime history analysis on a state window."""
    state_counts = compute_state_counts(history)
    oscillation_ratio = compute_persistence_ratio(history, "oscillatory")
    transition_count = compute_transition_count(history)
    history_behavior = classify_history_behavior(transition_count, oscillation_ratio)
    return {
        "state_counts": state_counts,
        "oscillation_ratio": oscillation_ratio,
        "transition_count": transition_count,
        "history_behavior": history_behavior,
    }
