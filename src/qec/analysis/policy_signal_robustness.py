"""v99.9.0 — Policy consistency, cycle detection, signal robustness & trajectory validation.

Adds (v99.8.0):
- Cycle detection in strategy/regime history to penalize oscillatory loops
- Geometric mean modulation to prevent multiplicative signal collapse
- Light signal decorrelation (normalize to reduce redundancy)
- Adaptive thresholds (deterministic percentile-based)
- Modulation stability clamp
- Integrated scoring with cycle penalty

Adds (v99.9.0):
- Trajectory score integration into final scoring formula

All functions are:
- deterministic (identical inputs → identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- pure analysis signals

Dependencies: stdlib only.  No randomness, no mutation, no ML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Task 1 — Cycle Detection & Penalty
# ---------------------------------------------------------------------------


def detect_cycle(
    history: List[str],
    window: int = 5,
) -> bool:
    """Detect repeating patterns in recent regime/strategy labels.

    Scans the last *window* entries for length-2 or length-3 repeating
    patterns (e.g. [A, B, A, B] or [A, B, C, A, B, C]).

    A stable repetition (all identical) is NOT flagged as a cycle,
    since uniform stability is a valid state.

    Parameters
    ----------
    history : list of str
        Recent regime or strategy labels (oldest first).
    window : int
        Number of most-recent entries to examine (default 5).

    Returns
    -------
    bool
        True if a degenerate oscillatory cycle is detected.
    """
    if len(history) < 3:
        return False

    recent = list(history[-window:]) if len(history) > window else list(history)
    n = len(recent)
    if n < 3:
        return False

    # Skip if all entries identical (stable repetition, not a cycle)
    if len(set(recent)) <= 1:
        return False

    # Check for period-2 cycle: A B A B ...
    for period in (2, 3):
        if n < 2 * period:
            continue
        is_cycle = True
        pattern = recent[:period]
        for i in range(period, n):
            if recent[i] != pattern[i % period]:
                is_cycle = False
                break
        if is_cycle:
            return True

    return False


def compute_cycle_penalty(
    history: List[str],
    window: int = 5,
) -> float:
    """Compute a multiplicative penalty for oscillatory cycles.

    Returns 1.0 (no penalty) when no cycle is detected, or a value
    in [0.8, 1.0) when a cycle is present.  The penalty scales with
    how much of the window is covered by distinct labels (more distinct
    labels in a cycle → stronger penalty).

    Parameters
    ----------
    history : list of str
        Recent regime or strategy labels (oldest first).
    window : int
        Number of most-recent entries to examine (default 5).

    Returns
    -------
    float
        Penalty in [0.8, 1.0].  1.0 = no penalty.
    """
    if not detect_cycle(history, window=window):
        return 1.0

    recent = list(history[-window:]) if len(history) > window else list(history)
    n_unique = len(set(recent))

    # More distinct labels in cycle → stronger penalty
    # n_unique >= 2 (since detect_cycle excludes all-same)
    # Map n_unique ∈ [2, window] to penalty ∈ [0.95, 0.8]
    max_unique = max(len(recent), 2)
    # Linear interpolation: 2 unique → 0.95, max_unique → 0.8
    if max_unique <= 2:
        penalty = 0.9
    else:
        t = (n_unique - 2) / (max_unique - 2)
        penalty = 0.95 - 0.15 * t

    return max(0.8, min(1.0, penalty))


# ---------------------------------------------------------------------------
# Task 2 — Geometric Mean Modulation (Multiplicative Collapse Fix)
# ---------------------------------------------------------------------------


def compute_geometric_mean_modulation(
    energy: Optional[float] = None,
    phase: Optional[float] = None,
    coherence: Optional[float] = None,
    alignment: Optional[float] = None,
) -> float:
    """Compute modulation via geometric mean to prevent signal collapse.

    Replaces the raw product ``(1-energy) * phase * coherence * alignment``
    with::

        geometric_mean = ((1-energy) * phase * coherence * alignment) ** 0.25
        modulation = 0.5 + geometric_mean

    Falls back to 1.0 (neutral) if any required signal is missing.

    Parameters
    ----------
    energy : float, optional
        System energy in [0, 1].
    phase : float, optional
        Phase stability in [0, 1].
    coherence : float, optional
        Multi-scale coherence in [0, 1].
    alignment : float, optional
        Control alignment in [0, 1].

    Returns
    -------
    float
        Modulation factor, typically in [0.5, 1.5].
    """
    if energy is None or phase is None or coherence is None or alignment is None:
        return 1.0

    # Clamp inputs to [0, 1]
    e = max(0.0, min(1.0, float(energy)))
    p = max(0.0, min(1.0, float(phase)))
    c = max(0.0, min(1.0, float(coherence)))
    a = max(0.0, min(1.0, float(alignment)))

    product = (1.0 - e) * p * c * a

    # Geometric mean (4th root)
    # product >= 0 since all factors in [0,1]
    geometric_mean = product ** 0.25

    modulation = 0.5 + geometric_mean

    return modulation


# ---------------------------------------------------------------------------
# Task 3 — Light Signal Decorrelation
# ---------------------------------------------------------------------------


def normalize_signals(
    signals: Dict[str, float],
    keys: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Normalize selected signal values to reduce redundancy.

    Subtracts the mean and rescales to [0, 1] for specified keys.
    Only applied when there are >= 2 values to normalize.

    Parameters
    ----------
    signals : dict
        Signal name → value mapping.
    keys : list of str, optional
        Keys to normalize.  Defaults to ``["phase", "coherence", "alignment"]``.

    Returns
    -------
    dict
        New dict with normalized values (input not mutated).
    """
    if keys is None:
        keys = ["phase", "coherence", "alignment"]

    result = dict(signals)

    # Collect values for normalization
    active_keys = [k for k in keys if k in result]
    if len(active_keys) < 2:
        return result

    values = [float(result[k]) for k in active_keys]
    mean_val = sum(values) / len(values)
    centered = [v - mean_val for v in values]

    c_min = min(centered)
    c_max = max(centered)
    span = c_max - c_min

    if span < 1e-15:
        # All values identical — normalize to 0.5
        for k in active_keys:
            result[k] = 0.5
        return result

    for i, k in enumerate(active_keys):
        normalized = (centered[i] - c_min) / span
        result[k] = max(0.0, min(1.0, normalized))

    return result


# ---------------------------------------------------------------------------
# Task 4 — Adaptive Thresholds (Deterministic)
# ---------------------------------------------------------------------------


def compute_adaptive_threshold(
    history_values: List[float],
    percentile: float,
    fallback: float,
    min_samples: int = 5,
) -> float:
    """Compute a deterministic percentile-based threshold.

    Uses sorted history to compute the threshold at the given percentile.
    Falls back to a constant if insufficient data.

    Parameters
    ----------
    history_values : list of float
        Historical values to derive threshold from.
    percentile : float
        Percentile in [0, 100].
    fallback : float
        Value to return if fewer than *min_samples* entries.
    min_samples : int
        Minimum history length required (default 5).

    Returns
    -------
    float
        Threshold value.
    """
    if len(history_values) < min_samples:
        return float(fallback)

    sorted_vals = sorted(float(v) for v in history_values)
    n = len(sorted_vals)

    # Deterministic percentile via nearest-rank method
    p = max(0.0, min(100.0, float(percentile)))
    rank = (p / 100.0) * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    frac = rank - lower

    # Linear interpolation between adjacent ranks
    threshold = sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])

    return threshold


def compute_oscillation_threshold(
    oscillation_history: List[float],
    fallback: float = 0.7,
) -> float:
    """Compute adaptive oscillation threshold (75th percentile).

    Parameters
    ----------
    oscillation_history : list of float
        Historical oscillation strength values.
    fallback : float
        Default threshold if insufficient data (default 0.7).

    Returns
    -------
    float
        Threshold value.
    """
    return compute_adaptive_threshold(oscillation_history, 75.0, fallback)


def compute_phase_threshold(
    phase_history: List[float],
    fallback: float = 0.3,
) -> float:
    """Compute adaptive phase stability threshold (25th percentile).

    Parameters
    ----------
    phase_history : list of float
        Historical phase stability values.
    fallback : float
        Default threshold if insufficient data (default 0.3).

    Returns
    -------
    float
        Threshold value.
    """
    return compute_adaptive_threshold(phase_history, 25.0, fallback)


# ---------------------------------------------------------------------------
# Task 5 — Modulation Stability Clamp
# ---------------------------------------------------------------------------


def clamp_modulation(
    modulation: float,
    inner_low: float = 0.6,
    inner_high: float = 1.4,
    outer_low: float = 0.5,
    outer_high: float = 1.5,
) -> float:
    """Apply two-tier clamping to modulation factor.

    Inner clamp [0.6, 1.4] is the preferred operational range.
    Outer clamp [0.5, 1.5] is the hard boundary.

    The inner clamp is applied first, then the outer as a safety net.

    Parameters
    ----------
    modulation : float
        Raw modulation value.
    inner_low, inner_high : float
        Inner (preferred) clamp bounds.
    outer_low, outer_high : float
        Outer (hard) clamp bounds.

    Returns
    -------
    float
        Clamped modulation value.
    """
    clamped = max(inner_low, min(inner_high, float(modulation)))
    clamped = max(outer_low, min(outer_high, clamped))
    return clamped


# ---------------------------------------------------------------------------
# Task 6 — Integrated Scoring
# ---------------------------------------------------------------------------


def compute_robust_score(
    base_score: float,
    stability_weight: float = 1.0,
    transition_bias: float = 1.0,
    multi_step_factor: float = 1.0,
    adaptation_modulation: float = 1.0,
    cycle_penalty: float = 1.0,
    trajectory_score: float = 1.0,
) -> float:
    """Compute final score with all v99.9.0 factors.

    Formula::

        final_score = (base_score
                       * stability_weight
                       * transition_bias
                       * multi_step_factor
                       * adaptation_modulation
                       * cycle_penalty
                       * trajectory_score)

    Result clamped to [0.0, 1.0].

    Parameters
    ----------
    base_score : float
        Base strategy score.
    stability_weight : float
        Regime stability weight (default 1.0).
    transition_bias : float
        Transition learning bias (default 1.0).
    multi_step_factor : float
        Multi-step lookahead factor (default 1.0).
    adaptation_modulation : float
        Physics-informed modulation (default 1.0).
    cycle_penalty : float
        Cycle detection penalty (default 1.0).
    trajectory_score : float
        Trajectory validation score (default 1.0).

    Returns
    -------
    float
        Final score in [0.0, 1.0].
    """
    score = (
        float(base_score)
        * float(stability_weight)
        * float(transition_bias)
        * float(multi_step_factor)
        * float(adaptation_modulation)
        * float(cycle_penalty)
        * float(trajectory_score)
    )
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Task 7 — Output Visibility
# ---------------------------------------------------------------------------


def compute_robustness_diagnostics(
    regime_history: List[str],
    energy: Optional[float] = None,
    phase: Optional[float] = None,
    coherence: Optional[float] = None,
    alignment: Optional[float] = None,
    window: int = 5,
) -> Dict[str, Any]:
    """Compute all v99.8.0 robustness diagnostics in one call.

    Returns a dict exposing cycle detection and modulation state
    for observability and debugging.

    Parameters
    ----------
    regime_history : list of str
        Recent regime/strategy labels.
    energy, phase, coherence, alignment : float, optional
        Physics signal values for modulation computation.
    window : int
        Cycle detection window size.

    Returns
    -------
    dict
        Diagnostic output with keys:
        - ``cycle_detected``: bool
        - ``cycle_penalty``: float
        - ``modulation_raw``: float
        - ``modulation_adjusted``: float
    """
    cycle_detected = detect_cycle(regime_history, window=window)
    cycle_pen = compute_cycle_penalty(regime_history, window=window)

    modulation_raw = compute_geometric_mean_modulation(
        energy=energy,
        phase=phase,
        coherence=coherence,
        alignment=alignment,
    )
    modulation_adjusted = clamp_modulation(modulation_raw)

    return {
        "cycle_detected": cycle_detected,
        "cycle_penalty": cycle_pen,
        "modulation_raw": modulation_raw,
        "modulation_adjusted": modulation_adjusted,
    }


__all__ = [
    "detect_cycle",
    "compute_cycle_penalty",
    "compute_geometric_mean_modulation",
    "normalize_signals",
    "compute_adaptive_threshold",
    "compute_oscillation_threshold",
    "compute_phase_threshold",
    "clamp_modulation",
    "compute_robust_score",
    "compute_robustness_diagnostics",
]
