"""Deterministic convergence detection and analysis — v101.0.0.

Provides convergence detection and convergence signal computation
for score sequences. All functions are deterministic with safe
edge handling.

Dependencies: none (stdlib only).
"""

from __future__ import annotations

from typing import List, Optional


def detect_convergence(
    scores: List[float],
    window: int = 5,
    threshold: float = 0.01,
) -> Optional[int]:
    """Detect the first step at which scores converge.

    Convergence is detected when the maximum absolute difference within
    a sliding window of size *window* drops below *threshold*.

    Parameters
    ----------
    scores : list of float
        Per-step score values.
    window : int
        Size of the sliding window (must be >= 2).
    threshold : float
        Maximum allowed range within window for convergence.

    Returns
    -------
    int or None
        Step index at which convergence is first detected, or None
        if the sequence never converges.
    """
    if window < 2:
        window = 2
    if len(scores) < window:
        return None
    for i in range(window - 1, len(scores)):
        segment = scores[i - window + 1: i + 1]
        seg_min = segment[0]
        seg_max = segment[0]
        for v in segment[1:]:
            if v < seg_min:
                seg_min = v
            if v > seg_max:
                seg_max = v
        if seg_max - seg_min < threshold:
            return i
    return None


def compute_convergence_signal(
    scores: List[float],
    window: int = 5,
) -> float:
    """Compute a [0, 1] convergence signal from the tail of the sequence.

    The signal is based on the inverse of the range within the last
    *window* scores. A value of 1.0 indicates perfect convergence
    (all values identical); values near 0.0 indicate high variability.

    Parameters
    ----------
    scores : list of float
        Per-step score values.
    window : int
        Number of trailing scores to consider (must be >= 2).

    Returns
    -------
    float
        Convergence signal in [0, 1]. Returns 1.0 for empty or
        single-element inputs (trivially converged).
    """
    if len(scores) < 2:
        return 1.0
    if window < 2:
        window = 2
    window = min(window, len(scores))
    tail = scores[-window:]
    tail_min = tail[0]
    tail_max = tail[0]
    for v in tail[1:]:
        if v < tail_min:
            tail_min = v
        if v > tail_max:
            tail_max = v
    spread = tail_max - tail_min
    # Bounded inversion: 1 / (1 + spread)
    return 1.0 / (1.0 + spread)
