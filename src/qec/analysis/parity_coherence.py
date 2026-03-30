"""v108.0.0 — Deterministic parity/coherence analysis signals.

This module adds a compact, physics-facing abstraction layer for
trajectory-style analysis. It is deterministic, bounded, side-effect free,
and uses only simple inspectable rules over the provided trajectory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Minimum dominant-sign ratio required for stable parity labeling.
PARITY_STABLE_THRESHOLD = 0.6
# Minimum run length on both sides of a sign transition to count as jump.
PERSISTENT_JUMP_MIN_RUN = 2
# Explicit bounded stability-score weights (sum to 1.0).
STABILITY_WEIGHT_GLOBAL = 0.5
STABILITY_WEIGHT_COHERENCE = 0.3
STABILITY_WEIGHT_DISAGREEMENT = 0.2


def run_parity_coherence_analysis(
    trajectory: List[float],
    regime_history: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute deterministic parity/coherence summary metrics.

    Parameters
    ----------
    trajectory : list of float
        Ordered scalar trajectory signal.
    regime_history : list of str, optional
        Accepted for API compatibility; not required for v108.0.0 rules.
    metadata : dict, optional
        Accepted for pipeline context; not used in computations.

    Returns
    -------
    dict
        Core fields:

        - ``parity_state``: one of ``stable_positive``, ``stable_negative``,
          ``neutral``.
        - ``global_probe_score``: dominant non-zero sign ratio in [0, 1].
        - ``local_probe_score``: local sign-consistency score in [0, 1].
        - ``probe_disagreement``: absolute global/local mismatch in [0, 1].
        - ``coherence_length``: longest contiguous non-zero parity run (int >= 0).
        - ``parity_jump_detected``: true only for persistent opposite-sign runs.
        - ``parity_stability_score``: weighted bounded stability summary in [0, 1].

    Notes
    -----
    Computation uses first-difference signs only, with deterministic tie handling
    (zero delta => zero sign). No mutation of inputs occurs.
    """
    _ = regime_history
    _ = metadata

    signs = _trajectory_signs(trajectory)
    nonzero_signs = [s for s in signs if s != 0]

    positive_count = sum(1 for s in nonzero_signs if s > 0)
    negative_count = sum(1 for s in nonzero_signs if s < 0)
    nonzero_count = len(nonzero_signs)

    global_probe_score = _global_probe_score(positive_count, negative_count, nonzero_count)
    local_probe_score = _local_probe_score(nonzero_signs, global_probe_score)
    probe_disagreement = _clamp01(abs(global_probe_score - local_probe_score))

    coherence_length = _longest_nonzero_run(signs)
    parity_jump_detected = _has_persistent_jump(signs, min_run=PERSISTENT_JUMP_MIN_RUN)

    if nonzero_count > 0 and global_probe_score >= PARITY_STABLE_THRESHOLD:
        parity_state = "stable_positive" if positive_count >= negative_count else "stable_negative"
    else:
        parity_state = "neutral"

    coherence_ratio = (coherence_length / nonzero_count) if nonzero_count > 0 else 0.0
    parity_stability_score = _clamp01(
        STABILITY_WEIGHT_GLOBAL * global_probe_score
        + STABILITY_WEIGHT_COHERENCE * coherence_ratio
        + STABILITY_WEIGHT_DISAGREEMENT * (1.0 - probe_disagreement)
    )

    return {
        "parity_state": parity_state,
        "global_probe_score": round(global_probe_score, 12),
        "local_probe_score": round(local_probe_score, 12),
        "probe_disagreement": round(probe_disagreement, 12),
        "coherence_length": int(coherence_length),
        "parity_jump_detected": bool(parity_jump_detected),
        "parity_stability_score": round(parity_stability_score, 12),
    }


def _trajectory_signs(trajectory: List[float]) -> List[int]:
    if len(trajectory) < 2:
        return []
    signs: List[int] = []
    for i in range(len(trajectory) - 1):
        delta = float(trajectory[i + 1]) - float(trajectory[i])
        if delta > 0.0:
            signs.append(1)
        elif delta < 0.0:
            signs.append(-1)
        else:
            signs.append(0)
    return signs


def _global_probe_score(positive: int, negative: int, total_nonzero: int) -> float:
    if total_nonzero == 0:
        return 0.0
    return max(positive, negative) / total_nonzero


def _local_probe_score(nonzero_signs: List[int], fallback: float) -> float:
    n = len(nonzero_signs)
    if n <= 1:
        return fallback
    flips = 0
    for i in range(n - 1):
        if nonzero_signs[i] != nonzero_signs[i + 1]:
            flips += 1
    return 1.0 - (flips / (n - 1))


def _longest_nonzero_run(signs: List[int]) -> int:
    runs = _segment_sign_runs(signs)
    if not runs:
        return 0
    return max(length for _, length, _, _ in runs)


def _has_persistent_jump(signs: List[int], min_run: int) -> bool:
    if not signs or min_run <= 0:
        return False
    runs = _segment_sign_runs(signs)

    for i in range(len(runs) - 1):
        left_sign, left_len, _, left_end = runs[i]
        right_sign, right_len, right_start, _ = runs[i + 1]
        if right_start != left_end + 1:
            continue
        if left_sign != right_sign and left_len >= min_run and right_len >= min_run:
            return True
    return False


def _segment_sign_runs(signs: List[int]) -> List[tuple[int, int, int, int]]:
    """Return contiguous non-zero sign runs; zero values terminate runs."""
    runs: List[tuple[int, int, int, int]] = []
    current_sign = 0
    current_len = 0
    current_start = -1

    for idx, sign in enumerate(signs):
        if sign == 0:
            if current_sign != 0 and current_len > 0:
                runs.append((current_sign, current_len, current_start, idx - 1))
            current_sign = 0
            current_len = 0
            current_start = -1
            continue

        if sign == current_sign:
            current_len += 1
        else:
            if current_sign != 0 and current_len > 0:
                runs.append((current_sign, current_len, current_start, idx - 1))
            current_sign = sign
            current_len = 1
            current_start = idx

    if current_sign != 0 and current_len > 0:
        runs.append((current_sign, current_len, current_start, len(signs) - 1))
    return runs


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


__all__ = ["run_parity_coherence_analysis"]
