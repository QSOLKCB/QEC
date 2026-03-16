"""
Deterministic analysis metrics for ternary decoder outputs.

All outputs are numpy.float64.  No randomness.
"""

from __future__ import annotations

import numpy as np


def compute_ternary_stability(messages: np.ndarray) -> np.float64:
    """Compute the stability of ternary messages.

    Stability is the fraction of messages that are decided (non-zero).
    A fully decided message vector has stability 1.0.
    A fully undecided message vector has stability 0.0.

    Parameters
    ----------
    messages : np.ndarray of np.int8
        Ternary message vector, shape (n,).

    Returns
    -------
    np.float64
        Stability in [0.0, 1.0].
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    if arr.size == 0:
        return np.float64(0.0)
    decided = np.count_nonzero(arr)
    return np.float64(decided / arr.size)


def compute_ternary_entropy(messages: np.ndarray) -> np.float64:
    """Compute the normalized entropy of ternary messages.

    Uses the empirical distribution over {-1, 0, +1} states.
    Maximum entropy is log2(3) for a uniform distribution.
    Result is normalized to [0.0, 1.0].

    Parameters
    ----------
    messages : np.ndarray of np.int8
        Ternary message vector, shape (n,).

    Returns
    -------
    np.float64
        Normalized entropy in [0.0, 1.0].
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    if arr.size == 0:
        return np.float64(0.0)

    counts = np.zeros(3, dtype=np.int64)
    counts[0] = np.sum(arr == -1)
    counts[1] = np.sum(arr == 0)
    counts[2] = np.sum(arr == 1)

    total = np.float64(arr.size)
    probs = counts.astype(np.float64) / total

    # Compute entropy, avoiding log2(0)
    entropy = np.float64(0.0)
    for p in probs:
        if p > 0.0:
            entropy -= p * np.log2(p)

    max_entropy = np.log2(np.float64(3.0))
    return np.float64(entropy / max_entropy)


def compute_ternary_conflict_density(messages: np.ndarray) -> np.float64:
    """Compute the conflict density of ternary messages.

    Conflict density is the fraction of adjacent message pairs that
    disagree (one is +1 and the other is -1).

    Parameters
    ----------
    messages : np.ndarray of np.int8
        Ternary message vector, shape (n,).

    Returns
    -------
    np.float64
        Conflict density in [0.0, 1.0].
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    if arr.size <= 1:
        return np.float64(0.0)

    num_pairs = arr.size - 1
    conflicts = 0
    for i in range(num_pairs):
        if arr[i] != 0 and arr[i + 1] != 0 and arr[i] != arr[i + 1]:
            conflicts += 1

    return np.float64(conflicts / num_pairs)
