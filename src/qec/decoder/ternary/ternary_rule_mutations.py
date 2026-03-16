"""
Deterministic mutation of ternary decoder rules.

Provides mutated rule variants derived from existing decoder rules,
enabling exploration of new rule behaviors alongside the base registry.
All outputs are np.int8.  No randomness.

This module does not modify the existing BP decoder or ternary decoder.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def flip_zero_bias_rule(messages: np.ndarray) -> np.int8:
    """Flip-zero-bias rule: bias toward +1 on ties instead of neutral.

    Behavior:
    - If input contains both +1 and -1 -> return 0 (conflict)
    - If sum == 0 -> return +1 (bias instead of neutral)
    - Else -> sign(sum)

    Parameters
    ----------
    messages : np.ndarray of np.int8
        Incoming ternary messages with values in {-1, 0, +1}.

    Returns
    -------
    np.int8
        Flip-zero-bias result.
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    has_pos = bool(np.any(arr == 1))
    has_neg = bool(np.any(arr == -1))
    if has_pos and has_neg:
        return np.int8(0)
    total = np.sum(arr, dtype=np.int64)
    if total == 0:
        return np.int8(1)
    return np.int8(np.sign(total))


def conservative_rule(messages: np.ndarray) -> np.int8:
    """Conservative rule: only output +/-1 if strong agreement.

    Behavior:
    - Only output +/-1 if |sum| >= 2
    - Otherwise return 0

    Parameters
    ----------
    messages : np.ndarray of np.int8
        Incoming ternary messages with values in {-1, 0, +1}.

    Returns
    -------
    np.int8
        Conservative result.
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    total = np.sum(arr, dtype=np.int64)
    if abs(int(total)) >= 2:
        return np.int8(np.sign(total))
    return np.int8(0)


def inverted_majority_rule(messages: np.ndarray) -> np.int8:
    """Inverted majority rule: return opposite of majority sign.

    Behavior:
    - Compute majority sign
    - Return opposite sign
    - If sum == 0 -> return 0

    Parameters
    ----------
    messages : np.ndarray of np.int8
        Incoming ternary messages with values in {-1, 0, +1}.

    Returns
    -------
    np.int8
        Inverted majority result.
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    total = np.sum(arr, dtype=np.int64)
    if total == 0:
        return np.int8(0)
    return np.int8(-np.sign(total))


def generate_mutated_rules() -> dict[str, Callable[[np.ndarray], np.int8]]:
    """Return deterministic dictionary of mutated rule variants.

    Keys are sorted lexicographically.

    Returns
    -------
    dict[str, Callable[[np.ndarray], np.int8]]
        Mapping from rule name to rule function.
    """
    rules = {
        "conservative": conservative_rule,
        "flip_zero_bias": flip_zero_bias_rule,
        "inverted_majority": inverted_majority_rule,
    }
    return dict(sorted(rules.items()))
