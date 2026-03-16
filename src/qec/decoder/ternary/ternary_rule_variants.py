"""
Deterministic rule variants for ternary message-passing decoding.

Provides alternative update rules for the ternary decoder sandbox.
Each rule maps a ternary message vector to a single ternary output.
All outputs are np.int8.  No randomness.

This module does not modify the existing BP decoder or ternary decoder.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np


def majority_rule(messages: np.ndarray) -> np.int8:
    """Standard majority vote over ternary messages.

    Parameters
    ----------
    messages : np.ndarray of np.int8
        Incoming ternary messages with values in {-1, 0, +1}.

    Returns
    -------
    np.int8
        Majority vote result: sign of the sum, ties resolve to 0.
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    total = np.sum(arr, dtype=np.int64)
    return np.int8(np.sign(total))


def damped_majority_rule(messages: np.ndarray) -> np.int8:
    """Majority rule with deterministic damping threshold.

    Returns 0 when the absolute sum is below the damping threshold (2),
    otherwise returns the sign of the sum.

    Parameters
    ----------
    messages : np.ndarray of np.int8
        Incoming ternary messages with values in {-1, 0, +1}.

    Returns
    -------
    np.int8
        Damped majority result.
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    total = np.sum(arr, dtype=np.int64)
    if abs(int(total)) < 2:
        return np.int8(0)
    return np.int8(np.sign(total))


def conflict_averse_rule(messages: np.ndarray) -> np.int8:
    """Bias toward zero when conflicts appear.

    If both +1 and -1 are present among the messages, returns 0.
    Otherwise returns the majority vote.

    Parameters
    ----------
    messages : np.ndarray of np.int8
        Incoming ternary messages with values in {-1, 0, +1}.

    Returns
    -------
    np.int8
        Conflict-averse result.
    """
    arr = np.asarray(messages, dtype=np.int8).ravel()
    has_pos = bool(np.any(arr == 1))
    has_neg = bool(np.any(arr == -1))
    if has_pos and has_neg:
        return np.int8(0)
    total = np.sum(arr, dtype=np.int64)
    return np.int8(np.sign(total))


def parity_pressure_rule(messages: np.ndarray) -> np.int8:
    """Parity pressure rule based on sign of the sum.

    This is functionally equivalent to majority_rule for ternary
    messages in {-1, 0, +1}. Implemented as a thin wrapper to avoid
    duplication and ambiguity between rules.
    """
    return majority_rule(messages)


# Deterministic rule registry with stable insertion order.
RULE_REGISTRY: OrderedDict[str, Callable[[np.ndarray], np.int8]] = OrderedDict([
    ("conflict_averse", conflict_averse_rule),
    ("damped_majority", damped_majority_rule),
    ("majority", majority_rule),
    ("parity_pressure", parity_pressure_rule),
])
