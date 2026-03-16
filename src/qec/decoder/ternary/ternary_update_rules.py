"""
Deterministic ternary update rules for variable and check nodes.

All rules are pure functions with no randomness or side effects.
"""

from __future__ import annotations

import numpy as np


def variable_node_update(
    incoming_messages: np.ndarray,
    channel_value: np.int8,
) -> np.int8:
    """Deterministic variable node update.

    Combines incoming check-to-variable messages with the channel value
    by majority vote.  Ties resolve to 0 (undecided).

    Parameters
    ----------
    incoming_messages : np.ndarray of np.int8
        Messages from neighboring check nodes, shape (degree,).
    channel_value : np.int8
        Channel observation for this variable node.

    Returns
    -------
    np.int8
        Updated ternary message.
    """
    msgs = np.asarray(incoming_messages, dtype=np.int8)
    total = np.int64(channel_value) + np.sum(msgs, dtype=np.int64)
    return np.int8(np.sign(total))


def check_node_update(incoming_messages: np.ndarray) -> np.int8:
    """Deterministic check node update.

    Enforces parity constraint over ternary states.  If any incoming
    message is undecided (0), the output is 0.  Otherwise the output
    is the product of all incoming messages (ternary parity).

    Parameters
    ----------
    incoming_messages : np.ndarray of np.int8
        Messages from neighboring variable nodes, shape (degree,).

    Returns
    -------
    np.int8
        Updated ternary message.
    """
    msgs = np.asarray(incoming_messages, dtype=np.int8)
    if msgs.size == 0:
        return np.int8(0)
    if np.any(msgs == 0):
        return np.int8(0)
    parity = np.int8(1)
    for m in msgs:
        parity = np.int8(parity * m)
    return parity
