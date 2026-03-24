"""Ternary message passing — experimental design layer.

Implements a simple deterministic message-passing scheme over ternary
states. Each message carries a ``state`` in {-1, 0, +1} and a
``confidence`` in [0, 1]. Aggregation uses majority vote weighted by
confidence, with ties resolved to neutral (0).

Dependencies: numpy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def make_message(state: int, confidence: float) -> Dict[str, float]:
    """Create a ternary message.

    Parameters
    ----------
    state : int
        Ternary state, must be in {-1, 0, +1}.
    confidence : float
        Confidence weight in [0, 1].

    Returns
    -------
    dict
        ``{"state": int, "confidence": float}``
    """
    if state not in (-1, 0, 1):
        raise ValueError(f"state must be in {{-1, 0, +1}}, got {state}")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    return {"state": int(state), "confidence": float(confidence)}


def aggregate_messages(messages: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate a list of ternary messages deterministically.

    Computes confidence-weighted sums for positive and negative states.
    The result state is determined by comparing these sums; ties
    (including empty input) resolve to neutral (0).

    Parameters
    ----------
    messages : list of dict
        Each dict has ``"state"`` (int) and ``"confidence"`` (float).

    Returns
    -------
    dict
        Aggregated message with ``"state"`` and ``"confidence"``.
    """
    if not messages:
        return {"state": 0, "confidence": 0.0}

    pos_weight = 0.0
    neg_weight = 0.0
    neut_weight = 0.0

    for msg in messages:
        s = msg["state"]
        c = msg["confidence"]
        if s == 1:
            pos_weight += c
        elif s == -1:
            neg_weight += c
        else:
            neut_weight += c

    total = pos_weight + neg_weight + neut_weight
    if total == 0.0:
        return {"state": 0, "confidence": 0.0}

    # Majority by weighted vote
    if pos_weight > neg_weight and pos_weight > neut_weight:
        state = 1
        confidence = pos_weight / total
    elif neg_weight > pos_weight and neg_weight > neut_weight:
        state = -1
        confidence = neg_weight / total
    else:
        # Tie or neutral dominates → neutral
        state = 0
        confidence = neut_weight / total if neut_weight >= max(pos_weight, neg_weight) else 0.0

    return {"state": state, "confidence": float(np.clip(confidence, 0.0, 1.0))}


def run_message_passing_round(
    states: np.ndarray,
    confidences: np.ndarray,
    adjacency: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute one round of ternary message passing.

    For each node, collect messages from its neighbours (defined by
    the adjacency matrix), aggregate them, and update the node.

    Parameters
    ----------
    states : np.ndarray
        1-D int array of ternary states {-1, 0, +1}, length N.
    confidences : np.ndarray
        1-D float array of confidences [0, 1], length N.
    adjacency : np.ndarray
        N×N binary adjacency matrix (symmetric, 0/1).

    Returns
    -------
    new_states : np.ndarray
        Updated ternary states.
    new_confidences : np.ndarray
        Updated confidences.
    """
    n = len(states)
    new_states = np.zeros(n, dtype=np.int8)
    new_conf = np.zeros(n, dtype=np.float64)

    for i in range(n):
        neighbours = np.where(adjacency[i] > 0)[0]
        if len(neighbours) == 0:
            new_states[i] = states[i]
            new_conf[i] = confidences[i]
            continue

        msgs = [
            make_message(int(states[j]), float(confidences[j]))
            for j in neighbours
        ]
        agg = aggregate_messages(msgs)
        new_states[i] = agg["state"]
        new_conf[i] = agg["confidence"]

    return new_states, new_conf
