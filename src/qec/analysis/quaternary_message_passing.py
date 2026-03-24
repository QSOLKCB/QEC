"""Quaternary message passing — experimental design layer.

Implements a simple deterministic message-passing scheme over quaternary
states. Each message carries a ``state`` in {-1.0, -0.5, 0.5, 1.0} and a
``confidence`` in [0, 1]. Aggregation uses deterministic weighted average,
snapping the result back to the nearest quaternary state.

Dependencies: numpy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from qec.analysis.quaternary_quantization import QUATERNARY_STATES, _QUATERNARY_ARRAY


_VALID_STATES = set(QUATERNARY_STATES)


def make_message(state: float, confidence: float) -> Dict[str, float]:
    """Create a quaternary message.

    Parameters
    ----------
    state : float
        Quaternary state, must be in {-1.0, -0.5, 0.5, 1.0}.
    confidence : float
        Confidence weight in [0, 1].

    Returns
    -------
    dict
        ``{"state": float, "confidence": float}``
    """
    if state not in _VALID_STATES:
        raise ValueError(
            f"state must be in {{-1.0, -0.5, 0.5, 1.0}}, got {state}"
        )
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    return {"state": float(state), "confidence": float(confidence)}


def _snap_to_nearest(value: float) -> float:
    """Snap a value to the nearest quaternary state (deterministic)."""
    distances = np.abs(_QUATERNARY_ARRAY - value)
    idx = int(np.argmin(distances))
    return float(_QUATERNARY_ARRAY[idx])


def aggregate_messages(messages: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate a list of quaternary messages deterministically.

    Computes a confidence-weighted average of the state values,
    then snaps the result to the nearest quaternary state.

    Parameters
    ----------
    messages : list of dict
        Each dict has ``"state"`` (float) and ``"confidence"`` (float).

    Returns
    -------
    dict
        Aggregated message with ``"state"`` and ``"confidence"``.
    """
    if not messages:
        return {"state": -0.5, "confidence": 0.0}

    total_weight = 0.0
    weighted_sum = 0.0

    for msg in messages:
        s = msg["state"]
        c = msg["confidence"]
        weighted_sum += s * c
        total_weight += c

    if total_weight == 0.0:
        return {"state": -0.5, "confidence": 0.0}

    avg = weighted_sum / total_weight
    snapped = _snap_to_nearest(avg)

    # Confidence: proportion of total weight that agreed with the result sign
    agree_weight = 0.0
    for msg in messages:
        if (msg["state"] > 0) == (snapped > 0):
            agree_weight += msg["confidence"]

    confidence = agree_weight / total_weight

    return {"state": snapped, "confidence": float(np.clip(confidence, 0.0, 1.0))}


def run_message_passing_round(
    states: np.ndarray,
    confidences: np.ndarray,
    adjacency: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute one round of quaternary message passing.

    For each node, collect messages from its neighbours (defined by
    the adjacency matrix), aggregate them, and update the node.

    Parameters
    ----------
    states : np.ndarray
        1-D float array of quaternary states {-1.0, -0.5, 0.5, 1.0}, length N.
    confidences : np.ndarray
        1-D float array of confidences [0, 1], length N.
    adjacency : np.ndarray
        N×N binary adjacency matrix (symmetric, 0/1).

    Returns
    -------
    new_states : np.ndarray
        Updated quaternary states.
    new_confidences : np.ndarray
        Updated confidences.
    """
    n = len(states)
    new_states = np.zeros(n, dtype=np.float64)
    new_conf = np.zeros(n, dtype=np.float64)

    for i in range(n):
        neighbours = np.where(adjacency[i] > 0)[0]
        if len(neighbours) == 0:
            new_states[i] = states[i]
            new_conf[i] = confidences[i]
            continue

        msgs = [
            make_message(float(states[j]), float(confidences[j]))
            for j in neighbours
        ]
        agg = aggregate_messages(msgs)
        new_states[i] = agg["state"]
        new_conf[i] = agg["confidence"]

    return new_states, new_conf
