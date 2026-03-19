"""
v81.0.0 — Deterministic Replay & Verification Engine.

Replays an FSM run step-by-step and verifies that execution is identical.
Produces a cryptographic-style hash chain over state transitions, enabling
tamper detection and deterministic reproducibility proofs.

This is state machine replication without the network:
    executed → replayed → verified

Layer 8 — Controller.
Does not modify FSM logic.  Fully deterministic.  Read-only verification.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from qec.controller.qec_fsm import QECFSM


# ---------------------------------------------------------------------------
# Deterministic serialization
# ---------------------------------------------------------------------------

def serialize_state(entry: Dict[str, Any]) -> str:
    """Serialize a history entry to a canonical string.

    Rules:
        - Sorted keys
        - Fixed float formatting (12 decimal places)
        - No randomness
        - Stable string output across runs

    Parameters
    ----------
    entry : dict
        A single FSM history entry.

    Returns
    -------
    str
        Canonical string representation.
    """
    return json.dumps(
        entry,
        sort_keys=True,
        default=_canonical_default,
        separators=(",", ":"),
    )


def _canonical_default(obj: Any) -> Any:
    """JSON default handler for canonical serialization."""
    if isinstance(obj, float):
        return f"{obj:.12e}"
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Non-serializable type: {type(obj)}")


# ---------------------------------------------------------------------------
# Hash chain
# ---------------------------------------------------------------------------

def build_hash_chain(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a SHA-256 hash chain over FSM history entries.

    For each step:
        hash_i = sha256(hash_{i-1} + serialize(state_i))

    The genesis hash is sha256(b"genesis").

    Parameters
    ----------
    history : list[dict]
        FSM history trace.

    Returns
    -------
    dict
        ``final_hash`` (str), ``step_hashes`` (list[str]).
    """
    prev_hash = hashlib.sha256(b"genesis").hexdigest()
    step_hashes: List[str] = []

    for entry in history:
        payload = prev_hash + serialize_state(entry)
        current_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        step_hashes.append(current_hash)
        prev_hash = current_hash

    return {
        "final_hash": prev_hash if step_hashes else hashlib.sha256(b"genesis").hexdigest(),
        "step_hashes": step_hashes,
    }


# ---------------------------------------------------------------------------
# History comparison
# ---------------------------------------------------------------------------

_COMPARE_KEYS = (
    "from_state",
    "to_state",
    "stability_score",
    "phase",
    "epsilon",
    "reject_cycle",
    "decision",
)


def compare_histories(
    h1: List[Dict[str, Any]],
    h2: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare two FSM history traces for equality.

    Checks: same length, same states, same stability_score, same epsilon,
    same thresholds, same decisions.

    Parameters
    ----------
    h1, h2 : list[dict]
        Two FSM history traces to compare.

    Returns
    -------
    dict
        ``match`` (bool), ``mismatch_index`` (int | None).
    """
    if len(h1) != len(h2):
        return {"match": False, "mismatch_index": min(len(h1), len(h2))}

    for i, (a, b) in enumerate(zip(h1, h2)):
        for key in _COMPARE_KEYS:
            va = a.get(key)
            vb = b.get(key)
            if isinstance(va, float) and isinstance(vb, float):
                if f"{va:.12e}" != f"{vb:.12e}":
                    return {"match": False, "mismatch_index": i}
            elif va != vb:
                return {"match": False, "mismatch_index": i}
        # Compare thresholds if present.
        ta = a.get("thresholds")
        tb = b.get("thresholds")
        if ta != tb:
            if ta is not None and tb is not None:
                for tk in sorted(set(list(ta.keys()) + list(tb.keys()))):
                    va = ta.get(tk)
                    vb = tb.get(tk)
                    if isinstance(va, float) and isinstance(vb, float):
                        if f"{va:.12e}" != f"{vb:.12e}":
                            return {"match": False, "mismatch_index": i}
                    elif va != vb:
                        return {"match": False, "mismatch_index": i}
            elif ta is not None or tb is not None:
                return {"match": False, "mismatch_index": i}

    return {"match": True, "mismatch_index": None}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_fsm(
    initial_input: Dict[str, Any],
    config: Dict[str, Any],
    max_steps: int = 20,
) -> Dict[str, Any]:
    """Re-run the FSM deterministically and return the result.

    Parameters
    ----------
    initial_input : dict
        Original FSM input data.
    config : dict
        FSM configuration.
    max_steps : int
        Maximum number of FSM steps.

    Returns
    -------
    dict
        FSM result with ``final_state``, ``steps``, ``history``.
    """
    fsm = QECFSM(config=dict(config))
    return fsm.run(initial_input, max_steps=max_steps)


# ---------------------------------------------------------------------------
# Main verification API
# ---------------------------------------------------------------------------

def verify_run(
    initial_input: Dict[str, Any],
    history: List[Dict[str, Any]],
    config: Dict[str, Any],
    max_steps: int = 20,
) -> Dict[str, Any]:
    """Replay an FSM run and verify it matches the original history.

    Parameters
    ----------
    initial_input : dict
        Original input data.
    history : list[dict]
        Original FSM history trace to verify against.
    config : dict
        FSM configuration used in the original run.
    max_steps : int
        Maximum steps for replay.

    Returns
    -------
    dict
        ``match`` (bool), ``final_hash`` (str), ``steps`` (int),
        ``mismatch_index`` (int | None).
    """
    replay_result = replay_fsm(initial_input, config, max_steps=max_steps)
    replay_history = replay_result["history"]

    comparison = compare_histories(history, replay_history)
    chain = build_hash_chain(replay_history)

    return {
        "match": comparison["match"],
        "final_hash": chain["final_hash"],
        "steps": replay_result["steps"],
        "mismatch_index": comparison["mismatch_index"],
    }
