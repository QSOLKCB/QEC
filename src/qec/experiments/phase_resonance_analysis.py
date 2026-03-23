"""v87.3.0 — Resonance Locks + Attractor Field Analysis.

Post-processing layer on top of trajectory motifs and motif graphs.
Detects resonance locks (consecutive same-state regions), computes
lock strength, builds an attractor field from the state graph, and
classifies the resonance field type.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Step 1 — Resonance Lock Detection
# ---------------------------------------------------------------------------


def detect_resonance_locks(
    series: List[Tuple[int, ...]],
    drift: List[float],
) -> Dict[str, Any]:
    """Detect resonance locks in a ternary trajectory.

    A lock is a contiguous region where the state does not change
    (same consecutive state) OR the spectral drift is zero.

    Parameters
    ----------
    series:
        Ordered list of ternary-encoded tuples.
    drift:
        Spectral drift values (len = len(series) - 1).

    Returns
    -------
    dict with ``locks`` (list of lock spans), ``n_locks``, ``mean_lock_length``.
    """
    if len(series) < 2:
        return {"locks": [], "n_locks": 0, "mean_lock_length": 0.0}

    # Build a boolean mask: position i is "locked" if series[i] == series[i+1]
    # or drift[i] == 0.
    n = len(series) - 1
    locked = []
    for i in range(n):
        same_state = series[i] == series[i + 1]
        zero_drift = (i < len(drift) and drift[i] == 0.0)
        locked.append(same_state or zero_drift)

    # Extract contiguous lock spans.
    locks: List[Dict[str, Any]] = []
    i = 0
    while i < len(locked):
        if locked[i]:
            start = i
            while i < len(locked) and locked[i]:
                i += 1
            end = i  # exclusive in the transition index, inclusive in state index
            length = end - start + 1  # number of states in the lock
            locks.append({"start": start, "end": end, "length": length})
        else:
            i += 1

    n_locks = len(locks)
    mean_lock_length = (
        sum(lk["length"] for lk in locks) / n_locks if n_locks > 0 else 0.0
    )

    return {
        "locks": locks,
        "n_locks": n_locks,
        "mean_lock_length": mean_lock_length,
    }


# ---------------------------------------------------------------------------
# Step 2 — Lock Strength
# ---------------------------------------------------------------------------


def compute_lock_strength(drift: List[float]) -> float:
    """Compute lock strength from spectral drift.

    strength = 1 - (mean(drift) / max(drift))

    Clamped to [0, 1].  Returns 1.0 for empty or all-zero drift.

    Parameters
    ----------
    drift:
        Spectral drift values.

    Returns
    -------
    Lock strength in [0, 1].
    """
    if not drift:
        return 1.0

    max_drift = max(drift)
    if max_drift == 0.0:
        return 1.0

    mean_drift = sum(drift) / len(drift)
    strength = 1.0 - (mean_drift / max_drift)
    return max(0.0, min(1.0, strength))


# ---------------------------------------------------------------------------
# Step 3 — Attractor Field
# ---------------------------------------------------------------------------


def build_attractor_field(
    state_graph: Dict[str, Any],
    series: List[Tuple[int, ...]],
) -> Dict[str, Any]:
    """Build an attractor field from the state graph and trajectory.

    Per node computes:
      - visit_count: number of times the state appears in series
      - self_loop: 1 if the node has a self-loop edge, else 0
      - in_degree: number of incoming edges
      - out_degree: number of outgoing edges

    Score = visit_count + 2 * self_loop + in_degree - out_degree
    Normalized to [0, 1].  A node is an attractor if its score == 1.0
    (i.e., it has the maximum raw score).

    Parameters
    ----------
    state_graph:
        Dict with ``nodes`` and ``edges`` from motif graph analysis.
    series:
        Ternary trajectory series.

    Returns
    -------
    dict with ``nodes`` (list of scored node dicts) and ``n_attractors``.
    """
    nodes = state_graph.get("nodes", [])
    edges = state_graph.get("edges", [])

    if not nodes:
        return {"nodes": [], "n_attractors": 0}

    # Visit counts.
    visit_count: Dict[Tuple[int, ...], int] = {}
    for s in series:
        visit_count[s] = visit_count.get(s, 0) + 1

    # Self-loops.
    self_loop: Dict[Tuple[int, ...], int] = {n: 0 for n in nodes}
    for e in edges:
        if e["from"] == e["to"]:
            self_loop[e["from"]] = 1

    # In-degree and out-degree.
    in_degree: Dict[Tuple[int, ...], int] = {n: 0 for n in nodes}
    out_degree: Dict[Tuple[int, ...], int] = {n: 0 for n in nodes}
    for e in edges:
        out_degree[e["from"]] = out_degree.get(e["from"], 0) + 1
        in_degree[e["to"]] = in_degree.get(e["to"], 0) + 1

    # Raw scores.
    raw_scores: List[float] = []
    for n in nodes:
        vc = visit_count.get(n, 0)
        sl = self_loop.get(n, 0)
        ind = in_degree.get(n, 0)
        outd = out_degree.get(n, 0)
        score = vc + 2 * sl + ind - outd
        raw_scores.append(float(score))

    # Normalize to [0, 1].
    max_score = max(raw_scores) if raw_scores else 0.0
    min_score = min(raw_scores) if raw_scores else 0.0
    score_range = max_score - min_score

    node_results: List[Dict[str, Any]] = []
    for i, n in enumerate(nodes):
        if score_range > 0:
            normalized = (raw_scores[i] - min_score) / score_range
        else:
            normalized = 1.0  # all equal → all maximal
        node_results.append({
            "state": n,
            "score": normalized,
            "is_attractor": normalized == 1.0,
        })

    n_attractors = sum(1 for nr in node_results if nr["is_attractor"])

    return {
        "nodes": node_results,
        "n_attractors": n_attractors,
    }


# ---------------------------------------------------------------------------
# Step 4 — Field Classification
# ---------------------------------------------------------------------------


def classify_resonance_field(
    lock_strength: float,
    attractor_field: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify the resonance field type.

    Rules (applied in order):
      - If n_attractors == 0 → "dispersed", confidence = 1 - lock_strength
      - If lock_strength >= 0.8 and n_attractors == 1 → "single_attractor",
        confidence = lock_strength
      - If lock_strength >= 0.8 and n_attractors > 1 → "multi_attractor",
        confidence = lock_strength * 0.8
      - If lock_strength >= 0.4 → "resonant", confidence = lock_strength
      - Otherwise → "transient", confidence = 1 - lock_strength

    Parameters
    ----------
    lock_strength:
        Scalar in [0, 1].
    attractor_field:
        Dict with ``n_attractors``.

    Returns
    -------
    dict with ``field_type`` (str) and ``confidence`` (float in [0, 1]).
    """
    n_attractors = attractor_field.get("n_attractors", 0)

    if n_attractors == 0:
        return {
            "field_type": "dispersed",
            "confidence": max(0.0, min(1.0, 1.0 - lock_strength)),
        }

    if lock_strength >= 0.8:
        if n_attractors == 1:
            return {
                "field_type": "single_attractor",
                "confidence": lock_strength,
            }
        return {
            "field_type": "multi_attractor",
            "confidence": max(0.0, min(1.0, lock_strength * 0.8)),
        }

    if lock_strength >= 0.4:
        return {
            "field_type": "resonant",
            "confidence": lock_strength,
        }

    return {
        "field_type": "transient",
        "confidence": max(0.0, min(1.0, 1.0 - lock_strength)),
    }


# ---------------------------------------------------------------------------
# Step 5 — Full Pipeline
# ---------------------------------------------------------------------------


def run_resonance_analysis(
    series: List[Tuple[int, ...]],
    drift: List[float],
    state_graph: Dict[str, Any],
) -> Dict[str, Any]:
    """Run full resonance analysis pipeline.

    Parameters
    ----------
    series:
        Ternary-encoded trajectory.
    drift:
        Spectral drift values.
    state_graph:
        State graph dict from motif graph analysis.

    Returns
    -------
    dict with ``locks``, ``lock_strength``, ``attractor_field``,
    ``field_classification``.
    """
    locks = detect_resonance_locks(series, drift)
    lock_strength = compute_lock_strength(drift)
    attractor_field = build_attractor_field(state_graph, series)
    field_classification = classify_resonance_field(lock_strength, attractor_field)

    return {
        "locks": locks,
        "lock_strength": lock_strength,
        "attractor_field": attractor_field,
        "field_classification": field_classification,
    }
