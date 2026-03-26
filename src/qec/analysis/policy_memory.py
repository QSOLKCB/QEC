"""v103.7.0 — Policy memory for experience replay and persistence.

Provides:
- in-memory policy storage with score tracking
- deterministic ranking of historical policy performance
- policy replay and comparison against current best
- JSON-compatible export/import for persistence

Implements experience replay for control policies: instead of
discarding the best policy after a run, store it, reuse it,
and improve over time.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- rule-based only (no stochastic selection, no learning)

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ROUND_PRECISION = 12


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


# ---------------------------------------------------------------------------
# Public API — Memory Initialization
# ---------------------------------------------------------------------------


def init_policy_memory() -> Dict[str, Any]:
    """Create an empty policy memory structure.

    Returns
    -------
    dict
        Memory dict with a ``"policies"`` key mapping to an empty dict.

    Example
    -------
    >>> mem = init_policy_memory()
    >>> mem
    {'policies': {}}
    """
    return {"policies": {}}


# ---------------------------------------------------------------------------
# Public API — Store Policy Results
# ---------------------------------------------------------------------------


def update_policy_memory(
    memory: Dict[str, Any],
    policy: Any,
    score: float,
) -> Dict[str, Any]:
    """Record a policy's performance score in memory.

    Creates a new memory dict with the updated entry — does NOT
    mutate the input *memory*.

    Parameters
    ----------
    memory : dict
        Existing policy memory (from ``init_policy_memory`` or prior call).
    policy : Policy
        The policy object to record.  Must have a ``.name`` attribute
        and a ``.to_dict()`` method.
    score : float
        Objective score achieved by the policy.

    Returns
    -------
    dict
        New memory dict with the updated policy entry.
    """
    score = _round(score)
    name = policy.name

    # Deep-copy existing policies.
    new_policies: Dict[str, Dict[str, Any]] = {}
    for k, v in sorted(memory.get("policies", {}).items()):
        new_policies[k] = {
            "policy_dict": dict(v["policy_dict"]),
            "scores": list(v["scores"]),
            "avg_score": v["avg_score"],
            "uses": v["uses"],
        }

    if name in new_policies:
        entry = new_policies[name]
        new_scores = list(entry["scores"]) + [score]
        avg = _round(sum(new_scores) / len(new_scores))
        new_policies[name] = {
            "policy_dict": entry["policy_dict"],
            "scores": new_scores,
            "avg_score": avg,
            "uses": entry["uses"] + 1,
        }
    else:
        new_policies[name] = {
            "policy_dict": policy.to_dict(),
            "scores": [score],
            "avg_score": score,
            "uses": 1,
        }

    return {"policies": new_policies}


# ---------------------------------------------------------------------------
# Public API — Retrieve Best Policies
# ---------------------------------------------------------------------------


def get_top_policies(
    memory: Dict[str, Any],
    k: int = DEFAULT_TOP_K,
) -> List[Any]:
    """Retrieve the top-k policies from memory by performance.

    Sorting rule (deterministic):
    - ``avg_score`` descending
    - ``uses`` descending (more experienced policies preferred)
    - ``name`` ascending (lexicographic tiebreaker)

    Parameters
    ----------
    memory : dict
        Policy memory.
    k : int
        Maximum number of policies to return.

    Returns
    -------
    list of Policy
        Top-k policies reconstructed from stored dicts.
    """
    from qec.analysis.policy import Policy

    policies_data = memory.get("policies", {})
    if not policies_data:
        return []

    # Build sortable list: (neg_avg_score, neg_uses, name, entry).
    items = []
    for name in sorted(policies_data.keys()):
        entry = policies_data[name]
        items.append((
            -entry["avg_score"],
            -entry["uses"],
            name,
            entry,
        ))

    items.sort()

    result = []
    for _, _, name, entry in items[:k]:
        result.append(Policy.from_dict(name, entry["policy_dict"]))

    return result


# ---------------------------------------------------------------------------
# Public API — Policy Replay
# ---------------------------------------------------------------------------


def replay_policies(
    runs: List[Dict[str, Any]],
    memory: Dict[str, Any],
    objective: Dict[str, Any],
    *,
    k: int = DEFAULT_TOP_K,
    multistate_result: Optional[Dict[str, Any]] = None,
    coupled_result: Optional[Dict[str, Any]] = None,
    max_steps: int = 5,
) -> Dict[str, Any]:
    """Replay top-k historical policies and compare against current best.

    Runs hierarchical control for each top-k policy from memory,
    compares their scores against the current best score, and
    reports whether replay improved performance.

    Parameters
    ----------
    runs : list of dict
        Run data (each with ``"strategies"`` key).
    memory : dict
        Policy memory.
    objective : dict
        Global objective weights.
    k : int
        Number of top policies to replay.
    multistate_result : dict, optional
        Precomputed multistate result.
    coupled_result : dict, optional
        Precomputed coupled dynamics result.
    max_steps : int
        Maximum meta-control steps per replay.

    Returns
    -------
    dict
        Contains:

        - ``"replayed"`` : dict — mapping policy name to replay score
        - ``"current_best"`` : float — best score from current run
        - ``"replay_best"`` : float — best score from replayed policies
        - ``"improved"`` : bool — whether replay beat current best
    """
    from qec.analysis.meta_control import run_meta_control
    from qec.analysis.strategy_adapter import (
        run_coupled_dynamics_analysis,
        run_multistate_analysis,
    )

    if multistate_result is None:
        multistate_result = run_multistate_analysis(runs)
    if coupled_result is None:
        coupled_result = run_coupled_dynamics_analysis(
            runs,
            multistate_result=multistate_result,
        )

    top = get_top_policies(memory, k=k)

    if not top:
        return {
            "replayed": {},
            "current_best": 0.0,
            "replay_best": 0.0,
            "improved": False,
        }

    # Run meta-control with current built-in policies for baseline.
    from qec.analysis.policy import get_policy

    current_policies = [
        get_policy("stability_first"),
        get_policy("sync_first"),
        get_policy("balanced"),
    ]
    current_result = run_meta_control(
        runs,
        current_policies,
        objective,
        max_steps=max_steps,
        multistate_result=multistate_result,
        coupled_result=coupled_result,
    )
    current_scores = current_result.get("scores", [0.0])
    current_best = _round(current_scores[-1]) if current_scores else 0.0

    # Replay each top policy.
    replayed: Dict[str, float] = {}
    for policy in top:
        replay_result = run_meta_control(
            runs,
            [policy],
            objective,
            max_steps=max_steps,
            multistate_result=multistate_result,
            coupled_result=coupled_result,
        )
        replay_scores = replay_result.get("scores", [0.0])
        replayed[policy.name] = _round(
            replay_scores[-1] if replay_scores else 0.0,
        )

    # Determine replay best.
    replay_best = _round(max(replayed.values())) if replayed else 0.0

    return {
        "replayed": replayed,
        "current_best": current_best,
        "replay_best": replay_best,
        "improved": replay_best > current_best,
    }


# ---------------------------------------------------------------------------
# Public API — Export / Import (JSON-compatible, no file I/O)
# ---------------------------------------------------------------------------


def export_policy_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Export policy memory as a JSON-compatible dict.

    Parameters
    ----------
    memory : dict
        Policy memory.

    Returns
    -------
    dict
        A deep copy suitable for ``json.dumps``.
    """
    exported: Dict[str, Any] = {"policies": {}}
    for name in sorted(memory.get("policies", {}).keys()):
        entry = memory["policies"][name]
        exported["policies"][name] = {
            "policy_dict": dict(entry["policy_dict"]),
            "scores": list(entry["scores"]),
            "avg_score": entry["avg_score"],
            "uses": entry["uses"],
        }
    return exported


def import_policy_memory(data: Dict[str, Any]) -> Dict[str, Any]:
    """Import policy memory from a JSON-compatible dict.

    Validates structure and recomputes averages for integrity.

    Parameters
    ----------
    data : dict
        Previously exported memory dict.

    Returns
    -------
    dict
        Validated policy memory.

    Raises
    ------
    ValueError
        If *data* has an invalid structure.
    """
    if not isinstance(data, dict) or "policies" not in data:
        raise ValueError("Invalid policy memory format: missing 'policies' key")

    policies: Dict[str, Dict[str, Any]] = {}
    for name in sorted(data["policies"].keys()):
        entry = data["policies"][name]

        if not isinstance(entry, dict):
            raise ValueError(f"Invalid entry for policy {name!r}")

        policy_dict = dict(entry.get("policy_dict", {}))
        scores = list(entry.get("scores", []))
        uses = int(entry.get("uses", len(scores)))

        # Recompute average for integrity.
        if scores:
            avg_score = _round(sum(scores) / len(scores))
        else:
            avg_score = 0.0

        policies[name] = {
            "policy_dict": policy_dict,
            "scores": scores,
            "avg_score": avg_score,
            "uses": uses,
        }

    return {"policies": policies}


# ---------------------------------------------------------------------------
# Public API — Formatting
# ---------------------------------------------------------------------------


def format_policy_memory_summary(
    memory: Dict[str, Any],
    replay_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Format policy memory and optional replay results as a summary.

    Parameters
    ----------
    memory : dict
        Policy memory.
    replay_result : dict, optional
        Output of ``replay_policies``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    lines.append("=== Policy Memory ===")
    lines.append("")

    policies_data = memory.get("policies", {})

    if not policies_data:
        lines.append("No policies stored.")
        return "\n".join(lines)

    lines.append("Stored Policies:")
    for name in sorted(policies_data.keys()):
        entry = policies_data[name]
        avg = entry["avg_score"]
        uses = entry["uses"]
        lines.append(f"  {name} -> avg_score: {avg:.2f} ({uses} runs)")

    # Show top policies.
    top = get_top_policies(memory, k=DEFAULT_TOP_K)
    if top:
        lines.append("")
        lines.append("Top Policies:")
        for i, policy in enumerate(top, 1):
            lines.append(f"  {i}. {policy.name}")

    # Show replay results if available.
    if replay_result is not None:
        lines.append("")
        lines.append("Replay Result:")
        current = replay_result.get("current_best", 0.0)
        replay_best = replay_result.get("replay_best", 0.0)
        improved = replay_result.get("improved", False)
        lines.append(f"  Current: {current:.2f}")
        lines.append(f"  Replay Best: {replay_best:.2f}")
        lines.append(f"  Improved: {improved}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API — Archetype Access
# ---------------------------------------------------------------------------


def get_archetypes(
    memory: Dict[str, Any],
    k: int = DEFAULT_TOP_K,
) -> List[Any]:
    """Retrieve top-k archetype policies from memory.

    Extracts archetypes by clustering stored policies, ranks them
    by average cluster member score, and returns the top *k*.

    Parameters
    ----------
    memory : dict
        Policy memory.
    k : int
        Maximum number of archetypes to return.

    Returns
    -------
    list of Policy
        Top-k ranked archetype policies.
    """
    from qec.analysis.policy_clustering import (
        extract_policy_archetypes,
        rank_archetypes,
    )

    archetypes = extract_policy_archetypes(memory)
    if not archetypes:
        return []

    ranked = rank_archetypes(archetypes, memory)
    return ranked[:k]


__all__ = [
    "DEFAULT_TOP_K",
    "ROUND_PRECISION",
    "export_policy_memory",
    "format_policy_memory_summary",
    "get_archetypes",
    "get_top_policies",
    "import_policy_memory",
    "init_policy_memory",
    "replay_policies",
    "update_policy_memory",
]
