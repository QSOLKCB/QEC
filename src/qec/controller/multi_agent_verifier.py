"""Deterministic multi-agent verification wrapper.

Runs ``verify_run`` multiple times with identical inputs and checks that
all agents produce the same result.  No networking, no threads — just a
loop and a comparison.

Version: v81.2.0
"""

from __future__ import annotations

from typing import Any, Dict, List

from qec.controller.replay_engine import verify_run


def run_agents(
    initial_input: Dict[str, Any],
    history: List[Dict[str, Any]],
    config: Dict[str, Any],
    n_agents: int,
) -> List[Dict[str, Any]]:
    """Run *n_agents* independent ``verify_run`` invocations."""
    results: List[Dict[str, Any]] = []
    for _ in range(n_agents):
        r = verify_run(initial_input, history, config)
        results.append(r)
    return results


def compute_consensus(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare final hashes across agent results and report agreement."""
    hashes = [r["final_hash"] for r in results]
    first = hashes[0]
    matches = [h == first for h in hashes]
    agreement = sum(matches) / len(matches)
    return {
        "consensus": agreement == 1.0,
        "agreement_ratio": agreement,
        "consensus_hash": first,
    }


def verify_multi_agent(
    initial_input: Dict[str, Any],
    history: List[Dict[str, Any]],
    config: Dict[str, Any],
    proof: Dict[str, Any] | None = None,
    n_agents: int = 3,
) -> Dict[str, Any]:
    """Run multi-agent verification and return consensus report.

    Parameters
    ----------
    initial_input : dict
        Original input data.
    history : list[dict]
        Original FSM history trace.
    config : dict
        FSM configuration.
    proof : dict or None
        Execution proof (reserved for future use).
    n_agents : int
        Number of independent verification runs.

    Returns
    -------
    dict
        Keys: ``n_agents``, ``consensus``, ``agreement_ratio``,
        ``consensus_hash``, ``results``.
    """
    results = run_agents(initial_input, history, config, n_agents)
    consensus = compute_consensus(results)
    return {
        "n_agents": n_agents,
        "consensus": consensus["consensus"],
        "agreement_ratio": consensus["agreement_ratio"],
        "consensus_hash": consensus["consensus_hash"],
        "results": results,
    }
