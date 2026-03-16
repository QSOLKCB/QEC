"""Deterministic memory for tracking conjecture support over time."""

from __future__ import annotations

from typing import Any

import numpy as np


def initialize_theory_memory() -> dict[str, Any]:
    """Create an empty deterministic theory memory container."""
    return {"conjectures": {}}


def _status(rec: dict[str, Any]) -> str:
    if int(rec.get("failure_count", 0)) > 0:
        return "rejected"
    if int(rec.get("survival_count", 0)) == 0:
        return "candidate"
    mean_support = np.float64(rec.get("mean_support_score", 0.0))
    if float(mean_support) >= 0.8:
        return "supported"
    if float(mean_support) >= 0.5:
        return "fragile"
    return "candidate"


def update_theory_memory(
    theory_memory: dict[str, Any],
    conjectures: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    counterexamples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Update per-conjecture deterministic support/failure tallies."""
    mem = dict(theory_memory)
    records = dict(mem.get("conjectures", {}))
    validation_map = {str(v.get("conjecture_id", "")): v for v in validations}
    counterexample_count: dict[str, int] = {}
    for ce in counterexamples:
        cid = str(ce.get("conjecture_id", ""))
        counterexample_count[cid] = counterexample_count.get(cid, 0) + 1

    for conjecture in sorted(conjectures, key=lambda c: str(c.get("conjecture_id", ""))):
        cid = str(conjecture.get("conjecture_id", ""))
        prev = dict(records.get(cid, {}))
        times_validated_prev = int(prev.get("times_validated", 0))
        prev_mean = np.float64(prev.get("mean_support_score", 0.0))

        val = validation_map.get(cid)
        increment = 1 if val is not None else 0
        support = np.float64(val.get("support_score", 0.0)) if val is not None else np.float64(0.0)
        passes = bool(val.get("passes_tolerance", False)) if val is not None else False
        times_validated = times_validated_prev + increment
        if times_validated > 0 and increment > 0:
            mean_support = (prev_mean * np.float64(times_validated_prev) + support) / np.float64(times_validated)
        else:
            mean_support = prev_mean

        rec = {
            "conjecture_id": cid,
            "equation_string": str(conjecture.get("equation_string", prev.get("equation_string", ""))),
            "times_validated": int(times_validated),
            "mean_support_score": float(np.float64(mean_support)),
            "best_support_score": float(max(np.float64(prev.get("best_support_score", 0.0)), support)),
            "survival_count": int(prev.get("survival_count", 0) + (1 if increment and passes else 0)),
            "failure_count": int(prev.get("failure_count", 0) + (1 if increment and not passes else 0)),
            "num_counterexamples": int(prev.get("num_counterexamples", 0) + counterexample_count.get(cid, 0)),
        }
        rec["status"] = _status(rec)
        records[cid] = rec

    mem["conjectures"] = dict(sorted(records.items(), key=lambda item: item[0]))
    return mem


def summarize_theory_memory(theory_memory: dict[str, Any]) -> list[dict[str, Any]]:
    """Return sorted per-conjecture summary records."""
    records = theory_memory.get("conjectures", {})
    return [dict(records[k]) for k in sorted(records.keys())]
