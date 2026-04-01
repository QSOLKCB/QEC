"""Deterministic forward-chaining law engine (v121.0.0).

This module provides a minimal deterministic law engine with bounded,
stable ordering semantics.

Design constraints:
- no randomness
- no global mutable state
- no mutation of input state
- deterministic selection among matching laws
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import copy


@dataclass(frozen=True)
class Law:
    """Single deterministic law definition."""

    law_id: str
    priority: int
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    enabled: bool = True


class LawEngine:
    """Deterministic forward-chaining engine executing first matching law."""

    def __init__(self, law_registry: Dict[str, Law]) -> None:
        self.law_registry: Dict[str, Law] = dict(law_registry)

    def register_law(self, law: Law) -> None:
        """Register or replace a law by id."""
        self.law_registry[law.law_id] = law

    def sort_laws(self) -> List[Law]:
        """Return laws sorted by descending priority then lexicographic law_id."""
        return sorted(
            self.law_registry.values(),
            key=lambda law: (-law.priority, law.law_id),
        )

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate laws deterministically and execute the first matching law.

        Output schema:
            {
                "law_triggered": bool,
                "executed_law_id": str | None,
                "priority": int | None,
                "action_result": dict,
                "first_matched_law_id": str | None,
                "evaluation_trace": tuple[str, ...],
            }
        """
        sorted_laws = self.sort_laws()

        first_matched_law_id: Optional[str] = None
        evaluation_trace: List[str] = []
        executed_law: Optional[Law] = None

        for law in sorted_laws:
            if not law.enabled:
                continue

            evaluation_trace.append(law.law_id)
            condition_state = copy.deepcopy(state)
            if law.condition(condition_state):
                first_matched_law_id = law.law_id
                if executed_law is None:
                    executed_law = law
                    break

        if executed_law is None:
            return {
                "law_triggered": False,
                "executed_law_id": None,
                "priority": None,
                "action_result": {},
                "first_matched_law_id": first_matched_law_id,
                "evaluation_trace": tuple(evaluation_trace),
            }

        action_state = copy.deepcopy(state)
        action_result = dict(executed_law.action(action_state))

        return {
            "law_triggered": True,
            "executed_law_id": executed_law.law_id,
            "priority": executed_law.priority,
            "action_result": action_result,
            "first_matched_law_id": first_matched_law_id,
            "evaluation_trace": tuple(evaluation_trace),
        }
