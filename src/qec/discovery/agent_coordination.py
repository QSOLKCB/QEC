"""Shared deterministic coordination state for cooperative discovery agents."""

from __future__ import annotations

from typing import Any

import numpy as np


class AgentCoordinationState:
    """Track shared targets, history, and messages for cooperating agents."""

    def __init__(self) -> None:
        self.agent_targets: dict[str, np.ndarray | None] = {}
        self.agent_history: dict[str, list[np.ndarray]] = {}
        self.generation_messages: dict[int, list[dict[str, Any]]] = {}
        self.frontier_assignments: dict[int, dict[str, np.ndarray | None]] = {}

    def update_target(self, agent_id: str, region: Any) -> None:
        aid = str(agent_id)
        self.agent_targets[aid] = None if region is None else np.asarray(region, dtype=np.float64)

    def record_history(self, agent_id: str, region: Any) -> None:
        if region is None:
            return
        aid = str(agent_id)
        self.agent_history.setdefault(aid, []).append(np.asarray(region, dtype=np.float64))

    def record_message(self, generation: int, message: dict[str, Any]) -> None:
        gen = int(generation)
        self.generation_messages.setdefault(gen, []).append(message)

    def record_frontier_assignments(
        self,
        generation: int,
        assignments: dict[str, np.ndarray | None],
    ) -> None:
        self.frontier_assignments[int(generation)] = {
            str(aid): None if region is None else np.asarray(region, dtype=np.float64)
            for aid, region in sorted(assignments.items(), key=lambda kv: kv[0])
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe deterministic snapshot of coordination state."""
        return {
            "agent_targets": {
                aid: None if region is None else region.astype(np.float64, copy=False).tolist()
                for aid, region in sorted(self.agent_targets.items(), key=lambda kv: kv[0])
            },
            "agent_history": {
                aid: [r.astype(np.float64, copy=False).tolist() for r in regions]
                for aid, regions in sorted(self.agent_history.items(), key=lambda kv: kv[0])
            },
            "generation_messages": {
                str(gen): list(messages)
                for gen, messages in sorted(self.generation_messages.items(), key=lambda kv: kv[0])
            },
            "frontier_assignments": {
                str(gen): {
                    aid: None if region is None else region.astype(np.float64, copy=False).tolist()
                    for aid, region in sorted(assignments.items(), key=lambda kv: kv[0])
                }
                for gen, assignments in sorted(self.frontier_assignments.items(), key=lambda kv: kv[0])
            },
        }
