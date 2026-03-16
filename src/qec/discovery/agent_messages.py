"""Basic deterministic agent messaging primitives for cooperation."""

from __future__ import annotations

from typing import Any


BASIN_FOUND = "BASIN_FOUND"
REGION_EXPLORED = "REGION_EXPLORED"
ESCAPE_EVENT = "ESCAPE_EVENT"
FRONTIER_EXPLORED = "FRONTIER_EXPLORED"


class AgentMessage:
    """Simple deterministic carrier for inter-agent discovery signals."""

    def __init__(
        self,
        agent_id: str,
        message_type: str,
        payload: Any = None,
        generation: int = 0,
    ) -> None:
        self.agent_id = str(agent_id)
        self.message_type = str(message_type)
        self.payload = payload
        self.generation = int(generation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "generation": self.generation,
        }
