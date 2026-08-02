# SPDX-License-Identifier: MPL-2.0
"""Hash-chained exchange event log."""

from __future__ import annotations

from dataclasses import dataclass

from qec.sonify.canonical import canonical_sha256


@dataclass(frozen=True)
class Event:
    sequence: int
    tick: int
    device: str
    from_state: str
    to_state: str
    action: str
    details: dict[str, object]
    previous_event_sha256: str | None
    event_sha256: str

    @classmethod
    def build(
        cls,
        *,
        sequence: int,
        tick: int,
        device: str,
        from_state: str,
        to_state: str,
        action: str,
        details: dict[str, object],
        previous_event_sha256: str | None,
    ) -> "Event":
        unsigned = {
            "sequence": sequence,
            "tick": tick,
            "device": device,
            "from_state": from_state,
            "to_state": to_state,
            "action": action,
            "details": details,
            "previous_event_sha256": previous_event_sha256,
        }
        return cls(**unsigned, event_sha256=canonical_sha256(unsigned))

    def unsigned_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "tick": self.tick,
            "device": self.device,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "action": self.action,
            "details": self.details,
            "previous_event_sha256": self.previous_event_sha256,
        }

    def as_dict(self) -> dict[str, object]:
        payload = self.unsigned_dict()
        payload["event_sha256"] = self.event_sha256
        return payload


class EventLog:
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._tick = 0

    @property
    def tick(self) -> int:
        return self._tick

    def advance(self, ticks: int = 1) -> None:
        if type(ticks) is not int or ticks < 0:
            raise ValueError("ticks must be a non-negative exact integer")
        self._tick += ticks

    def append(
        self,
        *,
        device: str,
        from_state: str,
        to_state: str,
        action: str,
        details: dict[str, object] | None = None,
    ) -> Event:
        previous = self._events[-1].event_sha256 if self._events else None
        event = Event.build(
            sequence=len(self._events),
            tick=self._tick,
            device=device,
            from_state=from_state,
            to_state=to_state,
            action=action,
            details={} if details is None else details,
            previous_event_sha256=previous,
        )
        self._events.append(event)
        return event

    def as_tuple(self) -> tuple[Event, ...]:
        return tuple(self._events)
