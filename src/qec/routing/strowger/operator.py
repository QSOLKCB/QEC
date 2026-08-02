# SPDX-License-Identifier: MPL-2.0
"""Optional Operator Desk for supervised and manual exchange operation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from qec.sonify.canonical import require_nonempty_text

from .events import EventLog
from .model import DeviceState, ExchangeMode, TrunkState


class OperatorAction(str, Enum):
    INSPECT = "inspect"
    QUARANTINE_TRUNK = "quarantine_trunk"
    RELEASE_ROUTE = "release_route"
    REPLAY_REQUEST = "replay_request"
    MANUAL_STEP = "manual_step"
    SEIZE_TRUNK = "seize_trunk"


@dataclass(frozen=True)
class OperatorCommand:
    action: OperatorAction
    operator_id: str
    target: str
    reason: str
    value: int | None = None

    def __post_init__(self) -> None:
        require_nonempty_text(self.operator_id, "operator_id")
        require_nonempty_text(self.target, "target")
        require_nonempty_text(self.reason, "reason")
        if self.value is not None and type(self.value) is not int:
            raise TypeError("value must be an exact int when present")

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "action": self.action.value,
            "operator_id": self.operator_id,
            "target": self.target,
            "reason": self.reason,
        }
        if self.value is not None:
            payload["value"] = self.value
        return payload


class OperatorDesk:
    """Receipt-bound maintenance surface.

    It can inspect, quarantine, release, replay, seize, and step devices.
    It has no force-accept operation and cannot mutate decoder outputs.
    """

    def __init__(self, mode: ExchangeMode, event_log: EventLog) -> None:
        self.mode = mode
        self.event_log = event_log
        self.commands: list[OperatorCommand] = []

    def _require_enabled(self) -> None:
        if self.mode is ExchangeMode.AUTOMATIC:
            raise PermissionError("Operator Desk is disabled in automatic mode")

    def record(self, command: OperatorCommand) -> None:
        self._require_enabled()
        if command.action in {
            OperatorAction.MANUAL_STEP,
            OperatorAction.SEIZE_TRUNK,
        } and self.mode is not ExchangeMode.MANUAL:
            raise PermissionError(
                f"{command.action.value} requires manual exchange mode"
            )
        self.commands.append(command)
        self.event_log.append(
            device="operator-desk",
            from_state=DeviceState.HOME.value,
            to_state=DeviceState.HOME.value,
            action=f"operator_{command.action.value}",
            details=command.as_dict(),
        )

    def quarantine(
        self,
        trunk_states: dict[str, list[TrunkState]],
        *,
        selector: str,
        contact: int,
        operator_id: str,
        reason: str,
    ) -> None:
        self._require_enabled()
        states = trunk_states.get(selector)
        if states is None or not 0 <= contact < len(states):
            raise ValueError("unknown selector/contact")
        states[contact] = TrunkState.QUARANTINED
        self.record(
            OperatorCommand(
                action=OperatorAction.QUARANTINE_TRUNK,
                operator_id=operator_id,
                target=f"{selector}:{contact}",
                reason=reason,
            )
        )

    def inspect(self, *, operator_id: str, target: str, reason: str) -> None:
        self.record(
            OperatorCommand(
                action=OperatorAction.INSPECT,
                operator_id=operator_id,
                target=target,
                reason=reason,
            )
        )
