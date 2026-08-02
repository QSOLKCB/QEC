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


_STATEFUL_ACTIONS = {
    OperatorAction.QUARANTINE_TRUNK,
    OperatorAction.RELEASE_ROUTE,
    OperatorAction.SEIZE_TRUNK,
}


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
        if self.action is OperatorAction.MANUAL_STEP:
            if self.value is None or self.value < 1:
                raise ValueError("manual_step requires a positive exact value")
        elif self.value is not None:
            raise ValueError("value is only valid for manual_step")

    @classmethod
    def from_dict(cls, payload: object) -> "OperatorCommand":
        if not isinstance(payload, dict):
            raise ValueError("operator command must be an object")
        required = {"action", "operator_id", "target", "reason"}
        allowed = required | {"value"}
        if not required <= set(payload) or set(payload) - allowed:
            raise ValueError("operator command fields do not match the contract")
        try:
            action = OperatorAction(payload["action"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown operator action") from exc
        command = cls(
            action=action,
            operator_id=payload["operator_id"],
            target=payload["target"],
            reason=payload["reason"],
            value=payload.get("value"),
        )
        if command.as_dict() != payload:
            raise ValueError("operator command is not canonically encoded")
        return command

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
    """Receipt-bound maintenance surface without any force-accept path."""

    def __init__(self, mode: ExchangeMode, event_log: EventLog) -> None:
        self.mode = mode
        self.event_log = event_log
        self.commands: list[OperatorCommand] = []

    def _require_enabled(self) -> None:
        if self.mode is ExchangeMode.AUTOMATIC:
            raise PermissionError("Operator Desk is disabled in automatic mode")

    def _check_permission(self, command: OperatorCommand) -> None:
        self._require_enabled()
        if command.action in {
            OperatorAction.MANUAL_STEP,
            OperatorAction.SEIZE_TRUNK,
        } and self.mode is not ExchangeMode.MANUAL:
            raise PermissionError(
                f"{command.action.value} requires manual exchange mode"
            )

    @staticmethod
    def _resolve_target(
        trunk_states: dict[str, list[TrunkState]], target: str
    ) -> tuple[str, int, list[TrunkState]]:
        selector, separator, raw_contact = target.rpartition(":")
        if not separator or not selector:
            raise ValueError("state-changing operator target must be selector:contact")
        try:
            contact = int(raw_contact)
        except ValueError as exc:
            raise ValueError("operator contact must be an integer") from exc
        states = trunk_states.get(selector)
        if states is None or not 0 <= contact < len(states):
            raise ValueError("unknown selector/contact")
        return selector, contact, states

    def _record(self, command: OperatorCommand) -> None:
        self.commands.append(command)
        self.event_log.append(
            device="operator-desk",
            from_state=DeviceState.HOME.value,
            to_state=DeviceState.HOME.value,
            action=f"operator_{command.action.value}",
            details=command.as_dict(),
        )

    def record(self, command: OperatorCommand) -> None:
        """Record a non-state-changing command.

        Stateful commands must use ``apply`` so the recorded target and the
        resulting exchange state cannot diverge.
        """
        self._check_permission(command)
        if command.action in _STATEFUL_ACTIONS:
            raise ValueError("state-changing operator commands must use apply")
        self._record(command)

    def apply(
        self,
        command: OperatorCommand,
        trunk_states: dict[str, list[TrunkState]],
    ) -> None:
        self._check_permission(command)
        if command.action in _STATEFUL_ACTIONS:
            _selector, _contact, states = self._resolve_target(
                trunk_states, command.target
            )
            contact = int(command.target.rpartition(":")[2])
            if command.action is OperatorAction.QUARANTINE_TRUNK:
                states[contact] = TrunkState.QUARANTINED
            elif command.action is OperatorAction.SEIZE_TRUNK:
                states[contact] = TrunkState.BUSY
            else:
                states[contact] = TrunkState.FREE
        self._record(command)

    def quarantine(
        self,
        trunk_states: dict[str, list[TrunkState]],
        *,
        selector: str,
        contact: int,
        operator_id: str,
        reason: str,
    ) -> None:
        self.apply(
            OperatorCommand(
                action=OperatorAction.QUARANTINE_TRUNK,
                operator_id=operator_id,
                target=f"{selector}:{contact}",
                reason=reason,
            ),
            trunk_states,
        )

    def seize(
        self,
        trunk_states: dict[str, list[TrunkState]],
        *,
        selector: str,
        contact: int,
        operator_id: str,
        reason: str,
    ) -> None:
        self.apply(
            OperatorCommand(
                action=OperatorAction.SEIZE_TRUNK,
                operator_id=operator_id,
                target=f"{selector}:{contact}",
                reason=reason,
            ),
            trunk_states,
        )

    def release(
        self,
        trunk_states: dict[str, list[TrunkState]],
        *,
        selector: str,
        contact: int,
        operator_id: str,
        reason: str,
    ) -> None:
        self.apply(
            OperatorCommand(
                action=OperatorAction.RELEASE_ROUTE,
                operator_id=operator_id,
                target=f"{selector}:{contact}",
                reason=reason,
            ),
            trunk_states,
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

    def replay(self, *, operator_id: str, target: str, reason: str) -> None:
        self.record(
            OperatorCommand(
                action=OperatorAction.REPLAY_REQUEST,
                operator_id=operator_id,
                target=target,
                reason=reason,
            )
        )

    def manual_step(
        self,
        *,
        operator_id: str,
        target: str,
        reason: str,
        value: int,
    ) -> None:
        self.record(
            OperatorCommand(
                action=OperatorAction.MANUAL_STEP,
                operator_id=operator_id,
                target=target,
                reason=reason,
                value=value,
            )
        )
