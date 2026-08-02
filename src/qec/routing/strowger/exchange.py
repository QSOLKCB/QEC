# SPDX-License-Identifier: MPL-2.0
"""Deterministic linefinder, selector, connector, and tone-routing engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .events import EventLog
from .model import (
    DeviceState,
    ExchangeConfig,
    ExchangeMode,
    FaultPlan,
    RouteOutcome,
    RouteRequest,
    TrunkState,
)
from .operator import OperatorDesk
from .pulse import Pulse, PulseCodec
from .receipts import build_receipt
from .tones import derive_tone_signature, observe_with_offsets, verify_tones


@dataclass(frozen=True)
class RouteResult:
    outcome: RouteOutcome
    receipt: dict[str, object]


class StrowgerExchange:
    def __init__(
        self,
        config: ExchangeConfig,
        *,
        mode: ExchangeMode = ExchangeMode.AUTOMATIC,
        pulse_codec: PulseCodec | None = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self.pulse_codec = PulseCodec() if pulse_codec is None else pulse_codec
        self.linefinder_states = [TrunkState.FREE] * config.linefinders
        self.trunk_states = {
            stage.name: [TrunkState.FREE] * stage.trunks
            for stage in config.selectors
        }

    @staticmethod
    def _first_free(states: list[TrunkState]) -> int | None:
        for index, state in enumerate(states):
            if state is TrunkState.FREE:
                return index
        return None

    @staticmethod
    def _advance_to(log: EventLog, tick: int) -> None:
        if tick < log.tick:
            raise ValueError("pulse timing moved backwards")
        log.advance(tick - log.tick)

    def state_snapshot(self) -> dict[str, object]:
        return {
            "linefinders": [state.value for state in self.linefinder_states],
            "trunks": {
                stage.name: [state.value for state in self.trunk_states[stage.name]]
                for stage in self.config.selectors
            },
        }

    def _validate_fault_plan(
        self, request: RouteRequest, fault_plan: FaultPlan
    ) -> None:
        radices = self.config.route_radices
        for name, pairs in (
            ("missed_pulses", fault_plan.missed_pulses),
            ("duplicate_pulses", fault_plan.duplicate_pulses),
        ):
            for stage, ordinal in pairs:
                if stage >= len(radices):
                    raise ValueError(f"{name} references an unknown route stage")
                count = self.pulse_codec.pulse_count(
                    request.digits[stage], radices[stage]
                )
                if ordinal > count:
                    raise ValueError(f"{name} references a nonexistent pulse")
        if any(stage >= len(self.config.selectors) for stage in fault_plan.stuck_selectors):
            raise ValueError("stuck_selectors may reference selector stages only")

    def _audit_operator_state(
        self,
        *,
        initial_state: dict[str, object],
        commands: tuple,
    ) -> None:
        expected_trunks = {
            stage.name: [
                TrunkState(value)
                for value in initial_state["trunks"][stage.name]  # type: ignore[index]
            ]
            for stage in self.config.selectors
        }
        audit_desk = OperatorDesk(self.mode, EventLog())
        for command in commands:
            audit_desk.apply(command, expected_trunks)
        expected = {
            name: [state.value for state in states]
            for name, states in expected_trunks.items()
        }
        observed = self.state_snapshot()["trunks"]
        if observed != expected:
            raise ValueError(
                "operator state mutation was not explained by receipt-bound commands"
            )

    def _process_digit_pulses(
        self,
        *,
        log: EventLog,
        device: str,
        stage_index: int,
        digit: int,
        radix: int,
        pulses: list[Pulse],
        fault_plan: FaultPlan,
        stepping_state: DeviceState,
        begin_action: str,
    ) -> bool:
        log.append(
            device=device,
            from_state=DeviceState.HOME.value,
            to_state=DeviceState.RECEIVING_PULSES.value,
            action="receive_digit",
            details={
                "digit": digit,
                "expected_pulses": len(pulses),
                "radix": radix,
            },
        )
        self._advance_to(log, pulses[0].tick)
        log.append(
            device=device,
            from_state=DeviceState.RECEIVING_PULSES.value,
            to_state=stepping_state.value,
            action=begin_action,
            details={"pulse_count": len(pulses)},
        )
        actual_count = 0
        for pulse in pulses:
            self._advance_to(log, pulse.tick)
            pair = (stage_index, pulse.ordinal)
            if pair in fault_plan.missed_pulses:
                log.append(
                    device=device,
                    from_state=stepping_state.value,
                    to_state=stepping_state.value,
                    action="missed_pulse",
                    details={"ordinal": pulse.ordinal, "encoded_tick": pulse.tick},
                )
                continue
            actual_count += 1
            log.append(
                device=device,
                from_state=stepping_state.value,
                to_state=stepping_state.value,
                action="pulse_received",
                details={"ordinal": pulse.ordinal, "encoded_tick": pulse.tick},
            )
            if pair in fault_plan.duplicate_pulses:
                actual_count += 1
                log.append(
                    device=device,
                    from_state=stepping_state.value,
                    to_state=stepping_state.value,
                    action="duplicate_pulse",
                    details={"ordinal": pulse.ordinal, "encoded_tick": pulse.tick},
                )
        expected_count = self.pulse_codec.pulse_count(digit, radix)
        if actual_count != expected_count:
            log.append(
                device=device,
                from_state=stepping_state.value,
                to_state=DeviceState.FAULT.value,
                action="pulse_count_mismatch",
                details={"expected": expected_count, "observed": actual_count},
            )
            return False
        return True

    def route(
        self,
        request: RouteRequest,
        *,
        faults: FaultPlan | None = None,
        prepare_operator: Callable[
            [OperatorDesk, dict[str, list[TrunkState]]], None
        ] | None = None,
    ) -> RouteResult:
        request.validate_against(self.config)
        fault_plan = FaultPlan() if faults is None else faults
        self._validate_fault_plan(request, fault_plan)
        log = EventLog()
        desk = OperatorDesk(self.mode, log)
        initial_state = self.state_snapshot()
        if prepare_operator is not None:
            if self.mode is ExchangeMode.AUTOMATIC:
                raise PermissionError(
                    "prepare_operator requires supervised or manual mode"
                )
            prepare_operator(desk, self.trunk_states)
            self._audit_operator_state(
                initial_state=initial_state,
                commands=tuple(desk.commands),
            )

        pre_route_state = self.state_snapshot()
        linefinder = self._first_free(self.linefinder_states)
        if linefinder is None:
            receipt = build_receipt(
                config=self.config,
                request=request,
                mode=self.mode.value,
                initial_state=initial_state,
                pre_route_state=pre_route_state,
                linefinder=None,
                selector_trunks=(),
                connector=None,
                expected_tones=None,
                observed_tones=None,
                tone_verification=None,
                outcome=RouteOutcome.NO_LINEFINDER_AVAILABLE.value,
                events=log.as_tuple(),
                operator_commands=tuple(
                    command.as_dict() for command in desk.commands
                ),
                fault_plan=fault_plan.as_dict(),
            )
            return RouteResult(RouteOutcome.NO_LINEFINDER_AVAILABLE, receipt)

        self.linefinder_states[linefinder] = TrunkState.BUSY
        log.append(
            device=f"linefinder-{linefinder}",
            from_state=DeviceState.HOME.value,
            to_state=DeviceState.SEIZED.value,
            action="seize_first_free_linefinder",
            details={"request_id": request.request_id},
        )

        pulses = self.pulse_codec.encode(
            request.digits, self.config.route_radices
        )
        pulse_groups: dict[int, list[Pulse]] = {}
        for pulse in pulses:
            pulse_groups.setdefault(pulse.stage, []).append(pulse)

        selected: list[int] = []
        outcome = RouteOutcome.COMMITTED
        connector: tuple[int, int] | None = None
        expected = None
        observed = None
        verification = None

        for stage_index, stage in enumerate(self.config.selectors):
            device = f"selector-{stage_index}:{stage.name}"
            valid = self._process_digit_pulses(
                log=log,
                device=device,
                stage_index=stage_index,
                digit=request.digits[stage_index],
                radix=stage.radix,
                pulses=pulse_groups[stage_index],
                fault_plan=fault_plan,
                stepping_state=DeviceState.VERTICAL_STEPPING,
                begin_action="begin_vertical_stepping",
            )
            if not valid:
                outcome = RouteOutcome.SELECTOR_FAULT
                break
            if stage_index in fault_plan.stuck_selectors:
                log.append(
                    device=device,
                    from_state=DeviceState.VERTICAL_STEPPING.value,
                    to_state=DeviceState.FAULT.value,
                    action="selector_stuck",
                    details={"stage": stage_index},
                )
                outcome = RouteOutcome.SELECTOR_FAULT
                break
            log.append(
                device=device,
                from_state=DeviceState.VERTICAL_STEPPING.value,
                to_state=DeviceState.TRUNK_HUNTING.value,
                action="level_selected",
                details={"level": request.digits[stage_index]},
            )
            trunk = self._first_free(self.trunk_states[stage.name])
            if trunk is None:
                log.append(
                    device=device,
                    from_state=DeviceState.TRUNK_HUNTING.value,
                    to_state=DeviceState.BUSY.value,
                    action="all_trunks_busy",
                    details={"tested": len(self.trunk_states[stage.name])},
                )
                outcome = RouteOutcome.ALL_TRUNKS_BUSY
                break
            self.trunk_states[stage.name][trunk] = TrunkState.BUSY
            selected.append(trunk)
            log.append(
                device=device,
                from_state=DeviceState.TRUNK_HUNTING.value,
                to_state=DeviceState.CONNECTED.value,
                action="first_free_trunk_selected",
                details={"contact": trunk},
            )

        if outcome is RouteOutcome.COMMITTED:
            connector_stages = (
                (
                    len(self.config.selectors),
                    "connector-vertical",
                    request.digits[-2],
                    self.config.connector_vertical_radix,
                    DeviceState.VERTICAL_STEPPING,
                    "begin_vertical_stepping",
                    "vertical_coordinate_selected",
                ),
                (
                    len(self.config.selectors) + 1,
                    "connector-rotary",
                    request.digits[-1],
                    self.config.connector_rotary_radix,
                    DeviceState.ROTARY_STEPPING,
                    "begin_rotary_stepping",
                    "rotary_coordinate_selected",
                ),
            )
            for (
                stage_index,
                device,
                digit,
                radix,
                stepping_state,
                begin_action,
                selected_action,
            ) in connector_stages:
                valid = self._process_digit_pulses(
                    log=log,
                    device=device,
                    stage_index=stage_index,
                    digit=digit,
                    radix=radix,
                    pulses=pulse_groups[stage_index],
                    fault_plan=fault_plan,
                    stepping_state=stepping_state,
                    begin_action=begin_action,
                )
                if not valid:
                    outcome = RouteOutcome.CONNECTOR_FAULT
                    break
                log.append(
                    device=device,
                    from_state=stepping_state.value,
                    to_state=DeviceState.CONTACT_TEST.value,
                    action=selected_action,
                    details={"coordinate": digit},
                )

        if outcome is RouteOutcome.COMMITTED:
            connector = (request.digits[-2], request.digits[-1])
            log.append(
                device="connector",
                from_state=DeviceState.HOME.value,
                to_state=DeviceState.CONNECTED.value,
                action="two_axis_destination_selected",
                details={
                    "vertical": connector[0],
                    "rotary": connector[1],
                    "destination": request.destination,
                },
            )
            expected = derive_tone_signature(
                config=self.config,
                digits=request.digits,
                trunk_path=tuple(selected),
                destination=request.destination,
            )
            observed = observe_with_offsets(expected, fault_plan.tone_offsets_hz)
            verification = verify_tones(
                expected,
                observed,
                tolerance_hz=self.config.tone_tolerance_hz,
            )
            log.append(
                device="tone-verifier",
                from_state=DeviceState.CONNECTED.value,
                to_state=DeviceState.VERIFYING.value,
                action="verify_route_tones",
                details=verification,
            )
            if verification["verified"]:
                log.append(
                    device="exchange",
                    from_state=DeviceState.VERIFYING.value,
                    to_state=DeviceState.COMMITTED.value,
                    action="commit_verified_route",
                    details={"destination": request.destination},
                )
            else:
                outcome = RouteOutcome.TONE_MISMATCH
                log.append(
                    device="exchange",
                    from_state=DeviceState.VERIFYING.value,
                    to_state=DeviceState.FAULT.value,
                    action="reject_tone_mismatch",
                    details={"destination": request.destination},
                )

        receipt = build_receipt(
            config=self.config,
            request=request,
            mode=self.mode.value,
            initial_state=initial_state,
            pre_route_state=pre_route_state,
            linefinder=linefinder,
            selector_trunks=tuple(selected),
            connector=connector,
            expected_tones=expected.as_dict() if expected else None,
            observed_tones=observed.as_dict() if observed else None,
            tone_verification=verification,
            outcome=outcome.value,
            events=log.as_tuple(),
            operator_commands=tuple(command.as_dict() for command in desk.commands),
            fault_plan=fault_plan.as_dict(),
        )

        self.linefinder_states[linefinder] = TrunkState.FREE
        for stage, trunk in zip(self.config.selectors, selected):
            if self.trunk_states[stage.name][trunk] is TrunkState.BUSY:
                self.trunk_states[stage.name][trunk] = TrunkState.FREE
        return RouteResult(outcome=outcome, receipt=receipt)
