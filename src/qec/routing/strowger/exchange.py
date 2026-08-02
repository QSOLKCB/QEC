# SPDX-License-Identifier: MPL-2.0
"""Deterministic linefinder, selector, connector, and tone-routing engine."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

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
from .pulse import PulseCodec
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
        log = EventLog()
        desk = OperatorDesk(self.mode, log)
        if prepare_operator is not None:
            if self.mode is ExchangeMode.AUTOMATIC:
                raise PermissionError(
                    "prepare_operator requires supervised or manual mode"
                )
            prepare_operator(desk, self.trunk_states)

        linefinder = self._first_free(self.linefinder_states)
        if linefinder is None:
            receipt = build_receipt(
                config=self.config,
                request=request,
                mode=self.mode.value,
                linefinder=None,
                selector_trunks=(),
                connector=None,
                expected_tones=None,
                observed_tones=None,
                tone_verification=None,
                outcome=RouteOutcome.ALL_TRUNKS_BUSY.value,
                events=log.as_tuple(),
                operator_commands=tuple(
                    command.as_dict() for command in desk.commands
                ),
                fault_plan=fault_plan.as_dict(),
            )
            return RouteResult(RouteOutcome.ALL_TRUNKS_BUSY, receipt)

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
        pulse_groups: dict[int, list[int]] = {}
        for pulse in pulses:
            pulse_groups.setdefault(pulse.stage, []).append(pulse.ordinal)

        selected: list[int] = []
        outcome = RouteOutcome.COMMITTED
        connector: tuple[int, int] | None = None
        expected = None
        observed = None
        verification = None

        for stage_index, stage in enumerate(self.config.selectors):
            device = f"selector-{stage_index}:{stage.name}"
            expected_pulses = pulse_groups[stage_index]
            actual_count = 0
            log.append(
                device=device,
                from_state=DeviceState.HOME.value,
                to_state=DeviceState.RECEIVING_PULSES.value,
                action="receive_digit",
                details={
                    "digit": request.digits[stage_index],
                    "expected_pulses": len(expected_pulses),
                    "radix": stage.radix,
                },
            )
            for ordinal in expected_pulses:
                if (stage_index, ordinal) in fault_plan.missed_pulses:
                    log.append(
                        device=device,
                        from_state=DeviceState.RECEIVING_PULSES.value,
                        to_state=DeviceState.RECEIVING_PULSES.value,
                        action="missed_pulse",
                        details={"ordinal": ordinal},
                    )
                    continue
                actual_count += 1
                if (stage_index, ordinal) in fault_plan.duplicate_pulses:
                    actual_count += 1
                    log.append(
                        device=device,
                        from_state=DeviceState.VERTICAL_STEPPING.value,
                        to_state=DeviceState.VERTICAL_STEPPING.value,
                        action="duplicate_pulse",
                        details={"ordinal": ordinal},
                    )
            if stage_index in fault_plan.stuck_selectors:
                log.append(
                    device=device,
                    from_state=DeviceState.VERTICAL_STEPPING.value,
                    to_state=DeviceState.FAULT.value,
                    action="selector_stuck",
                    details={"actual_pulses": actual_count},
                )
                outcome = RouteOutcome.SELECTOR_FAULT
                break
            expected_count = self.pulse_codec.pulse_count(
                request.digits[stage_index], stage.radix
            )
            if actual_count != expected_count:
                log.append(
                    device=device,
                    from_state=DeviceState.VERTICAL_STEPPING.value,
                    to_state=DeviceState.FAULT.value,
                    action="pulse_count_mismatch",
                    details={
                        "expected": expected_count,
                        "observed": actual_count,
                    },
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
            observed = observe_with_offsets(
                expected, fault_plan.tone_offsets_hz
            )
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
            linefinder=linefinder,
            selector_trunks=tuple(selected),
            connector=connector,
            expected_tones=expected.as_dict() if expected else None,
            observed_tones=observed.as_dict() if observed else None,
            tone_verification=verification,
            outcome=outcome.value,
            events=log.as_tuple(),
            operator_commands=tuple(
                command.as_dict() for command in desk.commands
            ),
            fault_plan=fault_plan.as_dict(),
        )

        self.linefinder_states[linefinder] = TrunkState.FREE
        for stage, trunk in zip(self.config.selectors, selected):
            if self.trunk_states[stage.name][trunk] is TrunkState.BUSY:
                self.trunk_states[stage.name][trunk] = TrunkState.FREE
        return RouteResult(outcome=outcome, receipt=receipt)
