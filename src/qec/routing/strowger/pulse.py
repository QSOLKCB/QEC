# SPDX-License-Identifier: MPL-2.0
"""Mixed-radix pulse encoding for deterministic selector stepping."""

from __future__ import annotations

from dataclasses import dataclass

from qec.sonify.canonical import require_int


@dataclass(frozen=True)
class Pulse:
    stage: int
    ordinal: int
    tick: int

    def as_dict(self) -> dict[str, int]:
        return {"stage": self.stage, "ordinal": self.ordinal, "tick": self.tick}


class PulseCodec:
    """Encode zero as a full-radix pulse train, as historical dial systems did."""

    def __init__(
        self,
        *,
        pulse_ticks: int = 1,
        inter_pulse_ticks: int = 1,
        digit_gap_ticks: int = 4,
    ) -> None:
        self.pulse_ticks = require_int(pulse_ticks, "pulse_ticks", minimum=1)
        self.inter_pulse_ticks = require_int(
            inter_pulse_ticks, "inter_pulse_ticks", minimum=1
        )
        self.digit_gap_ticks = require_int(
            digit_gap_ticks, "digit_gap_ticks", minimum=1
        )

    @staticmethod
    def pulse_count(digit: int, radix: int) -> int:
        require_int(digit, "digit", minimum=0)
        require_int(radix, "radix", minimum=2)
        if digit >= radix:
            raise ValueError("digit must be smaller than radix")
        return radix if digit == 0 else digit

    def encode(
        self, digits: tuple[int, ...], radices: tuple[int, ...]
    ) -> tuple[Pulse, ...]:
        if len(digits) != len(radices):
            raise ValueError("digits and radices must have equal length")
        pulses: list[Pulse] = []
        tick = 0
        for stage, (digit, radix) in enumerate(zip(digits, radices)):
            count = self.pulse_count(digit, radix)
            for ordinal in range(1, count + 1):
                pulses.append(Pulse(stage=stage, ordinal=ordinal, tick=tick))
                tick += self.pulse_ticks + self.inter_pulse_ticks
            tick += self.digit_gap_ticks
        return tuple(pulses)
