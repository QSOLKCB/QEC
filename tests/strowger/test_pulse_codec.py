# SPDX-License-Identifier: MPL-2.0
"""Mixed-radix pulse codec tests."""

from qec.routing.strowger import PulseCodec


def test_zero_uses_full_radix_pulse_train() -> None:
    codec = PulseCodec()
    pulses = codec.encode((0, 2), (10, 3))
    assert len([pulse for pulse in pulses if pulse.stage == 0]) == 10
    assert len([pulse for pulse in pulses if pulse.stage == 1]) == 2


def test_pulse_timing_is_deterministic() -> None:
    codec = PulseCodec(pulse_ticks=2, inter_pulse_ticks=1, digit_gap_ticks=5)
    first = codec.encode((2, 1), (3, 4))
    second = codec.encode((2, 1), (3, 4))
    assert first == second
    assert [pulse.tick for pulse in first] == [0, 3, 11]
