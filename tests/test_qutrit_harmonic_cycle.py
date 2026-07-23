"""End-to-end tests for exact correction through harmonic observation."""

from __future__ import annotations

import pytest

from qec.decoder.qutrit import (
    ExactDecoder,
    paulis_of_weight,
    shor_nine_qutrit_code,
    ternary_golay_qutrit_code,
)
from qec.decoder.qutrit.cycle import run_harmonic_cycle
from qec.sonify.qutrit_harmonics import encode_harmonics, phasor


@pytest.mark.parametrize(
    ("factory", "max_weight"),
    [
        (shor_nine_qutrit_code, 1),
        (ternary_golay_qutrit_code, 2),
    ],
)
def test_exact_errors_survive_harmonic_round_trip(factory, max_weight):
    code = factory()
    decoder = ExactDecoder(code, max_weight=max_weight)
    count = 0
    for weight in range(1, max_weight + 1):
        for error in paulis_of_weight(code.n, weight):
            result = run_harmonic_cycle(decoder, error)
            assert result.accepted
            assert result.success
            count += 1
    assert count > 0


def test_inconsistent_sound_is_never_used_for_correction():
    code = shor_nine_qutrit_code()
    decoder = ExactDecoder(code)
    error = next(paulis_of_weight(code.n, 1))
    syndrome = code.syndrome(error)
    samples = encode_harmonics(syndrome)
    samples[2] = (phasor((syndrome[0] + 1) % 3, 2),) + samples[2][1:]

    result = run_harmonic_cycle(
        decoder,
        error,
        samples=samples,
        tolerance=2.0,
    )
    assert not result.accepted
    assert result.correction is None
    assert result.residual == error
    assert not result.success
