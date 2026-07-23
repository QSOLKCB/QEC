"""Independent guards against false correction and readout positives."""

from __future__ import annotations

import numpy as np
import pytest

from qec.decoder.qutrit import (
    ExactDecoder,
    Pauli,
    cyclic_five_qutrit_code,
    paulis_of_weight,
    shor_nine_qutrit_code,
    ternary_golay_qutrit_code,
)
from qec.decoder.qutrit.cycle import run_harmonic_cycle
from qec.sonify.qutrit_harmonics import encode_harmonics, phasor


@pytest.mark.parametrize(
    ("factory", "invalid_radius"),
    [
        (cyclic_five_qutrit_code, 2),
        (shor_nine_qutrit_code, 2),
        (ternary_golay_qutrit_code, 3),
    ],
)
def test_decoder_rejects_a_radius_beyond_the_code_distance(
    factory,
    invalid_radius,
):
    with pytest.raises(ValueError, match="correction conditions"):
        ExactDecoder(factory(), max_weight=invalid_radius)


def test_symplectic_sign_matches_direct_qutrit_weyl_matrices():
    omega = np.exp(2j * np.pi / 3)
    shift = np.roll(np.eye(3, dtype=complex), 1, axis=0)
    phase = np.diag([1, omega, omega**2])

    for x1 in range(3):
        for z1 in range(3):
            left = np.linalg.matrix_power(shift, x1) @ np.linalg.matrix_power(
                phase,
                z1,
            )
            for x2 in range(3):
                for z2 in range(3):
                    right = np.linalg.matrix_power(
                        shift,
                        x2,
                    ) @ np.linalg.matrix_power(phase, z2)
                    symplectic = (x1 * z2 - z1 * x2) % 3
                    np.testing.assert_allclose(
                        left @ right,
                        omega ** (-symplectic) * (right @ left),
                        atol=1e-12,
                    )


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            cyclic_five_qutrit_code,
            (2, 0, 2, 2),
        ),
        (
            shor_nine_qutrit_code,
            (2, 0, 0, 0, 0, 0, 2, 0),
        ),
        (
            ternary_golay_qutrit_code,
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        ),
    ],
)
def test_independent_golden_syndromes(factory, expected):
    code = factory()
    error = Pauli(
        (1,) + (0,) * (code.n - 1),
        (2,) + (0,) * (code.n - 1),
    )
    assert code.syndrome(error) == expected


@pytest.mark.parametrize(
    ("factory", "max_weight"),
    [
        (cyclic_five_qutrit_code, 1),
        (shor_nine_qutrit_code, 1),
        (ternary_golay_qutrit_code, 2),
    ],
)
def test_every_certified_error_rejects_each_harmonic_fault(
    factory,
    max_weight,
):
    code = factory()
    decoder = ExactDecoder(code, max_weight=max_weight)
    for weight in range(1, max_weight + 1):
        for error in paulis_of_weight(code.n, weight):
            syndrome = code.syndrome(error)

            discord = encode_harmonics(syndrome)
            discord[2] = (
                phasor((syndrome[0] + 1) % 3, 2),
            ) + discord[2][1:]
            result = run_harmonic_cycle(
                decoder,
                error,
                samples=discord,
                tolerance=2.0,
            )
            assert not result.accepted
            assert result.correction is None

            dark_break = encode_harmonics(syndrome)
            dark_break[3] = (0j,) + dark_break[3][1:]
            result = run_harmonic_cycle(
                decoder,
                error,
                samples=dark_break,
            )
            assert not result.accepted
            assert result.correction is None


@pytest.mark.parametrize(
    "factory",
    [cyclic_five_qutrit_code, shor_nine_qutrit_code],
)
def test_out_of_radius_logical_miscorrection_is_not_reported_as_success(
    factory,
):
    code = factory()
    decoder = ExactDecoder(code, max_weight=1)
    found_logical_failure = False
    for error in paulis_of_weight(code.n, 2):
        try:
            result = decoder.correct(error)
        except ValueError:
            continue
        if not code.is_stabilizer(result.residual):
            assert not result.success
            found_logical_failure = True
            break
    assert found_logical_failure
