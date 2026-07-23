"""Exact algebraic tests for the qutrit stabilizer decoder."""

from __future__ import annotations

from itertools import product

import pytest

from qec.decoder.qutrit import (
    ExactDecoder,
    Pauli,
    QutritStabilizerCode,
    UncorrectableSyndrome,
    cyclic_five_qutrit_code,
    exact_distance,
    paulis_of_weight,
    shor_nine_qutrit_code,
    ternary_golay_checks,
    ternary_golay_qutrit_code,
)
from qec.decoder.qutrit.gf3 import rank


@pytest.mark.parametrize(
    ("factory", "parameters"),
    [
        (cyclic_five_qutrit_code, (5, 1, 3)),
        (shor_nine_qutrit_code, (9, 1, 3)),
        (ternary_golay_qutrit_code, (11, 1, 5)),
    ],
)
def test_code_parameters(factory, parameters):
    code = factory()
    assert (code.n, code.k, code.distance_hint) == parameters
    assert rank(code.stabilizers) == code.n - code.k


@pytest.mark.parametrize(
    "factory",
    [cyclic_five_qutrit_code, shor_nine_qutrit_code],
)
def test_distance_three_is_found_exactly(factory):
    assert exact_distance(factory(), max_weight=3) == 3


@pytest.mark.parametrize(
    "factory",
    [cyclic_five_qutrit_code, shor_nine_qutrit_code],
)
def test_every_single_qutrit_pauli_is_corrected(factory):
    code = factory()
    decoder = ExactDecoder(code, max_weight=1)
    errors = tuple(paulis_of_weight(code.n, 1))
    assert len(errors) == code.n * 8
    assert all(decoder.correct(error).success for error in errors)


def test_every_weight_two_golay_error_is_corrected():
    code = ternary_golay_qutrit_code()
    decoder = ExactDecoder(code, max_weight=2)
    errors = (
        tuple(paulis_of_weight(code.n, 1))
        + tuple(paulis_of_weight(code.n, 2))
    )
    assert len(errors) == 11 * 8 + 55 * 64
    assert all(decoder.correct(error).success for error in errors)


def test_golay_css_distance_certificate():
    checks = ternary_golay_checks()
    assert all(
        sum(a * b for a, b in zip(left, right)) % 3 == 0
        for left in checks
        for right in checks
    )

    stabilizer_words = {
        tuple(
            sum(
                coefficient * row[index]
                for coefficient, row in zip(coefficients, checks)
            )
            % 3
            for index in range(11)
        )
        for coefficients in product(range(3), repeat=5)
    }
    assert len(stabilizer_words) == 3**5

    classical_minimum = 11
    logical_minimum = 11
    for prefix in product(range(3), repeat=6):
        suffix = tuple(
            -sum(row[index] * prefix[index] for index in range(6)) % 3
            for row in checks
        )
        word = prefix + suffix
        weight = sum(value != 0 for value in word)
        if weight:
            classical_minimum = min(classical_minimum, weight)
            if word not in stabilizer_words:
                logical_minimum = min(logical_minimum, weight)
    assert classical_minimum == 5
    assert logical_minimum == 5


def test_shor_degeneracy_resolves_to_a_stabilizer():
    code = shor_nine_qutrit_code()
    decoder = ExactDecoder(code)
    first = Pauli((0,) * 9, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    second = Pauli((0,) * 9, (0, 1, 0, 0, 0, 0, 0, 0, 0))
    assert code.syndrome(first) == code.syndrome(second)
    assert decoder.correct(second).success


def test_unknown_bounded_syndrome_fails_closed():
    code = cyclic_five_qutrit_code()
    decoder = ExactDecoder(code)
    known = {
        code.syndrome(error)
        for error in paulis_of_weight(code.n, 1)
    }
    unknown = next(
        syndrome
        for syndrome in product(range(3), repeat=len(code.stabilizers))
        if syndrome not in known and any(syndrome)
    )
    with pytest.raises(UncorrectableSyndrome):
        decoder.decode(unknown)


def test_noncommuting_checks_are_rejected():
    with pytest.raises(ValueError, match="do not commute"):
        QutritStabilizerCode(
            (
                (1, 0, 0, 0),
                (0, 0, 1, 0),
            )
        )
