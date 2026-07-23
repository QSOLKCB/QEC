"""Small qutrit codes used as exact reference oracles."""

from __future__ import annotations

from .gf3 import Matrix
from .stabilizer import QutritStabilizerCode


def cyclic_five_qutrit_code() -> QutritStabilizerCode:
    """Return a cyclic [[5,1,3]]_3 stabilizer code."""
    rows: list[tuple[int, ...]] = []
    for shift in range(4):
        x, z = [0] * 5, [0] * 5
        for site in (0, 3):
            x[(site + shift) % 5] = 1
        for site in (1, 2):
            z[(site + shift) % 5] = 1
        rows.append(tuple(x + z))
    return QutritStabilizerCode(
        tuple(rows),
        name="cyclic-[[5,1,3]]_3",
        distance_hint=3,
    )


def shor_nine_qutrit_code() -> QutritStabilizerCode:
    """Return the generalized qutrit Shor [[9,1,3]]_3 code."""
    rows: list[tuple[int, ...]] = []
    for block in range(3):
        for left, right in ((0, 1), (1, 2)):
            x, z = [0] * 9, [0] * 9
            z[3 * block + left] = 1
            z[3 * block + right] = 2
            rows.append(tuple(x + z))

    for left_block, right_block in ((0, 1), (1, 2)):
        x, z = [0] * 9, [0] * 9
        for offset in range(3):
            x[3 * left_block + offset] = 1
            x[3 * right_block + offset] = 2
        rows.append(tuple(x + z))

    return QutritStabilizerCode(
        tuple(rows),
        name="shor-[[9,1,3]]_3",
        distance_hint=3,
    )


def ternary_golay_checks() -> Matrix:
    """Return checks for the classical perfect [11,6,5]_3 Golay code."""
    return (
        (2, 2, 2, 1, 1, 0, 1, 0, 0, 0, 0),
        (2, 2, 1, 2, 0, 1, 0, 1, 0, 0, 0),
        (2, 1, 2, 0, 2, 1, 0, 0, 1, 0, 0),
        (2, 1, 0, 2, 1, 2, 0, 0, 0, 1, 0),
        (2, 0, 1, 1, 2, 2, 0, 0, 0, 0, 1),
    )


def ternary_golay_qutrit_code() -> QutritStabilizerCode:
    """Return the CSS [[11,1,5]]_3 quantum ternary Golay code."""
    checks = ternary_golay_checks()
    zero = (0,) * 11
    stabilizers = tuple(
        (check + zero)
        for check in checks
    ) + tuple(
        (zero + check)
        for check in checks
    )
    return QutritStabilizerCode(
        stabilizers,
        name="golay-[[11,1,5]]_3",
        distance_hint=5,
    )
