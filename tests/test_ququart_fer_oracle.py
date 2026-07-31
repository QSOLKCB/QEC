from decimal import Decimal

from qec.benchmark.ququart_battery.oracle import (
    exact_fer_row,
    exact_weight_enumerator,
)


EXPECTED = (
    (0, 1, 1, 0, 0),
    (1, 75, 75, 0, 0),
    (2, 2250, 0, 1800, 450),
    (3, 33750, 300, 23400, 10050),
    (4, 253125, 5175, 178200, 69750),
    (5, 759375, 13905, 533880, 211590),
)


def test_exact_oracle_classifies_full_packed_pauli_basis():
    rows = exact_weight_enumerator()
    observed = tuple(
        (
            row["weight"],
            row["patterns"],
            row["corrected"],
            row["detected_uncorrectable"],
            row["logical_failure"],
        )
        for row in rows
    )
    assert observed == EXPECTED
    assert sum(row["patterns"] for row in rows) == 16 ** 5


def test_exact_fer_is_zero_at_zero_and_quadratic_at_small_p():
    assert Decimal(exact_fer_row("0")["frame_error_rate"]) == 0
    p = Decimal("0.00001")
    fer = Decimal(exact_fer_row(str(p))["frame_error_rate"])
    leading = Decimal(10) * p * p
    assert abs(fer / leading - Decimal(1)) < Decimal("0.0002")


def test_exact_probabilities_sum_to_one():
    for error_rate in ("0.03", "1"):
        row = exact_fer_row(error_rate)
        total = Decimal(row["success_rate"]) + Decimal(row["frame_error_rate"])
        assert abs(total - Decimal(1)) < Decimal("1e-16")
