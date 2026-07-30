"""Exact finite FER oracle for the packed [[5,1,3]]_4 code.

The oracle classifies every element of the five-site packed Pauli basis. It
uses a 1024-state single-lane table and combines the two encoded-qubit lanes,
so the full 16**5 pattern space is covered without Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache
from itertools import product
from typing import Iterable

from qec.decoder.ququart.codes import packed_five_ququart_code
from qec.decoder.ququart.gf2 import Vector, add, in_row_span
from qec.decoder.ququart.packed import symplectic

DEFAULT_ERROR_RATES = (
    "0.00001",
    "0.00003",
    "0.0001",
    "0.0003",
    "0.001",
    "0.003",
    "0.01",
    "0.03",
    "0.1",
    "0.2",
)

_LOCAL_BITS = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}
_LOCAL_LABELS = tuple(_LOCAL_BITS)


@dataclass(frozen=True)
class _LaneState:
    support_mask: int
    syndrome: tuple[int, ...]
    correction_site: int | None
    stabilizer_residual: bool


def _lane_vector(word: tuple[str, ...]) -> Vector:
    return (
        tuple(_LOCAL_BITS[label][0] for label in word)
        + tuple(_LOCAL_BITS[label][1] for label in word)
    )


def _single_lane_corrections() -> dict[tuple[int, ...], tuple[Vector, int | None]]:
    code = packed_five_ququart_code()
    width = code.n
    zero = (0,) * (2 * width)
    result: dict[tuple[int, ...], tuple[Vector, int | None]] = {
        (0,) * len(code.base_stabilizers): (zero, None)
    }
    for site in range(width):
        for label in ("X", "Y", "Z"):
            x = [0] * width
            z = [0] * width
            x[site], z[site] = _LOCAL_BITS[label]
            candidate = tuple(x + z)
            syndrome = tuple(
                symplectic(check, candidate)
                for check in code.base_stabilizers
            )
            if syndrome in result:
                raise ValueError("five-qubit lane has a duplicate weight-one syndrome")
            result[syndrome] = (candidate, site)
    expected = 2 ** len(code.base_stabilizers)
    if len(result) != expected:
        raise ValueError(
            f"single-lane syndrome table is incomplete: {len(result)} != {expected}"
        )
    return result


@lru_cache(maxsize=1)
def _lane_states() -> tuple[_LaneState, ...]:
    code = packed_five_ququart_code()
    corrections = _single_lane_corrections()
    states: list[_LaneState] = []
    for word in product(_LOCAL_LABELS, repeat=code.n):
        vector = _lane_vector(word)
        syndrome = tuple(
            symplectic(check, vector)
            for check in code.base_stabilizers
        )
        correction, site = corrections[syndrome]
        residual = add(vector, correction)
        support_mask = sum(
            1 << index
            for index, label in enumerate(word)
            if label != "I"
        )
        states.append(
            _LaneState(
                support_mask=support_mask,
                syndrome=syndrome,
                correction_site=site,
                stabilizer_residual=in_row_span(
                    residual,
                    code.base_stabilizers,
                ),
            )
        )
    return tuple(states)


def _binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    result = 1
    for value in range(1, min(k, n - k) + 1):
        result = result * (n - value + 1) // value
    return result


@lru_cache(maxsize=1)
def exact_weight_enumerator() -> tuple[dict[str, int], ...]:
    """Classify all 16**5 packed Pauli patterns by physical-ququart weight."""

    code = packed_five_ququart_code()
    counts = {
        weight: {
            "corrected": 0,
            "detected_uncorrectable": 0,
            "logical_failure": 0,
        }
        for weight in range(code.n + 1)
    }
    states = _lane_states()
    for lane0 in states:
        for lane1 in states:
            weight = (lane0.support_mask | lane1.support_mask).bit_count()
            accepted = (
                lane0.correction_site is None
                or lane1.correction_site is None
                or lane0.correction_site == lane1.correction_site
            )
            if not accepted:
                counts[weight]["detected_uncorrectable"] += 1
            elif lane0.stabilizer_residual and lane1.stabilizer_residual:
                counts[weight]["corrected"] += 1
            else:
                counts[weight]["logical_failure"] += 1

    rows: list[dict[str, int]] = []
    for weight in range(code.n + 1):
        row = counts[weight]
        total = (
            row["corrected"]
            + row["detected_uncorrectable"]
            + row["logical_failure"]
        )
        expected = 1 if weight == 0 else _binomial(code.n, weight) * (15 ** weight)
        if total != expected:
            raise AssertionError(
                f"weight-{weight} enumerator mismatch: {total} != {expected}"
            )
        rows.append({
            "weight": weight,
            "patterns": total,
            "corrected": row["corrected"],
            "detected_uncorrectable": row["detected_uncorrectable"],
            "logical_failure": row["logical_failure"],
            "frame_failures": (
                row["detected_uncorrectable"] + row["logical_failure"]
            ),
        })

    if sum(row["patterns"] for row in rows) != 16 ** code.n:
        raise AssertionError("full packed Pauli basis was not enumerated")
    return tuple(rows)


def _rate(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    return format(value, ".18E")


def exact_fer_row(error_rate: str | Decimal) -> dict[str, str]:
    """Evaluate the exact iid packed-depolarizing FER polynomial."""

    p = Decimal(error_rate)
    if p < 0 or p > 1:
        raise ValueError("physical error rate must be in [0, 1]")
    enumerator = exact_weight_enumerator()
    weights = tuple(row["weight"] for row in enumerator)
    n = max(weights)
    if weights != tuple(range(n + 1)):
        raise AssertionError("weight enumerator must cover every weight from 0 through n")

    with localcontext() as context:
        context.prec = 60
        local = p / Decimal(15)
        one_minus = Decimal(1) - p
        corrected = Decimal(0)
        detected = Decimal(0)
        logical = Decimal(0)
        for row in enumerator:
            weight = row["weight"]
            probability_per_pattern = (
                (local ** weight) * (one_minus ** (n - weight))
            )
            corrected += Decimal(row["corrected"]) * probability_per_pattern
            detected += (
                Decimal(row["detected_uncorrectable"])
                * probability_per_pattern
            )
            logical += (
                Decimal(row["logical_failure"])
                * probability_per_pattern
            )
        failure = detected + logical
        total = corrected + failure
        if abs(total - Decimal(1)) > Decimal("1e-45"):
            raise AssertionError(f"probabilities do not sum to one: {total}")
        return {
            "physical_error_rate": str(p),
            "frame_error_rate": _rate(failure),
            "success_rate": _rate(corrected),
            "detected_uncorrectable_rate": _rate(detected),
            "logical_failure_rate": _rate(logical),
        }


def exact_fer_curve(
    error_rates: Iterable[str] = DEFAULT_ERROR_RATES,
) -> tuple[dict[str, str], ...]:
    return tuple(exact_fer_row(rate) for rate in error_rates)
