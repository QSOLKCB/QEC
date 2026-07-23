"""Exact prime-dimensional stabilizer model used only by benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from math import comb
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

Vector = tuple[int, ...]
Matrix = tuple[Vector, ...]
Syndrome = tuple[int, ...]


def rank(rows: Sequence[Sequence[int]], modulus: int) -> int:
    """Return exact row rank over a prime field."""
    work = [[value % modulus for value in row] for row in rows]
    if not work:
        return 0
    width = len(work[0])
    if any(len(row) != width for row in work):
        raise ValueError("matrix must be rectangular")

    pivot_row = 0
    for column in range(width):
        pivot = next(
            (row for row in range(pivot_row, len(work)) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        inverse = pow(work[pivot_row][column], -1, modulus)
        work[pivot_row] = [
            inverse * value % modulus for value in work[pivot_row]
        ]
        for row, values in enumerate(work):
            if row == pivot_row or not values[column]:
                continue
            factor = values[column]
            work[row] = [
                (value - factor * base) % modulus
                for value, base in zip(values, work[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def symplectic(left: Vector, right: Vector, modulus: int) -> int:
    """Return x·z' - z·x' over the declared prime field."""
    middle = len(left) // 2
    return (
        sum(a * b for a, b in zip(left[:middle], right[middle:]))
        - sum(a * b for a, b in zip(left[middle:], right[:middle]))
    ) % modulus


def paulis_of_weight(width: int, weight: int, modulus: int) -> Iterator[Vector]:
    """Enumerate exact-weight generalized Paulis in canonical order."""
    local = tuple(
        (x, z)
        for x in range(modulus)
        for z in range(modulus)
        if (x, z) != (0, 0)
    )
    for support in combinations(range(width), weight):
        for powers in product(local, repeat=weight):
            values = [0] * (2 * width)
            for site, (x_power, z_power) in zip(support, powers):
                values[site] = x_power
                values[width + site] = z_power
            yield tuple(values)


def error_pattern_count(width: int, weight: int, modulus: int) -> int:
    return comb(width, weight) * (modulus * modulus - 1) ** weight


def _row_span(rows: Matrix, modulus: int) -> tuple[Vector, ...]:
    zero = (0,) * len(rows[0])
    span = {zero}
    for row in rows:
        additions = {
            tuple((base[i] + coefficient * row[i]) % modulus for i in range(len(row)))
            for base in span
            for coefficient in range(1, modulus)
        }
        span.update(additions)
    return tuple(sorted(span))


@dataclass(frozen=True)
class PrimeStabilizerModel:
    """Independent exact benchmark oracle for a bounded decoder."""

    code_id: str
    label: str
    family: str
    origin: str
    modulus: int
    n: int
    k: int
    distance: int
    radius: int
    checks: Matrix
    source_url: str
    _leaders: Mapping[Syndrome, Vector] = field(init=False, repr=False)
    _stabilizers: tuple[Vector, ...] = field(init=False, repr=False)
    _stabilizer_set: frozenset[Vector] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.modulus not in (2, 3):
            raise ValueError("benchmark supports only GF(2) and GF(3)")
        checks = tuple(
            tuple(value % self.modulus for value in row)
            for row in self.checks
        )
        if not checks or any(len(row) != 2 * self.n for row in checks):
            raise ValueError("checks must have independent (x | z) rows")
        if rank(checks, self.modulus) != self.n - self.k:
            raise ValueError("check rank does not match [[n,k,d]]")
        if any(
            symplectic(left, right, self.modulus)
            for left in checks
            for right in checks
        ):
            raise ValueError("stabilizer checks do not commute")
        if self.radius > (self.distance - 1) // 2:
            raise ValueError("decoder radius exceeds the distance guarantee")

        stabilizers = _row_span(checks, self.modulus)
        stabilizer_set = frozenset(stabilizers)
        leaders: dict[Syndrome, Vector] = {}
        for weight in range(self.radius + 1):
            for candidate in paulis_of_weight(self.n, weight, self.modulus):
                syndrome = self.syndrome(candidate, checks=checks)
                previous = leaders.get(syndrome)
                if previous is None:
                    leaders[syndrome] = candidate
                    continue
                difference = tuple(
                    (a - b) % self.modulus
                    for a, b in zip(candidate, previous)
                )
                if difference not in stabilizer_set:
                    raise ValueError("bounded error set is not exactly correctable")

        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "_leaders", MappingProxyType(leaders))
        object.__setattr__(self, "_stabilizers", stabilizers)
        object.__setattr__(self, "_stabilizer_set", stabilizer_set)

    @property
    def decoder_table_size(self) -> int:
        return len(self._leaders)

    @property
    def stabilizer_group_size(self) -> int:
        return len(self._stabilizers)

    def syndrome(
        self,
        error: Sequence[int],
        *,
        checks: Matrix | None = None,
    ) -> Syndrome:
        vector = tuple(value % self.modulus for value in error)
        if len(vector) != 2 * self.n:
            raise ValueError("error width does not match code")
        active_checks = self.checks if checks is None else checks
        return tuple(
            symplectic(check, vector, self.modulus)
            for check in active_checks
        )

    def classify(self, error: Sequence[int]) -> str:
        """Return corrected, rejected, or miscorrected."""
        vector = tuple(value % self.modulus for value in error)
        leader = self._leaders.get(self.syndrome(vector))
        if leader is None:
            return "rejected"
        residual = tuple(
            (value - correction) % self.modulus
            for value, correction in zip(vector, leader)
        )
        return "corrected" if residual in self._stabilizer_set else "miscorrected"

    def is_stabilizer(self, error: Sequence[int]) -> bool:
        vector = tuple(value % self.modulus for value in error)
        if len(vector) != 2 * self.n:
            raise ValueError("error width does not match code")
        return vector in self._stabilizer_set

    def success_weight_counts(self, *, operation_limit: int) -> tuple[int, ...] | None:
        """Return exact accepted-and-corrected Pauli counts by weight."""
        operations = self.decoder_table_size * self.stabilizer_group_size
        if operations > operation_limit:
            return None
        counts = [0] * (self.n + 1)
        for syndrome in sorted(self._leaders):
            leader = self._leaders[syndrome]
            for stabilizer in self._stabilizers:
                error = tuple(
                    (value + offset) % self.modulus
                    for value, offset in zip(leader, stabilizer)
                )
                counts[
                    sum(
                        error[site] != 0 or error[self.n + site] != 0
                        for site in range(self.n)
                    )
                ] += 1
        if sum(counts) != operations:
            raise AssertionError("coset enumerator cardinality drift")
        return tuple(counts)
