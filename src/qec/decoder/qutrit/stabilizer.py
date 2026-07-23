"""Qutrit Pauli and stabilizer algebra over GF(3)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterator, Sequence

from .gf3 import Matrix, Vector, add, in_row_span, matrix, negate, rank, vector

Syndrome = tuple[int, ...]


@dataclass(frozen=True)
class Pauli:
    """Qutrit Weyl error X^x Z^z, with global phase omitted."""

    x: Vector
    z: Vector

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", vector(self.x))
        object.__setattr__(self, "z", vector(self.z, width=len(self.x)))
        if not self.x:
            raise ValueError("Pauli error must act on at least one qutrit")

    @property
    def width(self) -> int:
        return len(self.x)

    @property
    def weight(self) -> int:
        return sum((x != 0 or z != 0) for x, z in zip(self.x, self.z))

    @property
    def symplectic_vector(self) -> Vector:
        return self.x + self.z

    def inverse(self) -> "Pauli":
        return Pauli(negate(self.x), negate(self.z))

    def compose(self, other: "Pauli") -> "Pauli":
        if self.width != other.width:
            raise ValueError("Pauli widths differ")
        return Pauli(add(self.x, other.x), add(self.z, other.z))


def symplectic(left: Sequence[int], right: Sequence[int]) -> int:
    """Return x·z' - z·x' over GF(3)."""
    if len(left) != len(right) or len(left) % 2:
        raise ValueError("symplectic vectors need equal even width")
    middle = len(left) // 2
    x, z = left[:middle], left[middle:]
    other_x, other_z = right[:middle], right[middle:]
    return (
        sum(a * b for a, b in zip(x, other_z))
        - sum(a * b for a, b in zip(z, other_x))
    ) % 3


@dataclass(frozen=True)
class QutritStabilizerCode:
    """An [[n,k,d]]_3 stabilizer code defined by independent generators."""

    stabilizers: Matrix
    name: str = "qutrit-stabilizer"
    distance_hint: int | None = None

    def __post_init__(self) -> None:
        checks = matrix(self.stabilizers)
        if len(checks[0]) % 2:
            raise ValueError("stabilizer rows must have (x | z) form")
        if rank(checks) != len(checks):
            raise ValueError("stabilizer generators must be independent")
        if any(symplectic(a, b) for a in checks for b in checks):
            raise ValueError("stabilizer generators do not commute")
        if len(checks) > len(checks[0]) // 2:
            raise ValueError("too many independent stabilizers")
        object.__setattr__(self, "stabilizers", checks)

    @property
    def n(self) -> int:
        return len(self.stabilizers[0]) // 2

    @property
    def k(self) -> int:
        return self.n - len(self.stabilizers)

    def syndrome(self, error: Pauli) -> Syndrome:
        if error.width != self.n:
            raise ValueError("error width does not match code")
        return tuple(
            symplectic(check, error.symplectic_vector)
            for check in self.stabilizers
        )

    def is_stabilizer(self, error: Pauli) -> bool:
        if error.width != self.n:
            raise ValueError("error width does not match code")
        return in_row_span(error.symplectic_vector, self.stabilizers)


def paulis_of_weight(width: int, weight: int) -> Iterator[Pauli]:
    """Enumerate all qutrit Paulis of one exact weight."""
    local = tuple(
        (x, z)
        for x in range(3)
        for z in range(3)
        if (x, z) != (0, 0)
    )
    for support in combinations(range(width), weight):
        for powers in product(local, repeat=weight):
            x, z = [0] * width, [0] * width
            for site, (x_power, z_power) in zip(support, powers):
                x[site], z[site] = x_power, z_power
            yield Pauli(tuple(x), tuple(z))


def exact_distance(
    code: QutritStabilizerCode,
    *,
    max_weight: int,
) -> int | None:
    """Find the first nontrivial logical Pauli up to ``max_weight``."""
    for weight in range(1, max_weight + 1):
        for candidate in paulis_of_weight(code.n, weight):
            if any(code.syndrome(candidate)):
                continue
            if not code.is_stabilizer(candidate):
                return weight
    return None
