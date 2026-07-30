"""Exact additive ququart QEC using two binary Pauli lanes per ququart."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterator, Sequence

from .gf2 import Matrix, Vector, add, in_row_span, matrix, rank, vector

BinarySyndrome = tuple[int, ...]
_LOCAL_BITS = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}
_LOCAL_LABELS = tuple(_LOCAL_BITS)


def symplectic(left: Sequence[int], right: Sequence[int]) -> int:
    """Return x·z' + z·x' over GF(2)."""
    if len(left) != len(right) or len(left) % 2:
        raise ValueError("symplectic vectors need equal even width")
    middle = len(left) // 2
    x, z = left[:middle], left[middle:]
    other_x, other_z = right[:middle], right[middle:]
    return (
        sum(a * b for a, b in zip(x, other_z))
        + sum(a * b for a, b in zip(z, other_x))
    ) % 2


@dataclass(frozen=True)
class PackedPauli:
    """Two-qubit Pauli error packed into each four-state physical unit.

    Global phase is omitted. A physical ququart site carries lane 0 and lane 1,
    matching the basis map |q0 q1> <-> |2*q0 + q1>.
    """

    lane0_x: Vector
    lane0_z: Vector
    lane1_x: Vector
    lane1_z: Vector

    def __post_init__(self) -> None:
        lane0_x = vector(self.lane0_x)
        if not lane0_x:
            raise ValueError("Pauli error must act on at least one ququart")
        width = len(lane0_x)
        object.__setattr__(self, "lane0_x", lane0_x)
        object.__setattr__(self, "lane0_z", vector(self.lane0_z, width=width))
        object.__setattr__(self, "lane1_x", vector(self.lane1_x, width=width))
        object.__setattr__(self, "lane1_z", vector(self.lane1_z, width=width))

    @property
    def width(self) -> int:
        return len(self.lane0_x)

    @property
    def weight(self) -> int:
        """Return physical-ququart weight, not binary-lane weight."""
        return sum(
            any(bits)
            for bits in zip(
                self.lane0_x,
                self.lane0_z,
                self.lane1_x,
                self.lane1_z,
            )
        )

    @property
    def lane0_vector(self) -> Vector:
        return self.lane0_x + self.lane0_z

    @property
    def lane1_vector(self) -> Vector:
        return self.lane1_x + self.lane1_z

    def inverse(self) -> "PackedPauli":
        """Every binary Pauli is self-inverse after global phase is omitted."""
        return self

    def compose(self, other: "PackedPauli") -> "PackedPauli":
        if self.width != other.width:
            raise ValueError("Pauli widths differ")
        return PackedPauli(
            add(self.lane0_x, other.lane0_x),
            add(self.lane0_z, other.lane0_z),
            add(self.lane1_x, other.lane1_x),
            add(self.lane1_z, other.lane1_z),
        )

    def local_labels(self, site: int) -> tuple[str, str]:
        """Return the two encoded-qubit Pauli labels at one ququart site."""
        if site < 0 or site >= self.width:
            raise IndexError("ququart site out of range")
        inverse = {bits: label for label, bits in _LOCAL_BITS.items()}
        return (
            inverse[(self.lane0_x[site], self.lane0_z[site])],
            inverse[(self.lane1_x[site], self.lane1_z[site])],
        )


@dataclass(frozen=True)
class PackedQuquartCode:
    """An additive [[n,k,d]]_4 code built from two identical qubit codes."""

    base_stabilizers: Matrix
    name: str = "packed-ququart-stabilizer"
    distance_hint: int | None = None

    def __post_init__(self) -> None:
        checks = matrix(self.base_stabilizers)
        if len(checks[0]) % 2:
            raise ValueError("stabilizer rows must have (x | z) form")
        if rank(checks) != len(checks):
            raise ValueError("stabilizer generators must be independent")
        if any(symplectic(a, b) for a in checks for b in checks):
            raise ValueError("stabilizer generators do not commute")
        n = len(checks[0]) // 2
        if len(checks) >= n:
            raise ValueError("base code must encode at least one qubit per lane")
        object.__setattr__(self, "base_stabilizers", checks)

    @property
    def n(self) -> int:
        return len(self.base_stabilizers[0]) // 2

    @property
    def lane_k(self) -> int:
        return self.n - len(self.base_stabilizers)

    @property
    def k(self) -> int:
        """Logical ququarts encoded when both lanes have the same dimension."""
        return self.lane_k

    @property
    def syndrome_width(self) -> int:
        return 2 * len(self.base_stabilizers)

    def syndrome(self, error: PackedPauli) -> BinarySyndrome:
        if error.width != self.n:
            raise ValueError("error width does not match code")
        lane0 = tuple(
            symplectic(check, error.lane0_vector)
            for check in self.base_stabilizers
        )
        lane1 = tuple(
            symplectic(check, error.lane1_vector)
            for check in self.base_stabilizers
        )
        return lane0 + lane1

    def is_stabilizer(self, error: PackedPauli) -> bool:
        if error.width != self.n:
            raise ValueError("error width does not match code")
        return (
            in_row_span(error.lane0_vector, self.base_stabilizers)
            and in_row_span(error.lane1_vector, self.base_stabilizers)
        )


def identity(width: int) -> PackedPauli:
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if width <= 0:
        raise ValueError("width must be positive")
    zero = (0,) * width
    return PackedPauli(zero, zero, zero, zero)


def local_pauli(width: int, site: int, lane0: str, lane1: str) -> PackedPauli:
    """Construct one local two-lane Pauli at a physical ququart site."""
    if width <= 0:
        raise ValueError("width must be positive")
    if site < 0 or site >= width:
        raise IndexError("ququart site out of range")
    try:
        lane0_bits = _LOCAL_BITS[lane0]
        lane1_bits = _LOCAL_BITS[lane1]
    except KeyError as error:
        raise ValueError("Pauli labels must be one of I, X, Y, Z") from error
    if lane0 == lane1 == "I":
        raise ValueError("local Pauli must be non-identity")

    vectors = [[0] * width for _ in range(4)]
    vectors[0][site], vectors[1][site] = lane0_bits
    vectors[2][site], vectors[3][site] = lane1_bits
    return PackedPauli(*(tuple(values) for values in vectors))


def paulis_of_weight(width: int, weight: int) -> Iterator[PackedPauli]:
    """Enumerate all packed Pauli errors of one exact ququart weight."""
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if isinstance(weight, bool) or not isinstance(weight, int):
        raise TypeError("weight must be an integer")
    if width <= 0:
        raise ValueError("width must be positive")
    if weight < 0 or weight > width:
        return

    local = tuple(
        (lane0, lane1)
        for lane0 in _LOCAL_LABELS
        for lane1 in _LOCAL_LABELS
        if (lane0, lane1) != ("I", "I")
    )
    if weight == 0:
        yield identity(width)
        return

    for support in combinations(range(width), weight):
        for labels in product(local, repeat=weight):
            result = identity(width)
            for site, (lane0, lane1) in zip(support, labels):
                result = result.compose(local_pauli(width, site, lane0, lane1))
            yield result


def exact_distance(
    code: PackedQuquartCode,
    *,
    max_weight: int,
) -> int | None:
    """Find the first nontrivial logical packed Pauli up to ``max_weight``."""
    for weight in range(1, max_weight + 1):
        for candidate in paulis_of_weight(code.n, weight):
            if any(code.syndrome(candidate)):
                continue
            if not code.is_stabilizer(candidate):
                return weight
    return None
