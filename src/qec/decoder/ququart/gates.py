"""Native ququart and encoded-two-qubit gate definitions."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

Matrix4 = tuple[tuple[complex, complex, complex, complex], ...]
Permutation4 = tuple[int, int, int, int]

X4: Matrix4 = (
    (0j, 0j, 0j, 1 + 0j),
    (1 + 0j, 0j, 0j, 0j),
    (0j, 1 + 0j, 0j, 0j),
    (0j, 0j, 1 + 0j, 0j),
)
Z4: Matrix4 = (
    (1 + 0j, 0j, 0j, 0j),
    (0j, 1j, 0j, 0j),
    (0j, 0j, -1 + 0j, 0j),
    (0j, 0j, 0j, -1j),
)
H4: Matrix4 = tuple(
    tuple(value / 2 for value in row)
    for row in (
        (1, 1, 1, 1),
        (1, 1j, -1, -1j),
        (1, -1, 1, -1),
        (1, -1j, -1, 1j),
    )
)

# Basis permutations under |q0 q1> <-> |2*q0 + q1>.
ENCODED_X_LANE0: Permutation4 = (2, 3, 0, 1)
ENCODED_X_LANE1: Permutation4 = (1, 0, 3, 2)
ENCODED_X_BOTH: Permutation4 = (3, 2, 1, 0)
INTERNAL_SWAP: Permutation4 = (0, 2, 1, 3)
NATIVE_X4: Permutation4 = (1, 2, 3, 0)


def _bit(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer bit")
    result = int(value)
    if result not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return result


def basis_to_qubits(state: int) -> tuple[int, int]:
    """Decode |0>,|1>,|2>,|3> as |00>,|01>,|10>,|11>."""
    if isinstance(state, bool) or not isinstance(state, Integral):
        raise TypeError("ququart basis state must be an integer")
    value = int(state)
    if value < 0 or value > 3:
        raise ValueError("ququart basis state must be in the range 0..3")
    return divmod(value, 2)


def qubits_to_basis(q0: int, q1: int) -> int:
    """Encode two logical bits as one four-state basis label."""
    return 2 * _bit(q0, name="q0") + _bit(q1, name="q1")


def apply_permutation(state: int, permutation: Sequence[int]) -> int:
    """Apply a validated four-state basis permutation."""
    basis_to_qubits(state)
    mapping = tuple(permutation)
    if sorted(mapping) != [0, 1, 2, 3]:
        raise ValueError("permutation must contain each state 0..3 exactly once")
    return mapping[int(state)]


def permutation_matrix(permutation: Sequence[int]) -> Matrix4:
    """Return the unitary matrix U satisfying U|j> = |permutation[j]>."""
    mapping = tuple(permutation)
    if sorted(mapping) != [0, 1, 2, 3]:
        raise ValueError("permutation must contain each state 0..3 exactly once")
    rows = [[0j for _ in range(4)] for _ in range(4)]
    for column, row in enumerate(mapping):
        rows[row][column] = 1 + 0j
    return tuple(tuple(row) for row in rows)  # type: ignore[return-value]
