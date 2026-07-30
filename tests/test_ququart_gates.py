import math

from qec.decoder.ququart import (
    ENCODED_X_BOTH,
    ENCODED_X_LANE0,
    ENCODED_X_LANE1,
    H4,
    INTERNAL_SWAP,
    NATIVE_X4,
    X4,
    Z4,
    apply_permutation,
    basis_to_qubits,
    permutation_matrix,
    qubits_to_basis,
)


def _matmul(left, right):
    return tuple(
        tuple(sum(left[r][k] * right[k][c] for k in range(4)) for c in range(4))
        for r in range(4)
    )


def _dagger(matrix):
    return tuple(tuple(matrix[c][r].conjugate() for c in range(4)) for r in range(4))


def _is_identity(matrix):
    return all(
        math.isclose(abs(matrix[r][c] - (1 if r == c else 0)), 0, abs_tol=1e-12)
        for r in range(4)
        for c in range(4)
    )


def test_basis_mapping_round_trip():
    assert [basis_to_qubits(state) for state in range(4)] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [
        qubits_to_basis(*basis_to_qubits(state)) for state in range(4)
    ] == list(range(4))


def test_native_and_encoded_permutations_are_distinct_and_exact():
    assert [apply_permutation(s, NATIVE_X4) for s in range(4)] == [1, 2, 3, 0]
    assert [apply_permutation(s, ENCODED_X_LANE0) for s in range(4)] == [2, 3, 0, 1]
    assert [apply_permutation(s, ENCODED_X_LANE1) for s in range(4)] == [1, 0, 3, 2]
    assert [apply_permutation(s, ENCODED_X_BOTH) for s in range(4)] == [3, 2, 1, 0]
    assert [apply_permutation(s, INTERNAL_SWAP) for s in range(4)] == [0, 2, 1, 3]
    assert permutation_matrix(NATIVE_X4) == X4


def test_x4_z4_h4_are_unitary():
    for gate in (X4, Z4, H4):
        assert _is_identity(_matmul(_dagger(gate), gate))
