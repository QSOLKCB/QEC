"""
qudit_stabilizer.py

Generic stabilizer utilities for qudits of arbitrary dimension d.

Implements:
- Generalized Pauli operators X_d, Z_d
- Multi-qudit Pauli construction from exponent vectors
- A QuditStabilizerCode class that:
    * stores stabilizer generators in Z_d exponent form
    * builds their matrix reps
    * measures stabilizer expectation values
    * extracts discrete syndromes in Z_d
"""

import numpy as np
from typing import List, Tuple


def generalized_X(d: int) -> np.ndarray:
    """Single-qudit shift operator X_d (d x d)."""
    X = np.zeros((d, d), dtype=complex)
    for j in range(d):
        X[(j + 1) % d, j] = 1.0
    return X


def generalized_Z(d: int) -> np.ndarray:
    """Single-qudit phase operator Z_d (d x d)."""
    Z = np.zeros((d, d), dtype=complex)
    omega = np.exp(2j * np.pi / d)
    for j in range(d):
        Z[j, j] = omega ** j
    return Z


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    """Kronecker product of a list of operators."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def pauli_from_exponents(
    d: int,
    a: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    """
    Build a multi-qudit generalized Pauli from exponent vectors.

    P(a,b) = prod_k Z_k^{b_k} X_k^{a_k}  in dimension d.

    Args:
        d: local dimension
        a: length-n vector of X exponents in Z_d
        b: length-n vector of Z exponents in Z_d

    Returns:
        (d^n x d^n) unitary matrix.
    """
    n = len(a)
    X = generalized_X(d)
    Z = generalized_Z(d)

    ops = []
    for ak, bk in zip(a, b):
        op = np.eye(d, dtype=complex)
        if bk % d != 0:
            op = np.linalg.matrix_power(Z, int(bk % d)) @ op
        if ak % d != 0:
            op = np.linalg.matrix_power(X, int(ak % d)) @ op
        ops.append(op)

    return kron_n(ops)


class QuditStabilizerCode:
    """
    Generic n-qudit stabilizer code over Z_d.

    Stabilizers are specified by exponent vectors (a,b) in Z_d^{2n},
    meaning:
        S_j = prod_k Z_k^{b_{j,k}} X_k^{a_{j,k}}.

    This class:
        - stores generators in exponent form
        - builds matrix representation
        - computes expectation values <psi|S_j|psi>
        - converts phases to discrete syndromes in Z_d
    """

    def __init__(
        self,
        d: int,
        n_qudits: int,
        generators: List[Tuple[np.ndarray, np.ndarray]]
    ):
        """
        Args:
            d: local dimension
            n_qudits: number of physical qudits
            generators: list of (a, b) exponent pairs defining stabilizers
        """
        self.d = d
        self.n = n_qudits
        self.generators = generators

        # Precompute single-site X,Z
        self._X = generalized_X(d)
        self._Z = generalized_Z(d)

        # Build matrix reps for stabilizers
        self.stabilizers = []
        for (a, b) in generators:
            self.stabilizers.append(pauli_from_exponents(d, a, b))

    def measure_stabilizers(self, state: np.ndarray) -> List[complex]:
        """
        Measure stabilizers on a state |psi> (non-projective expectation).

        Args:
            state: state vector in C^(d^n)

        Returns:
            List of complex expectation values.
        """
        norm = np.vdot(state, state)
        if norm == 0:
            raise ValueError("Zero state")

        vals = []
        for S in self.stabilizers:
            vals.append(np.vdot(state, S @ state) / norm)
        return vals

    def syndromes(self, state: np.ndarray) -> np.ndarray:
        """
        Convert stabilizer eigenvalues to integer syndromes in Z_d.

        For each eigenvalue z = e^{2πi k / d} (approx),
        we return k in {0,...,d-1}.

        Args:
            state: state vector

        Returns:
            syndrome: shape (m,) array in Z_d, where m=#generators.
        """
        vals = self.measure_stabilizers(state)
        d = self.d
        synd = []

        for z in vals:
            angle = np.angle(z)
            if angle < 0:
                angle += 2 * np.pi
            k = int(np.round(d * angle / (2 * np.pi))) % d
            synd.append(k)

        return np.array(synd, dtype=int)
