"""
qec_ququart.py

Generalized stabilizer code for ququarts (d = 4).

Implements a simple [[3,1]]_4 repetition-like code that protects
against single-ququart X-type shift errors using a qudit stabilizer
formalism over Z_4.

Codewords:
    |j_L> = |j, j, j>,  j in {0,1,2,3}
"""

import numpy as np

# Global constants
D = 4  # ququart dimension
omega = 1j  # exp(2πi/4) = i


def single_X():
    """Single-ququart shift operator X (4x4)."""
    X = np.zeros((D, D), dtype=complex)
    for j in range(D):
        X[(j + 1) % D, j] = 1.0
    return X


def single_Z():
    """Single-ququart phase operator Z (4x4)."""
    Z = np.zeros((D, D), dtype=complex)
    for j in range(D):
        Z[j, j] = omega ** j
    return Z


X_1q = single_X()
Z_1q = single_Z()


def kron_n(ops):
    """Kronecker product of a list of operators."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


class QuquartRepetitionCode3:
    """
    3-ququart [[3,1]]_4 stabilizer code.

    Stabilizers:
        S1 = Z1 Z2^{-1}
        S2 = Z2 Z3^{-1}

    Logical operators:
        X_L = X1 X2 X3
        Z_L = Z1  (equivalent to Z2 or Z3 in codespace)
    """

    def __init__(self):
        # Precompute n-ququart operators
        I = np.eye(D, dtype=complex)

        # Single-site actions embedded in 3-ququart space
        self.Z1 = kron_n([Z_1q, I, I])
        self.Z2 = kron_n([I, Z_1q, I])
        self.Z3 = kron_n([I, I, Z_1q])

        self.X1 = kron_n([X_1q, I, I])
        self.X2 = kron_n([I, X_1q, I])
        self.X3 = kron_n([I, I, X_1q])

        # Stabilizers
        self.S1 = self.Z1 @ np.linalg.inv(self.Z2)
        self.S2 = self.Z2 @ np.linalg.inv(self.Z3)

        # Logical ops
        self.XL = self.X1 @ self.X2 @ self.X3
        self.ZL = self.Z1

    def basis_state(self, j1, j2, j3):
        """|j1, j2, j3> in C^(4^3)."""
        vec = np.zeros(D**3, dtype=complex)
        idx = j1 * D**2 + j2 * D + j3
        vec[idx] = 1.0
        return vec

    def encode_logical(self, j):
        """
        Encode logical |j_L> = |j, j, j>.
        j in {0,1,2,3}.
        """
        j = int(j) % D
        return self.basis_state(j, j, j)

    def apply_X_error(self, state, site, power=1):
        """
        Apply X^power to given site (1,2,3).
        power in {0,1,2,3} mod 4.
        """
        power = power % D
        if power == 0:
            return state

        if site == 1:
            op = np.linalg.matrix_power(self.X1, power)
        elif site == 2:
            op = np.linalg.matrix_power(self.X2, power)
        elif site == 3:
            op = np.linalg.matrix_power(self.X3, power)
        else:
            raise ValueError("site must be 1,2,3")

        return op @ state

    def measure_stabilizers(self, state):
        """
        Measure S1, S2 eigenvalues (non-projective; just expectation).

        Returns:
            (e1, e2) complex eigenvalues (approx. 4th roots of unity).
        """
        norm = np.vdot(state, state)
        if norm == 0:
            raise ValueError("Zero state vector")

        s1_val = np.vdot(state, self.S1 @ state) / norm
        s2_val = np.vdot(state, self.S2 @ state) / norm
        return s1_val, s2_val

    def syndrome(self, state, tol=1e-6):
        """
        Map stabilizer eigenvalues to discrete syndrome in Z_4.

        Returns:
            (s1, s2) in {0,1,2,3}, where 0 means +1 eigenvalue.

        We round phases to nearest multiple of π/2 (4th roots).
        """
        s1_val, s2_val = self.measure_stabilizers(state)

        def phase_to_int(z):
            angle = np.angle(z)  # in [-π, π]
            # normalize to [0, 2π)
            if angle < 0:
                angle += 2 * np.pi
            m = int(np.round(2 * angle / np.pi)) % 4  # multiples of π/2
            return m

        return phase_to_int(s1_val), phase_to_int(s2_val)

    def decode_X_single_error(self, state):
        """
        Attempt to correct a single-ququart X^{±1} error.

        Returns:
            (corrected_state, info_dict)

        This uses a simple hard-coded syndrome table for X^±1 errors.
        """
        s1, s2 = self.syndrome(state)

        # Syndrome table (very simple):
        # Here we only handle ±1 shifts; X^2 needs an extended table.
        # s in {0,1,3} ~ {0, +, -} indicator modulo 4.
        info = {
            "syndrome": (s1, s2),
            "action": None
        }

        # No error
        if s1 == 0 and s2 == 0:
            info["action"] = "no_correction"
            return state, info

        # Crude mapping example:
        # You can refine this map after inspecting numerics / conventions.
        if s1 != 0 and s2 == 0:
            # Hit on site 1 or 2
            site = 1
            sign = 1 if s1 == 1 else -1
        elif s1 != 0 and s2 != 0:
            # Likely site 2
            site = 2
            # relative pattern encodes sign
            sign = 1 if s2 == 1 else -1
        elif s1 == 0 and s2 != 0:
            # Likely site 3
            site = 3
            sign = 1 if s2 == 3 else -1  # depends on chosen convention
        else:
            # Fallback: do nothing
            info["action"] = "unrecognized_syndrome"
            return state, info

        power = -sign  # invert the error
        corrected = self.apply_X_error(state, site, power=power)
        info["action"] = f"apply X^{power} on site {site}"
        return corrected, info
