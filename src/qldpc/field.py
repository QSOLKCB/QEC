"""
GF(2^e) finite field arithmetic.

Represents field elements as integers in [0, 2^e - 1] where bit i is the
coefficient of alpha^i in the polynomial basis {1, alpha, ..., alpha^{e-1}}.

Supports:
- Addition (XOR)
- Multiplication via lookup table
- Multiplicative inverse via lookup table
- Companion matrix (binary e x e representation of multiplication-by-element)
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


# Default irreducible polynomials (coefficient of x^e is implicit).
# Stored as integers: bit i is the coefficient of x^i for i < e,
# plus bit e for x^e.
_DEFAULT_IRREDUCIBLE_POLY = {
    2: 0b111,     # x^2 + x + 1
    3: 0b1011,    # x^3 + x + 1
    4: 0b10011,   # x^4 + x + 1
}


class GF2e:
    """
    Arithmetic over GF(2^e) with companion matrix support.

    Parameters
    ----------
    e : int
        Extension degree (field has 2^e elements).
    poly : int or None
        Irreducible polynomial as an integer.  Bit i is the coefficient
        of x^i.  Must include the x^e term (bit e set).
        If None, uses a built-in default for e in {2, 3, 4}.
    """

    def __init__(self, e: int, poly: Optional[int] = None):
        if e < 1:
            raise ValueError(f"Extension degree must be >= 1, got {e}")

        if poly is None:
            if e not in _DEFAULT_IRREDUCIBLE_POLY:
                raise ValueError(
                    f"GF(2^{e}) has no built-in polynomial. "
                    f"Supply one via the poly parameter. "
                    f"Built-in degrees: {sorted(_DEFAULT_IRREDUCIBLE_POLY)}"
                )
            poly = _DEFAULT_IRREDUCIBLE_POLY[e]

        # Validate polynomial has degree e
        if not ((poly >> e) & 1):
            raise ValueError(
                f"Polynomial 0b{poly:b} does not have degree {e} "
                f"(bit {e} not set)"
            )

        self.e = e
        self.q = 1 << e
        self.poly = poly
        self._build_tables()

    def _build_tables(self) -> None:
        """Pre-compute multiplication and inverse lookup tables.

        Validates that every nonzero element has a multiplicative inverse.
        If any element lacks an inverse, the polynomial is not irreducible
        or the multiplication logic is wrong.
        """
        q = self.q
        self.mul_table = np.zeros((q, q), dtype=np.int32)
        self.inv_table = np.zeros(q, dtype=np.int32)

        for a in range(q):
            for b in range(q):
                self.mul_table[a, b] = self._poly_mul(a, b)

        for a in range(1, q):
            found = False
            for b in range(1, q):
                if self.mul_table[a, b] == 1:
                    self.inv_table[a] = b
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Element {a} has no inverse -- polynomial "
                    f"0b{self.poly:b} may not be irreducible over GF(2)"
                )

    def _poly_mul(self, a: int, b: int) -> int:
        """Polynomial multiplication in GF(2)[x] mod irreducible poly."""
        result = 0
        for i in range(self.e):
            if (b >> i) & 1:
                result ^= a << i
        # Reduce modulo the irreducible polynomial
        for i in range(2 * self.e - 2, self.e - 1, -1):
            if (result >> i) & 1:
                result ^= self.poly << (i - self.e)
        return result & (self.q - 1)

    def add(self, a: int, b: int) -> int:
        """Add two field elements (XOR in GF(2^e))."""
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        """Multiply two field elements."""
        return int(self.mul_table[a, b])

    def inv(self, a: int) -> int:
        """Multiplicative inverse. Raises ZeroDivisionError for 0."""
        if a == 0:
            raise ZeroDivisionError("Zero has no inverse in GF(2^e)")
        return int(self.inv_table[a])

    def companion_matrix(self, element: int) -> np.ndarray:
        """
        Return the e x e binary matrix representing multiplication by
        *element* in the polynomial basis.

        Properties:
            C(a + b) = C(a) + C(b)  (mod 2)
            C(a * b) = C(a) @ C(b)  (mod 2)
            C(0)     = zero matrix
            C(1)     = identity matrix

        Replacing GF(2^e) entries with companion matrices in a parity-check
        pair preserves orthogonality over GF(2).
        """
        mat = np.zeros((self.e, self.e), dtype=np.uint8)
        for col in range(self.e):
            basis_vec = 1 << col
            product = self.mul(element, basis_vec)
            for row in range(self.e):
                mat[row, col] = (product >> row) & 1
        return mat

    def nonzero_elements(self) -> List[int]:
        """Return all nonzero field elements [1, 2, ..., q-1]."""
        return list(range(1, self.q))

    def __repr__(self) -> str:
        return f"GF2e(e={self.e}, q={self.q}, poly=0b{self.poly:b})"
