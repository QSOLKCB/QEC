"""GF(2^e) finite field arithmetic.

Elements are represented as integers whose binary digits are the
coefficients of the polynomial over GF(2).  Arithmetic is done modulo
an irreducible polynomial of degree *e*.

Irreducible polynomials used (one per extension degree):
    e=1 : x + 1  →  0b11  = 3   (but GF(2) is just {0,1})
    e=2 : x^2 + x + 1  →  0b111 = 7
    e=3 : x^3 + x + 1  →  0b1011 = 11
    e=4 : x^4 + x + 1  →  0b10011 = 19
    e=5 : x^5 + x^2 + 1  →  0b100101 = 37
    e=6 : x^6 + x + 1  →  0b1000011 = 67
    e=7 : x^7 + x^3 + 1  →  0b10001001 = 137
    e=8 : x^8 + x^4 + x^3 + x^2 + 1  →  0b100011101 = 285
"""

from __future__ import annotations

from typing import Dict, Tuple

# Primitive / irreducible polynomials for GF(2^e), keyed by e.
_IRREDUCIBLE: Dict[int, int] = {
    1: 0b11,
    2: 0b111,
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    7: 0b10001001,
    8: 0b100011101,
}


class GF2e:
    """Galois field GF(2^e).

    Parameters
    ----------
    e : int
        Extension degree.  Must be in [1..8].
    """

    def __init__(self, e: int) -> None:
        if e not in _IRREDUCIBLE:
            raise ValueError(f"Unsupported extension degree e={e}; must be in {sorted(_IRREDUCIBLE)}")
        self.e = e
        self.mod = _IRREDUCIBLE[e]
        self.order = 1 << e  # 2^e elements: 0 .. order-1
        self.q = self.order  # compatibility alias for tests

        # Pre-compute log / exp tables for fast multiply & inverse.
        self._exp: list[int] = [0] * (2 * self.order)
        self._log: list[int] = [0] * self.order

        # Find a primitive element (alpha).  For the chosen irreducible
        # polynomials alpha=2 (i.e. the polynomial 'x') is primitive.
        alpha = 2 if e > 1 else 1
        val = 1
        for i in range(self.order - 1):
            self._exp[i] = val
            self._log[val] = i
            val = self._mul_raw(val, alpha)
        # Extend exp table for easy modular lookup.
        for i in range(self.order - 1, 2 * self.order):
            self._exp[i] = self._exp[i - (self.order - 1)]

        # Lazily initialized tables for compatibility with canonical API.
        self._mul_table = None
        self._inv_table = None

    # ------------------------------------------------------------------
    # Low-level carry-less multiply mod irreducible
    # ------------------------------------------------------------------
    def _mul_raw(self, a: int, b: int) -> int:
        """Multiply two field elements (ints) without using the log table."""
        p = 0
        for _ in range(self.e):
            if b & 1:
                p ^= a
            a <<= 1
            if a & (1 << self.e):
                a ^= self.mod
            b >>= 1
        return p

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, a: int, b: int) -> int:
        """Add two elements (XOR in GF(2^e))."""
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        """Multiply two elements."""
        if a == 0 or b == 0:
            return 0
        return self._exp[self._log[a] + self._log[b]]

    def inv(self, a: int) -> int:
        """Multiplicative inverse.  Raises *ValueError* for zero."""
        if a == 0:
            raise ValueError("Zero has no multiplicative inverse in GF(2^e).")
        # a^{-1} = a^{2^e - 2}  or via log table:
        return self._exp[(self.order - 1) - self._log[a]]

    def div(self, a: int, b: int) -> int:
        """a / b."""
        return self.mul(a, self.inv(b))

    def elements(self) -> range:
        """All field elements 0 .. 2^e - 1."""
        return range(self.order)

    def nonzero_elements(self) -> range:
        """Non-zero field elements 1 .. 2^e - 1."""
        return range(1, self.order)

    @property
    def mul_table(self):
        """q x q multiplication table, computed lazily on first access."""
        if self._mul_table is None:
            import numpy as np
            table = np.zeros((self.order, self.order), dtype=np.int32)
            for a in range(self.order):
                for b in range(self.order):
                    table[a, b] = self.mul(a, b)
            self._mul_table = table
        return self._mul_table

    @property
    def inv_table(self):
        """Table of multiplicative inverses, computed lazily on first access."""
        if self._inv_table is None:
            import numpy as np
            table = np.zeros(self.order, dtype=np.int32)
            for a in range(1, self.order):
                table[a] = self.inv(a)
            self._inv_table = table
        return self._inv_table

    def companion_matrix(self, element: int):
        """e x e binary matrix representing multiplication by *element*."""
        import numpy as np
        mat = np.zeros((self.e, self.e), dtype=np.uint8)
        for col in range(self.e):
            basis_vec = 1 << col
            product = self.mul(element, basis_vec)
            for row in range(self.e):
                mat[row, col] = (product >> row) & 1
        return mat
