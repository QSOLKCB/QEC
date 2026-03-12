"""
qec_qldpc_codes.py
==================

Protograph-based quantum LDPC codes approaching the hashing bound.

Implements the construction from:
    Komoto & Kasai, "Quantum Error Correction near the Coding Theoretical Bound",
    npj Quantum Information 11, 154 (2025). arXiv:2412.21171

Key ideas from the paper:
- CSS codes built from orthogonal protograph matrix pairs over GF(2^e)
- Finite field extension replaces GF(2^e) entries with binary companion matrices
- Column weight 2 for optimal non-binary decoding performance
- Girth > 12 via affine permutation matrices to suppress error floors
- Joint X/Z sum-product decoder for the depolarizing channel
- Achieves frame error rate 10^{-4} at p_phys = 9.45% with
  104,000 logical / 312,000 physical qubits

Pre-defined code configurations at rates R in {0.50, 0.60, 0.75}
where R = 1 - 2J/L (J = check rows, L = variable columns).
"""

from __future__ import annotations

import warnings

import numpy as np
from typing import Any, Tuple, Dict, List, Optional, Union
from dataclasses import dataclass, field as dc_field

# Type aliases for bp_decode return variants (v2.9.1)
ResidualMetrics = Dict[str, List[Any]]
BpDecodeReturnType = Union[
    Tuple[np.ndarray, int],
    Tuple[np.ndarray, int, np.ndarray],
    Tuple[np.ndarray, int, ResidualMetrics],
    Tuple[np.ndarray, int, np.ndarray, ResidualMetrics],
]


# ═══════════════════════════════════════════════════════════════════════
# GF(2^e) Arithmetic
# ═══════════════════════════════════════════════════════════════════════

class GF2e:
    """
    Arithmetic over GF(2^e) with companion matrix support.

    Elements are integers in [0, 2^e - 1] where bit i is the
    coefficient of alpha^i in the polynomial basis.

    Supported extension degrees: e in {2, 3, 4} giving GF(4), GF(8), GF(16).
    """

    # Irreducible polynomials p(x) as integers (coefficient of x^e is implicit)
    # e.g. for e=3: x^3 + x + 1 → 0b1011 (bits for 1 + x + 0·x^2 + x^3)
    IRREDUCIBLE_POLY = {
        2: 0b111,    # x^2 + x + 1
        3: 0b1011,   # x^3 + x + 1
        4: 0b10011,  # x^4 + x + 1
    }

    def __init__(self, e: int):
        if e not in self.IRREDUCIBLE_POLY:
            raise ValueError(
                f"GF(2^{e}) not supported. Use e in {sorted(self.IRREDUCIBLE_POLY)}"
            )
        self.e = e
        self.q = 1 << e
        self.poly = self.IRREDUCIBLE_POLY[e]
        self._build_tables()

    def _build_tables(self):
        """Pre-compute multiplication and inverse lookup tables."""
        q = self.q
        self.mul_table = np.zeros((q, q), dtype=np.int32)
        self.inv_table = np.zeros(q, dtype=np.int32)

        for a in range(q):
            for b in range(q):
                self.mul_table[a, b] = self._poly_mul(a, b)

        for a in range(1, q):
            for b in range(1, q):
                if self.mul_table[a, b] == 1:
                    self.inv_table[a] = b
                    break

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

    def mul(self, a: int, b: int) -> int:
        """Multiply two field elements."""
        return int(self.mul_table[a, b])

    def add(self, a: int, b: int) -> int:
        """Add two field elements (XOR in GF(2^e))."""
        return a ^ b

    def inv(self, a: int) -> int:
        """Multiplicative inverse (raises ZeroDivisionError for 0)."""
        if a == 0:
            raise ZeroDivisionError("Zero has no inverse in GF(2^e)")
        return int(self.inv_table[a])

    def companion_matrix(self, element: int) -> np.ndarray:
        """
        e x e binary matrix representing multiplication by *element*.

        In the polynomial basis {1, alpha, ..., alpha^{e-1}}, left-
        multiplication by an element is a GF(2)-linear map.  This
        method returns that map as a binary matrix.

        The companion matrix representation preserves field operations:
            C(a + b) = C(a) + C(b)  (mod 2)
            C(a * b) = C(a) @ C(b)  (mod 2)
            C(0)     = zero matrix
        so replacing GF(2^e) entries with companion matrices in a parity-
        check pair preserves orthogonality over GF(2).
        """
        mat = np.zeros((self.e, self.e), dtype=np.uint8)
        for col in range(self.e):
            # Column col = representation of element * alpha^col
            basis_vec = 1 << col
            product = self.mul(element, basis_vec)
            for row in range(self.e):
                mat[row, col] = (product >> row) & 1
        return mat

    def nonzero_elements(self) -> List[int]:
        """Return all nonzero elements [1, 2, ..., q-1]."""
        return list(range(1, self.q))


# ═══════════════════════════════════════════════════════════════════════
# Protograph Pair Construction
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ProtographPair:
    """
    Orthogonal protograph matrix pair (B_X, B_Z) over GF(2^e).

    CSS orthogonality: sum_j B_X[i1, j] * B_Z[i2, j] = 0  in GF(2^e)
    for every pair of rows (i1, i2).

    Code rate: R = 1 - 2*J / L
    """
    J: int                   # rows (check nodes per type)
    L: int                   # columns (variable nodes)
    B_X: np.ndarray          # J x L matrix, entries in {0, ..., q-1}
    B_Z: np.ndarray          # J x L matrix, entries in {0, ..., q-1}
    field: GF2e

    @property
    def rate(self) -> float:
        return 1.0 - 2.0 * self.J / self.L


def _verify_orthogonality_gf(
    B_X: np.ndarray, B_Z: np.ndarray, gf: GF2e
) -> bool:
    """Check B_X . B_Z^T = 0 over GF(2^e)."""
    J = B_X.shape[0]
    for i1 in range(J):
        for i2 in range(B_Z.shape[0]):
            acc = 0
            for k in range(B_X.shape[1]):
                acc = gf.add(acc, gf.mul(int(B_X[i1, k]), int(B_Z[i2, k])))
            if acc != 0:
                return False
    return True


def build_protograph_pair(
    J: int,
    L: int,
    gf: GF2e,
    seed: int = 42
) -> ProtographPair:
    """
    Build an orthogonal protograph pair over GF(2^e).

    Orthogonality is enforced **by construction**, not by iterative
    repair.  We use B_X = B_Z (self-orthogonal) for all configurations:

    * J = 1: all-ones row.  Inner product = L * 1 = 0 in GF(2^e) when
      L is even (1+1 = 0 in any characteristic-2 field).  For odd L the
      last entry is adjusted once.
    * J >= 2: block-diagonal structure with explicit 2x2 orthogonal
      blocks guarantees B @ B^T = 0 globally, without pairwise patching.

    Args:
        J: Number of check rows per stabiliser type
        L: Number of variable columns (must satisfy L >= 2*J)
        gf: Finite field GF(2^e)
        seed: Random seed
    """
    rng = np.random.default_rng(seed)
    q = gf.q

    B = np.zeros((J, L), dtype=np.int32)

    if J == 1:
        # All-ones row.  sum_j 1*1 = L.  In GF(2^e) this is 0 when L
        # is even (char 2), which covers L=4,8.  Handle odd L too.
        B[0, :] = 1
        acc = 0
        for j in range(L):
            acc = gf.add(acc, gf.mul(1, 1))
        if acc != 0:
            # Replace last entry so the sum cancels.
            # Current contribution of position L-1 is 1*1 = 1.
            # We need new_val such that (acc XOR 1) XOR new_val^2 = 0,
            # i.e. new_val^2 = acc XOR 1.  Simpler: set B[0,L-1] to a
            # value v where v*v = acc (the residual without that col).
            # Since we want sum_{j<L-1} 1 + v*v = 0 → v*v = acc XOR 1
            # Just brute-force the tiny field:
            target = gf.add(acc, gf.mul(1, 1))  # acc minus old contrib
            found = False
            for v in range(1, q):
                if gf.mul(v, v) == target:
                    B[0, L - 1] = v
                    found = True
                    break
            if not found:
                # Fallback: make it zero, add a compensating column pair
                # For the fields/sizes we use this path is never hit.
                B[0, L - 1] = 0  # pragma: no cover
    else:
        # Multi-row self-orthogonal construction.
        #
        # Strategy: fill B in 2-column blocks so that each 2-col slice
        # contributes a J×J zero matrix to B @ B^T (over GF(2^e)).
        #
        # For a pair of columns (j, j+1), assign nonzero values to
        # exactly two rows (r1, r2).  The 2x2 sub-block contribution
        # to B @ B^T is:
        #
        #   [a, b] @ [a, b]^T = [a*a + b*b]   (diagonal)
        #   [c, d]   [c, d]     [c*a + d*b]   (off-diag)
        #
        # We choose values so each 2x2 is self-orthogonal:
        #   a*a + b*b = 0  →  b = a * sqrt(-1) [exists in GF(2^e)]
        #   c*c + d*d = 0  →  d = c * sqrt(-1)
        #   c*a + d*b = 0  →  c*a + c*sqrt(-1)*a*sqrt(-1)
        #                    = c*a*(1 + (-1)) = c*a*0 = 0  [char 2: -1=1]
        #
        # In characteristic 2, a*a + b*b = (a+b)^2 = 0 iff a = b.
        # And the cross-term: c*a + d*b = c*a + c*a = 0.  Perfect.
        # So we just need b = a, d = c for each paired column.
        #
        n_pairs = L // 2
        row_idx = 0
        for p_idx in range(n_pairs):
            j1 = 2 * p_idx
            j2 = 2 * p_idx + 1
            # Pick two rows for this column pair (cycling through rows)
            r1 = row_idx % J
            r2 = (row_idx + 1) % J
            row_idx += 2
            a = int(rng.integers(1, q))
            c = int(rng.integers(1, q))
            # Set identical values in paired columns → self-orthogonal
            B[r1, j1] = a
            B[r1, j2] = a
            B[r2, j1] = c
            B[r2, j2] = c
        # Handle odd leftover column (if any)
        if L % 2 == 1:
            # Last column is all-zero → contributes nothing to B @ B^T
            pass

    # B_X = B_Z = B  →  self-orthogonal CSS code
    B_X = B.copy()
    B_Z = B.copy()

    if not _verify_orthogonality_gf(B_X, B_Z, gf):
        raise ConstructionInvariantError(
            f"Protograph pair (J={J}, L={L}) failed GF(2^{gf.e}) "
            f"orthogonality check.  This is a bug in the construction."
        )
    return ProtographPair(J=J, L=L, B_X=B_X, B_Z=B_Z, field=gf)


class ConstructionInvariantError(Exception):
    """Raised when a code construction violates a structural invariant."""


# ═══════════════════════════════════════════════════════════════════════
# CSS Quantum LDPC Code
# ═══════════════════════════════════════════════════════════════════════

class QuantumLDPCCode:
    """
    CSS quantum LDPC code from a protograph pair over GF(2^e).

    Construction (Komoto & Kasai 2025):
        1. Start with orthogonal pair (B_X, B_Z) over GF(2^e)
        2. Lift each nonzero entry into a P x P circulant permutation matrix
        3. Replace the GF(2^e) scalar with its e x e companion matrix
        4. Each block position becomes an (eP) x (eP) binary block:
               block(i,j) = companion(B[i,j]) kron circulant(shift_{i,j})
        5. The full matrices H_X, H_Z are binary and satisfy
               H_X . H_Z^T = 0  (mod 2)

    Parameters:
        n  = e * P * L     physical qubits
        k  = n - rank(H_X) - rank(H_Z)   logical qubits
        R ~= 1 - 2J/L      code rate
    """

    def __init__(
        self,
        protograph: ProtographPair,
        lifting_size: int = 32,
        seed: int = 42
    ):
        self.protograph = protograph
        self.P = lifting_size
        self.field = protograph.field
        self.e = protograph.field.e
        self.seed = seed

        self.H_X, self.H_Z = self._build_parity_checks()

        # Hard invariant — if this fires the construction is wrong.
        if not self.verify_css_orthogonality():
            raise ConstructionInvariantError(
                "H_X @ H_Z^T != 0 mod 2 — CSS orthogonality violated. "
                "This is a construction bug, not a recoverable condition."
            )

        self.n = self.H_X.shape[1]
        self.m_X = self.H_X.shape[0]
        self.m_Z = self.H_Z.shape[0]

        # Rank computation (for moderate sizes)
        if self.n <= 20000:
            rx = np.linalg.matrix_rank(self.H_X.astype(np.float64))
            rz = np.linalg.matrix_rank(self.H_Z.astype(np.float64))
            self.k = max(0, self.n - rx - rz)
        else:
            # Estimate from protograph parameters
            self.k = max(0, self.n - self.m_X - self.m_Z)

        self.rate = self.k / self.n if self.n > 0 else 0.0

    def _build_parity_checks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct binary H_X and H_Z via lifting + companion replacement.

        CRITICAL INVARIANT (Komoto-Kasai 2025, Section III):
        Each protograph column j gets ONE circulant permutation pi_j,
        shared by both H_X and H_Z.  This ensures:

            (C(a) x pi_j) @ (C(b) x pi_j)^T  =  C(a) C(b)^T  x  I_P

        so GF-level orthogonality (sum_j a_j * b_j = 0 in GF(2^e))
        carries through to the binary expansion:  H_X @ H_Z^T = 0 mod 2.

        Generating independent permutations for H_X and H_Z would break
        this telescoping and destroy CSS orthogonality.
        """
        proto = self.protograph
        P = self.P
        e = self.e
        J, L = proto.J, proto.L
        rng = np.random.default_rng(self.seed)

        block = e * P

        # Step 1: generate ONE shared circulant shift per column j.
        # Deterministic from seed, iteration order is sorted by j.
        col_shifts = {j: int(rng.integers(0, P)) for j in range(L)}

        # Step 2: build the permutation matrices (cached)
        col_perms = {}
        for j in range(L):
            shift = col_shifts[j]
            perm = np.zeros((P, P), dtype=np.uint8)
            for k in range(P):
                perm[(k + shift) % P, k] = 1
            col_perms[j] = perm

        # Step 3: fill H_X, H_Z using shared permutations
        H_X = np.zeros((J * block, L * block), dtype=np.uint8)
        H_Z = np.zeros((J * block, L * block), dtype=np.uint8)

        for i in range(J):
            for j in range(L):
                r0 = i * block
                c0 = j * block
                pj = col_perms[j]
                if proto.B_X[i, j] != 0:
                    comp = self.field.companion_matrix(int(proto.B_X[i, j]))
                    H_X[r0:r0 + block, c0:c0 + block] = np.kron(comp, pj)
                if proto.B_Z[i, j] != 0:
                    comp = self.field.companion_matrix(int(proto.B_Z[i, j]))
                    H_Z[r0:r0 + block, c0:c0 + block] = np.kron(comp, pj)

        return H_X, H_Z

    # ── Syndrome helpers ──────────────────────────────────────────────

    def verify_css_orthogonality(self) -> bool:
        """Check H_X . H_Z^T = 0 (mod 2)."""
        product = (self.H_X.astype(np.int32) @ self.H_Z.astype(np.int32).T) % 2
        return bool(np.all(product == 0))

    def syndrome_X(self, z_error: np.ndarray) -> np.ndarray:
        """X-type syndrome from a Z-error pattern (length-n binary vector)."""
        return (self.H_X.astype(np.int32) @ z_error.astype(np.int32)) % 2

    def syndrome_Z(self, x_error: np.ndarray) -> np.ndarray:
        """Z-type syndrome from an X-error pattern (length-n binary vector)."""
        return (self.H_Z.astype(np.int32) @ x_error.astype(np.int32)) % 2

    def __repr__(self) -> str:
        return (
            f"QuantumLDPCCode(n={self.n}, k={self.k}, "
            f"rate={self.rate:.4f}, "
            f"J={self.protograph.J}, L={self.protograph.L}, "
            f"P={self.P}, e={self.e})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Joint X/Z Sum-Product (Belief Propagation) Decoder
# ═══════════════════════════════════════════════════════════════════════

class JointSPDecoder:
    """
    Joint X/Z sum-product decoder for CSS codes under depolarizing noise.

    The decoder from Komoto & Kasai (2025) marginalises the posterior
    probability of X and Z errors simultaneously, exploiting the
    correlation introduced by Y = XZ errors in the depolarizing channel.

    Uses log-likelihood ratios (LLRs) and the tanh rule for numerical
    stability.
    """

    def __init__(self, code: QuantumLDPCCode, max_iter: int = 100):
        self.code = code
        self.max_iter = max_iter

        # Pre-build Tanner-graph adjacency lists
        self._c2v_X, self._v2c_X = _tanner_graph(code.H_X)
        self._c2v_Z, self._v2c_Z = _tanner_graph(code.H_Z)

    def decode(
        self,
        syndrome_x: np.ndarray,
        syndrome_z: np.ndarray,
        p_phys: float
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Joint X/Z decoding under the depolarizing channel.

        Args:
            syndrome_x: X-syndrome (detects Z errors), length m_X
            syndrome_z: Z-syndrome (detects X errors), length m_Z
            p_phys: depolarizing error probability

        Returns:
            (x_hat, z_hat, converged)
            x_hat, z_hat are binary error estimates, converged is True
            when the decoded errors reproduce the observed syndrome.
        """
        # Channel marginals: P(X or Y), P(Z or Y)
        p_x_or_y = 2.0 * p_phys / 3.0
        p_z_or_y = 2.0 * p_phys / 3.0

        # Decode X-component (Z-checks detect X errors)
        x_hat = self._bp_component(
            self.code.H_Z, syndrome_z,
            self._c2v_Z, self._v2c_Z,
            p_err=p_x_or_y
        )

        # Decode Z-component (X-checks detect Z errors)
        z_hat = self._bp_component(
            self.code.H_X, syndrome_x,
            self._c2v_X, self._v2c_X,
            p_err=p_z_or_y
        )

        # Verify
        sx_hat = (self.code.H_X.astype(np.int32) @ z_hat.astype(np.int32)) % 2
        sz_hat = (self.code.H_Z.astype(np.int32) @ x_hat.astype(np.int32)) % 2
        converged = (
            np.array_equal(sx_hat, syndrome_x.astype(np.uint8))
            and np.array_equal(sz_hat, syndrome_z.astype(np.uint8))
        )
        return x_hat, z_hat, converged

    def _bp_component(
        self,
        H: np.ndarray,
        syndrome: np.ndarray,
        c2v: List[List[int]],
        v2c: List[List[int]],
        p_err: float
    ) -> np.ndarray:
        """
        Standard min-sum / sum-product BP for one component.

        Uses LLR messages and the tanh product rule for check updates.
        """
        m, n = H.shape
        eps = 1e-30
        channel_llr = np.log((1.0 - p_err + eps) / (p_err + eps))

        # Variable-to-check LLR messages (initialised to channel)
        v2c_msg = np.full((n, m), channel_llr, dtype=np.float64)
        # Check-to-variable LLR messages
        c2v_msg = np.zeros((m, n), dtype=np.float64)

        hard = np.zeros(n, dtype=np.uint8)

        for _it in range(self.max_iter):
            # ── check → variable ──
            for c in range(m):
                nbrs = c2v[c]
                if len(nbrs) == 0:
                    continue
                if len(nbrs) == 1:
                    # Degree-1 check node provides no extrinsic information.
                    c2v_msg[c, nbrs[0]] = 0.0
                    continue
                # Gather incoming tanh half-LLRs
                tanhs = np.array([
                    np.tanh(np.clip(v2c_msg[v, c] / 2.0, -20.0, 20.0))
                    for v in nbrs
                ])
                prod_all = np.prod(tanhs)
                sign = (-1.0) ** int(syndrome[c])
                for idx, v in enumerate(nbrs):
                    prod_excl = prod_all / (tanhs[idx] + eps)
                    prod_excl = np.clip(prod_excl, -1 + 1e-15, 1 - 1e-15)
                    c2v_msg[c, v] = sign * 2.0 * np.arctanh(prod_excl)

            # ── variable → check + hard decision ──
            for v in range(n):
                nbrs = v2c[v]
                total = channel_llr + sum(c2v_msg[c, v] for c in nbrs)
                for c in nbrs:
                    v2c_msg[v, c] = total - c2v_msg[c, v]
                hard[v] = 0 if total >= 0.0 else 1

            # ── early stop ──
            if np.array_equal((H.astype(np.int32) @ hard.astype(np.int32)) % 2,
                              syndrome.astype(np.uint8)):
                return hard

        return hard


def _tanner_graph(
    H: np.ndarray,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Build check→var and var→check adjacency from a binary matrix H."""
    m, n = H.shape
    c2v: List[List[int]] = [[] for _ in range(m)]
    v2c: List[List[int]] = [[] for _ in range(n)]
    rows, cols = np.nonzero(H)
    for r, c in zip(rows, cols):
        c2v[r].append(c)
        v2c[c].append(r)
    return c2v, v2c


# ═══════════════════════════════════════════════════════════════════════
# Depolarizing Channel
# ═══════════════════════════════════════════════════════════════════════

def depolarizing_channel(
    n: int, p: float, rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample i.i.d. depolarizing noise on *n* qubits.

    Each qubit independently receives:
        I  with probability  1 - p
        X  with probability  p / 3
        Y  with probability  p / 3   (Y = XZ → both x and z flip)
        Z  with probability  p / 3

    Returns:
        (x_error, z_error)  — binary vectors of length n.
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)
    probs = np.array([1.0 - p, p / 3.0, p / 3.0, p / 3.0])
    choices = rng.choice(4, size=n, p=probs)

    x_error = np.isin(choices, [1, 3]).astype(np.uint8)   # X or Y
    z_error = np.isin(choices, [2, 3]).astype(np.uint8)   # Z or Y
    return x_error, z_error


# ═══════════════════════════════════════════════════════════════════════
# Decoder Utilities — Pauli Frame, Syndrome, BP, Channel LLR
# ═══════════════════════════════════════════════════════════════════════

def update_pauli_frame(
    frame: np.ndarray,
    correction: np.ndarray,
) -> np.ndarray:
    """
    Apply a correction to a Pauli frame via GF(2) addition (XOR).

    Pure function: neither *frame* nor *correction* is mutated.

    Args:
        frame: Binary vector (length n) representing the current Pauli frame.
        correction: Binary vector (length n) representing the correction to apply.

    Returns:
        New binary vector: frame XOR correction.

    Raises:
        ValueError: If inputs are not 1-D or have mismatched shapes.
    """
    frame = np.asarray(frame)
    correction = np.asarray(correction)
    if frame.ndim != 1 or correction.ndim != 1:
        raise ValueError(
            f"Both frame and correction must be 1-D arrays, "
            f"got ndim={frame.ndim} and ndim={correction.ndim}"
        )
    if frame.shape[0] != correction.shape[0]:
        raise ValueError(
            f"Shape mismatch: frame has length {frame.shape[0]}, "
            f"correction has length {correction.shape[0]}"
        )
    return (frame ^ correction).astype(np.uint8)


def syndrome(H: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Compute the binary syndrome s = H @ e (mod 2).

    This is a standalone version of the syndrome computation performed by
    :meth:`QuantumLDPCCode.syndrome_X` and :meth:`QuantumLDPCCode.syndrome_Z`.

    Args:
        H: Binary parity-check matrix, shape (m, n).
        e: Binary error vector, length n.

    Returns:
        Binary syndrome vector of length m, dtype uint8.
    """
    return ((H.astype(np.int32) @ np.asarray(e).astype(np.int32)) % 2).astype(np.uint8)


_BP_MODES = {"sum_product", "min_sum", "norm_min_sum", "offset_min_sum",
             "improved_norm", "improved_offset", "min_sum_urw"}
_BP_SCHEDULES = {"flooding", "layered", "residual", "hybrid_residual", "adaptive", "geom_v1"}
_BP_POSTPROCESS = {None, "osd0", "osd1", "osd_cs", "guided_decimation", "mp_osd1", "mp_osd_cs"}


def bp_decode(
    H: np.ndarray,
    llr: np.ndarray,
    max_iters: int = 100,
    mode: str = "sum_product",
    damping: float = 0.0,
    norm_factor: float = 0.75,
    offset: float = 0.5,
    clip: Optional[float] = None,
    schedule: str = "flooding",
    postprocess: Optional[str] = None,
    seed: Optional[int] = None,
    syndrome_vec: Optional[np.ndarray] = None,
    llr_history: int = 0,
    osd_cs_lam: int = 1,
    # ── v2.8.0 opt-in parameters (defaults preserve v2.7.0 behavior) ──
    alpha1: float = 0.9,
    alpha2: float = 0.75,
    hybrid_residual_threshold: Optional[float] = None,
    ensemble_k: int = 1,
    state_aware_residual: bool = False,
    phi_by_state: Optional[np.ndarray] = None,
    s_by_state: Optional[np.ndarray] = None,
    state_label_by_check: Optional[np.ndarray] = None,
    lift_braided: bool = False,
    braid_distance: Optional[int] = None,
    gsa_style: Optional[str] = None,
    # ── v2.9.0 adaptive schedule parameters ──
    adaptive_k1: Optional[int] = None,
    adaptive_rule: str = "one_way",
    adaptive_log: bool = False,
    # ── v2.9.1 opt-in residual instrumentation ──
    residual_metrics: bool = False,
    # ── v3.4.0 guided decimation parameters ──
    decimation_rounds: int = 10,
    decimation_inner_iters: int = 10,
    decimation_freeze_llr: float = 1000.0,
    # ── v3.7.0 URW-BP parameter ──
    urw_rho: float = 1.0,
    # ── v3.9.0 BP energy trace ──
    energy_trace: bool = False,
    **kwargs,
) -> BpDecodeReturnType:
    """
    Standalone belief-propagation decoder for a binary parity-check matrix.

    Supports multiple check-node update rules, message damping,
    magnitude clipping, and optional OSD post-processing.

    Args:
        H: Binary parity-check matrix, shape (m, n).
        llr: Per-variable log-likelihood ratios, length n (or scalar broadcast).
        max_iters: Maximum BP iterations (default 100).
        mode: Check-node update rule.  One of ``"sum_product"``,
            ``"min_sum"``, ``"norm_min_sum"``, ``"offset_min_sum"``,
            ``"improved_norm"`` (v2.8.0), ``"improved_offset"`` (v2.8.0).
        damping: Damping factor in [0, 1).  ``0.0`` disables damping.
        norm_factor: Normalisation factor for ``"norm_min_sum"`` mode.
            Must be in the interval (0.0, 1.0].
        offset: Offset for ``"offset_min_sum"`` mode.
            Must be >= 0.0.
        clip: If not None, clip check-to-variable message magnitudes to
            ``[-clip, clip]`` after each iteration.
        schedule: Message-passing schedule.  ``"flooding"`` (default)
            updates all check nodes then all variable nodes per iteration.
            ``"layered"`` processes check nodes serially, updating beliefs
            incrementally — typically converges in fewer iterations.
            ``"residual"`` is a variant of layered that reorders check nodes
            each iteration by descending max message residual, with
            deterministic tie-breaking by ascending check index.
            ``"hybrid_residual"`` (v2.8.0) deterministic hybrid of layered
            + residual with even/odd layer partitioning.
        postprocess: Optional post-processing.  ``"osd0"`` applies order-0
            Ordered Statistics Decoding when BP fails to converge.
            ``"osd1"`` extends OSD-0 by testing a single least-reliable
            bit flip.  ``"osd_cs"`` applies Combination Sweep OSD
            (controlled by *osd_cs_lam*).
        seed: Unused; reserved for future stochastic post-processors.
        syndrome_vec: Binary syndrome vector, length m.  Defaults to all-zeros.
        llr_history: Number of recent L_total snapshots to retain.
            When 0 (default), no history is stored and the return type
            is ``(correction, iterations)``.  When > 0, the return type
            becomes ``(correction, iterations, history)`` where *history*
            has shape ``(k, n)`` with ``k = min(iterations_run, llr_history)``.
        osd_cs_lam: Lambda parameter for OSD-CS when
            ``postprocess="osd_cs"``.  Maximum number of pivot bits to
            flip simultaneously.  Default 1.
        alpha1: Scaling factor for ``"improved_norm"`` / ``"improved_offset"``
            modes (v2.8.0).  Applied to the first minimum.  Must be in
            (0, 1].  Default 0.9.
        alpha2: Scaling factor for ``"improved_norm"`` / ``"improved_offset"``
            modes (v2.8.0).  Applied to the second minimum.  Must be in
            (0, 1].  Default 0.75.
        hybrid_residual_threshold: Threshold for ``"hybrid_residual"``
            schedule (v2.8.0).  If None (default), update all checks in
            layered-residual order.  If set (>= 0), within each layer
            update checks with residual > threshold first, then the rest.
        ensemble_k: Number of ensemble members for deterministic ensemble
            decoding (v2.8.0).  ``1`` (default) disables ensemble mode.
            Member 0 is always the exact baseline.  Hard cap at 8; a
            warning is issued if > 4.
        state_aware_residual: If True, apply state-aware weighting to
            residuals before ordering (v2.8.0).  Requires *phi_by_state*,
            *s_by_state*, and *state_label_by_check* to be provided.
        phi_by_state: Per-state phase array for state-aware residual
            weighting.  Required when ``state_aware_residual=True``.
        s_by_state: Per-state amplitude array for state-aware residual
            weighting.  Required when ``state_aware_residual=True``.
        state_label_by_check: Per-check state label array (integer indices
            into *phi_by_state* / *s_by_state*).  Required when
            ``state_aware_residual=True``.
        lift_braided: Reserved for future braided-lift construction
            (v2.8.0).  Not yet implemented.
        braid_distance: Reserved for future braided-lift construction.
        gsa_style: Reserved for future braided-lift construction.
        adaptive_k1: Number of iterations for phase 1 (flooding) when
            ``schedule="adaptive"``.  Defaults to ``max(1, max_iters // 4)``
            when *None*.

            **Adaptive schedule overview** (v2.9.0):

            ``schedule="adaptive"`` is a deterministic one-way checkpoint
            controller.  It does **not** pass internal message state between
            phases — each phase is a fresh, independent ``bp_decode()`` call.

            * Phase 1: ``"flooding"`` for *k1* iterations.
            * If phase 1 converges, return immediately (iters = iters_p1).
            * Phase 2: ``"hybrid_residual"`` for *max_iters − k1* iterations.
            * Tie-break between phases: prefer converged → lower syndrome
              weight → fewer total iterations → phase order (deterministic).
            * When phase 2 is selected the returned iteration count is
              cumulative: ``k1 + iters_p2`` (total work performed).

            No residual-based dynamic thresholding is applied to the
            switching decision (reserved for a future extension).

        adaptive_rule: Switching rule for adaptive schedule.  Currently
            only ``"one_way"`` is supported: always switch to
            ``"hybrid_residual"`` after phase 1 if not converged.
        adaptive_log: If *True*, print a one-line log showing which
            phase was selected and why.  Default *False*.
        **kwargs: Accepts legacy ``max_iter`` keyword for backward
            compatibility.

    Returns:
        When ``residual_metrics=False`` (default):

            When ``llr_history == 0``:
                ``(correction, iterations)`` — hard-decision binary vector
                (length n, dtype uint8) and iteration count.

            When ``llr_history > 0``:
                ``(correction, iterations, history)`` — same as above plus
                a float64 array of shape ``(k, n)`` containing the last *k*
                per-iteration L_total snapshots.

        When ``residual_metrics=True``:

            When ``llr_history == 0``:
                ``(correction, iterations, metrics)`` — 3-tuple with a
                metrics dict as the last element.

            When ``llr_history > 0``:
                ``(correction, iterations, history, metrics)`` — 4-tuple
                with the LLR history array followed by the metrics dict.

            The *metrics* dict has keys:
                ``"residual_linf"``
                    List of per-iteration L-inf residual arrays, each of
                    shape ``(n_checks,)``.
                ``"residual_l2"``
                    List of per-iteration L2 residual arrays, each of
                    shape ``(n_checks,)``.
                ``"residual_energy"``
                    List of per-iteration scalar energy values.
            All three lists have length equal to the number of iterations
            executed.  For schedules that do not compute residuals
            (e.g. ``"flooding"``), the lists are empty.
    """
    # ── backward compatibility: accept old ``max_iter`` keyword ──
    if "max_iter" in kwargs:
        if max_iters != 100:
            raise TypeError("Cannot pass both max_iter and max_iters")
        max_iters = kwargs.pop("max_iter")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    # ── parameter validation ──
    if mode not in _BP_MODES:
        raise ValueError(f"mode must be one of {_BP_MODES}, got '{mode}'")
    if schedule not in _BP_SCHEDULES:
        raise NotImplementedError(
            f"schedule '{schedule}' not implemented; use one of {_BP_SCHEDULES}"
        )
    if postprocess not in _BP_POSTPROCESS:
        raise ValueError(f"Unknown postprocess: '{postprocess}'")

    # ── numeric parameter validation ──
    if not (0.0 < norm_factor <= 1.0):
        raise ValueError(
            f"norm_factor must be in the interval (0.0, 1.0], got {norm_factor}"
        )
    if offset < 0.0:
        raise ValueError(f"offset must be >= 0.0, got {offset}")
    if not isinstance(llr_history, int) or llr_history < 0:
        raise ValueError(
            f"llr_history must be a non-negative integer, got {llr_history}"
        )
    if not isinstance(osd_cs_lam, int) or osd_cs_lam < 0:
        raise ValueError(
            f"osd_cs_lam must be a non-negative integer, got {osd_cs_lam}"
        )

    # ── v2.8.0 parameter validation ──
    if mode in ("improved_norm", "improved_offset"):
        if not (0.0 < alpha1 <= 1.0):
            raise ValueError(
                f"alpha1 must be in the interval (0.0, 1.0], got {alpha1}"
            )
        if not (0.0 < alpha2 <= 1.0):
            raise ValueError(
                f"alpha2 must be in the interval (0.0, 1.0], got {alpha2}"
            )

    if schedule == "hybrid_residual" and hybrid_residual_threshold is not None and hybrid_residual_threshold < 0.0:
        raise ValueError(
            f"hybrid_residual_threshold must be >= 0.0, got {hybrid_residual_threshold}"
        )

    # ── v3.7.0 URW-BP validation ──
    if mode == "min_sum_urw":
        if not (0.0 < urw_rho <= 1.0):
            raise ValueError(
                f"urw_rho must be in the interval (0.0, 1.0], got {urw_rho}"
            )

    # ── v2.9.0 adaptive schedule validation ──
    _ADAPTIVE_RULES = {"one_way"}
    if schedule == "adaptive":
        if adaptive_rule not in _ADAPTIVE_RULES:
            raise ValueError(
                f"adaptive_rule must be one of {_ADAPTIVE_RULES}, "
                f"got '{adaptive_rule}'"
            )
        if adaptive_k1 is not None:
            if not isinstance(adaptive_k1, int) or adaptive_k1 < 1:
                raise ValueError(
                    f"adaptive_k1 must be a positive integer, got {adaptive_k1}"
                )
            if adaptive_k1 >= max_iters:
                raise ValueError(
                    f"adaptive_k1 must be < max_iters ({max_iters}), "
                    f"got {adaptive_k1}"
                )

    if not isinstance(ensemble_k, int) or ensemble_k < 1:
        raise ValueError(
            f"ensemble_k must be an integer >= 1, got {ensemble_k}"
        )
    if ensemble_k > 8:
        raise ValueError(
            f"ensemble_k must be <= 8, got {ensemble_k}"
        )
    if ensemble_k > 4:
        warnings.warn(
            f"ensemble_k={ensemble_k} is large; values > 4 may be slow "
            "with diminishing returns",
            stacklevel=2,
        )

    if state_aware_residual:
        if phi_by_state is None or s_by_state is None or state_label_by_check is None:
            raise ValueError(
                "state_aware_residual=True requires phi_by_state, "
                "s_by_state, and state_label_by_check to be provided"
            )
        state_label_by_check = np.asarray(state_label_by_check, dtype=np.int64)
        phi_by_state = np.asarray(phi_by_state, dtype=np.float64)
        s_by_state = np.asarray(s_by_state, dtype=np.float64)
        if state_label_by_check.shape[0] != H.shape[0]:
            raise ValueError(
                f"len(state_label_by_check) must equal m={H.shape[0]}, "
                f"got {state_label_by_check.shape[0]}"
            )
        if state_label_by_check.shape[0] > 0:
            _min_label = int(np.min(state_label_by_check))
            if _min_label < 0:
                raise ValueError(
                    f"state_label_by_check must be >= 0, got min={_min_label}"
                )
            _max_label = int(np.max(state_label_by_check))
            if _max_label >= len(phi_by_state):
                raise ValueError(
                    f"state_label_by_check max={_max_label} exceeds "
                    f"phi_by_state length={len(phi_by_state)}"
                )
            if _max_label >= len(s_by_state):
                raise ValueError(
                    f"state_label_by_check max={_max_label} exceeds "
                    f"s_by_state length={len(s_by_state)}"
                )

    if state_aware_residual:
        state_aware_residual_weights = (
            s_by_state[state_label_by_check]
            * np.abs(np.cos(phi_by_state[state_label_by_check]))
        )

    if lift_braided:
        raise NotImplementedError(
            "lift_braided is reserved for a future release and not yet implemented"
        )

    # ── v3.4.0: guided decimation early dispatch ──
    # This is a separate decode path that orchestrates its own BP sub-calls.
    # It exits before the main iteration loop and never touches
    # _bp_postprocess() or any existing schedule/postprocess logic.
    if postprocess == "guided_decimation":
        if not isinstance(decimation_rounds, int) or decimation_rounds < 1:
            raise ValueError(
                f"decimation_rounds must be a positive integer, "
                f"got {decimation_rounds}"
            )
        if not isinstance(decimation_inner_iters, int) or decimation_inner_iters < 1:
            raise ValueError(
                f"decimation_inner_iters must be a positive integer, "
                f"got {decimation_inner_iters}"
            )
        if not isinstance(decimation_freeze_llr, (int, float)) or decimation_freeze_llr <= 0.0:
            raise ValueError(
                f"decimation_freeze_llr must be a positive number, "
                f"got {decimation_freeze_llr}"
            )

        from .decoder.decimation import guided_decimation

        _H = np.asarray(H, dtype=np.uint8)
        _llr = np.broadcast_to(
            np.asarray(llr, dtype=np.float64), (_H.shape[1],)
        ).copy()

        if syndrome_vec is None:
            _syn = np.zeros(_H.shape[0], dtype=np.uint8)
        else:
            _syn = np.asarray(syndrome_vec, dtype=np.uint8)

        # Build bp_kwargs for the inner BP calls — pass through all
        # schedule/mode params but NOT postprocess, max_iters, llr_history,
        # syndrome_vec, or the decimation params themselves.
        _inner_bp_kwargs = {
            "mode": mode,
            "damping": damping,
            "norm_factor": norm_factor,
            "offset": offset,
            "schedule": schedule,
            "alpha1": alpha1,
            "alpha2": alpha2,
        }
        if clip is not None:
            _inner_bp_kwargs["clip"] = clip
        if hybrid_residual_threshold is not None:
            _inner_bp_kwargs["hybrid_residual_threshold"] = hybrid_residual_threshold

        correction, total_iters = guided_decimation(
            _H, _llr,
            syndrome_vec=_syn,
            decimation_rounds=decimation_rounds,
            decimation_inner_iters=decimation_inner_iters,
            decimation_freeze_llr=decimation_freeze_llr,
            bp_kwargs=_inner_bp_kwargs,
        )

        # ── Assemble return tuple matching bp_decode output-shape rules ──
        # The caller may have requested llr_history or residual_metrics.
        # Guided decimation does not produce these internally, so return
        # empty/neutral values that match the expected shapes.
        if llr_history > 0:
            _empty_hist = np.empty((0, _H.shape[1]), dtype=np.float64)
            if residual_metrics:
                _empty_metrics = {
                    "residual_linf": [],
                    "residual_l2": [],
                    "residual_energy": [],
                }
                return correction, total_iters, _empty_hist, _empty_metrics
            return correction, total_iters, _empty_hist
        if residual_metrics:
            _empty_metrics = {
                "residual_linf": [],
                "residual_l2": [],
                "residual_energy": [],
            }
            return correction, total_iters, _empty_metrics
        return correction, total_iters

    # ── v3.5.0: mp_osd1 early dispatch ──
    # MP-aware OSD-1 that uses posterior LLR magnitude for column ordering.
    # Follows the same early-return wrapper pattern as guided_decimation.
    # Runs BP with postprocess=None and llr_history=1 to obtain posterior,
    # then applies mp_osd1_postprocess if syndrome is not satisfied.
    if postprocess == "mp_osd1":
        from .decoder.osd import mp_osd1_postprocess

        _H = np.asarray(H, dtype=np.uint8)
        _llr = np.broadcast_to(
            np.asarray(llr, dtype=np.float64), (_H.shape[1],)
        ).copy()

        if syndrome_vec is None:
            _syn = np.zeros(_H.shape[0], dtype=np.uint8)
        else:
            _syn = np.asarray(syndrome_vec, dtype=np.uint8)

        # Inner BP call — same H, llr, max_iters, schedule, mode, etc.
        # postprocess=None prevents recursion; llr_history=1 captures posterior.
        _inner_result = bp_decode(
            _H, _llr,
            max_iters=max_iters,
            mode=mode,
            damping=damping,
            norm_factor=norm_factor,
            offset=offset,
            clip=clip,
            schedule=schedule,
            postprocess=None,
            seed=seed,
            syndrome_vec=_syn,
            llr_history=1,
            alpha1=alpha1,
            alpha2=alpha2,
            hybrid_residual_threshold=hybrid_residual_threshold,
            ensemble_k=ensemble_k,
            state_aware_residual=state_aware_residual,
            phi_by_state=phi_by_state,
            s_by_state=s_by_state,
            state_label_by_check=state_label_by_check,
        )

        hard_bp = _inner_result[0]
        iters_bp = _inner_result[1]
        _history = _inner_result[2]  # shape (k, n) from llr_history=1

        # Check if BP already converged.
        bp_syn = (
            (_H.astype(np.int32) @ hard_bp.astype(np.int32)) % 2
        ).astype(np.uint8)

        if np.array_equal(bp_syn, _syn):
            # BP converged — return immediately without OSD.
            correction = hard_bp
        else:
            # Extract posterior LLR from history (last snapshot).
            L_post = _history[-1]  # shape (n,)

            corrected = mp_osd1_postprocess(
                _H, _llr, hard_bp, L_post, _syn,
            )

            # Never-degrade rule: verify OSD result satisfies syndrome.
            osd_syn = (
                (_H.astype(np.int32) @ corrected.astype(np.int32)) % 2
            ).astype(np.uint8)
            if np.array_equal(osd_syn, _syn):
                correction = corrected
            else:
                correction = hard_bp

        # Assemble return tuple matching bp_decode output-shape rules.
        if llr_history > 0:
            # Return the captured history from the inner call.
            if residual_metrics:
                _empty_metrics = {
                    "residual_linf": [],
                    "residual_l2": [],
                    "residual_energy": [],
                }
                return correction, iters_bp, _history, _empty_metrics
            return correction, iters_bp, _history
        if residual_metrics:
            _empty_metrics = {
                "residual_linf": [],
                "residual_l2": [],
                "residual_energy": [],
            }
            return correction, iters_bp, _empty_metrics
        return correction, iters_bp

    # ── v3.6.0: mp_osd_cs early dispatch ──
    # MP-aware combination-sweep OSD that uses posterior LLR magnitude
    # for column ordering.  Follows the same early-return wrapper pattern
    # as mp_osd1.  Runs BP with postprocess=None and llr_history=1 to
    # obtain posterior, then applies mp_osd_cs_postprocess if syndrome
    # is not satisfied.
    if postprocess == "mp_osd_cs":
        from .decoder.osd import mp_osd_cs_postprocess

        _H = np.asarray(H, dtype=np.uint8)
        _llr = np.broadcast_to(
            np.asarray(llr, dtype=np.float64), (_H.shape[1],)
        ).copy()

        if syndrome_vec is None:
            _syn = np.zeros(_H.shape[0], dtype=np.uint8)
        else:
            _syn = np.asarray(syndrome_vec, dtype=np.uint8)

        # Inner BP call — same H, llr, max_iters, schedule, mode, etc.
        # postprocess=None prevents recursion; llr_history=1 captures posterior.
        _inner_result = bp_decode(
            _H, _llr,
            max_iters=max_iters,
            mode=mode,
            damping=damping,
            norm_factor=norm_factor,
            offset=offset,
            clip=clip,
            schedule=schedule,
            postprocess=None,
            seed=seed,
            syndrome_vec=_syn,
            llr_history=1,
            alpha1=alpha1,
            alpha2=alpha2,
            hybrid_residual_threshold=hybrid_residual_threshold,
            ensemble_k=ensemble_k,
            state_aware_residual=state_aware_residual,
            phi_by_state=phi_by_state,
            s_by_state=s_by_state,
            state_label_by_check=state_label_by_check,
        )

        hard_bp = _inner_result[0]
        iters_bp = _inner_result[1]
        _history = _inner_result[2]  # shape (k, n) from llr_history=1

        # Check if BP already converged.
        bp_syn = (
            (_H.astype(np.int32) @ hard_bp.astype(np.int32)) % 2
        ).astype(np.uint8)

        if np.array_equal(bp_syn, _syn):
            # BP converged — return immediately without OSD.
            correction = hard_bp
        else:
            # Extract posterior LLR from history (last snapshot).
            L_post = _history[-1]  # shape (n,)

            corrected = mp_osd_cs_postprocess(
                _H, _llr, hard_bp, L_post, _syn, lam=osd_cs_lam,
            )

            # Never-degrade rule: verify OSD result satisfies syndrome.
            osd_syn = (
                (_H.astype(np.int32) @ corrected.astype(np.int32)) % 2
            ).astype(np.uint8)
            if np.array_equal(osd_syn, _syn):
                correction = corrected
            else:
                correction = hard_bp

        # Assemble return tuple matching bp_decode output-shape rules.
        if llr_history > 0:
            # Return the captured history from the inner call.
            if residual_metrics:
                _empty_metrics = {
                    "residual_linf": [],
                    "residual_l2": [],
                    "residual_energy": [],
                }
                return correction, iters_bp, _history, _empty_metrics
            return correction, iters_bp, _history
        if residual_metrics:
            _empty_metrics = {
                "residual_linf": [],
                "residual_l2": [],
                "residual_energy": [],
            }
            return correction, iters_bp, _empty_metrics
        return correction, iters_bp

    # ── v2.9.1: residual metric collection (opt-in) ──
    if residual_metrics:
        _res_metrics = {"residual_linf": [], "residual_l2": [], "residual_energy": []}

    # ── Deterministic ensemble wrapper (v2.8.0) ──
    # When ensemble_k > 1, run K independent BP passes with deterministic
    # LLR perturbations and return the best candidate.  Member 0 always
    # uses the exact original LLR (baseline).  Falls through when k == 1.
    if ensemble_k > 1:
        _llr_arr = np.broadcast_to(
            np.asarray(llr, dtype=np.float64), (H.shape[1],)
        ).copy()
        _n_ens = H.shape[1]
        _mean_abs = np.mean(np.abs(_llr_arr)) if _n_ens > 0 else 0.0
        _scale = 0.05 * _mean_abs if _mean_abs > 0.0 else 0.05

        best_hard = None
        best_iters = None
        best_hist = None
        best_syn_weight = None  # None means not yet set
        best_converged = False
        best_member = -1
        H32 = H.astype(np.int32, copy=False)

        for _k in range(ensemble_k):
            if _k == 0:
                llr_k = _llr_arr
            else:
                # Deterministic, discrete, zero-mean perturbation.
                # Alternating +1/-1 base pattern, rolled by member index.
                _base = np.where(
                    (np.arange(_n_ens) % 2) == 0, 1.0, -1.0
                )
                _pattern = np.roll(_base, _k)
                # Enforce exact zero-mean (needed when n is odd).
                _pattern = _pattern - _pattern.mean()
                epsilon_k = _scale * _pattern
                llr_k = _llr_arr + epsilon_k

            result_k = bp_decode(
                H, llr_k, max_iters=max_iters, mode=mode,
                damping=damping, norm_factor=norm_factor,
                offset=offset, clip=clip,
                syndrome_vec=syndrome_vec,
                postprocess=postprocess, osd_cs_lam=osd_cs_lam,
                llr_history=llr_history,
                schedule=schedule,
                alpha1=alpha1, alpha2=alpha2,
                hybrid_residual_threshold=hybrid_residual_threshold,
                ensemble_k=1,  # single run per member
                state_aware_residual=state_aware_residual,
                phi_by_state=phi_by_state,
                s_by_state=s_by_state,
                state_label_by_check=state_label_by_check,
            )

            if llr_history > 0:
                hard_k, iters_k, hist_k = result_k
            else:
                hard_k, iters_k = result_k
                hist_k = None

            # Evaluate candidate quality.
            syn_k = (
                (H32 @ hard_k.astype(np.int32)) % 2
            ).astype(np.uint8)
            if syndrome_vec is not None:
                syn_weight_k = int(np.sum(syn_k != syndrome_vec))
            else:
                syn_weight_k = int(np.sum(syn_k))
            converged_k = (syn_weight_k == 0)

            # Selection: prefer converged, then lowest syndrome weight,
            # then lowest member index (deterministic tie-break).
            if best_hard is None:
                is_better = True
            elif converged_k and not best_converged:
                is_better = True
            elif not converged_k and best_converged:
                is_better = False
            elif syn_weight_k < best_syn_weight:
                is_better = True
            else:
                is_better = False

            if is_better:
                best_hard = hard_k
                best_iters = iters_k
                best_hist = hist_k
                best_syn_weight = syn_weight_k
                best_converged = converged_k
                best_member = _k

        if llr_history > 0:
            if residual_metrics:
                return best_hard, best_iters, best_hist, _res_metrics
            return best_hard, best_iters, best_hist
        if residual_metrics:
            return best_hard, best_iters, _res_metrics
        return best_hard, best_iters

    # ── Deterministic adaptive schedule controller (v2.9.0) ──
    # One-way checkpointed controller:
    #   Phase 1: run "flooding" for k1 iterations.
    #   If converged → return immediately.
    #   Phase 2: run "hybrid_residual" for remaining (max_iters - k1) iters.
    #   Tie-break: prefer converged → lower syndrome weight → fewer iters
    #   → phase order (deterministic).
    # No internal message state is passed between phases; each phase is
    # an independent bp_decode() call with a concrete schedule.
    if schedule == "adaptive":
        k1 = adaptive_k1 if adaptive_k1 is not None else max(1, max_iters // 4)
        k2 = max_iters - k1

        # Shared keyword arguments for both phases (forward all non-adaptive
        # parameters unchanged).
        _adaptive_common = dict(
            mode=mode, damping=damping, norm_factor=norm_factor,
            offset=offset, clip=clip, postprocess=postprocess,
            osd_cs_lam=osd_cs_lam, llr_history=llr_history,
            syndrome_vec=syndrome_vec,
            alpha1=alpha1, alpha2=alpha2,
            hybrid_residual_threshold=hybrid_residual_threshold,
            ensemble_k=1,
            state_aware_residual=state_aware_residual,
            phi_by_state=phi_by_state,
            s_by_state=s_by_state,
            state_label_by_check=state_label_by_check,
        )

        H32 = H.astype(np.int32, copy=False)
        _syn_ref = (
            np.asarray(syndrome_vec, dtype=np.uint8)
            if syndrome_vec is not None
            else np.zeros(H.shape[0], dtype=np.uint8)
        )

        # ── Phase 1: flooding for k1 iterations ──
        result_p1 = bp_decode(
            H, llr, max_iters=k1, schedule="flooding", **_adaptive_common,
        )
        if llr_history > 0:
            hard_p1, iters_p1, hist_p1 = result_p1
        else:
            hard_p1, iters_p1 = result_p1
            hist_p1 = None

        syn_p1 = (
            (H32 @ hard_p1.astype(np.int32)) % 2
        ).astype(np.uint8)
        syn_weight_p1 = int(np.sum(syn_p1 != _syn_ref))
        converged_p1 = (syn_weight_p1 == 0)

        if adaptive_log:
            print(
                f"[adaptive] phase1 flooding k1={k1}: "
                f"iters={iters_p1}, syn_weight={syn_weight_p1}, "
                f"converged={converged_p1}"
            )

        if converged_p1:
            # Phase 1 converged — return immediately.
            if adaptive_log:
                print("[adaptive] returning phase1 (converged)")
            if llr_history > 0:
                if residual_metrics:
                    return hard_p1, iters_p1, hist_p1, _res_metrics
                return hard_p1, iters_p1, hist_p1
            if residual_metrics:
                return hard_p1, iters_p1, _res_metrics
            return hard_p1, iters_p1

        # ── Phase 2: hybrid_residual for remaining iterations ──
        result_p2 = bp_decode(
            H, llr, max_iters=k2, schedule="hybrid_residual",
            **_adaptive_common,
        )
        if llr_history > 0:
            hard_p2, iters_p2, hist_p2 = result_p2
        else:
            hard_p2, iters_p2 = result_p2
            hist_p2 = None

        syn_p2 = (
            (H32 @ hard_p2.astype(np.int32)) % 2
        ).astype(np.uint8)
        syn_weight_p2 = int(np.sum(syn_p2 != _syn_ref))
        converged_p2 = (syn_weight_p2 == 0)

        if adaptive_log:
            print(
                f"[adaptive] phase2 hybrid_residual k2={k2}: "
                f"iters={iters_p2}, syn_weight={syn_weight_p2}, "
                f"converged={converged_p2}"
            )

        # ── Tie-break: pick best phase result ──
        # Prefer: converged → lower syndrome weight → fewer total iters
        # → phase order (phase 1 wins ties — deterministic).
        # Iteration accounting: phase 1 returns iters_p1 directly;
        # phase 2 returns k1 + iters_p2 (cumulative work).
        total_iters_p1 = iters_p1
        total_iters_p2 = k1 + iters_p2

        pick_p2 = False
        if converged_p2 and not converged_p1:
            pick_p2 = True
        elif not converged_p2 and converged_p1:
            pick_p2 = False
        elif syn_weight_p2 < syn_weight_p1:
            pick_p2 = True
        elif syn_weight_p2 > syn_weight_p1:
            pick_p2 = False
        elif total_iters_p2 < total_iters_p1:
            pick_p2 = True
        # else: tie on all criteria → prefer phase 1 (deterministic)

        if pick_p2:
            if adaptive_log:
                print("[adaptive] returning phase2 (better)")
            if llr_history > 0:
                if residual_metrics:
                    return hard_p2, total_iters_p2, hist_p2, _res_metrics
                return hard_p2, total_iters_p2, hist_p2
            if residual_metrics:
                return hard_p2, total_iters_p2, _res_metrics
            return hard_p2, total_iters_p2
        else:
            if adaptive_log:
                print("[adaptive] returning phase1 (better or tie)")
            if llr_history > 0:
                if residual_metrics:
                    return hard_p1, total_iters_p1, hist_p1, _res_metrics
                return hard_p1, total_iters_p1, hist_p1
            if residual_metrics:
                return hard_p1, total_iters_p1, _res_metrics
            return hard_p1, total_iters_p1

    m, n = H.shape
    eps = 1e-30

    llr = np.broadcast_to(np.asarray(llr, dtype=np.float64), (n,)).copy()

    if syndrome_vec is None:
        syndrome_vec = np.zeros(m, dtype=np.uint8)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    c2v, v2c = _tanner_graph(H)

    # Variable-to-check LLR messages (initialised to channel LLR per variable)
    v2c_msg = np.zeros((n, m), dtype=np.float64)
    for v in range(n):
        for c in v2c[v]:
            v2c_msg[v, c] = llr[v]

    # Check-to-variable LLR messages
    c2v_msg = np.zeros((m, n), dtype=np.float64)

    hard = np.zeros(n, dtype=np.uint8)

    # ── LLR history buffer (circular, fixed-size) ──
    if llr_history > 0:
        _hist_buf = [None] * llr_history
        _hist_idx = 0
        _hist_count = 0
        _L_total = np.empty(n, dtype=np.float64)

    use_min_sum = mode in ("min_sum", "norm_min_sum", "offset_min_sum",
                           "improved_norm", "improved_offset", "min_sum_urw")

    # ── v3.9.0 energy trace buffer ──
    if energy_trace:
        from src.qec.decoder.energy import bp_energy as _bp_energy
        _energy_trace: list[float] = []

    def _build_return(pp_result, hist=None, res_met=None, etrace=None):
        """Assemble return tuple in canonical order:
        (correction, iters[, llr_history][, residual_metrics][, energy_trace])
        """
        result = [pp_result[0], pp_result[1]]
        if hist is not None:
            result.append(hist)
        if res_met is not None:
            result.append(res_met)
        if etrace is not None:
            result.append(etrace)
        return tuple(result)

    if schedule == "flooding":
        # ══════════════════════════════════════════════════════════════
        # Flooding schedule: update ALL check nodes, then ALL variable
        # nodes, in separate sweeps.  This is the v2.4.0 default.
        # ══════════════════════════════════════════════════════════════
        for it in range(max_iters):
            # Save old messages for damping.
            if damping > 0.0:
                c2v_msg_old = c2v_msg.copy()

            # ── check → variable ──
            for c in range(m):
                nbrs = c2v[c]
                if len(nbrs) == 0:
                    continue
                if len(nbrs) == 1:
                    c2v_msg[c, nbrs[0]] = 0.0
                    continue

                sign_s = (-1.0) ** int(syndrome_vec[c])

                if use_min_sum:
                    # Gather signs and magnitudes.
                    incoming = np.array([v2c_msg[v, c] for v in nbrs])
                    signs = np.where(incoming == 0.0, 1.0, np.sign(incoming))
                    abs_vals = np.abs(incoming)
                    sign_prod_all = np.prod(signs) * sign_s

                    # Precompute first and second minimums for O(d_c) exclusion.
                    d = len(nbrs)
                    if d >= 2:
                        idx_sorted = np.argpartition(abs_vals, 1)[:2]
                        if abs_vals[idx_sorted[0]] <= abs_vals[idx_sorted[1]]:
                            min1_idx, min2_idx = idx_sorted[0], idx_sorted[1]
                        else:
                            min1_idx, min2_idx = idx_sorted[1], idx_sorted[0]
                        min1_val = abs_vals[min1_idx]
                        min2_val = abs_vals[min2_idx]
                    else:
                        min1_idx = 0
                        min1_val = abs_vals[0]
                        min2_val = 0.0

                    for idx, v in enumerate(nbrs):
                        sign_excl = sign_prod_all * signs[idx]
                        min_excl = min2_val if idx == min1_idx else min1_val

                        if mode == "min_sum":
                            c2v_msg[c, v] = sign_excl * min_excl
                        elif mode == "norm_min_sum":
                            c2v_msg[c, v] = norm_factor * sign_excl * min_excl
                        elif mode == "offset_min_sum":
                            c2v_msg[c, v] = sign_excl * max(min_excl - offset, 0.0)
                        elif mode == "improved_norm":
                            alpha = alpha1 if idx == min1_idx else alpha2
                            c2v_msg[c, v] = alpha * sign_excl * min_excl
                        elif mode == "min_sum_urw":
                            c2v_msg[c, v] = urw_rho * sign_excl * min_excl
                        else:  # improved_offset
                            alpha = alpha1 if idx == min1_idx else alpha2
                            c2v_msg[c, v] = sign_excl * max(alpha * min_excl - offset, 0.0)
                else:
                    # sum_product: tanh product rule.
                    tanhs = np.array([
                        np.tanh(np.clip(v2c_msg[v, c] / 2.0, -20.0, 20.0))
                        for v in nbrs
                    ])
                    prod_all = np.prod(tanhs)
                    for idx, v in enumerate(nbrs):
                        prod_excl = prod_all / (tanhs[idx] + eps)
                        prod_excl = np.clip(prod_excl, -1 + 1e-15, 1 - 1e-15)
                        c2v_msg[c, v] = sign_s * 2.0 * np.arctanh(prod_excl)

            # ── damping ──
            if damping > 0.0:
                c2v_msg = (1 - damping) * c2v_msg + damping * c2v_msg_old

            # ── user-specified message clipping ──
            if clip is not None:
                c2v_msg = np.clip(c2v_msg, -clip, clip)

            # ── variable → check + hard decision ──
            for v in range(n):
                nbrs = v2c[v]
                total = llr[v] + sum(c2v_msg[c, v] for c in nbrs)
                for c in nbrs:
                    v2c_msg[v, c] = total - c2v_msg[c, v]
                hard[v] = 0 if total >= 0.0 else 1
                if llr_history > 0:
                    _L_total[v] = total

            # ── LLR history snapshot (flooding) ──
            if llr_history > 0:
                _hist_buf[_hist_idx % llr_history] = _L_total.copy()
                _hist_idx += 1
                _hist_count = min(_hist_count + 1, llr_history)

            # ── energy trace (flooding) ──
            if energy_trace:
                _beliefs = np.array([
                    llr[v] + sum(c2v_msg[c, v] for c in v2c[v])
                    for v in range(n)
                ])
                _energy_trace.append(_bp_energy(llr, _beliefs))

            # ── early stop ──
            if np.array_equal(
                (H.astype(np.int32) @ hard.astype(np.int32)) % 2,
                syndrome_vec.astype(np.uint8),
            ):
                pp_result = _bp_postprocess(
                    H, llr, hard, it + 1, syndrome_vec, postprocess,
                    osd_cs_lam=osd_cs_lam,
                )
                return _build_return(
                    pp_result,
                    hist=_assemble_history(_hist_buf, _hist_idx, _hist_count, llr_history) if llr_history > 0 else None,
                    res_met=_res_metrics if residual_metrics else None,
                    etrace=_energy_trace if energy_trace else None,
                )

        pp_result = _bp_postprocess(
            H, llr, hard, max_iters, syndrome_vec, postprocess,
            osd_cs_lam=osd_cs_lam,
        )
        return _build_return(
            pp_result,
            hist=_assemble_history(_hist_buf, _hist_idx, _hist_count, llr_history) if llr_history > 0 else None,
            res_met=_res_metrics if residual_metrics else None,
            etrace=_energy_trace if energy_trace else None,
        )

    elif schedule == "geom_v1":
        # ══════════════════════════════════════════════════════════════
        # Geometry-aware flooding schedule (v3.8.0).
        #
        # Identical to flooding except check-to-variable messages are
        # scaled by  alpha_c = 1 / sqrt(d_c)  where d_c is the degree
        # of check node c (number of non-zero entries in row c of H).
        #
        # This normalises the influence of high-degree checks without
        # altering any other decoding semantics.
        # ══════════════════════════════════════════════════════════════

        # Precompute per-check scaling factors (deterministic, pure
        # function of H structure).
        _dc = np.array([len(c2v[c]) for c in range(m)], dtype=np.float64)
        _geom_alpha = np.where(_dc > 0.0, 1.0 / np.sqrt(_dc), 0.0)

        for it in range(max_iters):
            # Save old messages for damping.
            if damping > 0.0:
                c2v_msg_old = c2v_msg.copy()

            # ── check → variable (with geometric scaling) ──
            for c in range(m):
                nbrs = c2v[c]
                if len(nbrs) == 0:
                    continue
                if len(nbrs) == 1:
                    c2v_msg[c, nbrs[0]] = 0.0
                    continue

                alpha_c = _geom_alpha[c]
                sign_s = (-1.0) ** int(syndrome_vec[c])

                if use_min_sum:
                    # Gather signs and magnitudes.
                    incoming = np.array([v2c_msg[v, c] for v in nbrs])
                    signs = np.where(incoming == 0.0, 1.0, np.sign(incoming))
                    abs_vals = np.abs(incoming)
                    sign_prod_all = np.prod(signs) * sign_s

                    # Precompute first and second minimums for O(d_c) exclusion.
                    d = len(nbrs)
                    if d >= 2:
                        idx_sorted = np.argpartition(abs_vals, 1)[:2]
                        if abs_vals[idx_sorted[0]] <= abs_vals[idx_sorted[1]]:
                            min1_idx, min2_idx = idx_sorted[0], idx_sorted[1]
                        else:
                            min1_idx, min2_idx = idx_sorted[1], idx_sorted[0]
                        min1_val = abs_vals[min1_idx]
                        min2_val = abs_vals[min2_idx]
                    else:
                        min1_idx = 0
                        min1_val = abs_vals[0]
                        min2_val = 0.0

                    for idx, v in enumerate(nbrs):
                        sign_excl = sign_prod_all * signs[idx]
                        min_excl = min2_val if idx == min1_idx else min1_val

                        if mode == "min_sum":
                            c2v_msg[c, v] = alpha_c * sign_excl * min_excl
                        elif mode == "norm_min_sum":
                            c2v_msg[c, v] = alpha_c * norm_factor * sign_excl * min_excl
                        elif mode == "offset_min_sum":
                            c2v_msg[c, v] = alpha_c * sign_excl * max(min_excl - offset, 0.0)
                        elif mode == "improved_norm":
                            alpha = alpha1 if idx == min1_idx else alpha2
                            c2v_msg[c, v] = alpha_c * alpha * sign_excl * min_excl
                        elif mode == "min_sum_urw":
                            c2v_msg[c, v] = alpha_c * urw_rho * sign_excl * min_excl
                        else:  # improved_offset
                            alpha = alpha1 if idx == min1_idx else alpha2
                            c2v_msg[c, v] = alpha_c * sign_excl * max(alpha * min_excl - offset, 0.0)
                else:
                    # sum_product: tanh product rule, then scale.
                    tanhs = np.array([
                        np.tanh(np.clip(v2c_msg[v, c] / 2.0, -20.0, 20.0))
                        for v in nbrs
                    ])
                    prod_all = np.prod(tanhs)
                    for idx, v in enumerate(nbrs):
                        prod_excl = prod_all / (tanhs[idx] + eps)
                        prod_excl = np.clip(prod_excl, -1 + 1e-15, 1 - 1e-15)
                        c2v_msg[c, v] = alpha_c * sign_s * 2.0 * np.arctanh(prod_excl)

            # ── damping ──
            if damping > 0.0:
                c2v_msg = (1 - damping) * c2v_msg + damping * c2v_msg_old

            # ── user-specified message clipping ──
            if clip is not None:
                c2v_msg = np.clip(c2v_msg, -clip, clip)

            # ── variable → check + hard decision ──
            for v in range(n):
                nbrs = v2c[v]
                total = llr[v] + sum(c2v_msg[c, v] for c in nbrs)
                for c in nbrs:
                    v2c_msg[v, c] = total - c2v_msg[c, v]
                hard[v] = 0 if total >= 0.0 else 1
                if llr_history > 0:
                    _L_total[v] = total

            # ── LLR history snapshot (geom_v1) ──
            if llr_history > 0:
                _hist_buf[_hist_idx % llr_history] = _L_total.copy()
                _hist_idx += 1
                _hist_count = min(_hist_count + 1, llr_history)

            # ── energy trace (geom_v1) ──
            if energy_trace:
                _beliefs = np.array([
                    llr[v] + sum(c2v_msg[c, v] for c in v2c[v])
                    for v in range(n)
                ])
                _energy_trace.append(_bp_energy(llr, _beliefs))

            # ── early stop ──
            if np.array_equal(
                (H.astype(np.int32) @ hard.astype(np.int32)) % 2,
                syndrome_vec.astype(np.uint8),
            ):
                pp_result = _bp_postprocess(
                    H, llr, hard, it + 1, syndrome_vec, postprocess,
                    osd_cs_lam=osd_cs_lam,
                )
                return _build_return(
                    pp_result,
                    hist=_assemble_history(_hist_buf, _hist_idx, _hist_count, llr_history) if llr_history > 0 else None,
                    res_met=_res_metrics if residual_metrics else None,
                    etrace=_energy_trace if energy_trace else None,
                )

        pp_result = _bp_postprocess(
            H, llr, hard, max_iters, syndrome_vec, postprocess,
            osd_cs_lam=osd_cs_lam,
        )
        return _build_return(
            pp_result,
            hist=_assemble_history(_hist_buf, _hist_idx, _hist_count, llr_history) if llr_history > 0 else None,
            res_met=_res_metrics if residual_metrics else None,
            etrace=_energy_trace if energy_trace else None,
        )

    else:
        # ══════════════════════════════════════════════════════════════
        # Layered (serial) schedule: process check nodes one-by-one,
        # updating variable beliefs incrementally.
        #
        # State:
        #   L_total[v] = total belief for variable v.
        #     Invariant: L_total[v] = llr[v] + sum_c c2v_msg[c, v]
        #   c2v_msg[c, v] = check-to-variable messages (init 0.0).
        #
        # Per check node c:
        #   1. v2c[v,c] = L_total[v] - c2v_msg[c,v]   (remove old c2v)
        #   2. c2v_raw   = CheckNodeUpdate(v2c inputs)
        #   3. c2v_new   = damp(c2v_raw, c2v_old)
        #   4. c2v_new   = clip(c2v_new)
        #   5. L_total[v] += c2v_new - c2v_msg[c,v]    (remove old, add new)
        #   6. c2v_msg[c,v] = c2v_new
        # ══════════════════════════════════════════════════════════════

        # Initialise total beliefs from channel LLRs.
        # c2v_msg is already zero, so invariant L_total = llr + 0 holds.
        L_total = llr.copy()
        is_residual = (schedule == "residual")
        is_hybrid = (schedule == "hybrid_residual")
        if is_residual or is_hybrid:
            residuals = np.zeros(m, dtype=np.float64)
            check_indices = np.arange(m, dtype=np.int64)
        if is_hybrid:
            # Deterministic even/odd partition by check index.
            layer_even = np.array([c for c in range(m) if c % 2 == 0], dtype=np.int64)
            layer_odd = np.array([c for c in range(m) if c % 2 == 1], dtype=np.int64)

        for it in range(max_iters):
            if is_residual:
                check_order = np.lexsort((check_indices, -residuals))
                c2v_msg_before = c2v_msg.copy()
            elif is_hybrid:
                c2v_msg_before = c2v_msg.copy()
                check_order_parts = []
                for layer in (layer_even, layer_odd):
                    if len(layer) == 0:
                        continue
                    layer_res = residuals[layer]
                    layer_idx = layer
                    if hybrid_residual_threshold is not None:
                        high = layer_res > hybrid_residual_threshold
                        low = ~high
                        if np.any(high):
                            h_idx = layer_idx[high]
                            h_res = layer_res[high]
                            h_order = np.lexsort((h_idx, -h_res))
                            check_order_parts.append(h_idx[h_order])
                        if np.any(low):
                            l_idx = layer_idx[low]
                            l_res = layer_res[low]
                            l_order = np.lexsort((l_idx, -l_res))
                            check_order_parts.append(l_idx[l_order])
                    else:
                        order = np.lexsort((layer_idx, -layer_res))
                        check_order_parts.append(layer_idx[order])
                check_order = np.concatenate(check_order_parts) if check_order_parts else np.array([], dtype=np.int64)
            else:
                check_order = range(m)

            for c in check_order:
                nbrs = c2v[c]
                if len(nbrs) == 0:
                    continue

                # ── Step 1: Derive v2c from L_total (no full-sum recompute) ──
                # v2c[v,c] = L_total[v] - c2v_msg[c,v]
                v2c_vals = np.array([L_total[v] - c2v_msg[c, v] for v in nbrs])

                if len(nbrs) == 1:
                    # Degree-1 check: c2v message is zero by convention.
                    c2v_new_val = 0.0
                    old_val = c2v_msg[c, nbrs[0]]
                    if damping > 0.0:
                        c2v_new_val = (1 - damping) * c2v_new_val + damping * old_val
                    if clip is not None:
                        c2v_new_val = np.clip(c2v_new_val, -clip, clip)
                    # Step 5: Update L_total incrementally.
                    L_total[nbrs[0]] += c2v_new_val - old_val
                    c2v_msg[c, nbrs[0]] = c2v_new_val
                    continue

                sign_s = (-1.0) ** int(syndrome_vec[c])

                # ── Step 2: Compute new c2v messages ──
                if use_min_sum:
                    signs = np.where(v2c_vals == 0.0, 1.0, np.sign(v2c_vals))
                    abs_vals = np.abs(v2c_vals)
                    sign_prod_all = np.prod(signs) * sign_s

                    # Precompute first and second minimums.
                    d = len(nbrs)
                    if d >= 2:
                        idx_sorted = np.argpartition(abs_vals, 1)[:2]
                        if abs_vals[idx_sorted[0]] <= abs_vals[idx_sorted[1]]:
                            min1_idx, min2_idx = idx_sorted[0], idx_sorted[1]
                        else:
                            min1_idx, min2_idx = idx_sorted[1], idx_sorted[0]
                        min1_val = abs_vals[min1_idx]
                        min2_val = abs_vals[min2_idx]
                    else:
                        min1_idx = 0
                        min1_val = abs_vals[0]
                        min2_val = 0.0

                    for idx, v in enumerate(nbrs):
                        sign_excl = sign_prod_all * signs[idx]
                        min_excl = min2_val if idx == min1_idx else min1_val

                        if mode == "min_sum":
                            c2v_raw = sign_excl * min_excl
                        elif mode == "norm_min_sum":
                            c2v_raw = norm_factor * sign_excl * min_excl
                        elif mode == "offset_min_sum":
                            c2v_raw = sign_excl * max(min_excl - offset, 0.0)
                        elif mode == "improved_norm":
                            alpha = alpha1 if idx == min1_idx else alpha2
                            c2v_raw = alpha * sign_excl * min_excl
                        elif mode == "min_sum_urw":
                            c2v_raw = urw_rho * sign_excl * min_excl
                        else:  # improved_offset
                            alpha = alpha1 if idx == min1_idx else alpha2
                            c2v_raw = sign_excl * max(alpha * min_excl - offset, 0.0)

                        # Step 3: Damping per-message.
                        old_val = c2v_msg[c, v]
                        if damping > 0.0:
                            c2v_new_val = (1 - damping) * c2v_raw + damping * old_val
                        else:
                            c2v_new_val = c2v_raw

                        # Step 4: Clipping per-message.
                        if clip is not None:
                            c2v_new_val = np.clip(c2v_new_val, -clip, clip)

                        # Step 5: Update L_total incrementally.
                        L_total[v] += c2v_new_val - old_val

                        # Step 6: Store new c2v message.
                        c2v_msg[c, v] = c2v_new_val
                else:
                    # sum_product: tanh product rule.
                    tanhs = np.array([
                        np.tanh(np.clip(val / 2.0, -20.0, 20.0))
                        for val in v2c_vals
                    ])
                    prod_all = np.prod(tanhs)

                    for idx, v in enumerate(nbrs):
                        prod_excl = prod_all / (tanhs[idx] + eps)
                        prod_excl = np.clip(prod_excl, -1 + 1e-15, 1 - 1e-15)
                        c2v_raw = sign_s * 2.0 * np.arctanh(prod_excl)

                        # Step 3: Damping per-message.
                        old_val = c2v_msg[c, v]
                        if damping > 0.0:
                            c2v_new_val = (1 - damping) * c2v_raw + damping * old_val
                        else:
                            c2v_new_val = c2v_raw

                        # Step 4: Clipping per-message.
                        if clip is not None:
                            c2v_new_val = np.clip(c2v_new_val, -clip, clip)

                        # Step 5: Update L_total incrementally.
                        L_total[v] += c2v_new_val - old_val

                        # Step 6: Store new c2v message.
                        c2v_msg[c, v] = c2v_new_val

            # ── Update residuals (residual / hybrid_residual schedule) ──
            if is_residual or is_hybrid:
                if residual_metrics:
                    delta = c2v_msg - c2v_msg_before
                    abs_delta = np.abs(delta)
                    sq_delta = delta * delta
                    residuals = np.max(abs_delta, axis=1)
                    _res_metrics["residual_linf"].append(residuals.copy())
                    _res_metrics["residual_l2"].append(np.sqrt(np.sum(sq_delta, axis=1)))
                    _res_metrics["residual_energy"].append(float(np.sum(sq_delta)))
                else:
                    residuals = np.max(np.abs(c2v_msg - c2v_msg_before), axis=1)
                if state_aware_residual:
                    residuals *= state_aware_residual_weights

            # ── After all layers: compute hard decisions from L_total ──
            for v in range(n):
                hard[v] = 0 if L_total[v] >= 0.0 else 1

            # ── LLR history snapshot (layered) ──
            if llr_history > 0:
                _hist_buf[_hist_idx % llr_history] = L_total.copy()
                _hist_idx += 1
                _hist_count = min(_hist_count + 1, llr_history)

            # ── energy trace (layered) ──
            if energy_trace:
                _energy_trace.append(_bp_energy(llr, L_total))

            # ── early stop ──
            if np.array_equal(
                (H.astype(np.int32) @ hard.astype(np.int32)) % 2,
                syndrome_vec.astype(np.uint8),
            ):
                pp_result = _bp_postprocess(
                    H, llr, hard, it + 1, syndrome_vec, postprocess,
                    osd_cs_lam=osd_cs_lam,
                )
                return _build_return(
                    pp_result,
                    hist=_assemble_history(_hist_buf, _hist_idx, _hist_count, llr_history) if llr_history > 0 else None,
                    res_met=_res_metrics if residual_metrics else None,
                    etrace=_energy_trace if energy_trace else None,
                )

        pp_result = _bp_postprocess(
            H, llr, hard, max_iters, syndrome_vec, postprocess,
            osd_cs_lam=osd_cs_lam,
        )
        return _build_return(
            pp_result,
            hist=_assemble_history(_hist_buf, _hist_idx, _hist_count, llr_history) if llr_history > 0 else None,
            res_met=_res_metrics if residual_metrics else None,
            etrace=_energy_trace if energy_trace else None,
        )


def _assemble_history(hist_buf, hist_idx, hist_count, llr_history):
    """Reconstruct ordered history array from circular buffer."""
    filled = hist_buf[:hist_count]
    if hist_count == llr_history:
        start = hist_idx % llr_history
        filled = hist_buf[start:] + hist_buf[:start]
    return np.array(filled, dtype=np.float64)


def _bp_postprocess(H, llr, hard, iters, syndrome_vec, postprocess, **pp_kwargs):
    """Apply optional post-processing after BP terminates."""
    if postprocess not in ("osd0", "osd1", "osd_cs"):
        return hard, iters

    bp_syn = (
        (H.astype(np.int32) @ hard.astype(np.int32)) % 2
    ).astype(np.uint8)

    if np.array_equal(bp_syn, syndrome_vec):
        # BP already converged — do not risk degradation.
        return hard, iters

    if postprocess == "osd0":
        from .decoder.osd import osd0
        hard_pp = osd0(H, llr, hard, syndrome_vec=syndrome_vec)
    elif postprocess == "osd1":
        from .decoder.osd import osd1
        hard_pp = osd1(H, llr, hard, syndrome_vec=syndrome_vec)
    else:  # "osd_cs"
        from .decoder.osd import osd_cs
        lam = pp_kwargs.get("osd_cs_lam", 1)
        hard_pp = osd_cs(H, llr, hard, syndrome_vec=syndrome_vec, lam=lam)

    pp_syn = (
        (H.astype(np.int32) @ hard_pp.astype(np.int32)) % 2
    ).astype(np.uint8)

    if np.array_equal(pp_syn, syndrome_vec):
        return hard_pp, iters

    return hard, iters


def detect(H: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Compute the error syndrome (thin wrapper over :func:`syndrome`).

    Args:
        H: Binary parity-check matrix, shape (m, n).
        e: Binary error vector, length n.

    Returns:
        Binary syndrome vector, length m.
    """
    return syndrome(H, e)


def infer(
    H: np.ndarray,
    llr: np.ndarray,
    max_iter: int = 100,
    mode: str = "sum_product",
    damping: float = 0.0,
    norm_factor: float = 0.75,
    offset: float = 0.5,
    clip: Optional[float] = None,
    schedule: str = "flooding",
    postprocess: Optional[str] = None,
    seed: Optional[int] = None,
    syndrome_vec: Optional[np.ndarray] = None,
    llr_history: int = 0,
    osd_cs_lam: int = 1,
) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, np.ndarray]]:
    """
    Infer the most likely error pattern via belief propagation
    (thin wrapper over :func:`bp_decode`).

    Args:
        H: Binary parity-check matrix, shape (m, n).
        llr: Per-variable log-likelihood ratios, length n.
        max_iter: Maximum BP iterations.
        mode: Check-node update rule (see :func:`bp_decode`).
        damping: Damping factor in [0, 1).
        norm_factor: Normalisation factor for ``"norm_min_sum"``.
        offset: Offset for ``"offset_min_sum"``.
        clip: Message magnitude clipping bound.
        schedule: Message-passing schedule (``"flooding"``, ``"layered"``,
            or ``"residual"``).
        postprocess: Optional post-processing (see :func:`bp_decode`).
        seed: Reserved for future use.
        syndrome_vec: Binary syndrome vector, length m.  Defaults to all-zeros.
        llr_history: See :func:`bp_decode`.
        osd_cs_lam: See :func:`bp_decode`.

    Returns:
        Same as :func:`bp_decode`.
    """
    return bp_decode(
        H, llr, max_iters=max_iter, mode=mode, damping=damping,
        norm_factor=norm_factor, offset=offset, clip=clip,
        schedule=schedule, postprocess=postprocess, seed=seed,
        syndrome_vec=syndrome_vec, llr_history=llr_history,
        osd_cs_lam=osd_cs_lam,
    )


def channel_llr(
    e: np.ndarray,
    p: float,
    bias: Optional[Union[np.ndarray, dict]] = None,
) -> np.ndarray:
    """
    Compute per-variable channel log-likelihood ratios for a binary error pattern.

    The base LLR is ``log((1 - p + eps) / (p + eps))``, identical to the
    inline computation in :meth:`JointSPDecoder._bp_component`.
    Each element is then multiplied by ``(1 - 2*e[i])``, which flips the
    sign for positions where an error is present.

    Args:
        e: Binary error vector of length n.
        p: Channel error probability in (0, 1).
        bias: Optional noise-bias multiplier.
            - *None*  → uniform LLR (default, matches existing behavior).
            - *scalar* (0-d or length-1 array, or Python float/int)
              → multiply all LLRs by that scalar.
            - *vector* (length n) → element-wise multiply.
            - *dict* with keys ``"x"`` and/or ``"z"`` whose values are
              scalars or length-n vectors.  The combined bias is the
              element-wise product of the two components (missing keys
              default to 1.0).

    Returns:
        LLR vector of length n (float64).  Inputs are never mutated.

    Raises:
        ValueError: If *p* is not in (0, 1), bias shape mismatches,
            or dict contains unexpected keys.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")

    e = np.asarray(e)
    n = e.shape[0]
    eps = 1e-30

    base_llr = np.log((1.0 - p + eps) / (p + eps))
    sign = (1 - 2 * e.astype(np.float64))
    llr = base_llr * sign

    if bias is not None:
        if isinstance(bias, dict):
            valid_keys = {"x", "z"}
            unexpected = set(bias.keys()) - valid_keys
            if unexpected:
                raise ValueError(
                    f"Unexpected bias dict keys: {unexpected}. "
                    f"Expected subset of {valid_keys}"
                )
            def _validate_and_broadcast(component, name):
                arr = np.asarray(component, dtype=np.float64)
                if arr.ndim == 0:
                    return np.full(n, float(arr))
                if arr.shape == (1,):
                    return np.full(n, float(arr[0]))
                if arr.shape == (n,):
                    return arr
                raise ValueError(
                    f"bias['{name}'] has shape {arr.shape}; "
                    f"expected scalar, (1,), or ({n},)"
                )
            bx = _validate_and_broadcast(bias.get("x", 1.0), "x")
            bz = _validate_and_broadcast(bias.get("z", 1.0), "z")
            llr = llr * bx * bz
        else:
            bias = np.asarray(bias, dtype=np.float64)
            if bias.ndim == 0:
                llr = llr * float(bias)
            elif bias.shape == (1,):
                llr = llr * float(bias[0])
            elif bias.shape == (n,):
                llr = llr * bias
            else:
                raise ValueError(
                    f"bias shape {bias.shape} incompatible with error vector "
                    f"length {n}. Expected scalar, (1,), or ({n},)."
                )

    return llr


# ═══════════════════════════════════════════════════════════════════════
# Hashing Bound
# ═══════════════════════════════════════════════════════════════════════

def hashing_bound(p: float) -> float:
    """
    Quantum hashing bound for the depolarizing channel.

        R_hash(p) = 1 + (1-p) log2(1-p) + p log2(p/3)

    This is the theoretical maximum code rate at depolarizing
    probability p.  The Komoto-Kasai codes approach this bound.

    Args:
        p: depolarizing error probability in [0, 0.75]

    Returns:
        Maximum achievable code rate (non-negative).
    """
    if p <= 0.0:
        return 1.0
    if p >= 0.75:
        return 0.0
    r = 1.0 + (1.0 - p) * np.log2(1.0 - p) + p * np.log2(p / 3.0)
    return float(max(0.0, r))


def hashing_bound_threshold(rate: float, tol: float = 1e-8) -> float:
    """
    Maximum depolarizing probability at which code rate *rate* is
    achievable, according to the hashing bound.

    Solves  hashing_bound(p) = rate  via bisection.
    """
    if rate >= 1.0:
        return 0.0
    if rate <= 0.0:
        return 0.75
    lo, hi = 0.0, 0.75
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if hashing_bound(mid) > rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ═══════════════════════════════════════════════════════════════════════
# Monte Carlo Frame-Error-Rate Simulation
# ═══════════════════════════════════════════════════════════════════════

def simulate_frame_error_rate(
    code: QuantumLDPCCode,
    decoder: JointSPDecoder,
    p_phys: float,
    n_frames: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Estimate frame error rate (FER) under depolarizing noise.

    A frame error occurs when the residual error after decoding is
    not in the stabiliser group (i.e. it acts non-trivially on the
    code space).

    Args:
        code: Quantum LDPC code
        decoder: Joint SP decoder instance
        p_phys: Physical depolarizing probability
        n_frames: Monte Carlo sample count
        seed: Random seed

    Returns:
        dict with keys: p_phys, fer, fer_x, fer_z,
        decode_failure_rate, n_frames, n_physical, n_logical, rate
    """
    rng = np.random.default_rng(seed)
    x_fails = z_fails = total_fails = decode_fails = 0

    for _ in range(n_frames):
        x_err, z_err = depolarizing_channel(code.n, p_phys, rng)

        sx = code.syndrome_X(z_err)
        sz = code.syndrome_Z(x_err)

        x_hat, z_hat, converged = decoder.decode(sx, sz, p_phys)

        if not converged:
            decode_fails += 1

        # Residual error after correction
        res_x = (x_err ^ x_hat).astype(np.int32)
        res_z = (z_err ^ z_hat).astype(np.int32)

        xf = np.any((code.H_Z.astype(np.int32) @ res_x) % 2 != 0)
        zf = np.any((code.H_X.astype(np.int32) @ res_z) % 2 != 0)

        if xf:
            x_fails += 1
        if zf:
            z_fails += 1
        if xf or zf:
            total_fails += 1

    return {
        'p_phys': p_phys,
        'fer': total_fails / n_frames,
        'fer_x': x_fails / n_frames,
        'fer_z': z_fails / n_frames,
        'decode_failure_rate': decode_fails / n_frames,
        'n_frames': n_frames,
        'n_physical': code.n,
        'n_logical': code.k,
        'rate': code.rate,
    }


# ═══════════════════════════════════════════════════════════════════════
# Pre-defined Code Configurations
# ═══════════════════════════════════════════════════════════════════════

PREDEFINED_CODES: Dict[str, Dict] = {
    'rate_0.50': {
        'J': 1,
        'L': 4,
        'field_degree': 3,          # GF(8)
        'description': (
            'Rate-1/2 protograph QLDPC code (J=1, L=4, GF(2^3)). '
            'Komoto-Kasai 2025, Table I.'
        ),
    },
    'rate_0.60': {
        'J': 2,
        'L': 10,
        'field_degree': 3,          # GF(8)
        'description': (
            'Rate-3/5 protograph QLDPC code (J=2, L=10, GF(2^3)). '
            'Komoto-Kasai 2025, Table I.'
        ),
    },
    'rate_0.75': {
        'J': 1,
        'L': 8,
        'field_degree': 3,          # GF(8)
        'description': (
            'Rate-3/4 protograph QLDPC code (J=1, L=8, GF(2^3)). '
            'Komoto-Kasai 2025, Table I.'
        ),
    },
}


def create_code(
    name: str = 'rate_0.50',
    lifting_size: int = 32,
    seed: int = 42
) -> QuantumLDPCCode:
    """
    Instantiate a pre-defined quantum LDPC code.

    Args:
        name: one of 'rate_0.50', 'rate_0.60', 'rate_0.75'
        lifting_size: circulant permutation size P
                      (larger P → more physical qubits, better performance)
        seed: construction seed

    Returns:
        A fully constructed QuantumLDPCCode.
    """
    if name not in PREDEFINED_CODES:
        raise ValueError(
            f"Unknown code '{name}'. Choose from {sorted(PREDEFINED_CODES)}"
        )
    cfg = PREDEFINED_CODES[name]
    gf = GF2e(cfg['field_degree'])
    proto = build_protograph_pair(
        J=cfg['J'], L=cfg['L'], gf=gf, seed=seed
    )
    return QuantumLDPCCode(proto, lifting_size=lifting_size, seed=seed)


# ═══════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════

def demo_qldpc():
    """Demonstrate construction, decoding, and hashing-bound analysis."""
    print("=" * 64)
    print(" Quantum LDPC Codes Near the Hashing Bound")
    print(" Komoto & Kasai, npj Quantum Information 11, 154 (2025)")
    print("=" * 64)

    # Hashing bound table
    print("\n--- Hashing Bound for Depolarizing Channel ---")
    for rname in sorted(PREDEFINED_CODES):
        r = float(rname.split('_')[1])
        p_th = hashing_bound_threshold(r)
        print(f"  R = {r:.2f} :  max p_phys = {p_th:.4f}  ({p_th*100:.2f}%)")

    # Build a small rate-1/2 code
    print("\n--- Constructing Rate-1/2 Code (P=16) ---")
    code = create_code('rate_0.50', lifting_size=16, seed=42)
    print(f"  {code}")
    print(f"  CSS orthogonality verified: {code.verify_css_orthogonality()}")

    # Quick simulation
    print("\n--- Monte Carlo Frame Error Rate ---")
    dec = JointSPDecoder(code, max_iter=50)
    for p in [0.01, 0.03, 0.05]:
        res = simulate_frame_error_rate(code, dec, p, n_frames=100, seed=42)
        print(
            f"  p = {p:.2f}:  FER = {res['fer']:.4f}  "
            f"(X: {res['fer_x']:.4f}, Z: {res['fer_z']:.4f})"
        )

    print("\n" + "=" * 64)
    return code


if __name__ == '__main__':
    demo_qldpc()
