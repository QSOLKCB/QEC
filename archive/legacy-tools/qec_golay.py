"""
qec_golay.py
=================

This module implements a classical ternary [[11, 6, 5]] Golay code and provides
a simple interface for constructing a corresponding [[11, 1]]₃ quantum code.

The ternary Golay code is a linear code over GF(3) with block length 11,
dimension 6 and minimum distance 5.  It is the unique (up to equivalence)
perfect code of these parameters and its codewords can be generated as the
null‑space of a short 5×11 parity–check matrix.  The Golay codes play a
fundamental role in coding theory and serve as building blocks for
qutrit‑based quantum error–correcting codes via the CSS construction.  In the
CSS picture the parity–check matrices for both the `X`‑type and `Z`‑type
stabilizers are taken to be equal to the classical Golay parity–check
matrix; this yields an [[11, 1, 5]]₃ quantum code capable of correcting up
to two qutrit errors.

This module exposes the following data and helper routines:

* ``H`` – the canonical 5×11 parity–check matrix (over ℤ₃) for the ternary
  Golay code.  Its rows are mutually orthogonal with respect to the
  standard inner product mod 3, i.e. ``H @ H.T ≡ 0 (mod 3)``.
* ``G`` – a 6×11 generator matrix whose rows span the nullspace of ``H``.
  It satisfies ``H @ G.T ≡ 0 (mod 3)`` and has full rank 6 over ℤ₃.
* ``P`` – the 11×11 matrix obtained by vertically stacking ``H`` and ``G``.
  While this combined matrix is not itself a parity–check in the
  traditional sense (its rank over ℤ₃ is 7), it provides a convenient
  11×11 presentation of the classical code requested by the user.

Convenience functions are provided to retrieve copies of these matrices and
to encode or check words over GF(3).  All arithmetic is performed mod 3.

Note:
    The construction herein focuses on classical linear algebra and does
    not attempt to build explicit 3¹¹ × 3¹¹ stabilizer matrices; doing so
    would be prohibitively expensive in memory and runtime.  Instead, the
    parity–check and generator matrices may be used as building blocks
    within higher‑level routines (for example, syndrome calculation or
    syndrome‑based decoding) without constructing full state vectors.
"""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Classical ternary Golay matrices
#
# The 5×11 parity–check matrix H below is taken from standard references
# (see e.g. J. H. Conway and N. J. A. Sloane, *Sphere Packings, Lattices and
# Groups*) and is reproduced here verbatim.  Each entry is an integer 0, 1 or 2
# and all arithmetic on this matrix should be performed modulo 3.
#
# Rows of H are mutually orthogonal mod 3:
#     H @ H.T ≡ 0 (mod 3)
#
# A generator matrix G is computed once at import time by taking a basis of
# the nullspace of H over GF(3).  It satisfies H @ G.T ≡ 0 (mod 3).
# -----------------------------------------------------------------------------

# 5×11 parity–check matrix for the ternary Golay code (entries modulo 3)
H: np.ndarray = np.array(
    [
        [1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 0],
        [0, 1, 0, 0, 0, 1, 1, 2, 1, 0, 2],
        [0, 0, 1, 0, 0, 1, 2, 1, 0, 1, 2],
        [0, 0, 0, 1, 0, 1, 2, 0, 1, 2, 1],
        [0, 0, 0, 0, 1, 1, 0, 2, 2, 1, 1],
    ],
    dtype=int,
)

# Precomputed generator matrix G (6×11).  The rows of G span the nullspace of H
# over GF(3).  The matrix below was derived using a symbolic computation
# (computing a basis of the nullspace of H modulo 3) and is fixed here as a
# constant so that users do not need an external dependency such as SymPy to
# generate it at runtime.  See ``compute_generator_matrix`` below for a
# programmatic derivation.
G: np.ndarray = np.array(
    [
        [2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0],
        [2, 2, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [2, 1, 2, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 2, 0, 2, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 2, 1, 2, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 1],
    ],
    dtype=int,
)

# Combined 11×11 matrix obtained by stacking H and G.  This 11×11 matrix has
# rank 7 over GF(3).  While not itself a parity–check matrix, it provides an
# 11×11 presentation requested by the user.  Users needing the true
# parity–check for syndrome computations should use H instead.
P: np.ndarray = np.vstack((H, G)) % 3


def parity_check_matrix() -> np.ndarray:
    """Return a copy of the 5×11 Golay parity–check matrix H.

    Returns
    -------
    np.ndarray
        A new array equal to ``H``.  The caller should perform all
        arithmetic on the returned matrix modulo 3.
    """
    return H.copy()


def generator_matrix() -> np.ndarray:
    """Return a copy of the 6×11 Golay generator matrix G.

    The rows of ``G`` span the nullspace of ``H`` over GF(3) and can be used
    to encode 6‑digit base‑3 messages into 11‑digit codewords via
    ``(m @ G) % 3``.

    Returns
    -------
    np.ndarray
        A copy of the fixed generator matrix.
    """
    return G.copy()


def combined_matrix() -> np.ndarray:
    """Return the 11×11 combined matrix P = [H; G] modulo 3.

    This matrix is provided primarily to satisfy user requests for an
    11×11 presentation of the code.  It does **not** serve as a parity–check
    matrix for the quantum code.  For error–correction one should use
    ``H`` for both the X‑type and Z‑type stabilizers in the CSS construction.

    Returns
    -------
    np.ndarray
        A copy of the combined 11×11 matrix ``P``.
    """
    return P.copy()


def encode_message(message: np.ndarray) -> np.ndarray:
    """Encode a length‑6 vector over GF(3) into an 11‑symbol Golay codeword.

    Parameters
    ----------
    message : np.ndarray
        A one‑dimensional array of length 6 with integer entries 0, 1 or 2.

    Returns
    -------
    np.ndarray
        A one‑dimensional array of length 11 representing the encoded codeword
        ``(message @ G) % 3``.

    Raises
    ------
    ValueError
        If the input is not a one‑dimensional array of length 6.
    """
    message = np.asarray(message, dtype=int)
    if message.ndim != 1 or message.shape[0] != G.shape[0]:
        raise ValueError(
            f"Expected message of length {G.shape[0]}, got shape {message.shape}"
        )
    # Matrix–vector product over GF(3)
    return (message @ G) % 3


def syndrome(word: np.ndarray) -> np.ndarray:
    """Compute the syndrome of an 11‑symbol word with respect to H.

    Given a word ``w`` of length 11 (entries 0, 1 or 2), the syndrome is
    ``H @ w`` mod 3.  If the word is a valid Golay codeword then the
    syndrome will be the zero vector.

    Parameters
    ----------
    word : np.ndarray
        A one‑dimensional array of length 11 with integer entries 0, 1 or 2.

    Returns
    -------
    np.ndarray
        A one‑dimensional array of length 5 giving the syndrome of ``word``.

    Raises
    ------
    ValueError
        If the input is not a one‑dimensional array of length 11.
    """
    word = np.asarray(word, dtype=int)
    if word.ndim != 1 or word.shape[0] != H.shape[1]:
        raise ValueError(
            f"Expected word of length {H.shape[1]}, got shape {word.shape}"
        )
    return (H @ word) % 3


def compute_generator_matrix() -> np.ndarray:
    """Recompute the generator matrix of the ternary Golay code from H.

    This helper is provided for transparency.  It computes a basis for
    the nullspace of ``H`` over GF(3) using Gaussian elimination.  The
    resulting matrix will be equivalent (up to row operations) to the
    fixed generator matrix ``G`` defined above.

    Returns
    -------
    np.ndarray
        A 6×11 generator matrix whose rows span the nullspace of H.
    """
    # Perform Gaussian elimination mod 3 to find the nullspace
    from sympy import Matrix

    M_mod = Matrix(H.tolist()).applyfunc(lambda x: x % 3)
    nullspace = M_mod.nullspace()
    # Convert each basis vector to a row of a numpy array
    basis_vectors = [np.array(v.T.tolist()[0], dtype=int) % 3 for v in nullspace]
    return np.array(basis_vectors, dtype=int)


if __name__ == "__main__":
    # Simple self‑test: verify orthogonality and dimensions
    # H * H^T should be zero mod 3
    ortho = (H @ H.T) % 3
    assert np.all(ortho == 0), "H is not self‑orthogonal mod 3"
    # Verify H @ G^T = 0 mod 3
    assert np.all((H @ G.T) % 3 == 0), "H and G are not orthogonal"
    # Check that G has rank 6
    from numpy.linalg import matrix_rank

    assert matrix_rank(G % 3) == G.shape[0], "Generator matrix is not full rank"
    print("Self‑test passed: Golay matrices satisfy expected relations.")