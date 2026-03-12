"""
v12.4.0 — Stabilizer Mapping for QuTiP.

Maps parity-check matrix rows to Pauli stabilizer operators
suitable for QuTiP Hamiltonian construction.

Layer 5 — Experiments.
Does not modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _require_qutip():
    """Import and return qutip, raising RuntimeError if unavailable."""
    try:
        import qutip
        return qutip
    except ImportError:
        raise RuntimeError(
            "QuTiP is required for stabilizer mapping but is not installed. "
            "Install it with: pip install qutip"
        )


def h_row_to_pauli_label(row: np.ndarray) -> str:
    """Convert a binary parity-check row to a Pauli-Z string label.

    Each 1 in the row corresponds to a Z operator on that qubit;
    each 0 corresponds to I (identity).

    Parameters
    ----------
    row : np.ndarray
        Binary row vector, shape (n,).

    Returns
    -------
    str
        Pauli string, e.g. "IZZI" for row [0, 1, 1, 0].
    """
    return "".join("Z" if int(v) != 0 else "I" for v in row)


def parity_check_to_stabilizers(
    H: np.ndarray,
) -> list[str]:
    """Convert parity-check matrix to list of Pauli stabilizer labels.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    list[str]
        List of m Pauli strings, one per check row.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    return [h_row_to_pauli_label(H_arr[ci]) for ci in range(H_arr.shape[0])]


def stabilizers_to_hamiltonian(
    H: np.ndarray,
    *,
    coupling: float = -1.0,
) -> Any:
    """Build a stabilizer Hamiltonian from a parity-check matrix.

    H_stab = coupling * sum_i S_i

    where each S_i is the tensor product of Pauli-Z and identity
    operators determined by the i-th row of H.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    coupling : float
        Coupling strength (default -1.0).

    Returns
    -------
    qutip.Qobj
        Hamiltonian as a QuTiP quantum object.
    """
    qutip = _require_qutip()
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if n == 0:
        raise ValueError("Parity-check matrix has zero columns")

    sigma_z = qutip.sigmaz()
    identity = qutip.qeye(2)

    hamiltonian = None

    for ci in range(m):
        ops = []
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                ops.append(sigma_z)
            else:
                ops.append(identity)

        stabilizer = qutip.tensor(ops)
        term = coupling * stabilizer

        if hamiltonian is None:
            hamiltonian = term
        else:
            hamiltonian = hamiltonian + term

    if hamiltonian is None:
        raise ValueError("Parity-check matrix has zero rows")

    return hamiltonian
