"""
v12.4.0 — QuTiP Experiment Harness.

Entry point for running quantum stabilizer experiments using QuTiP.
Constructs Hamiltonians from parity-check matrices and runs
time-evolution simulations.

Layer 5 — Experiments.
Does not modify the decoder (Layer 1).
Fully deterministic given identical inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .stabilizer_mapping import (
    parity_check_to_stabilizers,
    stabilizers_to_hamiltonian,
)


def _require_qutip():
    """Import and return qutip, raising RuntimeError if unavailable."""
    try:
        import qutip
        return qutip
    except ImportError:
        raise RuntimeError(
            "QuTiP is required for the experiment harness but is not "
            "installed. Install it with: pip install qutip"
        )


def run_stabilizer_energy_experiment(
    H: np.ndarray,
    *,
    coupling: float = -1.0,
) -> dict[str, Any]:
    """Compute ground state energy and stabilizer structure.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    coupling : float
        Coupling strength for Hamiltonian.

    Returns
    -------
    dict
        ground_energy : float
        num_stabilizers : int
        num_qubits : int
        stabilizer_labels : list[str]
    """
    qutip = _require_qutip()
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    hamiltonian = stabilizers_to_hamiltonian(H_arr, coupling=coupling)
    energies = hamiltonian.eigenenergies()

    return {
        "ground_energy": float(energies[0]),
        "num_stabilizers": m,
        "num_qubits": n,
        "stabilizer_labels": parity_check_to_stabilizers(H_arr),
    }


def run_time_evolution_experiment(
    H: np.ndarray,
    *,
    coupling: float = -1.0,
    t_max: float = 10.0,
    num_steps: int = 100,
) -> dict[str, Any]:
    """Simulate time evolution under the stabilizer Hamiltonian.

    Starts from the computational basis state |0...0> and tracks
    the expectation value of each stabilizer operator over time.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    coupling : float
        Coupling strength for Hamiltonian.
    t_max : float
        Maximum simulation time.
    num_steps : int
        Number of time steps.

    Returns
    -------
    dict
        times : list[float]
        stabilizer_expectations : list[list[float]]
            Shape (m, num_steps) — expectation of each stabilizer
            at each time step.
    """
    qutip = _require_qutip()
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    hamiltonian = stabilizers_to_hamiltonian(H_arr, coupling=coupling)

    # Initial state: |0...0>
    basis_states = [qutip.basis(2, 0) for _ in range(n)]
    psi0 = qutip.tensor(basis_states)

    tlist = np.linspace(0.0, t_max, num_steps).tolist()

    # Build stabilizer operators for expectation tracking.
    sigma_z = qutip.sigmaz()
    identity = qutip.qeye(2)

    e_ops = []
    for ci in range(m):
        ops = []
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                ops.append(sigma_z)
            else:
                ops.append(identity)
        e_ops.append(qutip.tensor(ops))

    result = qutip.sesolve(hamiltonian, psi0, tlist, e_ops=e_ops)

    stabilizer_expectations = []
    for expect_vals in result.expect:
        stabilizer_expectations.append(
            [round(float(v), 12) for v in expect_vals]
        )

    return {
        "times": [round(t, 12) for t in tlist],
        "stabilizer_expectations": stabilizer_expectations,
    }
