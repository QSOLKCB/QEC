"""
v12.4.0 — QuTiP Noise Experiment.

Simulates depolarizing noise on stabilizer structures and measures
stabilizer fidelity decay.

Layer 5 — Experiments.
Does not modify the decoder (Layer 1).
Fully deterministic given identical inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .stabilizer_mapping import stabilizers_to_hamiltonian


def _require_qutip():
    """Import and return qutip, raising RuntimeError if unavailable."""
    try:
        import qutip
        return qutip
    except ImportError:
        raise RuntimeError(
            "QuTiP is required for noise experiments but is not installed. "
            "Install it with: pip install qutip"
        )


def run_depolarizing_noise_experiment(
    H: np.ndarray,
    *,
    coupling: float = -1.0,
    gamma: float = 0.01,
    t_max: float = 10.0,
    num_steps: int = 100,
) -> dict[str, Any]:
    """Simulate stabilizer evolution under depolarizing noise.

    Uses Lindblad master equation with single-qubit depolarizing
    collapse operators on each qubit.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    coupling : float
        Coupling strength for Hamiltonian.
    gamma : float
        Depolarizing rate per qubit.
    t_max : float
        Maximum simulation time.
    num_steps : int
        Number of time steps.

    Returns
    -------
    dict
        times : list[float]
        stabilizer_expectations : list[list[float]]
        mean_stabilizer_fidelity : list[float]
            Mean absolute stabilizer expectation at each time step.
    """
    qutip = _require_qutip()
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    hamiltonian = stabilizers_to_hamiltonian(H_arr, coupling=coupling)

    # Initial state: |0...0><0...0|
    basis_states = [qutip.basis(2, 0) for _ in range(n)]
    psi0 = qutip.tensor(basis_states)
    rho0 = qutip.ket2dm(psi0)

    tlist = np.linspace(0.0, t_max, num_steps).tolist()

    # Build collapse operators: sqrt(gamma) * sigma_{x,y,z} on each qubit.
    c_ops = []
    identity = qutip.qeye(2)
    paulis = [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]

    rate = np.sqrt(gamma / 3.0)
    for vi in range(n):
        for pauli in paulis:
            ops = [identity] * n
            ops[vi] = rate * pauli
            c_ops.append(qutip.tensor(ops))

    # Build stabilizer operators for expectation tracking.
    sigma_z = qutip.sigmaz()
    e_ops = []
    for ci in range(m):
        ops = []
        for vi_idx in range(n):
            if H_arr[ci, vi_idx] != 0:
                ops.append(sigma_z)
            else:
                ops.append(identity)
        e_ops.append(qutip.tensor(ops))

    result = qutip.mesolve(hamiltonian, rho0, tlist, c_ops=c_ops, e_ops=e_ops)

    stabilizer_expectations = []
    for expect_vals in result.expect:
        stabilizer_expectations.append(
            [round(float(v), 12) for v in expect_vals]
        )

    # Mean absolute stabilizer fidelity at each time step.
    mean_fidelity = []
    for t_idx in range(num_steps):
        vals = [abs(stabilizer_expectations[ci][t_idx]) for ci in range(m)]
        mean_fidelity.append(round(sum(vals) / max(len(vals), 1), 12))

    return {
        "times": [round(t, 12) for t in tlist],
        "stabilizer_expectations": stabilizer_expectations,
        "mean_stabilizer_fidelity": mean_fidelity,
        "gamma": gamma,
        "num_qubits": n,
        "num_stabilizers": m,
    }
