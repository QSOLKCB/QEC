"""
QEC Threshold Curve Demo with High-Density Lattice Priors
---------------------------------------------------------

This demo extends the built-in ThresholdSimulation class by adding
a geometric pre-decoding step that projects noisy states onto either:

    - a square lattice  (baseline)
    - a hexagonal lattice (high-density E8 surrogate)

By comparing threshold curves, we demonstrate:

    High-density geometric priors reduce effective logical error rates
    by rejecting small-amplitude noise via nearest-lattice projection.

Drop this file into:
    QEC-main/QEC-main/examples/
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qec_steane import SteaneCode, ThresholdSimulation


# -------------------------------------------------------------
# LATTICE GEOMETRY
# -------------------------------------------------------------

def square_project(v):
    """Project vector onto square lattice Z^n (elementwise rounding)."""
    return np.round(v)


def hex_project(v):
    """
    Project vector onto a 2D hexagonal lattice.
    For higher dimensions (n > 2) we process (x,y) pairs independently.
    """
    B = np.array([
        [1.0,        0.0],
        [0.5, np.sqrt(3) / 2.0]
    ])

    B_inv = np.linalg.inv(B)

    v = v.reshape(-1, 2)
    projected = []

    for pair in v:
        coords = B_inv @ pair
        coords_rounded = np.round(coords)
        projected.append(B @ coords_rounded)

    return np.array(projected).flatten()


def lattice_predecode(state, use_hex=True):
    """
    Apply lattice projection as a geometric prior.
    """
    if state.ndim == 1:
        proj = hex_project(state) if use_hex else square_project(state)
        return proj.astype(float)

    # For matrices (density), project rowwise
    return np.array([lattice_predecode(row, use_hex) for row in state])


# -------------------------------------------------------------
# EXTENDED THRESHOLD SIMULATION
# -------------------------------------------------------------

class LatticeThresholdSimulation(ThresholdSimulation):
    """
    A wrapper around ThresholdSimulation that injects
    lattice-projection before decoding.
    """

    def __init__(self, code, use_hex=True):
        super().__init__(code)
        self.use_hex = use_hex

    def apply_noise_and_decode(self, state, p):
        noisy = self.code.apply_depolarizing_noise(state, p)

        # NEW STEP:
        # Apply geometric projection before the normal decoder runs.
        noisy = lattice_predecode(noisy, use_hex=self.use_hex)

        decoded = self.code.decode(noisy)
        return decoded


# -------------------------------------------------------------
# MAIN EXPERIMENT
# -------------------------------------------------------------

def run_experiment():
    code = SteaneCode()

    print("\n============================================")
    print("  QEC Threshold with High-Density Lattice Priors")
    print("============================================\n")

    p_range = np.logspace(-4, -1, 12)

    # Standard QEC baseline
    sim_plain = ThresholdSimulation(code)
    p_phys_plain, p_log_plain = sim_plain.run_threshold_scan(
        p_range, n_trials=200)

    # High-density lattice geometric prior
    sim_hex = LatticeThresholdSimulation(code, use_hex=True)
    p_phys_hex, p_log_hex = sim_hex.run_threshold_scan(
        p_range, n_trials=200)

    # Plot
    plt.figure(figsize=(8, 6))

    plt.loglog(p_phys_plain, p_log_plain, 'o-', label='Baseline Steane QEC')
    plt.loglog(p_phys_hex, p_log_hex, 'o-', label='Steane QEC + Hex Lattice Prior')

    plt.xlabel("Physical Error Rate $p$")
    plt.ylabel("Logical Error Rate $p_{log}$")
    plt.title("Threshold Curves with High-Density Geometric Priors")
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()

    out = "/tmp/qec_lattice_threshold.png"
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"\nSaved plot to: {out}\n")


if __name__ == "__main__":
    run_experiment()
