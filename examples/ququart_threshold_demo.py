"""
ququart_threshold_with_prior.py

Threshold-style scan for the 3-ququart [[3,1]]_4 code with and without
a high-density lattice prior in logical amplitude space.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qec_ququart import QuquartRepetitionCode3
from ququart_lattice_prior import QuquartLatticePrior


def logical_index_from_state(code: QuquartRepetitionCode3, state, tol=1e-8):
    overlaps = []
    for j in range(4):
        basis = code.encode_logical(j)
        overlaps.append(np.vdot(basis, state))

    overlaps = np.array(overlaps)
    probs = np.abs(overlaps) ** 2
    j_hat = int(np.argmax(probs))
    if probs[j_hat] > 1.0 - tol:
        return j_hat
    return None


def run_threshold_scan(p_values, n_trials=5000, use_prior=False, seed=1234):
    rng = np.random.default_rng(seed)
    code = QuquartRepetitionCode3()

    prior = None
    if use_prior:
        prior = QuquartLatticePrior(code, mode="d4", beta=2.0)

    p_log = []

    for p in p_values:
        errors = 0

        for _ in range(n_trials):
            # 1. Sample logical state
            j = rng.integers(0, 4)
            encoded = code.encode_logical(j)

            # 2. Apply X-type noise
            noisy = encoded.copy()
            if rng.random() < p:
                site = int(rng.integers(1, 4))  # 1,2,3
                sign = rng.choice([1, -1])
                noisy = code.apply_X_error(noisy, site=site, power=sign)

            # 3. Optional geometric prior
            if prior is not None:
                noisy = prior.apply_prior(noisy)

            # 4. Decode via stabilizer logic
            decoded, info = code.decode_X_single_error(noisy)

            # 5. Check logical outcome
            j_hat = logical_index_from_state(code, decoded)
            if j_hat is None or j_hat != j:
                errors += 1

        p_log.append(errors / n_trials)
        label = "with prior" if use_prior else "baseline"
        print(f"[{label}] p={p:.2e} -> p_log={p_log[-1]:.3e}")

    return np.array(p_values), np.array(p_log)


def main():
    p_vals = np.logspace(-4, -1, 10)

    # Baseline code
    p_phys_0, p_log_0 = run_threshold_scan(p_vals, n_trials=3000, use_prior=False)

    # With geometric lattice prior
    p_phys_1, p_log_1 = run_threshold_scan(p_vals, n_trials=3000, use_prior=True)

    plt.figure(figsize=(7, 5))
    plt.loglog(p_phys_0, p_log_0, "o-", label="[[3,1]]$_4$ baseline")
    plt.loglog(p_phys_1, p_log_1, "o-", label="[[3,1]]$_4$ + D4 lattice prior")

    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate p_log")
    plt.title("Ququart [[3,1]]$_4$ Code with High-Density Lattice Prior")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()

    out = "/tmp/ququart_lattice_prior_threshold.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {out}\n")


if __name__ == "__main__":
    main()
