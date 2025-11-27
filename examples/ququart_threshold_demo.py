"""
ququart_threshold_demo.py

Empirical threshold-style scan for the 3-ququart [[3,1]]_4 code.

Noise model:
    - With probability p: apply a random X^±1 to a random site (1,2,3)
    - With probability 1-p: no error

We:
    1. Encode a random logical state |j_L>, j in {0,1,2,3}
    2. Apply noise
    3. Attempt to decode single X^±1 errors
    4. Check if we recovered the correct logical state

This yields a logical error rate p_log as a function of p.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qec_ququart import QuquartRepetitionCode3


def logical_index_from_state(code: QuquartRepetitionCode3, state, tol=1e-8):
    """
    Given a state, determine which logical |j_L> it matches,
    or return None if not within tolerance.

    This is crude but sufficient for the repetition structure.
    """
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


def run_threshold_scan(p_values, n_trials=5000, seed=1234):
    rng = np.random.default_rng(seed)
    code = QuquartRepetitionCode3()

    p_log = []

    for p in p_values:
        errors = 0

        for _ in range(n_trials):
            # 1. Sample logical state
            j = rng.integers(0, 4)
            encoded = code.encode_logical(j)

            # 2. Apply noise
            noisy = encoded.copy()
            if rng.random() < p:
                site = int(rng.integers(1, 4))  # 1,2,3
                sign = rng.choice([1, -1])
                noisy = code.apply_X_error(noisy, site=site, power=sign)

            # 3. Decode
            decoded, info = code.decode_X_single_error(noisy)

            # 4. Check logical outcome
            j_hat = logical_index_from_state(code, decoded)
            if j_hat is None or j_hat != j:
                errors += 1

        p_log.append(errors / n_trials)
        print(f"p={p:.2e} -> p_log={p_log[-1]:.3e}")

    return np.array(p_values), np.array(p_log)


def main():
    p_vals = np.logspace(-4, -1, 10)
    p_phys, p_log = run_threshold_scan(p_vals, n_trials=3000)

    plt.figure(figsize=(7, 5))
    plt.loglog(p_phys, p_log, "o-", label="[[3,1]]$_4$ ququart repetition code")
    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate p_log")
    plt.title("Ququart [[3,1]]$_4$ Code – Threshold-style Scan")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()

    out = "/tmp/ququart_threshold.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {out}\n")


if __name__ == "__main__":
    main()
