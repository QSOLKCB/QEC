# =============================================================================
# Steane [[7,1,3]] Monte Carlo — Pure NumPy, Deterministic, Vectorized
# =============================================================================
# This module provides a clean, fast, dependency-minimal logical error
# estimator for the Steane code under iid depolarizing noise.
#
# Features:
#   • No Python loops in the hot path
#   • No SciPy / Qiskit / external quantum libs
#   • Memory-safe (optional: chunked mode)
#   • Deterministic reproducibility (seeded RNG)
#   • Auto-derived syndrome decoder (no hand-entered masks)
#
# This is the canonical baseline for QSOLKCB/QEC: small, honest, accurate.
# =============================================================================

import numpy as np


# ----------------------------------------------------------------------
# Static Steane geometry (classical [7,4,3] Hamming structure)
# ----------------------------------------------------------------------
H = np.array([
    [1,0,0,1,1,0,1],
    [0,1,0,1,0,1,1],
    [0,0,1,0,1,1,1]
], dtype=np.uint8)

LOGICAL_X = np.ones(7, dtype=np.uint8)

# All 2^7 possible X-error masks (0..127 encoded as bitmasks)
all_errors = np.arange(128, dtype=np.uint8)
error_vectors = ((all_errors[:, None] >> np.arange(7)) & 1).astype(np.uint8)

# Compute syndrome of each 7-bit pattern
syndromes = (error_vectors @ H.T) & 1
syndrome_int = (
    syndromes[:, 0] * 4 +
    syndromes[:, 1] * 2 +
    syndromes[:, 2]
).astype(np.uint8)

# Build consistent minimum weight Hamming decoder:
# syndrome → lowest-weight representative (0 or 1-qubit flips)
decoder = np.zeros(8, dtype=np.uint8)
weights = error_vectors.sum(axis=1)
for e_int, w, s in zip(all_errors, weights, syndrome_int):
    if w <= 1:
        decoder[s] = e_int

# Logical action lookup table: does error anticommute with logical X?
logical_flip = (error_vectors @ LOGICAL_X) & 1


# ----------------------------------------------------------------------
# Fast Monte Carlo engine
# ----------------------------------------------------------------------
def steane_logical_rate_fast(
    p_phys: float,
    *,
    shots: int = 2**22,
    seed: int | None = 12345,
    chunk: int | None = None
) -> float:
    """
    Monte Carlo estimator of the logical error probability for the Steane code
    under independent depolarizing noise.

    Args:
        p_phys (float): physical depolarizing error probability per qubit.
        shots (int): number of samples for Monte Carlo estimate.
        seed (int or None): RNG seed for reproducibility.
        chunk (int or None): optional chunk size to bound RAM usage.
                             If None, uses full shot array at once.

    Returns:
        float: estimated probability of any logical error (X, Z, or Y).
    """

    if p_phys <= 0.0:
        return 0.0

    # Depolarizing probabilities
    p_x = p_z = p_y = p_phys / 3.0
    p_i = 1.0 - p_phys
    probs = np.array([p_i, p_x, p_z, p_y], dtype=np.float64)

    rng = np.random.default_rng(seed)

    # If giant shot count, process in bounded slices
    if chunk is None:
        chunk = shots

    total_fail = 0
    processed = 0

    while processed < shots:
        n = min(chunk, shots - processed)

        # Sample Pauli errors: 0=I,1=X,2=Z,3=Y
        choices = rng.choice(4, size=(n, 7), p=probs)

        # X-type and Z-type masks
        Xmask = np.logical_or(choices == 1, choices == 3)
        Zmask = np.logical_or(choices == 2, choices == 3)

        # Pack masks into 7-bit integers
        Xerr_int = np.packbits(Xmask, axis=1, bitorder='little')[:, 0]
        Zerr_int = np.packbits(Zmask, axis=1, bitorder='little')[:, 0]

        # Syndrome → correction
        synd_X = syndrome_int[Xerr_int]
        synd_Z = syndrome_int[Zerr_int]

        corr_X = decoder[synd_X]
        corr_Z = decoder[synd_Z]

        # Residual errors after correction
        residual_X = Xerr_int ^ corr_X
        residual_Z = Zerr_int ^ corr_Z

        # Logical failures: X, Z, or both (Y)
        logical_X_fail = logical_flip[residual_X]
        logical_Z_fail = logical_flip[residual_Z]

        logical_any = np.logical_or(logical_X_fail, logical_Z_fail)
        total_fail += logical_any.sum()

        processed += n

    return total_fail / shots


# ----------------------------------------------------------------------
# Benchmark sweep helper (optional)
# ----------------------------------------------------------------------
def sweep(p_values, shots=2**22, seed=12345):
    """
    Convenience function: sweep Steane logical rate across p-values.
    Returns a list of dictionaries; caller can convert to DataFrame.
    """
    results = []
    for p in p_values:
        r = steane_logical_rate_fast(p, shots=shots, seed=seed)
        results.append({'p_phys': p, 'p_logical': r})
    return results


# ----------------------------------------------------------------------
# Module self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = 1e-3
    print("Testing Steane Monte Carlo at p =", p)
    est = steane_logical_rate_fast(p, shots=2**20)
    print("Estimated logical error rate:", est)
