"""
v9.4.0 — Discovery Decoder Benchmark.

Runs a decoder on a given parity-check matrix across multiple trials
and records success rate and iteration statistics.

Layer 6 — Benchmark.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no hidden randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

from qec.decoder.bp_decoder_reference import bp_decode


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _compute_syndrome(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute binary syndrome s = H @ x (mod 2)."""
    return (H.astype(np.int32) @ x.astype(np.int32)) % 2


def _generate_error_vector(n: int, p: float, seed: int) -> np.ndarray:
    """Generate deterministic binary error vector."""
    rng = np.random.RandomState(seed)
    return (rng.random(n) < p).astype(np.uint8)


def run_decoder_benchmark(
    H: np.ndarray,
    *,
    trials: int = 100,
    error_rate: float = 0.05,
    max_iters: int = 100,
    base_seed: int = 0,
) -> dict[str, Any]:
    """Run decoder benchmark on a parity-check matrix.

    Generates deterministic error vectors, computes syndromes, and
    runs BP decoding to measure success rate and iteration counts.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    trials : int
        Number of decoding trials.
    error_rate : float
        Bit-flip probability for error generation.
    max_iters : int
        Maximum BP iterations per trial.
    base_seed : int
        Deterministic base seed.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``bp_success_rate`` : float in [0, 1]
        - ``avg_iterations`` : float
        - ``trials`` : int
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    successes = 0
    iterations_list: list[int] = []

    for trial_idx in range(trials):
        trial_seed = _derive_seed(base_seed, f"trial_{trial_idx}")

        # Generate error vector
        error = _generate_error_vector(n, error_rate, trial_seed)

        # Compute syndrome
        syndrome = _compute_syndrome(H_arr, error)

        # Build channel LLR: log((1-p)/p) for 0-bits, -log((1-p)/p) for 1-bits
        llr_magnitude = float(np.log((1.0 - error_rate) / error_rate))
        llr = np.where(error == 0, llr_magnitude, -llr_magnitude)

        # Run BP decode
        result = bp_decode(
            H_arr,
            llr,
            max_iters=max_iters,
            syndrome_vec=syndrome,
        )

        hard_decision = result[0]
        iters = int(result[1])
        iterations_list.append(iters)

        # Check convergence: decoded syndrome must match
        decoded_syndrome = _compute_syndrome(H_arr, hard_decision)
        if np.array_equal(decoded_syndrome, syndrome.astype(np.uint8)):
            successes += 1

    return {
        "bp_success_rate": successes / trials if trials > 0 else 0.0,
        "avg_iterations": float(np.mean(iterations_list)) if iterations_list else 0.0,
        "trials": trials,
    }
