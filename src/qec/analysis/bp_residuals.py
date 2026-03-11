"""
v11.1.0 — BP Residual Analyzer.

Computes residual magnitude per variable node during short min-sum BP
runs.  The residual map identifies decoder failure regions where message
updates fail to converge.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


class BPResidualAnalyzer:
    """Compute BP residual magnitude per variable node.

    Runs a short min-sum style BP update on a parity-check matrix and
    tracks message change magnitude between iterations.  The accumulated
    residual per variable node identifies regions where decoding is
    unstable.

    Fully deterministic: same H, iterations, seed → identical output.
    Does not modify inputs.
    """

    def compute_residual_map(
        self,
        H: np.ndarray,
        iterations: int = 10,
        seed: int = 0,
    ) -> dict:
        """Compute residual magnitude per variable node.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).
        iterations : int
            Number of min-sum BP iterations (default 10).
        seed : int
            Deterministic seed for LLR initialization.

        Returns
        -------
        dict
            ``residual_map`` : np.ndarray of shape (n,) — accumulated
            residual per variable node.
            ``max_residual`` : float — maximum residual value.
            ``mean_residual`` : float — mean residual value.
        """
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape

        if m == 0 or n == 0:
            return {
                "residual_map": np.zeros(n, dtype=np.float64),
                "max_residual": 0.0,
                "mean_residual": 0.0,
            }

        # Initialize LLR values deterministically
        rng = np.random.RandomState(_derive_seed(seed, "bp_residual_init"))
        llr = rng.standard_normal(n).astype(np.float64)

        # Build edge lists for efficient message passing
        # check_to_var[ci] = sorted list of variable indices
        # var_to_check[vi] = sorted list of check indices
        check_to_var: list[list[int]] = [[] for _ in range(m)]
        var_to_check: list[list[int]] = [[] for _ in range(n)]

        # Use NumPy to find all nonzero entries in H_arr to avoid an O(m·n)
        # nested Python loop; this keeps the same complexity but pushes the
        # per-entry scanning into optimized NumPy code.
        rows, cols = np.nonzero(H_arr)
        for ci, vi in zip(rows, cols):
            check_to_var[ci].append(int(vi))
            var_to_check[vi].append(int(ci))

        # Initialize messages: check-to-variable messages
        # msg_c2v[(ci, vi)] = message from check ci to variable vi
        msg_c2v: dict[tuple[int, int], float] = {}
        for ci in range(m):
            for vi in check_to_var[ci]:
                msg_c2v[(ci, vi)] = 0.0

        # Accumulated residual per variable node
        residual_map = np.zeros(n, dtype=np.float64)

        for _it in range(iterations):
            # Variable-to-check messages
            msg_v2c: dict[tuple[int, int], float] = {}
            for vi in range(n):
                for ci in var_to_check[vi]:
                    # Sum of incoming check-to-var messages excluding ci
                    total = llr[vi]
                    for cj in var_to_check[vi]:
                        if cj != ci:
                            total += msg_c2v.get((cj, vi), 0.0)
                    msg_v2c[(vi, ci)] = total

            # Check-to-variable messages (min-sum)
            new_msg_c2v: dict[tuple[int, int], float] = {}
            for ci in range(m):
                vars_ci = check_to_var[ci]
                for vi in vars_ci:
                    # Min-sum: product of signs × minimum of abs values
                    sign = 1.0
                    min_abs = float("inf")
                    for vj in vars_ci:
                        if vj != vi:
                            val = msg_v2c.get((vj, ci), 0.0)
                            if val < 0:
                                sign *= -1.0
                            abs_val = abs(val)
                            if abs_val < min_abs:
                                min_abs = abs_val
                    if min_abs == float("inf"):
                        min_abs = 0.0
                    new_msg_c2v[(ci, vi)] = sign * min_abs

            # Compute LLR updates and track residuals
            new_llr = llr.copy()
            for vi in range(n):
                total = llr[vi]
                for ci in var_to_check[vi]:
                    total += new_msg_c2v.get((ci, vi), 0.0)
                new_llr[vi] = total

            # Residual = |LLR_new - LLR_old|
            residual = np.abs(new_llr - llr)
            residual_map += residual

            # Update state for next iteration
            llr = new_llr
            msg_c2v = new_msg_c2v

        max_residual = float(np.max(residual_map)) if n > 0 else 0.0
        mean_residual = float(np.mean(residual_map)) if n > 0 else 0.0

        return {
            "residual_map": residual_map,
            "max_residual": max_residual,
            "mean_residual": mean_residual,
        }
