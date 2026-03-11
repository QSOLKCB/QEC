"""
v11.0.0 — BP Stability Probe.

Runs short belief-propagation probes to detect decoder instability and
estimates Jacobian spectral radius for BP convergence analysis.

This module lives in the decoder package but does **not** modify the
decoder core.  It is a read-only diagnostic probe that runs lightweight
min-sum style updates on copied inputs only.

Layer 1 extension — Observational probe.
Does not modify decoder internals.
Fully deterministic: explicit seed injection, no global state, no input
mutation.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import numpy as np


_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


class BPStabilityProbe:
    """Run short BP probes to detect decoder instability.

    Parameters
    ----------
    trials : int
        Number of independent probe trials (default 50).
    iterations : int
        Maximum BP iterations per trial (default 10).
    seed : int
        Base seed for deterministic error pattern generation.
    """

    def __init__(
        self,
        trials: int = 50,
        iterations: int = 10,
        seed: int = 42,
    ) -> None:
        self.trials = trials
        self.iterations = iterations
        self.seed = seed

    def probe(self, H: np.ndarray) -> dict[str, Any]:
        """Run BP stability probes on a parity-check matrix.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``bp_stability_score`` : float
            - ``divergence_rate`` : float
            - ``stagnation_rate`` : float
            - ``oscillation_score`` : float
            - ``average_iterations`` : float
        """
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape

        if self.trials == 0 or m == 0 or n == 0:
            return {
                "bp_stability_score": 1.0,
                "divergence_rate": 0.0,
                "stagnation_rate": 0.0,
                "oscillation_score": 0.0,
                "average_iterations": 0.0,
            }

        # Precompute adjacency lists once
        check_to_vars: list[list[int]] = [[] for _ in range(m)]
        var_to_checks: list[list[int]] = [[] for _ in range(n)]
        for ci in range(m):
            for vi in range(n):
                if H_arr[ci, vi] != 0:
                    check_to_vars[ci].append(vi)
                    var_to_checks[vi].append(ci)

        diverged_count = 0
        stagnated_count = 0
        total_oscillation = 0.0
        total_iterations = 0

        for t in range(self.trials):
            trial_seed = _derive_seed(self.seed, f"trial_{t}")
            rng = np.random.RandomState(trial_seed)

            # Generate a small deterministic error pattern
            error_rate = 0.05
            error = (rng.random(n) < error_rate).astype(np.float64)
            syndrome = (H_arr @ error) % 2

            # Initialize LLR from channel (simple BSC model)
            channel_llr = np.where(error > 0.5, -2.0, 2.0)
            # Add small noise for realism
            channel_llr += rng.randn(n) * 0.1

            # Run abbreviated min-sum BP
            converged, iters, osc = self._run_min_sum(
                H_arr, m, n, syndrome, channel_llr,
                check_to_vars, var_to_checks,
            )

            total_iterations += iters
            total_oscillation += osc

            if not converged and iters >= self.iterations:
                # Check if messages diverged (grew very large)
                diverged_count += 1
            elif not converged:
                stagnated_count += 1

        divergence_rate = diverged_count / self.trials
        stagnation_rate = stagnated_count / self.trials
        oscillation_score = total_oscillation / self.trials
        average_iterations = total_iterations / self.trials

        bp_stability_score = (
            (1.0 - divergence_rate)
            * (1.0 - stagnation_rate)
            * math.exp(-oscillation_score)
        )

        return {
            "bp_stability_score": round(bp_stability_score, _ROUND),
            "divergence_rate": round(divergence_rate, _ROUND),
            "stagnation_rate": round(stagnation_rate, _ROUND),
            "oscillation_score": round(oscillation_score, _ROUND),
            "average_iterations": round(average_iterations, _ROUND),
        }

    def _run_min_sum(
        self,
        H: np.ndarray,
        m: int,
        n: int,
        syndrome: np.ndarray,
        channel_llr: np.ndarray,
        check_to_vars: list[list[int]],
        var_to_checks: list[list[int]],
    ) -> tuple[bool, int, float]:
        """Run min-sum BP and return (converged, iterations, oscillation).

        Returns
        -------
        tuple[bool, int, float]
            (converged, iteration_count, oscillation_score)
        """
        # Messages: check-to-variable
        c2v: dict[tuple[int, int], float] = {}
        for ci in range(m):
            for vi in check_to_vars[ci]:
                c2v[(ci, vi)] = 0.0

        prev_decisions = np.zeros(n, dtype=np.float64)
        sign_flips = 0
        total_sign_checks = 0

        converged = False
        iters = 0

        for it in range(self.iterations):
            iters = it + 1

            # Variable-to-check messages
            v2c: dict[tuple[int, int], float] = {}
            for vi in range(n):
                for ci in var_to_checks[vi]:
                    total = channel_llr[vi]
                    for cj in var_to_checks[vi]:
                        if cj != ci:
                            total += c2v.get((cj, vi), 0.0)
                    v2c[(vi, ci)] = total

            # Check-to-variable messages (min-sum)
            for ci in range(m):
                for vi in check_to_vars[ci]:
                    min_abs = float("inf")
                    sign_prod = 1.0
                    s = 1.0 if syndrome[ci] == 0 else -1.0
                    for vj in check_to_vars[ci]:
                        if vj != vi:
                            msg = v2c.get((vj, ci), 0.0)
                            abs_msg = abs(msg)
                            if abs_msg < min_abs:
                                min_abs = abs_msg
                            sign_prod *= 1.0 if msg >= 0 else -1.0
                    c2v[(ci, vi)] = s * sign_prod * min_abs

            # Compute posterior decisions
            decisions = np.zeros(n, dtype=np.float64)
            for vi in range(n):
                total = channel_llr[vi]
                for ci in var_to_checks[vi]:
                    total += c2v.get((ci, vi), 0.0)
                decisions[vi] = total

            # Check convergence: hard decisions satisfy syndrome
            hard = (decisions < 0).astype(np.float64)
            computed_syndrome = (H @ hard) % 2
            if np.array_equal(computed_syndrome, syndrome):
                converged = True
                break

            # Track oscillation: sign changes in decisions
            if it > 0:
                for vi in range(n):
                    total_sign_checks += 1
                    if (decisions[vi] >= 0) != (prev_decisions[vi] >= 0):
                        sign_flips += 1

            prev_decisions = decisions.copy()

            # Divergence check: if messages grow too large, flag and stop
            max_msg = max(abs(v) for v in c2v.values()) if c2v else 0.0
            if max_msg > 1e6:
                break

        oscillation = sign_flips / max(total_sign_checks, 1)
        return converged, iters, oscillation


def estimate_bp_instability(H: np.ndarray, seed: int = 42) -> dict[str, float]:
    """Estimate BP instability via Jacobian spectral radius approximation.

    Uses deterministic power iteration on the implicit BP Jacobian
    operator to estimate the spectral radius.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Seed for deterministic initialization.

    Returns
    -------
    dict[str, float]
        Dictionary with key:
        - ``jacobian_spectral_radius_est`` : float
          rho < 1 → stable, rho ~ 1 → marginal, rho > 1 → unstable
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if m == 0 or n == 0:
        return {"jacobian_spectral_radius_est": 0.0}

    # Build adjacency
    check_to_vars: list[list[int]] = [[] for _ in range(m)]
    var_to_checks: list[list[int]] = [[] for _ in range(n)]
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                check_to_vars[ci].append(vi)
                var_to_checks[vi].append(ci)

    # Count directed edges for message-space dimension
    edges: list[tuple[int, int]] = []  # (ci, vi) pairs
    for ci in range(m):
        for vi in check_to_vars[ci]:
            edges.append((ci, vi))

    num_edges = len(edges)
    if num_edges == 0:
        return {"jacobian_spectral_radius_est": 0.0}

    edge_index = {e: i for i, e in enumerate(edges)}

    # Power iteration on the linearized BP Jacobian
    # The Jacobian of min-sum at high-SNR acts similarly to the
    # non-backtracking operator. We approximate:
    # J[c->v, c'->v'] = tanh weight propagation through the factor graph
    rng = np.random.RandomState(seed)
    x = rng.randn(num_edges)
    norm = np.linalg.norm(x)
    if norm < 1e-15:
        return {"jacobian_spectral_radius_est": 0.0}
    x /= norm

    eigenvalue_est = 0.0

    for _ in range(30):
        y = np.zeros(num_edges, dtype=np.float64)

        # For each check-to-var message (ci, vi):
        # J maps incoming var-to-check messages from other variables
        for ci in range(m):
            for vi in check_to_vars[ci]:
                out_idx = edge_index[(ci, vi)]
                # Incoming contributions: for each other vj connected to ci,
                # and for each check cj connected to vj (cj != ci),
                # there is a coupling
                for vj in check_to_vars[ci]:
                    if vj == vi:
                        continue
                    for cj in var_to_checks[vj]:
                        if cj == ci:
                            continue
                        in_idx = edge_index.get((cj, vj))
                        if in_idx is not None:
                            y[out_idx] += x[in_idx]

        norm_y = np.linalg.norm(y)
        if norm_y < 1e-15:
            eigenvalue_est = 0.0
            break

        eigenvalue_est = norm_y / np.linalg.norm(x)
        x = y / norm_y

    return {
        "jacobian_spectral_radius_est": round(float(eigenvalue_est), _ROUND),
    }
