"""
v11.0.0 — Trapping Set Detector.

Detects small elementary trapping sets (ETS) in LDPC/QLDPC Tanner graphs.

An (a, b) elementary trapping set is a variable-node subset of size *a*
whose induced subgraph has exactly *b* unsatisfied (odd-degree) check
nodes, and all induced check nodes have degree 1 or 2 within the subset.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


_DEFAULT_MAX_A = 6
_DEFAULT_MAX_B = 4
_COMBINATORIAL_BUDGET = 500_000


class TrappingSetDetector:
    """Detect small elementary trapping sets in a Tanner graph.

    Parameters
    ----------
    max_a : int
        Maximum variable-node subset size to search (default 6).
    max_b : int
        Maximum number of unsatisfied checks to report (default 4).
    budget : int
        Maximum number of candidate subsets to evaluate before stopping,
        as a runtime safety bound for large matrices.
    """

    def __init__(
        self,
        max_a: int = _DEFAULT_MAX_A,
        max_b: int = _DEFAULT_MAX_B,
        budget: int = _COMBINATORIAL_BUDGET,
    ) -> None:
        self.max_a = max_a
        self.max_b = max_b
        self.budget = budget

    def detect(self, H: np.ndarray) -> dict[str, Any]:
        """Detect elementary trapping sets in a parity-check matrix.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``min_size`` : int — smallest ETS variable-subset size found
            - ``counts`` : dict — {(a, b): count} for each ETS type
            - ``total`` : int — total number of ETS found
            - ``variable_participation`` : list[int] — per-variable count
              of how many ETS each variable node participates in
        """
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape

        # Build adjacency: for each variable node, list of connected checks
        var_to_checks: list[list[int]] = []
        for vi in range(n):
            checks = []
            for ci in range(m):
                if H_arr[ci, vi] != 0:
                    checks.append(ci)
            var_to_checks.append(checks)

        counts: dict[tuple[int, int], int] = {}
        variable_participation = [0] * n
        min_size = self.max_a + 1
        total = 0
        evaluated = 0

        for a in range(1, self.max_a + 1):
            if evaluated >= self.budget:
                break
            for subset in combinations(range(n), a):
                evaluated += 1
                if evaluated > self.budget:
                    break

                # Collect all checks connected to any variable in the subset
                # and count how many variables in the subset each check touches
                check_degree: dict[int, int] = {}
                for vi in subset:
                    for ci in var_to_checks[vi]:
                        check_degree[ci] = check_degree.get(ci, 0) + 1

                # Elementary trapping set: all induced checks have degree 1 or 2
                is_elementary = True
                for deg in check_degree.values():
                    if deg > 2:
                        is_elementary = False
                        break

                if not is_elementary:
                    continue

                # Count unsatisfied checks (odd degree within subset)
                b = 0
                for deg in check_degree.values():
                    if deg % 2 == 1:
                        b += 1

                if b > self.max_b:
                    continue

                # Valid (a, b) ETS found
                key = (a, b)
                counts[key] = counts.get(key, 0) + 1
                total += 1
                if a < min_size:
                    min_size = a

                for vi in subset:
                    variable_participation[vi] += 1

        if min_size > self.max_a:
            min_size = 0

        return {
            "min_size": min_size,
            "counts": counts,
            "total": total,
            "variable_participation": variable_participation,
        }
