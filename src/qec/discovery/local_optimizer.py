"""
v11.5.0 — Local Graph Optimizer.

Performs deterministic local edge rewiring to improve Tanner graph fitness
without changing code size.  Used as the local search component in
memetic evolutionary discovery.

Operators:
  1. Absorbing-set repair — break internal edges of candidate absorbing sets
  2. Residual hot-spot smoothing — rewire high-residual variable edges
  3. Cycle irregularity reduction — rewire edges in highest-twist cycles
  4. Bethe-Hessian stability improvement — rewire unstable-mode nodes
  5. Residual cluster smoothing — improve cluster escape connectivity

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no hidden randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

from src.qec.fitness.fitness_engine import FitnessEngine
from src.qec.analysis.absorbing_sets import AbsorbingSetPredictor
from src.qec.analysis.bp_residuals import BPResidualAnalyzer
from src.qec.analysis.cycle_topology import CycleTopologyAnalyzer
from src.qec.analysis.bethe_hessian import BetheHessianAnalyzer
from src.qec.analysis.residual_clusters import ResidualClusterAnalyzer


_DEFAULT_MAX_STEPS = 10
_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _edge_count(H: np.ndarray) -> int:
    """Count non-zero entries in a binary matrix."""
    return int(np.count_nonzero(H))


def _try_rewire(
    H: np.ndarray,
    row_from: int,
    col_from: int,
    row_to: int,
    col_to: int,
) -> np.ndarray | None:
    """Attempt a single edge rewire: move (row_from, col_from) to (row_to, col_to).

    Returns the rewired matrix or None if the rewire is invalid.
    Preserves edge count and prevents node isolation.
    """
    if H[row_from, col_from] == 0:
        return None
    if H[row_to, col_to] != 0:
        return None
    if row_from == row_to and col_from == col_to:
        return None

    H_new = H.copy()
    H_new[row_from, col_from] = 0
    H_new[row_to, col_to] = 1

    # Check no row or column becomes all-zero
    if H_new[row_from, :].sum() < 1:
        return None
    if H_new[:, col_from].sum() < 1:
        return None

    return H_new


class LocalGraphOptimizer:
    """Deterministic local graph rewiring optimizer.

    Iteratively applies small local rewiring operations to improve
    a Tanner graph's fitness.  Each step evaluates candidate rewires
    from four operators and accepts the best fitness-improving move.

    Parameters
    ----------
    max_steps : int
        Maximum number of local improvement steps (default 10).
    seed : int
        Base seed for deterministic operator scheduling.
    """

    def __init__(
        self,
        max_steps: int = _DEFAULT_MAX_STEPS,
        seed: int = 42,
    ) -> None:
        self.max_steps = max_steps
        self.seed = seed
        self._absorbing_predictor = AbsorbingSetPredictor()
        self._residual_analyzer = BPResidualAnalyzer()
        self._cycle_analyzer = CycleTopologyAnalyzer()
        self._bethe_analyzer = BetheHessianAnalyzer()
        self._cluster_analyzer = ResidualClusterAnalyzer()

    def optimize(
        self,
        H: np.ndarray,
        fitness_engine: FitnessEngine,
    ) -> np.ndarray:
        """Perform local optimization on a parity-check matrix.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).
        fitness_engine : FitnessEngine
            Fitness evaluator for candidate matrices.

        Returns
        -------
        np.ndarray
            Locally optimized parity-check matrix.  Shape and edge
            count are preserved.  If no improving rewire is found
            in any step, the original matrix is returned unchanged.
        """
        H_current = np.array(H, dtype=np.float64)
        original_shape = H_current.shape
        original_edges = _edge_count(H_current)

        current_result = fitness_engine.evaluate(H_current)
        current_fitness = current_result["composite"]

        operators = [
            self._absorbing_set_repair,
            self._residual_hotspot_smoothing,
            self._cycle_irregularity_reduction,
            self._bethe_hessian_improvement,
            self._residual_cluster_smoothing,
        ]

        for step in range(self.max_steps):
            step_seed = _derive_seed(self.seed, f"local_step_{step}")
            op_idx = step % len(operators)
            operator = operators[op_idx]

            candidates = operator(H_current, step_seed)

            best_candidate = None
            best_fitness = current_fitness

            for cand in candidates:
                # Validate shape and edge count
                if cand.shape != original_shape:
                    continue
                if _edge_count(cand) != original_edges:
                    continue

                cand_result = fitness_engine.evaluate(cand)
                cand_fitness = cand_result["composite"]

                if cand_fitness > best_fitness:
                    best_fitness = cand_fitness
                    best_candidate = cand

            if best_candidate is not None:
                H_current = best_candidate
                current_fitness = best_fitness

        return H_current

    def _absorbing_set_repair(
        self,
        H: np.ndarray,
        seed: int,
    ) -> list[np.ndarray]:
        """Generate candidate rewires that break absorbing-set structure.

        Identifies candidate absorbing sets and tries to move an internal
        edge to outside the set.
        """
        candidates: list[np.ndarray] = []
        rng = np.random.RandomState(seed)

        prediction = self._absorbing_predictor.predict(H)
        absorbing_sets = prediction.get("candidate_sets", [])

        if not absorbing_sets:
            return candidates

        # Sort by risk descending for determinism
        sorted_sets = sorted(
            absorbing_sets,
            key=lambda s: (-s.get("risk_score", 0.0),),
        )

        # Process top candidate set
        top_set = sorted_sets[0]
        set_vars = sorted(top_set.get("variables", []))

        if not set_vars:
            return candidates

        m, n = H.shape
        all_vars = list(range(n))
        outside_vars = sorted(set(all_vars) - set(set_vars))

        if not outside_vars:
            return candidates

        # Try rewiring edges from inside to outside the set
        for var_in in set_vars:
            checks = sorted(np.where(H[:, var_in] == 1)[0].tolist())
            if len(checks) <= 1:
                continue
            # Pick check deterministically
            check_idx = checks[rng.randint(0, len(checks))]

            # Try connecting to an outside variable
            for var_out in outside_vars[:3]:
                result = _try_rewire(H, check_idx, var_in, check_idx, var_out)
                if result is not None:
                    candidates.append(result)
                    break

            if len(candidates) >= 3:
                break

        return candidates

    def _residual_hotspot_smoothing(
        self,
        H: np.ndarray,
        seed: int,
    ) -> list[np.ndarray]:
        """Generate candidate rewires that smooth BP residual hot spots.

        Identifies high-residual variable nodes and rewires their edges
        toward low-residual regions.
        """
        candidates: list[np.ndarray] = []
        rng = np.random.RandomState(seed)

        residuals = self._residual_analyzer.compute_residual_map(
            H, iterations=5, seed=seed,
        )
        residual_map = residuals.get("residual_map", np.array([]))

        if len(residual_map) == 0:
            return candidates

        m, n = H.shape
        if len(residual_map) != n:
            return candidates

        # Sort variables by residual magnitude (descending)
        sorted_vars = sorted(
            range(n), key=lambda v: -residual_map[v],
        )

        # Top 3 high-residual and bottom 3 low-residual
        high_vars = sorted_vars[:3]
        low_vars = sorted_vars[-3:]

        for var_high in high_vars:
            checks_high = sorted(np.where(H[:, var_high] == 1)[0].tolist())
            if len(checks_high) <= 1:
                continue

            check_idx = checks_high[rng.randint(0, len(checks_high))]

            for var_low in low_vars:
                if var_low == var_high:
                    continue
                result = _try_rewire(H, check_idx, var_high, check_idx, var_low)
                if result is not None:
                    candidates.append(result)
                    break

            if len(candidates) >= 3:
                break

        return candidates

    def _cycle_irregularity_reduction(
        self,
        H: np.ndarray,
        seed: int,
    ) -> list[np.ndarray]:
        """Generate candidate rewires reducing cycle degree-twist.

        Uses CycleTopologyAnalyzer to find high-twist cycles,
        then rewires an edge to reduce irregularity.
        """
        candidates: list[np.ndarray] = []
        rng = np.random.RandomState(seed)

        m, n = H.shape

        # Compute variable degrees
        col_degrees = H.sum(axis=0).astype(int)
        # Compute degree variance contribution per variable
        mean_deg = col_degrees.mean() if n > 0 else 0
        degree_deviation = np.abs(col_degrees - mean_deg)

        # Target highest-deviation variables for rewiring
        sorted_vars = sorted(
            range(n), key=lambda v: -degree_deviation[v],
        )

        high_dev_vars = sorted_vars[:3]
        low_dev_vars = sorted_vars[-3:]

        for var_high in high_dev_vars:
            if degree_deviation[var_high] == 0:
                continue

            checks = sorted(np.where(H[:, var_high] == 1)[0].tolist())
            if len(checks) <= 1:
                continue

            check_idx = checks[rng.randint(0, len(checks))]

            for var_low in low_dev_vars:
                if var_low == var_high:
                    continue
                result = _try_rewire(H, check_idx, var_high, check_idx, var_low)
                if result is not None:
                    candidates.append(result)
                    break

            if len(candidates) >= 3:
                break

        return candidates

    def _bethe_hessian_improvement(
        self,
        H: np.ndarray,
        seed: int,
    ) -> list[np.ndarray]:
        """Generate candidate rewires improving Bethe-Hessian stability.

        Identifies nodes contributing to the most unstable eigenvector
        and rewires their edges to reduce instability.
        """
        candidates: list[np.ndarray] = []
        rng = np.random.RandomState(seed)

        m, n = H.shape
        if n < 3:
            return candidates

        # Compute variable-node adjacency
        HtH = H.T @ H
        np.fill_diagonal(HtH, 0)
        A = (HtH > 0).astype(np.float64)

        degrees = A.sum(axis=1)
        avg_deg = degrees.mean() if n > 0 else 1.0
        r = max(np.sqrt(max(avg_deg - 1, 0.01)), 0.1)

        # Build Bethe Hessian: H_B = (r^2 - 1)I - rA + D
        D = np.diag(degrees)
        I = np.eye(n)
        H_B = (r**2 - 1) * I - r * A + D

        # Find smallest eigenvalue eigenvector
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(H_B)
        except np.linalg.LinAlgError:
            return candidates

        # Eigenvector for smallest eigenvalue
        min_idx = int(np.argmin(eigenvalues))
        v_min = np.abs(eigenvectors[:, min_idx])

        # Nodes with highest participation in unstable mode
        sorted_vars = sorted(range(n), key=lambda i: -v_min[i])
        high_participation = sorted_vars[:3]
        low_participation = sorted_vars[-3:]

        for var_high in high_participation:
            checks = sorted(np.where(H[:, var_high] == 1)[0].tolist())
            if len(checks) <= 1:
                continue

            check_idx = checks[rng.randint(0, len(checks))]

            for var_low in low_participation:
                if var_low == var_high:
                    continue
                result = _try_rewire(H, check_idx, var_high, check_idx, var_low)
                if result is not None:
                    candidates.append(result)
                    break

            if len(candidates) >= 3:
                break

        return candidates

    def _residual_cluster_smoothing(
        self,
        H: np.ndarray,
        seed: int,
    ) -> list[np.ndarray]:
        """Generate candidate rewires that improve cluster escape connectivity.

        Identifies residual clusters and rewires edges from cluster-interior
        variables to low-residual variables outside the cluster, improving
        the graph's ability to escape decoder failure basins.
        """
        candidates: list[np.ndarray] = []
        rng = np.random.RandomState(seed)

        m, n = H.shape

        # Compute residual map
        residuals = self._residual_analyzer.compute_residual_map(
            H, iterations=5, seed=seed,
        )
        residual_map = residuals.get("residual_map", np.array([]))

        if len(residual_map) == 0 or len(residual_map) != n:
            return candidates

        # Find clusters
        cluster_result = self._cluster_analyzer.find_clusters(
            H, residual_map=residual_map,
        )
        clusters = cluster_result.get("clusters", [])

        if not clusters:
            return candidates

        # Target the highest-risk cluster
        target = clusters[0]
        cluster_vars = set(target["variables"])

        # Variables outside cluster, sorted by residual ascending
        outside_vars = sorted(
            [vi for vi in range(n) if vi not in cluster_vars],
            key=lambda vi: residual_map[vi],
        )

        if not outside_vars:
            return candidates

        # Rank cluster variables by residual descending
        cluster_ranked = sorted(
            target["variables"],
            key=lambda vi: -residual_map[vi],
        )

        for var_high in cluster_ranked[:3]:
            checks = sorted(np.where(H[:, var_high] == 1)[0].tolist())
            if len(checks) <= 1:
                continue

            check_idx = checks[rng.randint(0, len(checks))]

            for var_low in outside_vars[:3]:
                if var_low == var_high:
                    continue
                result = _try_rewire(H, check_idx, var_high, check_idx, var_low)
                if result is not None:
                    candidates.append(result)
                    break

            if len(candidates) >= 3:
                break

        return candidates
