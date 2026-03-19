"""
v11.3.0 — Fitness Engine.

Computes weighted composite fitness scores from spectral metrics for
LDPC/QLDPC parity-check matrices.  Supports caching by matrix hash.

v11 extension: optional decoder-aware mode adds trapping-set penalty,
BP stability score, and Jacobian instability penalty.

v11.2 extension: adds Bethe Hessian stability score to decoder-aware
composite fitness.

v11.3 extension: adds absorbing-set risk and twisted-cycle fraction
to decoder-aware composite fitness.

Layer 3 — Fitness.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from qec.fitness.spectral_metrics import (
    compute_nbt_spectral_radius,
    compute_girth_spectrum,
    compute_ace_spectrum,
    estimate_eigenvector_ipr,
)


_ROUND = 12


def _matrix_hash(H: np.ndarray) -> str:
    """Compute a deterministic content hash for a parity-check matrix."""
    data = np.asarray(H, dtype=np.float64).tobytes()
    return hashlib.sha256(data).hexdigest()


class FitnessEngine:
    """Computes composite fitness scores for parity-check matrices.

    The composite fitness is a weighted combination of:
    - girth (maximize)
    - NBT spectral radius (minimize)
    - ACE spectrum variance (minimize)
    - expansion coefficient (maximize)
    - cycle density (minimize)
    - sparsity (maintain)

    Parameters
    ----------
    weights : dict[str, float] or None
        Optional custom weights for fitness components.
        Default weights are used if not provided.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        decoder_aware: bool = False,
        bp_trials: int = 50,
        bp_iterations: int = 10,
        seed: int = 42,
    ) -> None:
        self._weights = weights or {
            "girth": 3.0,
            "nbt_spectral_radius": -2.0,
            "ace_variance": -1.5,
            "expansion": 2.0,
            "cycle_density": -1.0,
            "sparsity": 0.5,
        }
        self._decoder_aware = decoder_aware
        self._bp_trials = bp_trials
        self._bp_iterations = bp_iterations
        self._seed = seed
        self._cache: dict[str, dict[str, Any]] = {}

        # Lazy-initialize decoder-aware components
        self._trapping_detector = None
        self._bp_probe = None

    def evaluate(self, H: np.ndarray) -> dict[str, Any]:
        """Compute composite fitness for a parity-check matrix.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``composite`` : float — weighted composite score
            - ``components`` : dict — individual weighted components
            - ``metrics`` : dict — raw metric values
        """
        H_arr = np.asarray(H, dtype=np.float64)
        h = _matrix_hash(H_arr)

        if h in self._cache:
            return self._cache[h]

        metrics = self._compute_metrics(H_arr)

        if self._decoder_aware:
            da_metrics = self._compute_decoder_aware_metrics(H_arr)
            metrics.update(da_metrics)

        components = self._compute_components(metrics)
        composite = sum(components.values())

        result = {
            "composite": round(composite, _ROUND),
            "components": {k: round(v, _ROUND) for k, v in sorted(components.items())},
            "metrics": {k: round(v, _ROUND) if isinstance(v, float) else v
                        for k, v in sorted(metrics.items())},
        }

        self._cache[h] = result
        return result

    def _compute_metrics(self, H: np.ndarray) -> dict[str, Any]:
        """Compute all raw metrics."""
        m, n = H.shape

        nbt_radius = compute_nbt_spectral_radius(H)
        girth_result = compute_girth_spectrum(H)
        ace = compute_ace_spectrum(H)
        ipr_result = estimate_eigenvector_ipr(H)

        # Expansion coefficient: ratio of unique 2-hop neighbours to
        # expected for a tree-like graph
        total_edges = float(H.sum())
        if total_edges > 0 and n > 0:
            avg_var_deg = total_edges / n
            avg_check_deg = total_edges / m if m > 0 else 0.0
            # Expected 2-hop reach for tree-like
            expected = avg_var_deg * (avg_check_deg - 1)
            # Actual average 2-hop reach
            HtH = H.T @ H
            np.fill_diagonal(HtH, 0)
            actual_reach = np.mean(np.sum(HtH > 0, axis=1))
            expansion = actual_reach / max(expected, 1.0)
        else:
            expansion = 0.0

        # Cycle density: total short cycles normalised by edges
        total_cycles = sum(girth_result["cycle_counts"].values())
        cycle_density = total_cycles / max(total_edges, 1.0)

        # Sparsity: fraction of non-zero entries
        sparsity = total_edges / max(m * n, 1)

        # ACE variance
        ace_var = float(np.var(ace)) if len(ace) > 0 else 0.0

        return {
            "nbt_spectral_radius": nbt_radius,
            "girth": girth_result["girth"],
            "cycle_counts": girth_result["cycle_counts"],
            "ace_variance": ace_var,
            "ace_min": float(np.min(ace)) if len(ace) > 0 else 0.0,
            "ace_mean": float(np.mean(ace)) if len(ace) > 0 else 0.0,
            "expansion": expansion,
            "cycle_density": cycle_density,
            "sparsity": sparsity,
            "mean_ipr": ipr_result["mean_ipr"],
            "max_ipr": ipr_result["max_ipr"],
        }

    def _compute_decoder_aware_metrics(self, H: np.ndarray) -> dict[str, Any]:
        """Compute decoder-aware metrics: trapping sets, BP stability, Jacobian, Bethe Hessian,
        absorbing-set risk, cycle topology."""
        from qec.analysis.trapping_sets import TrappingSetDetector
        from qec.analysis.bethe_hessian import BetheHessianAnalyzer
        from qec.analysis.absorbing_sets import AbsorbingSetPredictor
        from qec.analysis.cycle_topology import CycleTopologyAnalyzer
        from qec.decoder.stability_probe import BPStabilityProbe, estimate_bp_instability

        if self._trapping_detector is None:
            self._trapping_detector = TrappingSetDetector()
        if self._bp_probe is None:
            self._bp_probe = BPStabilityProbe(
                trials=self._bp_trials,
                iterations=self._bp_iterations,
                seed=self._seed,
            )

        ts_result = self._trapping_detector.detect(H)
        bp_result = self._bp_probe.probe(H)
        jac_result = estimate_bp_instability(H, seed=self._seed)

        # Bethe Hessian stability
        bh_analyzer = BetheHessianAnalyzer()
        bh_result = bh_analyzer.compute_bethe_hessian_stability(H)

        # Absorbing-set prediction (v11.3)
        abs_predictor = AbsorbingSetPredictor()
        abs_result = abs_predictor.predict(H)

        # Cycle topology (v11.3)
        ct_analyzer = CycleTopologyAnalyzer()
        ct_result = ct_analyzer.analyze(H)

        # Trapping-set penalty: normalized total count scaled by min size
        total_ts = ts_result["total"]
        min_size = ts_result["min_size"]
        n = H.shape[1]
        if n > 0 and total_ts > 0:
            trapping_set_penalty = total_ts / n * (1.0 / max(min_size, 1))
        else:
            trapping_set_penalty = 0.0

        # Jacobian instability penalty: how much rho exceeds 1.0
        jac_rho = jac_result["jacobian_spectral_radius_est"]
        jacobian_instability_penalty = max(0.0, jac_rho - 1.0)

        return {
            "trapping_set_total": total_ts,
            "trapping_set_min_size": min_size,
            "trapping_set_penalty": trapping_set_penalty,
            "bp_stability_score": bp_result["bp_stability_score"],
            "divergence_rate": bp_result["divergence_rate"],
            "stagnation_rate": bp_result["stagnation_rate"],
            "oscillation_score": bp_result["oscillation_score"],
            "jacobian_spectral_radius_est": jac_rho,
            "jacobian_instability_penalty": jacobian_instability_penalty,
            "bethe_hessian_min_eigenvalue": bh_result["bethe_hessian_min_eigenvalue"],
            "bethe_hessian_stability_score": bh_result["bethe_hessian_stability_score"],
            "absorbing_set_risk": abs_result["absorbing_set_risk"],
            "num_candidate_absorbing_sets": abs_result["num_candidate_absorbing_sets"],
            "min_candidate_absorbing_set_size": abs_result["min_candidate_size"],
            "twisted_cycle_fraction": ct_result["twisted_cycle_fraction"],
        }

    def _compute_components(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Compute weighted component scores from metrics."""
        w = self._weights

        components = {
            "girth": w.get("girth", 0.0) * float(metrics["girth"]),
            "nbt_spectral_radius": w.get("nbt_spectral_radius", 0.0) * metrics["nbt_spectral_radius"],
            "ace_variance": w.get("ace_variance", 0.0) * metrics["ace_variance"],
            "expansion": w.get("expansion", 0.0) * metrics["expansion"],
            "cycle_density": w.get("cycle_density", 0.0) * metrics["cycle_density"],
            "sparsity": w.get("sparsity", 0.0) * metrics["sparsity"],
        }

        if self._decoder_aware:
            # Decoder-aware components
            bp_score = metrics.get("bp_stability_score", 1.0)
            ts_penalty = metrics.get("trapping_set_penalty", 0.0)
            jac_penalty = metrics.get("jacobian_instability_penalty", 0.0)
            bh_stability = metrics.get("bethe_hessian_stability_score", 0.0)

            components["bp_stability"] = w.get("bp_stability", 2.0) * bp_score
            components["trapping_set_penalty"] = w.get("trapping_set_penalty", -3.0) * ts_penalty
            components["jacobian_instability_penalty"] = w.get("jacobian_instability_penalty", -1.5) * jac_penalty
            components["bethe_hessian_stability"] = w.get("bethe_hessian_stability", 1.5) * bh_stability

            # v11.3 absorbing-set and cycle topology components
            abs_risk = metrics.get("absorbing_set_risk", 0.0)
            twisted_frac = metrics.get("twisted_cycle_fraction", 0.0)
            components["absorbing_set_risk"] = w.get("absorbing_set_risk", -2.0) * abs_risk
            components["twisted_cycle_fraction"] = w.get("twisted_cycle_fraction", -1.0) * twisted_frac

        return components

    def clear_cache(self) -> None:
        """Clear the fitness evaluation cache."""
        self._cache.clear()
