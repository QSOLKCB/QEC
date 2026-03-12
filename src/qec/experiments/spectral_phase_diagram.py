"""
v12.5.0 — Spectral Instability Phase Diagram Generator.

Maps decoder behaviour across spectral space by running BP decoding
on a collection of Tanner graphs at multiple error rates, then
recording Frame Error Rate (FER) alongside spectral metrics.

Produces a phase diagram:
  x-axis → NB spectral radius
  y-axis → error rate
  color  → FER

Consumes:
  - NonBacktrackingFlowAnalyzer  (v12.0)
  - EigenvectorLocalizationAnalyzer  (v12.2)
  - FlowAlignmentAnalyzer  (v12.1)

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic: all
randomness via explicit seed injection, no global state, no
input mutation.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import numpy as np

from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from src.qec.analysis.eigenvector_localization import EigenvectorLocalizationAnalyzer
from src.qec.analysis.flow_alignment import FlowAlignmentAnalyzer
from src.qec.analysis.nb_trapping_set_predictor import NBTrappingSetPredictor
from src.qec.experiments.tanner_graph_repair import (
    _experimental_bp_flooding,
    _compute_syndrome,
)


_ROUND = 12


# ── Deterministic seed derivation ─────────────────────────────────


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


# ── Spectral metric extraction ────────────────────────────────────


def extract_spectral_metrics(H: np.ndarray) -> dict[str, Any]:
    """Extract NB spectral metrics from a parity-check matrix.

    Returns spectral_radius, IPR, and flow_alignment (set to 0.0
    when no residual map is available).

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: spectral_radius, ipr, flow_alignment.
    """
    H_arr = np.asarray(H, dtype=np.float64)

    analyzer = NonBacktrackingFlowAnalyzer()
    flow = analyzer.compute_flow(H_arr)

    ipr_result = EigenvectorLocalizationAnalyzer.compute_ipr(
        flow["variable_flow"],
    )

    trapping = NBTrappingSetPredictor().predict_trapping_regions(H_arr)

    return {
        "spectral_radius": round(float(flow["max_flow"]), _ROUND),
        "ipr": round(float(ipr_result["ipr"]), _ROUND),
        "flow_alignment": 0.0,
        "trapping_risk": round(float(trapping["risk_score"]), _ROUND),
    }


def extract_spectral_metrics_with_residual(
    H: np.ndarray,
    residual_map: np.ndarray,
) -> dict[str, Any]:
    """Extract NB spectral metrics including flow alignment.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    residual_map : np.ndarray
        BP residual magnitude per variable node, shape (n,).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: spectral_radius, ipr, flow_alignment.
    """
    H_arr = np.asarray(H, dtype=np.float64)

    analyzer = NonBacktrackingFlowAnalyzer()
    flow = analyzer.compute_flow(H_arr, residual_map=residual_map)

    ipr_result = EigenvectorLocalizationAnalyzer.compute_ipr(
        flow["variable_flow"],
    )

    alignment_score = 0.0
    if "flow_alignment" in flow:
        alignment_score = float(flow["flow_alignment"]["alignment_score"])

    trapping = NBTrappingSetPredictor().predict_trapping_regions(H_arr)

    return {
        "spectral_radius": round(float(flow["max_flow"]), _ROUND),
        "ipr": round(float(ipr_result["ipr"]), _ROUND),
        "flow_alignment": round(alignment_score, _ROUND),
        "trapping_risk": round(float(trapping["risk_score"]), _ROUND),
    }


# ── Decoder simulation ────────────────────────────────────────────


def _run_decoder_trial(
    H: np.ndarray,
    error_rate: float,
    trial_seed: int,
    max_iters: int = 100,
) -> dict[str, Any]:
    """Run a single deterministic BP decode trial.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    error_rate : float
        BSC channel error probability.
    trial_seed : int
        Deterministic seed for this trial.
    max_iters : int
        Maximum BP iterations.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: success, iterations, residual_norm.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    rng = np.random.default_rng(trial_seed)
    error_vector = (rng.random(n) < error_rate).astype(np.uint8)
    syndrome_vec = _compute_syndrome(H_arr, error_vector)

    p = error_rate
    if p <= 0.0 or p >= 1.0:
        p = 0.5
    log_ratio = math.log((1.0 - p) / p)
    llr = np.where(error_vector > 0, -log_ratio, log_ratio).astype(np.float64)

    correction, iterations, residual_norms = _experimental_bp_flooding(
        H_arr, llr, syndrome_vec, max_iters,
    )

    corrected_syndrome = _compute_syndrome(H_arr, correction)
    success = bool(np.array_equal(
        corrected_syndrome,
        syndrome_vec.astype(np.uint8),
    ))

    residual_norm = residual_norms[-1] if residual_norms else 0.0

    return {
        "success": success,
        "iterations": iterations,
        "residual_norm": round(float(residual_norm), _ROUND),
    }


# ── Phase diagram generator ───────────────────────────────────────


class SpectralPhaseDiagramGenerator:
    """Generate spectral instability phase diagrams.

    Maps decoder FER across spectral space by running BP decoding
    on a collection of Tanner graphs at multiple error rates.

    Parameters
    ----------
    max_iters : int
        Maximum BP iterations per trial (default 100).
    base_seed : int
        Base seed for deterministic trial generation (default 42).
    """

    def __init__(
        self,
        *,
        max_iters: int = 100,
        base_seed: int = 42,
    ) -> None:
        self.max_iters = max_iters
        self.base_seed = base_seed

    def generate_phase_diagram(
        self,
        graphs: list[np.ndarray],
        error_rates: list[float],
        trials_per_point: int,
    ) -> dict[str, Any]:
        """Generate a spectral instability phase diagram.

        For each graph, computes spectral metrics. For each
        (graph, error_rate) pair, runs ``trials_per_point`` decoder
        trials and records FER.

        Parameters
        ----------
        graphs : list[np.ndarray]
            List of binary parity-check matrices.
        error_rates : list[float]
            Channel error rates to sweep.
        trials_per_point : int
            Number of decode trials per (graph, error_rate) point.

        Returns
        -------
        dict[str, Any]
            JSON-serializable result with key ``points`` containing
            a list of phase diagram data points.
        """
        points: list[dict[str, Any]] = []

        for graph_idx, H in enumerate(graphs):
            H_arr = np.asarray(H, dtype=np.float64)

            # Extract spectral metrics once per graph.
            metrics = extract_spectral_metrics(H_arr)

            for error_rate in error_rates:
                failures = 0

                for trial_idx in range(trials_per_point):
                    trial_seed = _derive_seed(
                        self.base_seed,
                        f"graph_{graph_idx}_er_{error_rate}_trial_{trial_idx}",
                    )
                    result = _run_decoder_trial(
                        H_arr, error_rate, trial_seed, self.max_iters,
                    )
                    if not result["success"]:
                        failures += 1

                fer = failures / trials_per_point if trials_per_point > 0 else 0.0

                points.append({
                    "spectral_radius": metrics["spectral_radius"],
                    "error_rate": round(float(error_rate), _ROUND),
                    "FER": round(float(fer), _ROUND),
                    "fer": round(float(fer), _ROUND),
                    "IPR": metrics["ipr"],
                    "ipr": metrics["ipr"],
                    "flow_alignment": metrics["flow_alignment"],
                    "trapping_risk": metrics["trapping_risk"],
                })

        # Sort deterministically by (spectral_radius, error_rate).
        points.sort(key=lambda p: (p["spectral_radius"], p["error_rate"]))

        return {"points": points}
