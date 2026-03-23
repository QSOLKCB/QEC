"""Temporal spectral analysis over trajectories of phase maps.

Extends v86 single-snapshot spectral analysis to ordered sequences
of phase maps, extracting spectral evolution, convergence behaviour,
and temporal phase transitions.
"""

from typing import Any, Dict, List

import numpy as np

from .phase_spectral_analysis import run_phase_spectral_analysis


# -- per-step helpers ------------------------------------------------


def compute_spectral_drift(spectra: List[Dict[str, Any]]) -> List[float]:
    """L2 norm of eigenvalue difference between consecutive steps.

    Shorter eigenvalue vectors are zero-padded to match the longer one.
    """
    drift: List[float] = []
    for i in range(1, len(spectra)):
        prev = np.array(spectra[i - 1]["eigenvalues"], dtype=np.float64)
        curr = np.array(spectra[i]["eigenvalues"], dtype=np.float64)
        max_len = max(len(prev), len(curr))
        if len(prev) < max_len:
            prev = np.concatenate([prev, np.zeros(max_len - len(prev))])
        if len(curr) < max_len:
            curr = np.concatenate([curr, np.zeros(max_len - len(curr))])
        drift.append(float(np.linalg.norm(curr - prev)))
    return drift


def detect_temporal_transitions(
    drift: List[float], threshold: float = 1e-6
) -> List[Dict[str, Any]]:
    """Return time indices where spectral drift exceeds *threshold*."""
    transitions: List[Dict[str, Any]] = []
    for t, d in enumerate(drift):
        if d > threshold:
            transitions.append({"time_index": t, "drift": d})
    return transitions


def classify_trajectory(
    drift: List[float], lambda_max: List[float]
) -> str:
    """Classify trajectory as convergent / oscillatory / divergent.

    Rules (deterministic, applied in order):
    * If fewer than 2 drift values → ``"undetermined"``.
    * If drift is monotonically non-increasing → ``"convergent"``.
    * If drift is monotonically non-decreasing **and** not constant →
      ``"divergent"``.
    * Otherwise → ``"oscillatory"``.
    """
    if len(drift) < 2:
        return "undetermined"

    non_increasing = all(drift[i] >= drift[i + 1] for i in range(len(drift) - 1))
    if non_increasing:
        return "convergent"

    non_decreasing = all(drift[i] <= drift[i + 1] for i in range(len(drift) - 1))
    constant = all(drift[i] == drift[i + 1] for i in range(len(drift) - 1))
    if non_decreasing and not constant:
        return "divergent"

    return "oscillatory"


# -- main entry point ------------------------------------------------


def run_phase_trajectory_analysis(
    phase_maps: List[Dict[str, Any]],
    transition_threshold: float = 1e-6,
) -> Dict[str, Any]:
    """Analyse a trajectory of phase maps.

    Parameters
    ----------
    phase_maps:
        Ordered sequence of phase-map dicts (each with ``"nodes"`` and
        ``"edges"`` keys).
    transition_threshold:
        Drift value above which a temporal transition is recorded.

    Returns
    -------
    dict with keys:
        n_steps, spectral_trajectory, drift, lambda_max,
        rank_evolution, degeneracy_evolution, temporal_transitions,
        trajectory_type.
    """
    spectra: List[Dict[str, Any]] = []
    for pm in phase_maps:
        result = run_phase_spectral_analysis(pm)
        spectra.append(result["spectrum"])

    drift = compute_spectral_drift(spectra)
    lambda_max = [float(s["max_eigenvalue"]) for s in spectra]
    rank_evolution = [int(s["rank"]) for s in spectra]
    degeneracy_evolution = [int(s["n_degenerate_modes"]) for s in spectra]
    transitions = detect_temporal_transitions(drift, threshold=transition_threshold)
    trajectory_type = classify_trajectory(drift, lambda_max)

    return {
        "n_steps": len(phase_maps),
        "spectral_trajectory": spectra,
        "drift": drift,
        "lambda_max": lambda_max,
        "rank_evolution": rank_evolution,
        "degeneracy_evolution": degeneracy_evolution,
        "temporal_transitions": transitions,
        "trajectory_type": trajectory_type,
    }
