"""Temporal spectral analysis over trajectories of phase maps.

Extends v86 single-snapshot spectral analysis to ordered sequences
of phase maps, extracting spectral evolution, convergence behaviour,
and temporal phase transitions.
"""

from typing import Any, Dict, List

import numpy as np

from .phase_spectral_analysis import run_phase_spectral_analysis
from .phase_syndrome_analysis import run_syndrome_analysis
from .phase_syndrome_decoder import decode_syndrome_trajectory
from .phase_syndrome_geometry import run_syndrome_geometry_analysis
from .phase_geometric_dynamics import run_geometric_dynamics
from .phase_motif_graph import run_motif_graph_analysis
from .phase_trajectory_motifs import run_trajectory_motif_analysis
from .phase_resonance_analysis import run_resonance_analysis
from .phase_basin_analysis import run_basin_analysis
from .trajectory_clustering import run_trajectory_clustering


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

    # Syndrome analysis over phase-map nodes (one synthetic result per step).
    node_results = []
    for pm in phase_maps:
        for node in pm.get("nodes", []):
            node_results.append({
                "class": node.get("dominant_class", ""),
                "phase": node.get("dominant_phase", ""),
                "compatibility": node.get("mean_compatibility", 0.0),
                "score": node.get("mean_score", 0.0),
            })
    syndrome_analysis = run_syndrome_analysis(node_results)
    syndrome_geometry = run_syndrome_geometry_analysis(node_results)

    series = syndrome_geometry["ternary_series"]["encoded"]
    trajectory_motifs = run_trajectory_motif_analysis(series)
    motif_graph = run_motif_graph_analysis(series)

    resonance_analysis = run_resonance_analysis(
        series,
        drift,
        motif_graph["state_graph"],
    )

    basin_analysis = run_basin_analysis(
        motif_graph["state_graph"],
        resonance_analysis["attractor_field"],
    )

    # Build per-step trajectory for clustering: single trajectory = series.
    trajectory_states = {0: series}

    return {
        "n_steps": len(phase_maps),
        "spectral_trajectory": spectra,
        "drift": drift,
        "lambda_max": lambda_max,
        "rank_evolution": rank_evolution,
        "degeneracy_evolution": degeneracy_evolution,
        "temporal_transitions": transitions,
        "trajectory_type": trajectory_type,
        "syndrome_analysis": syndrome_analysis,
        "syndrome_decoder": decode_syndrome_trajectory(
            syndrome_analysis["series"]["encoded"],
            syndrome_analysis["transitions"],
        ),
        "syndrome_geometry": syndrome_geometry,
        "geometric_dynamics": run_geometric_dynamics(series),
        "trajectory_motifs": trajectory_motifs,
        "motif_graph": motif_graph,
        "resonance_analysis": resonance_analysis,
        "basin_analysis": basin_analysis,
        "trajectory_clustering": run_trajectory_clustering(
            trajectory_states,
            basin_analysis["mapping"],
        ),
    }
