"""
v8.0.0 — Stability Phase Diagram Experiment.

Maps BP stability regimes for Tanner graphs using a 2-D grid scan
over spectral radius and SIS (spectral instability score).

For each grid cell:
  1. Generate deterministic Tanner graph perturbations
  2. Compute spectral diagnostics
  3. Run deterministic BP decoding
  4. Record convergence results

Uses incremental NB spectrum updates when available to accelerate
evaluation.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic: no randomness,
no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from typing import Any

import numpy as np

from qec.diagnostics.spectral_nb import (
    _TannerGraph,
    _compute_eeec,
    _compute_sis,
    compute_nb_spectrum,
)
from qec.diagnostics._spectral_utils import compute_ipr
from qec.diagnostics.spectral_incremental import (
    update_nb_eigenpair_incremental,
)
from qec.diagnostics.spectral_repair import (
    apply_repair_candidate,
    propose_repair_candidates,
)
from qec.experiments.tanner_graph_repair import (
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


# ── Deterministic perturbation ────────────────────────────────────


def _generate_deterministic_perturbation(
    H: np.ndarray,
    seed: int,
    perturbation_index: int,
) -> np.ndarray:
    """Generate a deterministic Tanner graph perturbation.

    Applies a single degree-preserving edge swap to H, selected
    deterministically from the seed and perturbation index.

    Returns a copy — does not mutate H.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    sub_seed = _derive_seed(seed, f"perturbation_{perturbation_index}")

    # Find candidate edge swaps deterministically
    candidates = propose_repair_candidates(
        H_arr, top_k_edges=5, max_candidates=20,
    )

    if not candidates:
        return H_arr.copy()

    # Select candidate deterministically from seed
    idx = sub_seed % len(candidates)
    return apply_repair_candidate(H_arr, candidates[idx])


# ── Oscillation detection ─────────────────────────────────────────


def detect_metastable_bp_oscillation(
    residual_norms: list[float],
    *,
    window_size: int = 10,
    oscillation_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Detect metastable BP oscillation from residual norm history.

    Identifies BP states that oscillate rather than converge by
    checking for periodic patterns in the residual norm sequence.

    Parameters
    ----------
    residual_norms : list[float]
        Residual norms from BP iterations.
    window_size : int
        Window size for periodicity detection.
    oscillation_threshold : float
        Maximum norm difference to consider two values as matching.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``is_oscillatory`` : bool
        - ``oscillation_period`` : int (0 if not oscillatory)
        - ``mean_residual`` : float
        - ``residual_variance`` : float
    """
    n = len(residual_norms)
    if n < 2 * window_size:
        mean_r = sum(residual_norms) / n if n > 0 else 0.0
        var_r = (
            sum((r - mean_r) ** 2 for r in residual_norms) / n
            if n > 0 else 0.0
        )
        return {
            "is_oscillatory": False,
            "oscillation_period": 0,
            "mean_residual": round(mean_r, _ROUND),
            "residual_variance": round(var_r, _ROUND),
        }

    mean_r = sum(residual_norms) / n
    var_r = sum((r - mean_r) ** 2 for r in residual_norms) / n

    # Check for periodicity: try periods 2..window_size
    best_period = 0
    for period in range(2, window_size + 1):
        # Check if the last 2*period values show periodicity
        tail = residual_norms[-(2 * period):]
        matches = True
        for i in range(period):
            if abs(tail[i] - tail[i + period]) > oscillation_threshold:
                matches = False
                break
        if matches:
            best_period = period
            break

    is_oscillatory = best_period > 0 and residual_norms[-1] > oscillation_threshold

    return {
        "is_oscillatory": bool(is_oscillatory),
        "oscillation_period": best_period,
        "mean_residual": round(mean_r, _ROUND),
        "residual_variance": round(var_r, _ROUND),
    }


# ── Boundary estimation ──────────────────────────────────────────


def estimate_bp_stability_boundary(
    grid_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate the BP stability boundary from experiment grid results.

    Finds the boundary between converged and failed cells in the
    spectral_radius x SIS grid by identifying adjacent cells with
    different convergence outcomes.

    Parameters
    ----------
    grid_results : list[dict]
        List of grid cell results with spectral_radius, sis, and
        convergence_rate keys.

    Returns
    -------
    dict[str, Any]
        Boundary estimate with boundary_points and statistics.
    """
    if not grid_results:
        return {
            "boundary_points": [],
            "mean_boundary_spectral_radius": 0.0,
            "mean_boundary_sis": 0.0,
            "num_boundary_points": 0,
        }

    # Threshold: convergence_rate < 0.5 is "failed"
    boundary_points = []

    # Sort results by (spectral_radius, sis) for deterministic processing
    sorted_results = sorted(
        grid_results,
        key=lambda r: (r["spectral_radius_bin"], r["sis_bin"]),
    )

    # Build lookup by (sr_bin, sis_bin)
    cell_map: dict[tuple[int, int], dict] = {}
    for r in sorted_results:
        key = (r["spectral_radius_bin"], r["sis_bin"])
        cell_map[key] = r

    for r in sorted_results:
        sr_bin = r["spectral_radius_bin"]
        sis_bin = r["sis_bin"]
        conv_rate = r["convergence_rate"]

        # Check neighbors
        neighbors = [
            (sr_bin + 1, sis_bin),
            (sr_bin - 1, sis_bin),
            (sr_bin, sis_bin + 1),
            (sr_bin, sis_bin - 1),
        ]

        for nb_key in neighbors:
            nb = cell_map.get(nb_key)
            if nb is None:
                continue
            nb_conv = nb["convergence_rate"]
            # Boundary: one side converged, other side failed
            if (conv_rate >= 0.5) != (nb_conv >= 0.5):
                boundary_points.append({
                    "spectral_radius": round(
                        (r["spectral_radius_center"] +
                         nb["spectral_radius_center"]) / 2.0, _ROUND,
                    ),
                    "sis": round(
                        (r["sis_center"] + nb["sis_center"]) / 2.0, _ROUND,
                    ),
                })
                break  # One boundary point per cell

    # Deduplicate and sort
    seen: set[tuple[float, float]] = set()
    unique_points = []
    for bp in boundary_points:
        key = (bp["spectral_radius"], bp["sis"])
        if key not in seen:
            seen.add(key)
            unique_points.append(bp)
    unique_points.sort(key=lambda p: (p["spectral_radius"], p["sis"]))

    if unique_points:
        mean_sr = sum(p["spectral_radius"] for p in unique_points) / len(unique_points)
        mean_sis = sum(p["sis"] for p in unique_points) / len(unique_points)
    else:
        mean_sr = 0.0
        mean_sis = 0.0

    return {
        "boundary_points": unique_points,
        "mean_boundary_spectral_radius": round(mean_sr, _ROUND),
        "mean_boundary_sis": round(mean_sis, _ROUND),
        "num_boundary_points": len(unique_points),
    }


def predict_spectral_stability_boundary(
    grid_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate the stability boundary from NB spectral metrics only.

    Uses SIS threshold to predict where BP will fail, without
    running any decodes.  Compares spectral predictions against
    actual convergence data in the grid.

    Parameters
    ----------
    grid_results : list[dict]
        Grid cell results with sis_center and convergence_rate.

    Returns
    -------
    dict[str, Any]
        Predicted boundary with threshold and accuracy.
    """
    if not grid_results:
        return {
            "predicted_sis_threshold": 0.0,
            "prediction_accuracy": 0.0,
            "num_cells": 0,
        }

    # Find SIS threshold that best separates converged from failed
    sis_values = sorted(set(
        round(r["sis_center"], _ROUND) for r in grid_results
    ))

    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in sis_values:
        correct = 0
        for r in grid_results:
            predicted_fail = r["sis_center"] > threshold
            actual_fail = r["convergence_rate"] < 0.5
            if predicted_fail == actual_fail:
                correct += 1
        accuracy = correct / len(grid_results)
        if accuracy > best_accuracy or (
            accuracy == best_accuracy and threshold < best_threshold
        ):
            best_accuracy = accuracy
            best_threshold = threshold

    return {
        "predicted_sis_threshold": round(best_threshold, _ROUND),
        "prediction_accuracy": round(best_accuracy, _ROUND),
        "num_cells": len(grid_results),
    }


# ── Repair boundary tracking ─────────────────────────────────────


def track_repair_boundary_shift(
    H: np.ndarray,
    grid_results_before: list[dict[str, Any]],
    grid_results_after: list[dict[str, Any]],
) -> dict[str, Any]:
    """Track how the stability boundary moves after graph repair.

    Compares boundary estimates before and after a repair step.

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix (for reference).
    grid_results_before : list[dict]
        Grid results before repair.
    grid_results_after : list[dict]
        Grid results after repair.

    Returns
    -------
    dict[str, Any]
        Boundary shift metrics.
    """
    boundary_before = estimate_bp_stability_boundary(grid_results_before)
    boundary_after = estimate_bp_stability_boundary(grid_results_after)

    delta_sr = round(
        boundary_after["mean_boundary_spectral_radius"]
        - boundary_before["mean_boundary_spectral_radius"],
        _ROUND,
    )
    delta_sis = round(
        boundary_after["mean_boundary_sis"]
        - boundary_before["mean_boundary_sis"],
        _ROUND,
    )

    return {
        "boundary_before": boundary_before,
        "boundary_after": boundary_after,
        "delta_mean_spectral_radius": delta_sr,
        "delta_mean_sis": delta_sis,
        "boundary_expanded": delta_sr > 0 or delta_sis > 0,
    }


# ── Unstable subgraph logging ─────────────────────────────────────


def log_most_unstable_subgraph(
    H: np.ndarray,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Identify the most unstable subgraph using NB eigenvector energy.

    Finds nodes and edges most responsible for instability by
    ranking them by eigenvector energy concentration.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    top_k : int
        Number of top nodes/edges to report.

    Returns
    -------
    dict[str, Any]
        Dictionary with top unstable nodes and edges.
    """
    from qec.diagnostics.nb_energy_heatmap import compute_nb_energy_heatmap

    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    heatmap = compute_nb_energy_heatmap(H_arr)
    spectrum = compute_nb_spectrum(H_arr)

    # Rank variable nodes by heat
    var_heat = heatmap["variable_node_heat"]
    var_ranking = sorted(
        [(i, h) for i, h in enumerate(var_heat)],
        key=lambda x: (-x[1], x[0]),
    )[:top_k]

    # Rank check nodes by heat
    chk_heat = heatmap["check_node_heat"]
    chk_ranking = sorted(
        [(i, h) for i, h in enumerate(chk_heat)],
        key=lambda x: (-x[1], x[0]),
    )[:top_k]

    # Rank edges by energy
    edge_energy = spectrum["edge_energy"]
    edge_ranking = sorted(
        [(i, round(float(e), _ROUND)) for i, e in enumerate(edge_energy)],
        key=lambda x: (-x[1], x[0]),
    )[:top_k]

    return {
        "top_variable_nodes": [
            {"node": i, "heat": round(h, _ROUND)} for i, h in var_ranking
        ],
        "top_check_nodes": [
            {"node": i, "heat": round(h, _ROUND)} for i, h in chk_ranking
        ],
        "top_edges": [
            {"edge_index": i, "energy": e} for i, e in edge_ranking
        ],
        "spectral_radius": spectrum["spectral_radius"],
        "ipr": spectrum["ipr"],
        "sis": spectrum["sis"],
    }


# ── ASCII phase diagram visualization ─────────────────────────────


def _render_ascii_phase_diagram(
    grid_results: list[dict[str, Any]],
    grid_resolution: int,
) -> str:
    """Render an ASCII phase diagram from grid results.

    Legend:
      + = BP converged (convergence_rate >= 0.5)
      - = BP failed (convergence_rate < 0.5)
      o = oscillatory
      * = boundary cell
    """
    # Build grid map
    cell_map: dict[tuple[int, int], dict] = {}
    for r in grid_results:
        key = (r["spectral_radius_bin"], r["sis_bin"])
        cell_map[key] = r

    # Find boundary cells
    boundary = estimate_bp_stability_boundary(grid_results)
    boundary_set: set[tuple[int, int]] = set()
    for r in grid_results:
        sr_bin = r["spectral_radius_bin"]
        sis_bin = r["sis_bin"]
        neighbors = [
            (sr_bin + 1, sis_bin), (sr_bin - 1, sis_bin),
            (sr_bin, sis_bin + 1), (sr_bin, sis_bin - 1),
        ]
        for nb_key in neighbors:
            nb = cell_map.get(nb_key)
            if nb is None:
                continue
            if (r["convergence_rate"] >= 0.5) != (nb["convergence_rate"] >= 0.5):
                boundary_set.add((sr_bin, sis_bin))
                break

    lines = []
    lines.append("Stability Phase Diagram")
    lines.append(f"Grid: {grid_resolution}x{grid_resolution}")
    lines.append("Legend: + converged  - failed  o oscillatory  * boundary")
    lines.append("")

    # Header row: SIS bins
    header = "SR\\SIS "
    for j in range(grid_resolution):
        header += f"{j:2d}"
    lines.append(header)

    for i in range(grid_resolution):
        row = f"  {i:2d}   "
        for j in range(grid_resolution):
            key = (i, j)
            cell = cell_map.get(key)
            if cell is None:
                row += " ."
                continue
            if key in boundary_set:
                row += " *"
            elif cell.get("has_oscillation", False):
                row += " o"
            elif cell["convergence_rate"] >= 0.5:
                row += " +"
            else:
                row += " -"
        lines.append(row)

    lines.append("")
    lines.append(f"Boundary points: {boundary['num_boundary_points']}")

    return "\n".join(lines)


def scale_ascii_phase_diagram(
    diagram: str,
    max_width: int = 80,
) -> str:
    """Scale an ASCII phase diagram to fit within max_width columns."""
    lines = diagram.split("\n")
    scaled = []
    for line in lines:
        if len(line) > max_width:
            scaled.append(line[:max_width])
        else:
            scaled.append(line)
    return "\n".join(scaled)


def highlight_bp_critical_line(
    diagram: str,
    grid_results: list[dict[str, Any]],
) -> str:
    """Mark the BP critical boundary with '*' characters.

    This is already done in _render_ascii_phase_diagram, but this
    function can re-highlight an existing diagram by replacing
    boundary characters.
    """
    # Already handled in render; return as-is
    return diagram


def render_ascii_stability_boundary(
    grid_results: list[dict[str, Any]],
    grid_resolution: int,
    critical_radius: float,
) -> str:
    """Render ASCII phase diagram with estimated critical radius marked.

    Marks the column corresponding to the critical spectral radius
    with '|' characters in the grid.

    Parameters
    ----------
    grid_results : list[dict]
        Grid cell results from phase diagram experiment.
    grid_resolution : int
        Number of bins along each axis.
    critical_radius : float
        Estimated critical spectral radius.

    Returns
    -------
    str
        ASCII phase diagram with stability boundary marked.
    """
    cell_map: dict[tuple[int, int], dict] = {}
    sr_centers: list[float] = []
    for r in grid_results:
        key = (r["spectral_radius_bin"], r["sis_bin"])
        cell_map[key] = r
        sr_centers.append(r.get("spectral_radius_center", 0.0))

    # Find the SR bin closest to critical_radius
    critical_bin = 0
    min_dist = float("inf")
    for r in grid_results:
        dist = abs(r.get("spectral_radius_center", 0.0) - critical_radius)
        if dist < min_dist:
            min_dist = dist
            critical_bin = r["spectral_radius_bin"]

    lines = []
    lines.append("Stability Phase Diagram (with critical boundary)")
    lines.append(f"Grid: {grid_resolution}x{grid_resolution}")
    lines.append("Legend: + converged  - failed  | boundary")
    lines.append("")

    header = "SR\\SIS "
    for j in range(grid_resolution):
        header += f"{j:2d}"
    lines.append(header)

    for i in range(grid_resolution):
        row = f"  {i:2d}   "
        for j in range(grid_resolution):
            key = (i, j)
            cell = cell_map.get(key)
            if i == critical_bin:
                row += " |"
            elif cell is None:
                row += " ."
            elif cell["convergence_rate"] >= 0.5:
                row += " +"
            else:
                row += " -"
        lines.append(row)

    lines.append("")
    lines.append(f"Critical radius: {critical_radius:.6f}")

    return "\n".join(lines)


# ── Spectral trajectory tracking ──────────────────────────────────


def _record_spectral_trajectory(
    H: np.ndarray,
    perturbation_steps: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Record spectral trajectory during perturbation sequence.

    Tracks lambda_NB(t), IPR(t), SIS(t), and decoder convergence
    over a sequence of perturbation steps.

    Uses incremental NB updates when possible.
    """
    trajectory = []
    H_current = np.asarray(H, dtype=np.float64).copy()
    previous_eigenvector = None

    for step in range(perturbation_steps):
        # Compute spectrum (incremental if possible)
        if previous_eigenvector is not None:
            incr = update_nb_eigenpair_incremental(
                H_current, previous_eigenvector,
            )
            if incr["converged"]:
                spectral_radius = incr["spectral_radius"]
                eigenvector = incr["eigenvector"]
                ipr = round(float(compute_ipr(eigenvector)), _ROUND)
                edge_energy = np.abs(eigenvector) ** 2
                eeec = round(float(_compute_eeec(edge_energy)), _ROUND)
                sis = round(float(_compute_sis(spectral_radius, ipr, eeec)), _ROUND)
            else:
                spectrum = compute_nb_spectrum(H_current)
                spectral_radius = spectrum["spectral_radius"]
                eigenvector = spectrum["eigenvector"]
                ipr = spectrum["ipr"]
                sis = spectrum["sis"]
        else:
            spectrum = compute_nb_spectrum(H_current)
            spectral_radius = spectrum["spectral_radius"]
            eigenvector = spectrum["eigenvector"]
            ipr = spectrum["ipr"]
            sis = spectrum["sis"]

        previous_eigenvector = eigenvector

        trajectory.append({
            "step": step,
            "spectral_radius": round(float(spectral_radius), _ROUND),
            "ipr": round(float(ipr), _ROUND),
            "sis": round(float(sis), _ROUND),
        })

        # Apply next perturbation
        step_seed = _derive_seed(base_seed, f"trajectory_{step}")
        H_current = _generate_deterministic_perturbation(
            H_current, step_seed, step,
        )

    return trajectory


# ── Main experiment ───────────────────────────────────────────────


def run_stability_phase_diagram_experiment(
    H: np.ndarray,
    grid_resolution: int = 20,
    perturbations_per_cell: int = 10,
    *,
    base_seed: int = 42,
    max_iters: int = 100,
    p: float = 0.05,
    snapshot_interval: int = 0,
    snapshot_dir: str = "artifacts/phase_diagram_snapshots",
) -> dict[str, Any]:
    """Run the stability phase diagram experiment.

    Maps BP stability regimes on a 2-D grid of spectral radius x SIS.
    For each grid cell, generates deterministic perturbations, computes
    spectral diagnostics, runs BP decoding, and records convergence.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    grid_resolution : int
        Number of bins along each axis.
    perturbations_per_cell : int
        Number of perturbations to evaluate per grid cell.
    base_seed : int
        Base seed for deterministic perturbation generation.
    max_iters : int
        Maximum BP iterations.
    p : float
        Channel error probability for LLR generation.
    snapshot_interval : int
        If > 0, save ASCII snapshots every this many perturbations.
    snapshot_dir : str
        Directory for ASCII snapshot files.

    Returns
    -------
    dict[str, Any]
        JSON-serializable experiment artifact.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    # Compute baseline spectrum
    baseline_spectrum = compute_nb_spectrum(H_arr)
    baseline_sr = baseline_spectrum["spectral_radius"]
    baseline_sis = baseline_spectrum["sis"]

    # Define grid ranges based on baseline with margins
    sr_min = max(0.0, baseline_sr * 0.5)
    sr_max = baseline_sr * 2.0
    sis_min = 0.0
    sis_max = max(baseline_sis * 3.0, 0.1)

    sr_step = (sr_max - sr_min) / grid_resolution if grid_resolution > 0 else 1.0
    sis_step = (sis_max - sis_min) / grid_resolution if grid_resolution > 0 else 1.0

    # Generate perturbations and classify into grid cells
    grid_results: list[dict[str, Any]] = []
    spectral_trajectories: list[dict[str, Any]] = []

    # Track per-cell data
    cell_data: dict[tuple[int, int], list[dict]] = {}

    total_perturbations = grid_resolution * perturbations_per_cell
    previous_eigenvector = baseline_spectrum["eigenvector"]

    snapshot_count = 0

    for perturbation_idx in range(total_perturbations):
        # Generate deterministic perturbation
        cell_seed = _derive_seed(base_seed, f"cell_{perturbation_idx}")
        H_perturbed = _generate_deterministic_perturbation(
            H_arr, cell_seed, perturbation_idx,
        )

        # Compute spectrum (try incremental first)
        incr = update_nb_eigenpair_incremental(
            H_perturbed, previous_eigenvector,
        )
        if incr["converged"]:
            sr = incr["spectral_radius"]
            eigenvector = incr["eigenvector"]
            ipr_val = round(float(compute_ipr(eigenvector)), _ROUND)
            edge_energy = np.abs(eigenvector) ** 2
            eeec_val = round(float(_compute_eeec(edge_energy)), _ROUND)
            sis_val = round(float(_compute_sis(sr, ipr_val, eeec_val)), _ROUND)
        else:
            spectrum = compute_nb_spectrum(H_perturbed)
            sr = spectrum["spectral_radius"]
            eigenvector = spectrum["eigenvector"]
            ipr_val = spectrum["ipr"]
            eeec_val = spectrum["eeec"]
            sis_val = spectrum["sis"]

        previous_eigenvector = eigenvector

        # Generate deterministic LLR and syndrome
        llr_seed = _derive_seed(cell_seed, "llr")
        rng = np.random.RandomState(llr_seed)
        error_vector = (rng.random(n) < p).astype(np.uint8)
        syndrome_vec = _compute_syndrome(H_perturbed, error_vector)
        llr = np.where(
            error_vector > 0,
            -math.log((1 - p) / p),
            math.log((1 - p) / p),
        ) * np.ones(n, dtype=np.float64)
        # Use channel model: positive LLR = likely 0, negative = likely 1
        llr = np.where(error_vector > 0, -abs(llr[0]), abs(llr[0]))

        # Run BP decoding
        correction, iterations, residual_norms = _experimental_bp_flooding(
            H_perturbed, llr, syndrome_vec, max_iters,
        )

        converged = bool(np.array_equal(
            _compute_syndrome(H_perturbed, correction),
            syndrome_vec.astype(np.uint8),
        ))

        # Detect oscillation
        osc = detect_metastable_bp_oscillation(residual_norms)

        # Compute residual norm
        residual_norm = residual_norms[-1] if residual_norms else 0.0

        # Classify into grid cell
        sr_bin = min(
            int((sr - sr_min) / sr_step) if sr_step > 0 else 0,
            grid_resolution - 1,
        )
        sis_bin = min(
            int((sis_val - sis_min) / sis_step) if sis_step > 0 else 0,
            grid_resolution - 1,
        )
        sr_bin = max(0, sr_bin)
        sis_bin = max(0, sis_bin)

        trial_result = {
            "spectral_radius": round(float(sr), _ROUND),
            "ipr": round(float(ipr_val), _ROUND),
            "eeec": round(float(eeec_val), _ROUND),
            "sis": round(float(sis_val), _ROUND),
            "decoder_converged": converged,
            "iterations_to_converge": iterations,
            "residual_norm": round(residual_norm, _ROUND),
            "has_oscillation": osc["is_oscillatory"],
            "oscillation_period": osc["oscillation_period"],
        }

        cell_key = (sr_bin, sis_bin)
        cell_data.setdefault(cell_key, []).append(trial_result)

        # Record trajectory point
        spectral_trajectories.append({
            "step": perturbation_idx,
            "spectral_radius": round(float(sr), _ROUND),
            "ipr": round(float(ipr_val), _ROUND),
            "sis": round(float(sis_val), _ROUND),
            "decoder_converged": converged,
        })

        # Snapshot
        if snapshot_interval > 0 and (perturbation_idx + 1) % snapshot_interval == 0:
            _save_snapshot(
                cell_data, grid_resolution, sr_min, sr_step,
                sis_min, sis_step, snapshot_dir, perturbation_idx + 1,
            )
            snapshot_count += 1

    # Aggregate cell data into grid results
    for (sr_bin, sis_bin), trials in sorted(cell_data.items()):
        num_converged = sum(1 for t in trials if t["decoder_converged"])
        num_oscillatory = sum(1 for t in trials if t["has_oscillation"])
        convergence_rate = num_converged / len(trials) if trials else 0.0

        mean_iters = (
            sum(t["iterations_to_converge"] for t in trials) / len(trials)
            if trials else 0.0
        )
        mean_residual = (
            sum(t["residual_norm"] for t in trials) / len(trials)
            if trials else 0.0
        )

        grid_results.append({
            "spectral_radius_bin": sr_bin,
            "sis_bin": sis_bin,
            "spectral_radius_center": round(
                sr_min + (sr_bin + 0.5) * sr_step, _ROUND,
            ),
            "sis_center": round(
                sis_min + (sis_bin + 0.5) * sis_step, _ROUND,
            ),
            "num_trials": len(trials),
            "convergence_rate": round(convergence_rate, _ROUND),
            "mean_iterations": round(mean_iters, _ROUND),
            "mean_residual_norm": round(mean_residual, _ROUND),
            "num_oscillatory": num_oscillatory,
            "has_oscillation": num_oscillatory > 0,
        })

    # Sort grid results deterministically
    grid_results.sort(
        key=lambda r: (r["spectral_radius_bin"], r["sis_bin"]),
    )

    # Boundary estimation
    measured_boundary = estimate_bp_stability_boundary(grid_results)
    predicted_boundary = predict_spectral_stability_boundary(grid_results)

    # ASCII visualization
    ascii_diagram = _render_ascii_phase_diagram(grid_results, grid_resolution)
    ascii_diagram = scale_ascii_phase_diagram(ascii_diagram)
    ascii_diagram = highlight_bp_critical_line(ascii_diagram, grid_results)

    # Unstable subgraph
    unstable_subgraph = log_most_unstable_subgraph(H_arr)

    return {
        "schema_version": "8.0.0",
        "grid_resolution": grid_resolution,
        "perturbations_per_cell": perturbations_per_cell,
        "total_perturbations": total_perturbations,
        "base_seed": base_seed,
        "baseline_spectral_radius": round(float(baseline_sr), _ROUND),
        "baseline_sis": round(float(baseline_sis), _ROUND),
        "grid_results": grid_results,
        "spectral_trajectories": spectral_trajectories,
        "measured_boundary": measured_boundary,
        "predicted_boundary": predicted_boundary,
        "ascii_phase_diagram": ascii_diagram,
        "unstable_subgraph": unstable_subgraph,
        "num_snapshots_saved": snapshot_count,
    }


def _save_snapshot(
    cell_data: dict[tuple[int, int], list[dict]],
    grid_resolution: int,
    sr_min: float,
    sr_step: float,
    sis_min: float,
    sis_step: float,
    snapshot_dir: str,
    step: int,
) -> None:
    """Save an ASCII phase diagram snapshot to disk."""
    # Build temporary grid results for rendering
    temp_results = []
    for (sr_bin, sis_bin), trials in sorted(cell_data.items()):
        num_converged = sum(1 for t in trials if t["decoder_converged"])
        convergence_rate = num_converged / len(trials) if trials else 0.0
        num_oscillatory = sum(1 for t in trials if t["has_oscillation"])

        temp_results.append({
            "spectral_radius_bin": sr_bin,
            "sis_bin": sis_bin,
            "spectral_radius_center": round(
                sr_min + (sr_bin + 0.5) * sr_step, _ROUND,
            ),
            "sis_center": round(
                sis_min + (sis_bin + 0.5) * sis_step, _ROUND,
            ),
            "convergence_rate": round(convergence_rate, _ROUND),
            "has_oscillation": num_oscillatory > 0,
        })

    diagram = _render_ascii_phase_diagram(temp_results, grid_resolution)
    diagram = scale_ascii_phase_diagram(diagram)

    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, f"phase_diagram_step_{step}.txt")
    with open(path, "w") as f:
        f.write(diagram)
        f.write("\n")


# ── Artifact serialization ────────────────────────────────────────


def serialize_phase_diagram_artifact(
    result: dict[str, Any],
    output_path: str = "artifacts/stability_phase_diagram.json",
) -> str:
    """Serialize stability phase diagram results to JSON.

    Parameters
    ----------
    result : dict
        Output from run_stability_phase_diagram_experiment.
    output_path : str
        Path for the JSON output file.

    Returns
    -------
    str
        The serialized JSON string.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    json_str = json.dumps(result, sort_keys=True, separators=(",", ":"))

    with open(output_path, "w") as f:
        f.write(json_str)
        f.write("\n")

    return json_str
