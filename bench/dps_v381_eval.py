"""
v4.0.0 — BP Free-Energy Landscape Diagnostics.

Extends v3.9.1 harness with:
  - Energy landscape classification per trial
  - Geometry-induced basin switching detection
  - Aggregate basin statistics per mode

Evaluates Distance Penalty Slope (DPS) across fourteen modes:
  baseline           — flooding, all interventions disabled
  rpc_only           — flooding, RPC augmentation
  geom_v1_only       — geom_v1 schedule
  rpc_geom           — geom_v1 + RPC
  centered           — centered field projection
  prior              — pseudo-prior injection
  centered_prior     — centered field + pseudo-prior
  geom_centered      — geom_v1 + centered field
  geom_centered_prior — geom_v1 + centered field + pseudo-prior
  rpc_centered       — RPC + centered field
  rpc_centered_prior — RPC + centered field + pseudo-prior
  centered_strong    — centered field + geometry_strength=2.0
  centered_normalized — centered field + normalize_geometry
  centered_prior_normalized — centered + prior + normalize_geometry

FER uses syndrome-consistency semantics:
  frame_error := syndrome(H, correction) != syndrome(H, error)

Deterministic: fixed seed, pre-generated error instances reused across
all modes.  No benchmark caching.  Deterministic loop ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on Python path so `src.*` imports resolve.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code
from qec.decoder.rpc import RPCConfig, StructuralConfig, build_rpc_augmented_system
from qec.decoder.decoder_interface import get_decoder

from qec.channel.geometry import (
    centered_syndrome_field,
    syndrome_field,
    pseudo_prior_bias,
    apply_pseudo_prior,
)

from qec.channel.geometry_post import apply_geometry_postprocessing

from qec.diagnostics.energy_landscape import (
    classify_energy_landscape,
    classify_basin_switch,
    compute_landscape_metrics,
    detect_basin_switch,
)

from qec.diagnostics.iteration_trace import (
    compute_iteration_trace_metrics,
)

from qec.diagnostics.bp_dynamics import (
    compute_bp_dynamics_metrics,
)

from qec.diagnostics.bp_regime_trace import (
    compute_bp_regime_trace,
)

from qec.diagnostics.bp_phase_diagram import (
    compute_bp_phase_diagram,
)

from qec.diagnostics.bp_freeze_detection import (
    compute_bp_freeze_detection,
)

from qec.diagnostics.bp_fixed_point_analysis import (
    compute_bp_fixed_point_analysis,
)

from qec.diagnostics.bp_basin_analysis import (
    compute_bp_basin_analysis,
)

from qec.diagnostics.bp_barrier_analysis import (
    compute_bp_barrier_analysis,
)

from qec.diagnostics.bp_boundary_analysis import (
    compute_bp_boundary_analysis,
)

from qec.diagnostics.spectral_boundary_alignment import (
    compute_spectral_boundary_alignment,
)
from qec.diagnostics.spectral_trapping_sets import (
    compute_spectral_trapping_sets,
)

from qec.diagnostics.bp_phase_space import (
    compute_bp_phase_space,
    compute_metastability_score,
)

from qec.diagnostics.ternary_decoder_topology import (
    compute_ternary_decoder_topology,
)

from qec.diagnostics.basin_probe import (
    probe_local_ternary_basin,
)

from qec.diagnostics.phase_diagram import (
    build_decoder_phase_diagram,
    make_phase_grid,
)

from qec.diagnostics.phase_boundary_analysis import (
    analyze_phase_boundaries,
)

from qec.diagnostics.non_backtracking_spectrum import (
    compute_non_backtracking_spectrum,
)
from qec.diagnostics.bethe_hessian import (
    compute_bethe_hessian,
)
from qec.diagnostics.bp_stability_proxy import (
    estimate_bp_stability,
)
from qec.diagnostics.bp_jacobian_estimator import (
    estimate_bp_jacobian_spectral_radius,
)
from qec.diagnostics.phase_heatmap import (
    print_phase_heatmap,
)
from qec.diagnostics.spectral_bp_alignment import (
    compute_spectral_bp_alignment,
)
from qec.diagnostics.spectral_failure_risk import (
    compute_spectral_failure_risk,
)
from qec.diagnostics.bp_stability_predictor import (
    compute_bp_stability_prediction,
)
from qec.experiments.bp_prediction_validation import (
    run_bp_prediction_validation,
)
from qec.experiments.spectral_decoder_controller import (
    run_spectral_decoder_control_experiment,
)
from qec.experiments.spectral_instability_phase_map import (
    compute_spectral_instability_score,
    run_spectral_phase_map_experiment,
    compute_phase_map_aggregate_metrics,
)
from qec.experiments.spectral_graph_repair_loop import (
    run_spectral_graph_repair_loop,
    compute_repair_loop_aggregate_metrics,
)
from qec.experiments.spectral_graph_design_rules import (
    run_spectral_graph_design_analysis,
)
from qec.experiments.spectral_graph_optimizer import (
    run_spectral_graph_optimization,
)
from qec.experiments.spectral_optimizer_sanity_experiment import (
    run_spectral_optimizer_sanity_experiment,
    print_sanity_report,
)

# ── Mode definitions ─────────────────────────────────────────────────

_RPC_ON = RPCConfig(enabled=True, max_rows=64, w_min=2, w_max=32)
_RPC_OFF = RPCConfig(enabled=False)

MODES: dict[str, dict[str, Any]] = {
    "baseline": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=_RPC_OFF),
    },
    "rpc_only": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=_RPC_ON),
    },
    "geom_v1_only": {
        "schedule": "geom_v1",
        "structural": StructuralConfig(rpc=_RPC_OFF),
    },
    "rpc_geom": {
        "schedule": "geom_v1",
        "structural": StructuralConfig(rpc=_RPC_ON),
    },
    # ── v3.9.0 channel-geometry modes ──
    "centered": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=_RPC_OFF, centered_field=True),
    },
    "prior": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=_RPC_OFF, pseudo_prior=True),
    },
    "centered_prior": {
        "schedule": "flooding",
        "structural": StructuralConfig(
            rpc=_RPC_OFF, centered_field=True, pseudo_prior=True,
        ),
    },
    "geom_centered": {
        "schedule": "geom_v1",
        "structural": StructuralConfig(rpc=_RPC_OFF, centered_field=True),
    },
    "geom_centered_prior": {
        "schedule": "geom_v1",
        "structural": StructuralConfig(
            rpc=_RPC_OFF, centered_field=True, pseudo_prior=True,
        ),
    },
    "rpc_centered": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=_RPC_ON, centered_field=True),
    },
    "rpc_centered_prior": {
        "schedule": "flooding",
        "structural": StructuralConfig(
            rpc=_RPC_ON, centered_field=True, pseudo_prior=True,
        ),
    },
    # ── v3.9.1 geometry field control modes ──
    "centered_strong": {
        "schedule": "flooding",
        "structural": StructuralConfig(
            rpc=_RPC_OFF, centered_field=True,
            pseudo_prior=False, geometry_strength=2.0,
        ),
    },
    "centered_normalized": {
        "schedule": "flooding",
        "structural": StructuralConfig(
            rpc=_RPC_OFF, centered_field=True,
            normalize_geometry=True,
        ),
    },
    "centered_prior_normalized": {
        "schedule": "flooding",
        "structural": StructuralConfig(
            rpc=_RPC_OFF, centered_field=True,
            pseudo_prior=True, normalize_geometry=True,
        ),
    },
}

MODE_ORDER = [
    "baseline", "rpc_only", "geom_v1_only", "rpc_geom",
    "centered", "prior", "centered_prior",
    "geom_centered", "geom_centered_prior",
    "rpc_centered", "rpc_centered_prior",
    "centered_strong", "centered_normalized", "centered_prior_normalized",
]

# ── Default parameters ───────────────────────────────────────────────

DEFAULT_SEED = 42
DEFAULT_DISTANCES = [3, 5, 7]
DEFAULT_P_VALUES = [0.01, 0.015, 0.02]
DEFAULT_TRIALS = 200
DEFAULT_MAX_ITERS = 50
DEFAULT_BP_MODE = "min_sum"


# ── Helpers ──────────────────────────────────────────────────────────

def _array_checksum(arr: np.ndarray) -> str:
    """Deterministic SHA-256 hex digest (first 12 chars) of an array."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


def _pre_generate_instances(
    H: np.ndarray,
    p: float,
    trials: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Pre-generate error instances for reuse across modes."""
    n = H.shape[1]
    instances = []
    for _ in range(trials):
        e = (rng.random(n) < p).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, p)
        instances.append({"e": e, "s": s, "llr": llr})
    return instances


# ── Per-trial activation audit record ────────────────────────────────

def _trial_audit(
    H_original: np.ndarray,
    H_used: np.ndarray,
    s_original: np.ndarray,
    s_used: np.ndarray,
    iters: int,
    s_weight: int,
    e_weight: int,
) -> dict[str, Any]:
    """Collect per-trial activation audit data."""
    original_rows = H_original.shape[0]
    augmented_rows = H_used.shape[0]
    added_rows = augmented_rows - original_rows
    return {
        "original_rows": original_rows,
        "augmented_rows": augmented_rows,
        "added_rows": added_rows,
        "H_checksum": _array_checksum(H_used),
        "syndrome_checksum": _array_checksum(s_used),
        "iter_count": iters,
        "syndrome_weight": s_weight,
        "error_weight": e_weight,
    }


def _summarize_audit(trials_audit: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from per-trial audit records."""
    added = [t["added_rows"] for t in trials_audit]
    augmented = [t["augmented_rows"] for t in trials_audit]
    iters_list = [t["iter_count"] for t in trials_audit]
    syndrome_weights = [t["syndrome_weight"] for t in trials_audit]

    n_trials = len(trials_audit)
    fraction_iters_eq_1 = sum(1 for i in iters_list if i == 1) / n_trials if n_trials else 0.0
    fraction_zero_syndrome = sum(1 for w in syndrome_weights if w == 0) / n_trials if n_trials else 0.0

    return {
        "added_rows_min": min(added),
        "added_rows_mean": sum(added) / n_trials,
        "added_rows_max": max(added),
        "augmented_rows_min": min(augmented),
        "augmented_rows_mean": sum(augmented) / n_trials,
        "augmented_rows_max": max(augmented),
        "H_checksum_first": trials_audit[0]["H_checksum"],
        "H_checksum_last": trials_audit[-1]["H_checksum"],
        "syndrome_checksum_first": trials_audit[0]["syndrome_checksum"],
        "syndrome_checksum_last": trials_audit[-1]["syndrome_checksum"],
        "mean_iters": sum(iters_list) / n_trials,
        "max_iters_observed": max(iters_list),
        "fraction_iters_eq_1": fraction_iters_eq_1,
        "fraction_zero_syndrome": fraction_zero_syndrome,
    }


# ── Single mode evaluation ──────────────────────────────────────────

def run_mode(
    mode_name: str,
    H: np.ndarray,
    instances: list[dict[str, Any]],
    max_iters: int = DEFAULT_MAX_ITERS,
    bp_mode: str = DEFAULT_BP_MODE,
    enable_energy_trace: bool = False,
    enable_landscape: bool = False,
    enable_iteration_diagnostics: bool = False,
    enable_bp_dynamics: bool = False,
    enable_bp_transitions: bool = False,
    enable_bp_phase_diagram: bool = False,
    enable_bp_freeze_detection: bool = False,
    decoder_fn=None,
    compare_decoders: bool = False,
    paired_seed: bool = False,
    paired_errors: bool = False,
    enable_bp_fixed_point_analysis: bool = False,
    enable_bp_basin_analysis: bool = False,
    enable_bp_landscape_map: bool = False,
    enable_bp_barrier_analysis: bool = False,
    enable_bp_boundary_analysis: bool = False,
    enable_bp_phase_space: bool = False,
    enable_ternary_topology: bool = False,
    enable_ternary_transition_metrics: bool = False,
    enable_ternary_basin_probe: bool = False,
    enable_bp_prediction_validation: bool = False,
) -> dict[str, Any]:
    """Run a single mode over pre-generated instances.

    Returns dict with keys: fer, frame_errors, trials, audit_summary.
    When *enable_energy_trace* or *enable_landscape* is True, also
    returns ``energy_traces``.  When *enable_landscape* is True, also
    returns ``landscape_metrics``, ``basin_switch``, and
    ``energy_delta``.  When *enable_iteration_diagnostics* is True,
    also returns ``iteration_diagnostics``.  When *enable_bp_dynamics*
    is True, also returns ``bp_dynamics``.  When
    *enable_bp_transitions* is True, also returns
    ``bp_regime_trace``, ``bp_transition_summary``, and
    ``bp_transition_counts``.  When *enable_bp_freeze_detection* is
    True, also returns ``bp_freeze_detection``.  When
    *enable_bp_fixed_point_analysis* is True, also returns
    ``bp_fixed_point_analysis`` and ``bp_fixed_point_summary``.
    When *enable_bp_basin_analysis* is True, also returns
    ``bp_basin_analysis`` and ``bp_basin_summary``.
    When *enable_bp_landscape_map* is True, also returns
    ``bp_landscape_map`` and ``bp_landscape_summary``.
    When *enable_bp_barrier_analysis* is True, also returns
    ``bp_barrier_analysis`` and ``bp_barrier_summary``.
    When *enable_bp_boundary_analysis* is True, also returns
    ``bp_boundary_analysis`` and ``bp_boundary_summary``.
    When *enable_bp_phase_space* is True, also returns
    ``bp_phase_space``.
    When *enable_ternary_topology* is True, also returns
    ``ternary_topology``.
    When *enable_ternary_transition_metrics* is True, transition
    metrics are included in ternary topology output.
    When *enable_ternary_basin_probe* is True, also returns
    ``ternary_basin_probe``.
    """
    if decoder_fn is None:
        decoder_fn = bp_decode

    mode_cfg = MODES[mode_name]
    schedule = mode_cfg["schedule"]
    structural: StructuralConfig = mode_cfg["structural"]

    # v5.8.0: basin probe implies ternary topology.
    if enable_ternary_basin_probe:
        enable_ternary_topology = True
    # v5.8.0: transition metrics implies ternary topology.
    if enable_ternary_transition_metrics:
        enable_ternary_topology = True
    # v5.7.0: ternary topology implies phase-space.
    if enable_ternary_topology:
        enable_bp_phase_space = True
    # v5.7.0: phase-space implies LLR history.
    if enable_bp_phase_space:
        enable_iteration_diagnostics = True
    # Landscape mode implies energy trace.
    if enable_landscape:
        enable_energy_trace = True
    # Phase diagram implies BP transitions.
    if enable_bp_phase_diagram:
        enable_bp_transitions = True
    # Barrier analysis implies landscape map.
    if enable_bp_barrier_analysis:
        enable_bp_landscape_map = True
    # Landscape map implies basin analysis.
    if enable_bp_landscape_map:
        enable_bp_basin_analysis = True
    # Basin analysis implies fixed-point analysis.
    if enable_bp_basin_analysis:
        enable_bp_fixed_point_analysis = True
    # Fixed-point analysis implies energy trace and iteration diagnostics.
    if enable_bp_fixed_point_analysis:
        enable_energy_trace = True
        enable_iteration_diagnostics = True
    # Freeze detection implies BP dynamics.
    if enable_bp_freeze_detection:
        enable_bp_dynamics = True
    # Iteration diagnostics implies energy trace.
    if enable_iteration_diagnostics:
        enable_energy_trace = True
    # BP transitions implies BP dynamics.
    if enable_bp_transitions:
        enable_bp_dynamics = True
    # BP dynamics implies energy trace and LLR history.
    if enable_bp_dynamics:
        enable_energy_trace = True
        enable_iteration_diagnostics = True

    frame_errors = 0
    residual_mismatches = 0
    trials_audit: list[dict[str, Any]] = []
    all_energy_traces: list[list[float]] = []
    all_landscape_metrics: list[dict[str, Any]] = []
    basin_switches = 0
    energy_deltas: list[float] = []
    all_basin_classifications: list[dict[str, Any]] = []
    all_iteration_diagnostics: list[dict[str, Any]] = []
    all_bp_dynamics: list[dict[str, Any]] = []
    all_bp_regime_traces: list[dict[str, Any]] = []
    all_bp_freeze_detections: list[dict[str, Any]] = []
    all_bp_fixed_point_analyses: list[dict[str, Any]] = []
    all_bp_basin_analyses: list[dict[str, Any]] = []
    all_bp_landscape_maps: list[dict[str, Any]] = []
    all_bp_barrier_analyses: list[dict[str, Any]] = []
    all_bp_boundary_analyses: list[dict[str, Any]] = []
    all_bp_phase_space: list[dict[str, Any]] = []
    all_ternary_topology: list[dict[str, Any]] = []
    all_ternary_basin_probes: list[dict[str, Any]] = []
    all_decoder_success: list[bool] = []
    comparison_results: list[dict[str, Any]] = []

    for trial_idx, inst in enumerate(instances):
        e = inst["e"]
        s = inst["s"]
        llr = inst["llr"]

        # Apply RPC augmentation if enabled.
        H_used = H
        s_used = s
        if structural.rpc.enabled:
            H_used, s_used = build_rpc_augmented_system(
                H, s, structural.rpc,
            )

          # ── Channel geometry interventions ──
        llr_used = llr
        if structural.centered_field:
            llr_used = centered_syndrome_field(H_used, s_used)
        elif structural.pseudo_prior:
            llr_used = syndrome_field(H_used, s_used)

        if structural.pseudo_prior:
            bias = pseudo_prior_bias(H_used, s_used)
            llr_used = apply_pseudo_prior(
                llr_used, bias, structural.pseudo_prior_strength,
            )

        llr_used = apply_geometry_postprocessing(llr_used, structural)
      
        # Decode.
        use_energy = enable_energy_trace or structural.energy_trace
        use_llr_history = max_iters if enable_iteration_diagnostics else 0
        result = decoder_fn(
            H_used, llr_used,
            max_iters=max_iters,
            mode=bp_mode,
            schedule=schedule,
            syndrome_vec=s_used,
            energy_trace=use_energy,
            llr_history=use_llr_history,
        )
        correction, iters = result[0], result[1]

        # ── Decoder comparison mode ──
        if compare_decoders:
            ref_fn = get_decoder("reference")
            exp_fn = get_decoder("experimental")
            # When paired_errors is active, explicitly copy inputs to
            # guarantee both decoders observe identical data.
            ref_llr = np.copy(llr_used) if paired_errors else llr_used
            exp_llr = np.copy(llr_used) if paired_errors else llr_used
            ref_result = ref_fn(
                H_used, ref_llr,
                max_iters=max_iters,
                mode=bp_mode,
                schedule=schedule,
                syndrome_vec=s_used,
            )
            exp_result = exp_fn(
                H_used, exp_llr,
                max_iters=max_iters,
                mode=bp_mode,
                schedule=schedule,
                syndrome_vec=s_used,
            )
            ref_corr, ref_iters = ref_result[0], ref_result[1]
            exp_corr, exp_iters = exp_result[0], exp_result[1]
            s_ref = syndrome(H, ref_corr)
            s_exp = syndrome(H, exp_corr)
            comparison_results.append({
                "reference": {
                    "success": bool(np.array_equal(s_ref, s)),
                    "iterations": int(ref_iters),
                    "syndrome_weight": int(np.sum(s_ref != s)),
                },
                "experimental": {
                    "success": bool(np.array_equal(s_exp, s)),
                    "iterations": int(exp_iters),
                    "syndrome_weight": int(np.sum(s_exp != s)),
                },
            })
        # Extract optional outputs based on what was requested.
        llr_hist = None
        trace = None
        if use_llr_history > 0 and use_energy:
            # Return order: (correction, iters, llr_history, energy_trace)
            llr_hist = result[2]
            trace = result[-1]
        elif use_llr_history > 0:
            llr_hist = result[2]
        elif use_energy:
            trace = result[-1]

        if trace is not None:
            all_energy_traces.append(trace)

            if enable_landscape and len(trace) >= 2:
                all_landscape_metrics.append(classify_energy_landscape(trace))

                basin = detect_basin_switch(
                    H_used, llr_used, correction, trace,
                    max_iters, bp_mode, schedule, s_used,
                )
                if basin["switch"]:
                    basin_switches += 1
                energy_deltas.append(abs(basin["energy_base"] - basin["energy_perturbed"]))

                # v4.2.0: landscape metrics (BSI, AD, EE) + classification.
                classification = compute_landscape_metrics(
                    H_used, llr_used, correction, trace,
                    max_iters, bp_mode, schedule, s_used,
                )
                all_basin_classifications.append(classification)

        # v4.3.0: iteration-trace diagnostics.
        if enable_iteration_diagnostics and trace is not None:
            llr_trace_list = (
                [llr_hist[i] for i in range(llr_hist.shape[0])]
                if llr_hist is not None and llr_hist.shape[0] > 0
                else []
            )
            iter_diag = compute_iteration_trace_metrics(
                llr_trace=llr_trace_list,
                energy_trace=list(trace),
                correction_vectors=None,
            )
            all_iteration_diagnostics.append(iter_diag)

            # v4.4.0: BP dynamics regime analysis.
            if enable_bp_dynamics:
                bp_dyn = compute_bp_dynamics_metrics(
                    llr_trace=llr_trace_list,
                    energy_trace=list(trace),
                    correction_vectors=None,
                )
                all_bp_dynamics.append(bp_dyn)

                # v4.5.0: BP regime transition analysis.
                if enable_bp_transitions:
                    rt = compute_bp_regime_trace(
                        llr_trace=llr_trace_list,
                        energy_trace=list(trace),
                        correction_vectors=None,
                    )
                    all_bp_regime_traces.append(rt)

                # v4.7.0: BP freeze detection.
                if enable_bp_freeze_detection:
                    fd = compute_bp_freeze_detection(
                        llr_trace=llr_trace_list,
                        energy_trace=list(trace),
                    )
                    all_bp_freeze_detections.append(fd)

        # v4.8.0: BP fixed-point trap analysis.
        if enable_bp_fixed_point_analysis and trace is not None:
            llr_trace_for_fp = (
                [llr_hist[i] for i in range(llr_hist.shape[0])]
                if llr_hist is not None and llr_hist.shape[0] > 0
                else []
            )
            energy_list = list(trace)
            # Build syndrome trace from per-iteration correction syndromes.
            # Use final syndrome weight from the correction.
            s_correction = syndrome(H, correction)
            final_sw = int(np.sum(s_correction != s))
            # Build per-iteration syndrome weights from energy trace length.
            syndrome_trace_fp = [final_sw] * len(energy_list)
            fp_result = compute_bp_fixed_point_analysis(
                llr_trace=llr_trace_for_fp if llr_trace_for_fp else [np.zeros(H.shape[1])],
                energy_trace=energy_list if energy_list else [0.0],
                syndrome_trace=syndrome_trace_fp if syndrome_trace_fp else [final_sw],
                final_syndrome_weight=final_sw,
            )
            all_bp_fixed_point_analyses.append(fp_result)

            # v4.9.0: BP basin-of-attraction analysis.
            if enable_bp_basin_analysis:
                basin_result = compute_bp_basin_analysis(
                    H=H,
                    llr=llr_used,
                    baseline_fixed_point_type=fp_result["fixed_point_type"],
                    max_iters=max_iters,
                    bp_mode=bp_mode,
                    schedule=schedule,
                    syndrome_vec=s_used,
                    syndrome_original=s,
                )
                all_bp_basin_analyses.append(basin_result)

            # v5.0.0: BP attractor landscape mapping.
            if enable_bp_landscape_map:
                from qec.diagnostics.bp_landscape_mapping import (
                    compute_bp_landscape_map,
                )
                landscape_map_result = compute_bp_landscape_map(
                    H=H,
                    llr=llr_used,
                    max_iters=max_iters,
                    bp_mode=bp_mode,
                    schedule=schedule,
                    syndrome_vec=s_used,
                    syndrome_original=s,
                )
                all_bp_landscape_maps.append(landscape_map_result)

            # v5.1.0: BP free-energy barrier estimation.
            if enable_bp_barrier_analysis:
                def _barrier_decode_fn(llr_vec):
                    """Decode wrapper for barrier analysis."""
                    _result = bp_decode(
                        H, llr_vec,
                        max_iters=max_iters,
                        mode=bp_mode,
                        schedule=schedule,
                        syndrome_vec=s_used,
                        energy_trace=True,
                        llr_history=max_iters,
                    )
                    _correction, _iters = _result[0], _result[1]
                    _llr_hist = _result[2]
                    _trace = _result[-1]

                    _llr_trace_list = (
                        [_llr_hist[i] for i in range(_llr_hist.shape[0])]
                        if _llr_hist is not None and _llr_hist.shape[0] > 0
                        else []
                    )
                    _energy_list = list(_trace) if _trace is not None else []

                    _s_correction = syndrome(H, _correction)
                    _final_sw = int(np.sum(_s_correction != s))

                    _syndrome_trace = [_final_sw] * len(_energy_list)

                    return {
                        "llr_trace": _llr_trace_list if _llr_trace_list else [np.zeros(H.shape[1])],
                        "energy_trace": _energy_list if _energy_list else [0.0],
                        "syndrome_trace": _syndrome_trace if _syndrome_trace else [_final_sw],
                        "final_syndrome_weight": _final_sw,
                    }

                barrier_result = compute_bp_barrier_analysis(
                    decode_fn=_barrier_decode_fn,
                    llr_init=llr_used,
                )
                all_bp_barrier_analyses.append(barrier_result)

            # v5.3.0: BP boundary analysis (attractor basin distance).
            if enable_bp_boundary_analysis:
                def _boundary_decode_fn(llr_vec):
                    """Decode wrapper for boundary analysis."""
                    _result = bp_decode(
                        H, llr_vec,
                        max_iters=max_iters,
                        mode=bp_mode,
                        schedule=schedule,
                        syndrome_vec=s_used,
                    )
                    return _result[0]

                boundary_result = compute_bp_boundary_analysis(
                    llr_vector=llr_used,
                    decoder_fn=_boundary_decode_fn,
                    parity_check_matrix=H,
                )
                all_bp_boundary_analyses.append(boundary_result)

        # v5.7.0: BP phase-space exploration.
        if enable_bp_phase_space and llr_hist is not None and llr_hist.shape[0] > 0:
            trajectory_states = [
                llr_hist[i].copy() for i in range(llr_hist.shape[0])
            ]
            ps_result = compute_bp_phase_space(trajectory_states)
            all_bp_phase_space.append(ps_result)

            # v5.7.0: Ternary topology classification.
            if enable_ternary_topology:
                s_correction_tt = syndrome(H, correction)
                final_sw_tt = int(np.sum(s_correction_tt != s))
                syndrome_residuals_tt = [final_sw_tt] * len(trajectory_states)
                tt_result = compute_ternary_decoder_topology(
                    phase_space_result=ps_result,
                    syndrome_residuals=syndrome_residuals_tt,
                )

                # v5.8.0: Add metastability score.
                if enable_ternary_transition_metrics:
                    ms = compute_metastability_score(ps_result["residual_norms"])
                    tt_result["metastability_score"] = ms

                all_ternary_topology.append(tt_result)

                # v5.8.0: Deterministic local basin probe.
                if enable_ternary_basin_probe and llr_hist is not None and llr_hist.shape[0] > 0:
                    final_llr = llr_hist[-1].copy()

                    def _basin_decode_fn(llr_in: np.ndarray) -> np.ndarray:
                        return decoder_fn(
                            H, llr_in,
                            max_iters=max_iters,
                            bp_mode=bp_mode,
                            schedule=schedule,
                        )

                    basin_result = probe_local_ternary_basin(
                        llr_vector=final_llr,
                        decode_function=_basin_decode_fn,
                        perturbation_scale=1e-3,
                        syndrome_function=lambda c: syndrome(H, c),
                        syndrome_target=s,
                    )
                    all_ternary_basin_probes.append(basin_result)

        # FER: syndrome-consistency semantics.
        # A frame error occurs when syndrome(H_original, correction) != s_original.
        s_correction = syndrome(H, correction)
        is_frame_error = not np.array_equal(s_correction, s)
        if is_frame_error:
            frame_errors += 1

        # v6.9.0: Per-trial decoder success for prediction validation.
        if enable_bp_prediction_validation:
            decoder_success_flag = not is_frame_error
            all_decoder_success.append(decoder_success_flag)
            # Inject into ternary topology result if available.
            if enable_ternary_topology and all_ternary_topology:
                all_ternary_topology[-1]["decoder_success"] = decoder_success_flag

        # Residual check (diagnostic only).
        residual = np.asarray(e) ^ np.asarray(correction)
        if np.any(residual):
            residual_mismatches += 1

        # Per-trial audit.
        s_weight = int(np.sum(s))
        e_weight = int(np.sum(e))
        audit = _trial_audit(H, H_used, s, s_used, iters, s_weight, e_weight)
        trials_audit.append(audit)

    n_trials = len(instances)
    fer = float(frame_errors) / n_trials if n_trials else 0.0

    out: dict[str, Any] = {
        "fer": fer,
        "frame_errors": frame_errors,
        "residual_mismatches": residual_mismatches,
        "trials": n_trials,
        "audit_summary": _summarize_audit(trials_audit),
    }
    if all_energy_traces:
        out["energy_traces"] = all_energy_traces
    if all_landscape_metrics:
        out["landscape_metrics"] = all_landscape_metrics
        n = len(instances)
        out["basin_switch"] = basin_switches > 0
        out["basin_switch_fraction"] = basin_switches / n if n else 0.0
        out["energy_delta"] = (
            sum(energy_deltas) / len(energy_deltas)
            if energy_deltas else 0.0
        )
    if all_basin_classifications:
        out["basin_classifications"] = all_basin_classifications
        # Aggregate class counts.
        class_counts: dict[str, int] = {}
        for bc in all_basin_classifications:
            cls = bc["basin_switch_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        out["basin_class_counts"] = class_counts
    if all_iteration_diagnostics:
        out["iteration_diagnostics"] = all_iteration_diagnostics
    if all_bp_dynamics:
        out["bp_dynamics"] = all_bp_dynamics
        # Aggregate regime counts.
        regime_counts: dict[str, int] = {}
        for bd in all_bp_dynamics:
            r = bd["regime"]
            regime_counts[r] = regime_counts.get(r, 0) + 1
        out["bp_regime_counts"] = regime_counts
    if all_bp_regime_traces:
        out["bp_regime_trace"] = all_bp_regime_traces
        # Aggregate transition summary and counts across all trials.
        agg_counts: dict[str, int] = {}
        total_events = 0
        total_transitions = 0
        total_iters = 0
        max_dwell_all = 0
        for rt in all_bp_regime_traces:
            for k, v in rt["transition_counts"].items():
                agg_counts[k] = agg_counts.get(k, 0) + v
            total_events += rt["summary"]["num_events"]
            total_transitions += len(rt["transitions"])
            total_iters += len(rt["regime_trace"])
            if rt["summary"]["max_dwell"] > max_dwell_all:
                max_dwell_all = rt["summary"]["max_dwell"]
        out["bp_transition_counts"] = {
            k: agg_counts[k] for k in sorted(agg_counts.keys())
        }
        out["bp_transition_summary"] = {
            "total_transitions": total_transitions,
            "total_events": total_events,
            "total_iters": total_iters,
            "max_dwell": max_dwell_all,
            "aggregate_switch_rate": (
                float(total_transitions) / float(total_iters)
                if total_iters > 0 else 0.0
            ),
        }
    if all_bp_freeze_detections:
        out["bp_freeze_detection"] = all_bp_freeze_detections
        # Aggregate: count how many trials had freeze detected.
        freeze_count = sum(
            1 for fd in all_bp_freeze_detections if fd["freeze_detected"]
        )
        out["bp_freeze_summary"] = {
            "freeze_count": freeze_count,
            "freeze_fraction": (
                float(freeze_count) / float(len(all_bp_freeze_detections))
                if all_bp_freeze_detections else 0.0
            ),
            "max_freeze_score": max(
                fd["freeze_score"] for fd in all_bp_freeze_detections
            ),
        }
    if all_bp_fixed_point_analyses:
        out["bp_fixed_point_analysis"] = all_bp_fixed_point_analyses
        # Aggregate fixed-point type counts.
        fp_type_counts: dict[str, int] = {}
        total_iters_to_fp: list[int] = []
        for fpa in all_bp_fixed_point_analyses:
            fpt = fpa["fixed_point_type"]
            fp_type_counts[fpt] = fp_type_counts.get(fpt, 0) + 1
            total_iters_to_fp.append(fpa["iterations_to_fixed_point"])
        n_fp = len(all_bp_fixed_point_analyses)
        out["bp_fixed_point_summary"] = {
            "correct_fixed_point_probability": float(
                fp_type_counts.get("correct_fixed_point", 0)
            ) / float(n_fp),
            "degenerate_fixed_point_probability": float(
                fp_type_counts.get("degenerate_fixed_point", 0)
            ) / float(n_fp),
            "incorrect_fixed_point_probability": float(
                fp_type_counts.get("incorrect_fixed_point", 0)
            ) / float(n_fp),
            "mean_iterations_to_fixed_point": (
                float(sum(total_iters_to_fp)) / float(n_fp)
            ),
            "no_convergence_probability": float(
                fp_type_counts.get("no_convergence", 0)
            ) / float(n_fp),
            "type_counts": {
                k: fp_type_counts[k]
                for k in sorted(fp_type_counts.keys())
            },
        }
    if all_bp_basin_analyses:
        out["bp_basin_analysis"] = all_bp_basin_analyses
        # Aggregate basin statistics.
        n_ba = len(all_bp_basin_analyses)
        total_correct = sum(ba["num_correct"] for ba in all_bp_basin_analyses)
        total_incorrect = sum(ba["num_incorrect"] for ba in all_bp_basin_analyses)
        total_degenerate = sum(ba["num_degenerate"] for ba in all_bp_basin_analyses)
        total_perturbations = sum(ba["num_perturbations"] for ba in all_bp_basin_analyses)
        boundary_values = [
            ba["basin_boundary_eps"]
            for ba in all_bp_basin_analyses
            if ba["basin_boundary_eps"] is not None
        ]
        out["bp_basin_summary"] = {
            "mean_basin_correct_probability": (
                float(total_correct) / float(total_perturbations)
                if total_perturbations > 0 else 0.0
            ),
            "mean_basin_incorrect_probability": (
                float(total_incorrect) / float(total_perturbations)
                if total_perturbations > 0 else 0.0
            ),
            "mean_basin_degenerate_probability": (
                float(total_degenerate) / float(total_perturbations)
                if total_perturbations > 0 else 0.0
            ),
            "num_trials_with_boundary": len(boundary_values),
            "mean_boundary_eps": (
                float(sum(boundary_values)) / float(len(boundary_values))
                if boundary_values else None
            ),
            "min_boundary_eps": (
                float(min(boundary_values)) if boundary_values else None
            ),
            "total_perturbations": total_perturbations,
            "total_trials": n_ba,
        }
    if all_bp_landscape_maps:
        out["bp_landscape_map"] = all_bp_landscape_maps
        # Aggregate landscape statistics.
        n_lm = len(all_bp_landscape_maps)
        total_attractors = sum(
            lm["num_attractors"] for lm in all_bp_landscape_maps
        )
        total_pseudocodewords = sum(
            lm["num_pseudocodewords"] for lm in all_bp_landscape_maps
        )
        basin_fractions = [
            lm["largest_basin_fraction"] for lm in all_bp_landscape_maps
        ]
        out["bp_landscape_summary"] = {
            "mean_largest_basin_fraction": (
                float(sum(basin_fractions)) / float(n_lm)
            ),
            "mean_num_attractors": float(total_attractors) / float(n_lm),
            "total_pseudocodewords": total_pseudocodewords,
            "total_trials": n_lm,
        }
    if all_bp_barrier_analyses:
        out["bp_barrier_analysis"] = all_bp_barrier_analyses
        # Aggregate barrier statistics.
        n_bar = len(all_bp_barrier_analyses)
        escaped_count = sum(
            1 for ba in all_bp_barrier_analyses if ba["escaped"]
        )
        barrier_eps_values = [
            ba["barrier_eps"] for ba in all_bp_barrier_analyses
            if ba["barrier_eps"] is not None
        ]
        total_trials_sum = sum(
            ba["num_trials"] for ba in all_bp_barrier_analyses
        )
        out["bp_barrier_summary"] = {
            "escape_probability": float(escaped_count) / float(n_bar),
            "mean_barrier_eps": (
                float(sum(barrier_eps_values)) / float(len(barrier_eps_values))
                if barrier_eps_values else None
            ),
            "num_trials": total_trials_sum,
        }
    if all_bp_boundary_analyses:
        out["bp_boundary_analysis"] = all_bp_boundary_analyses
        # Aggregate boundary statistics.
        n_bnd = len(all_bp_boundary_analyses)
        crossed_count = sum(
            1 for ba in all_bp_boundary_analyses if ba["boundary_crossed"]
        )
        boundary_eps_values = [
            ba["boundary_eps"] for ba in all_bp_boundary_analyses
            if ba["boundary_eps"] is not None
        ]
        total_boundary_directions = sum(
            ba["num_directions"] for ba in all_bp_boundary_analyses
        )
        out["bp_boundary_summary"] = {
            "boundary_cross_probability": float(crossed_count) / float(n_bnd),
            "mean_boundary_eps": (
                float(sum(boundary_eps_values)) / float(len(boundary_eps_values))
                if boundary_eps_values else None
            ),
            "num_directions": total_boundary_directions,
        }
    if all_bp_phase_space:
        out["bp_phase_space"] = all_bp_phase_space
    if all_ternary_topology:
        out["ternary_topology"] = all_ternary_topology
        # Aggregate ternary state counts.
        final_state_counts: dict[int, int] = {}
        for tt in all_ternary_topology:
            fs = tt["final_ternary_state"]
            final_state_counts[fs] = final_state_counts.get(fs, 0) + 1
        n_tt = len(all_ternary_topology)
        out["ternary_topology_summary"] = {
            "success_basin_count": final_state_counts.get(1, 0),
            "boundary_count": final_state_counts.get(0, 0),
            "failure_basin_count": final_state_counts.get(-1, 0),
            "success_basin_fraction": (
                float(final_state_counts.get(1, 0)) / float(n_tt)
            ),
            "boundary_fraction": (
                float(final_state_counts.get(0, 0)) / float(n_tt)
            ),
            "failure_basin_fraction": (
                float(final_state_counts.get(-1, 0)) / float(n_tt)
            ),
        }
    if all_ternary_basin_probes:
        out["ternary_basin_probe"] = all_ternary_basin_probes
    if all_decoder_success:
        out["per_trial_decoder_success"] = all_decoder_success
    if comparison_results:
        out["decoder_comparison"] = comparison_results
    return out


# ── DPS slope computation ───────────────────────────────────────────

_LOG_EPS = 1e-30


def compute_dps_slope(
    fer_by_distance: dict[int, float],
    eps: float = _LOG_EPS,
) -> float:
    """Compute linear slope of log10(FER + eps) vs distance."""
    distances = sorted(fer_by_distance.keys())
    if len(distances) < 2:
        return 0.0
    d_arr = np.array(distances, dtype=np.float64)
    lf_arr = np.array(
        [math.log10(fer_by_distance[d] + eps) for d in distances],
        dtype=np.float64,
    )
    A = np.vstack([d_arr, np.ones(len(d_arr))]).T
    coef, _, _, _ = np.linalg.lstsq(A, lf_arr, rcond=None)
    return float(coef[0])


# ── Full evaluation ─────────────────────────────────────────────────

def run_evaluation(
    seed: int = DEFAULT_SEED,
    distances: list[int] | None = None,
    p_values: list[float] | None = None,
    trials: int = DEFAULT_TRIALS,
    max_iters: int = DEFAULT_MAX_ITERS,
    bp_mode: str = DEFAULT_BP_MODE,
    enable_energy_trace: bool = False,
    enable_landscape: bool = False,
    enable_iteration_diagnostics: bool = False,
    enable_bp_dynamics: bool = False,
    enable_bp_transitions: bool = False,
    enable_bp_phase_diagram: bool = False,
    enable_bp_freeze_detection: bool = False,
    enable_bp_fixed_point_analysis: bool = False,
    enable_bp_basin_analysis: bool = False,
    enable_bp_landscape_map: bool = False,
    enable_bp_barrier_analysis: bool = False,
    enable_bp_boundary_analysis: bool = False,
    enable_tanner_spectral_analysis: bool = False,
    enable_spectral_boundary_alignment: bool = False,
    enable_spectral_trapping_sets: bool = False,
    enable_bp_phase_space: bool = False,
    enable_ternary_topology: bool = False,
    enable_ternary_transition_metrics: bool = False,
    enable_ternary_basin_probe: bool = False,
    enable_spectral_bp_alignment: bool = False,
    enable_spectral_failure_risk: bool = False,
    enable_risk_aware_damping_experiment: bool = False,
    enable_risk_guided_perturbation_experiment: bool = False,
    enable_tanner_graph_repair_experiment: bool = False,
    enable_spectral_graph_optimization: bool = False,
    enable_bp_stability_predictor: bool = False,
    enable_bp_prediction_validation: bool = False,
    enable_spectral_decoder_controller: bool = False,
    enable_spectral_cluster_control: bool = False,
    enable_spectral_phase_map: bool = False,
    enable_spectral_graph_repair_loop: bool = False,
    enable_spectral_multistep_repair: bool = False,
    enable_spectral_graph_design_analysis: bool = False,
    enable_spectral_graph_optimize: bool = False,
    decoder_fn=None,
    compare_decoders: bool = False,
    paired_seed: bool = False,
    paired_errors: bool = False,
) -> dict[str, Any]:
    """Run the full DPS evaluation across all modes, distances, and p values.

    Returns a dict with:
      results[mode_name][p][distance] = run_mode result
      slopes[mode_name][p] = DPS slope
      audit[mode_name][p][distance] = audit_summary
    """
    # v7.5.0: Spectral graph optimize implies spectral failure risk.
    if enable_spectral_graph_optimize:
        enable_spectral_failure_risk = True
    # v7.4.0: Spectral graph design analysis implies spectral failure risk.
    if enable_spectral_graph_design_analysis:
        enable_spectral_failure_risk = True
    # v7.3.x: Spectral multistep repair implies repair loop.
    if enable_spectral_multistep_repair:
        enable_spectral_graph_repair_loop = True
    # v7.3.0: Spectral graph repair loop implies spectral phase map.
    if enable_spectral_graph_repair_loop:
        enable_spectral_phase_map = True
    # v7.2.0: Spectral phase map implies BP stability predictor.
    if enable_spectral_phase_map:
        enable_bp_stability_predictor = True
    # v7.1.0: Spectral cluster control implies spectral decoder controller.
    if enable_spectral_cluster_control:
        enable_spectral_decoder_controller = True
    # v7.0.0: Spectral decoder controller implies BP stability predictor.
    if enable_spectral_decoder_controller:
        enable_bp_stability_predictor = True
    # v6.9.0: BP prediction validation implies BP stability predictor.
    if enable_bp_prediction_validation:
        enable_bp_stability_predictor = True
    # v6.8.0: BP stability predictor implies spectral failure risk.
    if enable_bp_stability_predictor:
        enable_spectral_failure_risk = True
    # v6.7.0: Spectral graph optimization implies spectral failure risk.
    if enable_spectral_graph_optimization:
        enable_spectral_failure_risk = True
    # v6.6.0: Tanner graph repair implies spectral failure risk.
    if enable_tanner_graph_repair_experiment:
        enable_spectral_failure_risk = True
    # v6.5.0: experiments imply spectral failure risk.
    if enable_risk_aware_damping_experiment:
        enable_spectral_failure_risk = True
    if enable_risk_guided_perturbation_experiment:
        enable_spectral_failure_risk = True

    # v6.4.0: spectral failure risk implies spectral-BP alignment.
    if enable_spectral_failure_risk:
        enable_spectral_bp_alignment = True

    # v6.3.0: spectral-BP alignment implies iteration diagnostics.
    if enable_spectral_bp_alignment:
        enable_iteration_diagnostics = True

    # v5.8.0: basin probe implies ternary topology.
    if enable_ternary_basin_probe:
        enable_ternary_topology = True
    # v5.8.0: transition metrics implies ternary topology.
    if enable_ternary_transition_metrics:
        enable_ternary_topology = True
    # v5.7.0: ternary topology implies phase-space.
    if enable_ternary_topology:
        enable_bp_phase_space = True

    # v5.5.0: spectral-boundary alignment implies dependencies.
    if enable_spectral_boundary_alignment:
        enable_tanner_spectral_analysis = True
        enable_bp_boundary_analysis = True
        enable_bp_barrier_analysis = True

    # v5.6.0: spectral trapping sets implies tanner spectral analysis.
    if enable_spectral_trapping_sets:
        enable_tanner_spectral_analysis = True

    if distances is None:
        distances = list(DEFAULT_DISTANCES)
    if p_values is None:
        p_values = list(DEFAULT_P_VALUES)

    distances = sorted(distances)
    p_values = sorted(p_values)

    results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
    slopes: dict[str, dict[float, float]] = {}
    audits: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}

    # Pre-generate all error instances per (distance, p), keyed for reuse.
    rng = np.random.default_rng(seed)
    codes: dict[int, np.ndarray] = {}
    all_instances: dict[tuple[int, float], list[dict[str, Any]]] = {}

    for distance in distances:
        H = create_code(name="rate_0.50", lifting_size=distance, seed=seed).H_X
        codes[distance] = H
        for p in p_values:
            all_instances[(distance, p)] = _pre_generate_instances(H, p, trials, rng)

    # v5.4.0: Tanner spectral analysis (once per code instance).
    tanner_spectral_results: dict[int, dict[str, Any]] = {}
    if enable_tanner_spectral_analysis:
        from qec.diagnostics.tanner_spectral_analysis import (
            compute_tanner_spectral_analysis,
        )
        for distance in distances:
            tanner_spectral_results[distance] = compute_tanner_spectral_analysis(
                codes[distance],
            )

    # v5.5.0 / v5.6.0: Extract spectral eigenvectors (shared, once per code).
    tanner_eigenvectors: dict[int, np.ndarray] = {}
    tanner_spectral_modes: dict[int, list[np.ndarray]] = {}
    if enable_spectral_boundary_alignment or enable_spectral_trapping_sets:
        for distance in distances:
            H = codes[distance]
            m, n = H.shape
            top = np.concatenate([np.zeros((n, n), dtype=np.float64), H.T.astype(np.float64)], axis=1)
            bottom = np.concatenate([H.astype(np.float64), np.zeros((m, m), dtype=np.float64)], axis=1)
            A = np.concatenate([top, bottom], axis=0)
            eigvals, eigvecs = np.linalg.eigh(A)
            sort_idx = np.argsort(-eigvals)
            eigvecs = eigvecs[:, sort_idx]
            tanner_eigenvectors[distance] = eigvecs
            top_k = min(3, eigvecs.shape[1])
            tanner_spectral_modes[distance] = [
                eigvecs[:n, i].copy() for i in range(top_k)
            ]

    # v6.1/v6.2/v6.3: NB localization and trapping candidates (once per code).
    nb_localization_results: dict[int, dict[str, Any]] = {}
    nb_trapping_results: dict[int, dict[str, Any]] = {}
    if enable_spectral_bp_alignment:
        from qec.diagnostics.nb_localization import (
            compute_nb_localization_metrics,
        )
        from qec.diagnostics.nb_trapping_candidates import (
            compute_nb_trapping_candidates,
        )
        for distance in distances:
            loc = compute_nb_localization_metrics(codes[distance])
            nb_localization_results[distance] = loc
            nb_trapping_results[distance] = compute_nb_trapping_candidates(
                codes[distance], loc,
            )

    # Deterministic loop order: modes → p_values → distances.
    for mode_name in MODE_ORDER:
        results[mode_name] = {}
        slopes[mode_name] = {}
        audits[mode_name] = {}

        for p in p_values:
            fer_by_distance: dict[int, float] = {}
            results[mode_name][p] = {}
            audits[mode_name][p] = {}

            for distance in distances:
                H = codes[distance]
                instances = all_instances[(distance, p)]

                result = run_mode(
                    mode_name, H, instances,
                    max_iters=max_iters, bp_mode=bp_mode,
                    enable_energy_trace=enable_energy_trace,
                    enable_landscape=enable_landscape,
                    enable_iteration_diagnostics=enable_iteration_diagnostics,
                    enable_bp_dynamics=enable_bp_dynamics,
                    enable_bp_transitions=enable_bp_transitions,
                    enable_bp_phase_diagram=enable_bp_phase_diagram,
                    enable_bp_freeze_detection=enable_bp_freeze_detection,
                    enable_bp_fixed_point_analysis=enable_bp_fixed_point_analysis,
                    enable_bp_basin_analysis=enable_bp_basin_analysis,
                    enable_bp_landscape_map=enable_bp_landscape_map,
                    enable_bp_barrier_analysis=enable_bp_barrier_analysis,
                    enable_bp_boundary_analysis=enable_bp_boundary_analysis,
                    enable_bp_phase_space=enable_bp_phase_space,
                    enable_ternary_topology=enable_ternary_topology,
                    enable_ternary_transition_metrics=enable_ternary_transition_metrics,
                    enable_ternary_basin_probe=enable_ternary_basin_probe,
                    enable_bp_prediction_validation=enable_bp_prediction_validation,
                    decoder_fn=decoder_fn,
                    compare_decoders=compare_decoders,
                    paired_seed=paired_seed,
                    paired_errors=paired_errors,
                )
                results[mode_name][p][distance] = result
                audits[mode_name][p][distance] = result["audit_summary"]
                fer_by_distance[distance] = result["fer"]

            slopes[mode_name][p] = compute_dps_slope(fer_by_distance)

    out: dict[str, Any] = {
        "results": results,
        "slopes": slopes,
        "audits": audits,
        "config": {
            "seed": seed,
            "distances": distances,
            "p_values": p_values,
            "trials": trials,
            "max_iters": max_iters,
            "bp_mode": bp_mode,
        },
    }

    # v5.4.0: Tanner spectral analysis results.
    if tanner_spectral_results:
        out["tanner_spectral_analysis"] = {
            int(d): tanner_spectral_results[d] for d in sorted(tanner_spectral_results.keys())
        }

    # v5.6.0: Spectral trapping-set analysis (once per code instance).
    if enable_spectral_trapping_sets and tanner_eigenvectors:
        trapping_set_results: dict[int, dict[str, Any]] = {}
        for distance in sorted(tanner_eigenvectors.keys()):
            H = codes[distance]
            n = H.shape[1]
            eigvecs = tanner_eigenvectors[distance]
            top_k = min(3, eigvecs.shape[1])
            modes_for_trapping = [eigvecs[:, i].copy() for i in range(top_k)]
            ts = compute_spectral_trapping_sets(modes_for_trapping, n)
            trapping_set_results[distance] = ts
        out["spectral_trapping_sets"] = {
            int(d): trapping_set_results[d]
            for d in sorted(trapping_set_results.keys())
        }

    # v5.5.0: Spectral–boundary alignment analysis.
    if enable_spectral_boundary_alignment:
        alignment_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        for mode_name in MODE_ORDER:
            alignment_results[mode_name] = {}
            for p in p_values:
                alignment_results[mode_name][p] = {}
                for distance in distances:
                    r = results[mode_name][p][distance]
                    spectral_modes = tanner_spectral_modes.get(distance, [])
                    # Extract boundary direction from boundary analysis.
                    boundary_dir = None
                    if "bp_boundary_analysis" in r:
                        for ba in r["bp_boundary_analysis"]:
                            if ba.get("boundary_direction") is not None:
                                boundary_dir = np.asarray(
                                    ba["boundary_direction"], dtype=np.float64
                                )
                                break
                    if boundary_dir is not None and spectral_modes:
                        sba = compute_spectral_boundary_alignment(
                            spectral_modes, boundary_dir,
                        )
                        # Extract barrier_eps and boundary_eps from summaries.
                        barrier_eps = None
                        boundary_eps = None
                        if "bp_barrier_summary" in r:
                            barrier_eps = r["bp_barrier_summary"].get("mean_barrier_eps")
                        if "bp_boundary_summary" in r:
                            boundary_eps = r["bp_boundary_summary"].get("mean_boundary_eps")
                        alignment_results[mode_name][p][distance] = {
                            "alignment_max": sba["max_alignment"],
                            "alignment_mean": sba["mean_alignment"],
                            "dominant_alignment_mode": sba["dominant_alignment_mode"],
                            "mode_count": sba["mode_count"],
                            "p": p,
                            "FER": r["fer"],
                            "boundary_eps": boundary_eps,
                            "barrier_eps": barrier_eps,
                        }
        out["spectral_boundary_alignment"] = alignment_results

    # v4.6.0: BP phase diagram aggregation.
    if enable_bp_phase_diagram:
        phase_run_results: list[dict[str, Any]] = []
        for mode_name in MODE_ORDER:
            for p in p_values:
                for distance in distances:
                    r = results[mode_name][p][distance]
                    if "bp_regime_trace" in r:
                        phase_run_results.append({
                            "distance": distance,
                            "noise": p,
                            "regime_trace_results": r["bp_regime_trace"],
                        })
        out["bp_phase_diagram"] = compute_bp_phase_diagram(phase_run_results)

    # v6.3.0: Spectral–BP attractor alignment analysis.
    if enable_spectral_bp_alignment and nb_trapping_results:
        sbpa_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        for mode_name in MODE_ORDER:
            sbpa_results[mode_name] = {}
            for p in p_values:
                sbpa_results[mode_name][p] = {}
                for distance in distances:
                    r = results[mode_name][p][distance]
                    trapping = nb_trapping_results.get(distance)
                    iter_diags = r.get("iteration_diagnostics", [])
                    if trapping is None or not iter_diags:
                        continue
                    # Per-trial alignment, then aggregate.
                    trial_alignments: list[dict[str, Any]] = []
                    for diag in iter_diags:
                        boi = diag.get("belief_oscillation_index", {})
                        boi_vec = boi.get("boi_vector")
                        if boi_vec is None:
                            continue
                        boi_arr = np.asarray(boi_vec, dtype=np.float64)
                        bp_scores = {
                            i: float(boi_arr[i]) for i in range(len(boi_arr))
                        }
                        ta = compute_spectral_bp_alignment(
                            trapping, bp_scores,
                        )
                        trial_alignments.append(ta)
                    if trial_alignments:
                        mean_align = sum(
                            t["spectral_bp_alignment_score"]
                            for t in trial_alignments
                        ) / len(trial_alignments)
                        mean_cand_overlap = sum(
                            t["candidate_node_overlap_fraction"]
                            for t in trial_alignments
                        ) / len(trial_alignments)
                        mean_cluster_overlap = sum(
                            t["max_cluster_alignment"]
                            for t in trial_alignments
                        ) / len(trial_alignments)
                        max_align = max(
                            t["spectral_bp_alignment_score"]
                            for t in trial_alignments
                        )
                        sbpa_results[mode_name][p][distance] = {
                            "mean_spectral_bp_alignment": round(mean_align, 12),
                            "mean_candidate_node_overlap_fraction": round(mean_cand_overlap, 12),
                            "mean_candidate_cluster_overlap_fraction": round(mean_cluster_overlap, 12),
                            "max_spectral_bp_alignment": round(max_align, 12),
                            "num_trials": len(trial_alignments),
                            "per_trial": trial_alignments,
                        }
        out["spectral_bp_alignment"] = sbpa_results

    # v6.4.0: Spectral failure risk scoring.
    if enable_spectral_failure_risk and nb_trapping_results and nb_localization_results:
        sfr_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        sbpa_data = out.get("spectral_bp_alignment", {})
        for mode_name in MODE_ORDER:
            sfr_results[mode_name] = {}
            sbpa_mode = sbpa_data.get(mode_name, {})
            for p in p_values:
                sfr_results[mode_name][p] = {}
                sbpa_p = sbpa_mode.get(p, {})
                for distance in distances:
                    loc = nb_localization_results.get(distance)
                    trapping = nb_trapping_results.get(distance)
                    sbpa_cell = sbpa_p.get(distance)
                    if loc is None or trapping is None or sbpa_cell is None:
                        continue
                    per_trial_alignments = sbpa_cell.get("per_trial", [])
                    if not per_trial_alignments:
                        continue
                    # Per-trial risk scores, then aggregate.
                    trial_risks: list[dict[str, Any]] = []
                    for ta in per_trial_alignments:
                        tr = compute_spectral_failure_risk(
                            loc, trapping, ta,
                        )
                        trial_risks.append(tr)
                    if trial_risks:
                        mean_cluster_risk = sum(
                            t["mean_cluster_risk"] for t in trial_risks
                        ) / len(trial_risks)
                        max_cluster_risk = max(
                            t["max_cluster_risk"] for t in trial_risks
                        )
                        mean_high_risk = sum(
                            t["num_high_risk_clusters"] for t in trial_risks
                        ) / len(trial_risks)
                        sfr_results[mode_name][p][distance] = {
                            "mean_cluster_risk": round(mean_cluster_risk, 12),
                            "max_cluster_risk": round(max_cluster_risk, 12),
                            "mean_num_high_risk_clusters": round(mean_high_risk, 12),
                            "num_trials": len(trial_risks),
                            "per_trial": trial_risks,
                        }
        out["spectral_failure_risk"] = sfr_results

    # v7.4.0: Spectral graph design analysis (per-code, before decoding).
    if enable_spectral_graph_design_analysis and "spectral_failure_risk" in out and nb_localization_results:
        from qec.diagnostics.non_backtracking_spectrum import (
            compute_non_backtracking_spectrum as _compute_nb_spectrum_design,
        )
        sgda_results: dict[int, dict[str, Any]] = {}
        sfr_data_sgda = out["spectral_failure_risk"]
        for distance in distances:
            H_sgda = codes[distance]
            m_sgda, n_sgda = H_sgda.shape
            num_edges_sgda = int(np.count_nonzero(H_sgda))
            avg_var_deg_sgda = num_edges_sgda / n_sgda if n_sgda > 0 else 0.0
            avg_chk_deg_sgda = num_edges_sgda / m_sgda if m_sgda > 0 else 0.0
            nb_spec_sgda = _compute_nb_spectrum_design(H_sgda)
            loc_sgda = nb_localization_results.get(distance, {})
            # Aggregate risk across modes: use first available mode's risk.
            risk_sgda: dict[str, Any] = {}
            instab_ratio_sgda = 0.0
            for mode_name in MODE_ORDER:
                sfr_mode_sgda = sfr_data_sgda.get(mode_name, {})
                for p in p_values:
                    sfr_cell_sgda = sfr_mode_sgda.get(p, {}).get(distance)
                    if sfr_cell_sgda is not None:
                        risk_sgda = sfr_cell_sgda
                        break
                if risk_sgda:
                    break
            sgda_results[distance] = run_spectral_graph_design_analysis(
                nb_spectrum_result=nb_spec_sgda,
                localization_result=loc_sgda,
                risk_result=risk_sgda,
                spectral_instability_ratio=instab_ratio_sgda,
                avg_variable_degree=round(avg_var_deg_sgda, 12),
                avg_check_degree=round(avg_chk_deg_sgda, 12),
            )
        out["spectral_graph_design_analysis"] = sgda_results

    # v6.8.0: BP stability predictor.
    if enable_bp_stability_predictor and "spectral_failure_risk" in out and nb_localization_results:
        bsp_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        sfr_data_bsp = out["spectral_failure_risk"]
        for mode_name in MODE_ORDER:
            bsp_results[mode_name] = {}
            sfr_mode_bsp = sfr_data_bsp.get(mode_name, {})
            for p in p_values:
                bsp_results[mode_name][p] = {}
                sfr_p_bsp = sfr_mode_bsp.get(p, {})
                for distance in distances:
                    sfr_cell_bsp = sfr_p_bsp.get(distance)
                    loc = nb_localization_results.get(distance)
                    if sfr_cell_bsp is None or loc is None:
                        continue
                    H = codes[distance]
                    m_h, n_h = H.shape
                    num_edges_h = int(np.count_nonzero(H))
                    avg_degree_h = (
                        (2.0 * num_edges_h) / (n_h + m_h)
                        if (n_h + m_h) > 0 else 0.0
                    )
                    graph_info = {
                        "num_variable_nodes": n_h,
                        "num_check_nodes": m_h,
                        "num_edges": num_edges_h,
                        "avg_degree": round(avg_degree_h, 12),
                    }
                    per_trial_risks = sfr_cell_bsp.get("per_trial", [])
                    # Build per-trial diagnostics dict from localization + risk.
                    nb_max_ipr = loc.get("nb_max_ipr", 0.0)
                    if nb_max_ipr is None:
                        # Derive from ipr_scores if available.
                        ipr_scores = loc.get("ipr_scores", [])
                        nb_max_ipr = max(ipr_scores) if ipr_scores else 0.0
                    nb_num_localized = loc.get("nb_num_localized_modes", 0)
                    if nb_num_localized is None:
                        localized_modes = loc.get("localized_modes", [])
                        nb_num_localized = len(localized_modes)
                    trial_predictions: list[dict[str, Any]] = []
                    for risk_t in per_trial_risks:
                        diag = {
                            "spectral_radius": sfr_cell_bsp.get(
                                "spectral_radius",
                                risk_t.get("max_cluster_risk", 0.0),
                            ),
                            "nb_max_ipr": nb_max_ipr,
                            "nb_num_localized_modes": nb_num_localized,
                            "max_cluster_risk": risk_t.get("max_cluster_risk", 0.0),
                        }
                        pred = compute_bp_stability_prediction(graph_info, diag)
                        trial_predictions.append(pred)
                    if trial_predictions:
                        n_pred = len(trial_predictions)
                        mean_bp_stability = sum(
                            t["bp_stability_score"] for t in trial_predictions
                        ) / n_pred
                        mean_bp_failure_risk = sum(
                            t["bp_failure_risk"] for t in trial_predictions
                        ) / n_pred
                        instability_count = sum(
                            1 for t in trial_predictions
                            if t["predicted_instability"]
                        )
                        mean_spectral_ratio = sum(
                            t["spectral_instability_ratio"]
                            for t in trial_predictions
                        ) / n_pred
                        bsp_results[mode_name][p][distance] = {
                            "mean_bp_stability_prediction": round(mean_bp_stability, 12),
                            "mean_bp_failure_risk": round(mean_bp_failure_risk, 12),
                            "instability_fraction": round(
                                instability_count / n_pred, 12,
                            ),
                            "mean_spectral_instability_ratio": round(
                                mean_spectral_ratio, 12,
                            ),
                            "num_trials": n_pred,
                            "per_trial": trial_predictions,
                        }
        out["bp_stability_predictor"] = bsp_results

    # v6.9.0: BP prediction validation.
    if enable_bp_prediction_validation and "bp_stability_predictor" in out:
        bpv_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        bsp_data_bpv = out["bp_stability_predictor"]
        for mode_name in MODE_ORDER:
            bpv_results[mode_name] = {}
            bsp_mode_bpv = bsp_data_bpv.get(mode_name, {})
            for p in p_values:
                bpv_results[mode_name][p] = {}
                bsp_p_bpv = bsp_mode_bpv.get(p, {})
                for distance in distances:
                    bsp_cell_bpv = bsp_p_bpv.get(distance)
                    if bsp_cell_bpv is None:
                        continue
                    mode_result = results[mode_name][p][distance]
                    per_trial_success = mode_result.get(
                        "per_trial_decoder_success", [],
                    )
                    per_trial_preds = bsp_cell_bpv.get("per_trial", [])
                    if not per_trial_success or not per_trial_preds:
                        continue
                    n_pair = min(len(per_trial_preds), len(per_trial_success))
                    trial_pairs: list[dict[str, Any]] = []
                    for i_pair in range(n_pair):
                        trial_pairs.append({
                            "bp_stability_prediction": per_trial_preds[i_pair],
                            "decoder_success": per_trial_success[i_pair],
                        })
                    validation = run_bp_prediction_validation(trial_pairs)
                    bpv_results[mode_name][p][distance] = validation
        out["bp_prediction_validation"] = bpv_results

    # v7.2.0: Spectral instability phase map.
    if enable_spectral_phase_map and "bp_stability_predictor" in out and nb_localization_results:
        spm_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        bsp_data_spm = out["bp_stability_predictor"]
        sfr_data_spm = out.get("spectral_failure_risk", {})
        for mode_name in MODE_ORDER:
            spm_results[mode_name] = {}
            bsp_mode_spm = bsp_data_spm.get(mode_name, {})
            sfr_mode_spm = sfr_data_spm.get(mode_name, {})
            for p in p_values:
                spm_results[mode_name][p] = {}
                bsp_p_spm = bsp_mode_spm.get(p, {})
                sfr_p_spm = sfr_mode_spm.get(p, {})
                for distance in distances:
                    bsp_cell_spm = bsp_p_spm.get(distance)
                    if bsp_cell_spm is None:
                        continue
                    H_spm = codes[distance]
                    m_spm, n_spm = H_spm.shape
                    num_edges_spm = int(np.count_nonzero(H_spm))
                    avg_var_deg = (
                        num_edges_spm / n_spm if n_spm > 0 else 0.0
                    )
                    avg_chk_deg = (
                        num_edges_spm / m_spm if m_spm > 0 else 0.0
                    )
                    loc_spm = nb_localization_results.get(distance, {})
                    nb_max_ipr_spm = loc_spm.get("nb_max_ipr", 0.0)
                    if nb_max_ipr_spm is None:
                        ipr_scores_spm = loc_spm.get("ipr_scores", [])
                        nb_max_ipr_spm = max(ipr_scores_spm) if ipr_scores_spm else 0.0
                    sfr_cell_spm = sfr_p_spm.get(distance, {})
                    per_trial_preds_spm = bsp_cell_spm.get("per_trial", [])
                    per_trial_risks_spm = sfr_cell_spm.get("per_trial", [])
                    mode_result_spm = results[mode_name][p][distance]
                    per_trial_success_spm = mode_result_spm.get(
                        "per_trial_decoder_success", [],
                    )
                    if not per_trial_preds_spm or not per_trial_success_spm:
                        continue
                    n_spm_trials = min(
                        len(per_trial_preds_spm), len(per_trial_success_spm),
                    )
                    trial_phase_map: list[dict[str, Any]] = []
                    for i_spm in range(n_spm_trials):
                        pred_spm = per_trial_preds_spm[i_spm]
                        # Get cluster risk scores for this trial.
                        cluster_risks: list[float] = []
                        if i_spm < len(per_trial_risks_spm):
                            risk_t_spm = per_trial_risks_spm[i_spm]
                            cluster_risks = risk_t_spm.get(
                                "cluster_risk_scores", [],
                            )
                        score = compute_spectral_instability_score(
                            nb_spectral_radius=pred_spm.get(
                                "spectral_radius", 0.0,
                            ),
                            spectral_instability_ratio=pred_spm.get(
                                "spectral_instability_ratio", 0.0,
                            ),
                            ipr_localization_score=float(nb_max_ipr_spm),
                            cluster_risk_scores=cluster_risks,
                            avg_variable_degree=round(avg_var_deg, 12),
                            avg_check_degree=round(avg_chk_deg, 12),
                        )
                        trial_result = run_spectral_phase_map_experiment(
                            spectral_instability_score=score,
                            decoder_success=per_trial_success_spm[i_spm],
                        )
                        trial_phase_map.append(trial_result)
                    if trial_phase_map:
                        aggregate = compute_phase_map_aggregate_metrics(
                            trial_phase_map,
                        )
                        aggregate["per_trial"] = trial_phase_map
                        spm_results[mode_name][p][distance] = aggregate
        out["spectral_phase_map"] = spm_results

    # v7.0.0 / v7.1.0: Spectral decoder controller experiment.
    if enable_spectral_decoder_controller and "bp_stability_predictor" in out and "spectral_failure_risk" in out:
        sdc_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        bsp_data_sdc = out["bp_stability_predictor"]
        sfr_data_sdc = out["spectral_failure_risk"]
        for mode_name in MODE_ORDER:
            sdc_results[mode_name] = {}
            bsp_mode_sdc = bsp_data_sdc.get(mode_name, {})
            sfr_mode_sdc = sfr_data_sdc.get(mode_name, {})
            for p in p_values:
                sdc_results[mode_name][p] = {}
                bsp_p_sdc = bsp_mode_sdc.get(p, {})
                sfr_p_sdc = sfr_mode_sdc.get(p, {})
                for distance in distances:
                    bsp_cell_sdc = bsp_p_sdc.get(distance)
                    sfr_cell_sdc = sfr_p_sdc.get(distance)
                    if bsp_cell_sdc is None or sfr_cell_sdc is None:
                        continue
                    per_trial_preds_sdc = bsp_cell_sdc.get("per_trial", [])
                    per_trial_risks_sdc = sfr_cell_sdc.get("per_trial", [])
                    if not per_trial_preds_sdc or not per_trial_risks_sdc:
                        continue
                    H = codes[distance]
                    instances = all_instances[(distance, p)]
                    # v7.1.0: Inject candidate_clusters for cluster control.
                    trapping_data_sdc = nb_trapping_results.get(distance, {})
                    cand_clusters_sdc = trapping_data_sdc.get(
                        "candidate_clusters", [],
                    )
                    trial_controller: list[dict[str, Any]] = []
                    n_ctrl = min(
                        len(instances),
                        len(per_trial_preds_sdc),
                        len(per_trial_risks_sdc),
                    )
                    for t_idx in range(n_ctrl):
                        inst = instances[t_idx]
                        # Enrich risk result with candidate_clusters.
                        trial_risk = per_trial_risks_sdc[t_idx]
                        if enable_spectral_cluster_control and cand_clusters_sdc:
                            trial_risk = dict(trial_risk)
                            trial_risk["candidate_clusters"] = cand_clusters_sdc
                        exp = run_spectral_decoder_control_experiment(
                            H,
                            inst["llr"],
                            inst["s"],
                            trial_risk,
                            per_trial_preds_sdc[t_idx],
                            max_iters=max_iters,
                            enable_cluster_control=enable_spectral_cluster_control,
                        )
                        trial_controller.append(exp)
                    if trial_controller:
                        n_tc = len(trial_controller)
                        mean_bp_failure_risk = sum(
                            t["bp_failure_risk"] for t in trial_controller
                        ) / n_tc
                        instability_count = sum(
                            1 for t in trial_controller
                            if t["predicted_instability"]
                        )
                        mean_delta_iters = sum(
                            t["delta_iterations"] for t in trial_controller
                        ) / n_tc
                        mean_delta_success = sum(
                            t["delta_success"] for t in trial_controller
                        ) / n_tc
                        controlled_success_count = sum(
                            1 for t in trial_controller
                            if t["controlled_metrics"]["success"]
                        )
                        baseline_success_count = sum(
                            1 for t in trial_controller
                            if t["baseline_metrics"]["success"]
                        )
                        # v7.1.0: Cluster scheduling metrics.
                        cluster_sizes = [
                            t["cluster_size"] for t in trial_controller
                        ]
                        cluster_fractions = [
                            t["cluster_priority_fraction"]
                            for t in trial_controller
                        ]
                        cluster_active_count = sum(
                            1 for t in trial_controller
                            if t["cluster_control_enabled"]
                        )
                        mean_cluster_size = round(
                            sum(cluster_sizes) / n_tc, 12,
                        )
                        mean_cluster_priority_fraction = round(
                            sum(cluster_fractions) / n_tc, 12,
                        )
                        cluster_schedule_activation_rate = round(
                            cluster_active_count / n_tc, 12,
                        )
                        sdc_results[mode_name][p][distance] = {
                            "mean_controller_bp_failure_risk": round(
                                mean_bp_failure_risk, 12,
                            ),
                            "controller_instability_fraction": round(
                                instability_count / n_tc, 12,
                            ),
                            "mean_controlled_delta_iterations": round(
                                mean_delta_iters, 12,
                            ),
                            "mean_controlled_delta_success": round(
                                mean_delta_success, 12,
                            ),
                            "mean_controller_accuracy": round(
                                controlled_success_count / n_tc, 12,
                            ),
                            "mean_controller_error_rate": round(
                                1.0 - (controlled_success_count / n_tc), 12,
                            ),
                            "mean_baseline_accuracy": round(
                                baseline_success_count / n_tc, 12,
                            ),
                            "mean_cluster_size": mean_cluster_size,
                            "mean_cluster_priority_fraction": (
                                mean_cluster_priority_fraction
                            ),
                            "cluster_schedule_activation_rate": (
                                cluster_schedule_activation_rate
                            ),
                            "num_trials": n_tc,
                            "per_trial": trial_controller,
                        }
        out["spectral_decoder_controller"] = sdc_results

    # v6.5.0: Risk-aware damping experiment.
    if enable_risk_aware_damping_experiment and "spectral_failure_risk" in out:
        from qec.experiments.risk_aware_damping import (
            run_risk_aware_damping_experiment,
        )
        rad_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        sfr_data = out["spectral_failure_risk"]
        for mode_name in MODE_ORDER:
            rad_results[mode_name] = {}
            sfr_mode = sfr_data.get(mode_name, {})
            for p in p_values:
                rad_results[mode_name][p] = {}
                sfr_p = sfr_mode.get(p, {})
                for distance in distances:
                    sfr_cell = sfr_p.get(distance)
                    if sfr_cell is None:
                        continue
                    per_trial_risks = sfr_cell.get("per_trial", [])
                    if not per_trial_risks:
                        continue
                    H = codes[distance]
                    instances = all_instances[(distance, p)]
                    trial_experiments: list[dict[str, Any]] = []
                    for t_idx, (inst, risk_t) in enumerate(
                        zip(instances, per_trial_risks),
                    ):
                        exp = run_risk_aware_damping_experiment(
                            H,
                            inst["llr"],
                            inst["s"],
                            risk_t,
                            max_iters=max_iters,
                        )
                        trial_experiments.append(exp)
                    if trial_experiments:
                        n_exp = len(trial_experiments)
                        mean_delta_iters = sum(
                            t["delta_iterations"] for t in trial_experiments
                        ) / n_exp
                        mean_delta_success = sum(
                            t["delta_success"] for t in trial_experiments
                        ) / n_exp
                        rad_results[mode_name][p][distance] = {
                            "mean_delta_iterations": round(mean_delta_iters, 12),
                            "mean_delta_success": round(mean_delta_success, 12),
                            "num_trials": n_exp,
                            "per_trial": trial_experiments,
                        }
        out["risk_aware_damping_experiment"] = rad_results

    # v6.5.0: Risk-guided perturbation experiment.
    if enable_risk_guided_perturbation_experiment and "spectral_failure_risk" in out:
        from qec.experiments.risk_guided_perturbation import (
            run_risk_guided_perturbation,
        )
        rgp_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        sfr_data = out["spectral_failure_risk"]
        for mode_name in MODE_ORDER:
            rgp_results[mode_name] = {}
            sfr_mode = sfr_data.get(mode_name, {})
            for p in p_values:
                rgp_results[mode_name][p] = {}
                sfr_p = sfr_mode.get(p, {})
                for distance in distances:
                    sfr_cell = sfr_p.get(distance)
                    if sfr_cell is None:
                        continue
                    per_trial_risks = sfr_cell.get("per_trial", [])
                    if not per_trial_risks:
                        continue
                    H = codes[distance]
                    instances = all_instances[(distance, p)]
                    trial_experiments: list[dict[str, Any]] = []
                    for t_idx, (inst, risk_t) in enumerate(
                        zip(instances, per_trial_risks),
                    ):
                        exp = run_risk_guided_perturbation(
                            H,
                            inst["llr"],
                            inst["s"],
                            risk_t,
                            max_iters=max_iters,
                        )
                        trial_experiments.append(exp)
                    if trial_experiments:
                        n_exp = len(trial_experiments)
                        mean_delta_iters = sum(
                            t["delta_iterations"] for t in trial_experiments
                        ) / n_exp
                        mean_delta_success = sum(
                            t["delta_success"] for t in trial_experiments
                        ) / n_exp
                        n_stalls = sum(
                            1 for t in trial_experiments
                            if t["stall_detected"]
                        )
                        n_perturbed = sum(
                            1 for t in trial_experiments
                            if t["perturbation_applied"]
                        )
                        rgp_results[mode_name][p][distance] = {
                            "mean_delta_iterations": round(mean_delta_iters, 12),
                            "mean_delta_success": round(mean_delta_success, 12),
                            "num_stalls_detected": n_stalls,
                            "num_perturbations_applied": n_perturbed,
                            "num_trials": n_exp,
                            "per_trial": trial_experiments,
                        }
        out["risk_guided_perturbation_experiment"] = rgp_results

    # v6.6.0: Tanner graph repair experiment.
    if enable_tanner_graph_repair_experiment and "spectral_failure_risk" in out:
        from qec.experiments.tanner_graph_repair import (
            run_tanner_graph_repair_experiment,
        )
        tgr_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        sfr_data = out["spectral_failure_risk"]
        for mode_name in MODE_ORDER:
            tgr_results[mode_name] = {}
            sfr_mode = sfr_data.get(mode_name, {})
            for p in p_values:
                tgr_results[mode_name][p] = {}
                sfr_p = sfr_mode.get(p, {})
                for distance in distances:
                    sfr_cell = sfr_p.get(distance)
                    if sfr_cell is None:
                        continue
                    per_trial_risks = sfr_cell.get("per_trial", [])
                    if not per_trial_risks:
                        continue
                    H = codes[distance]
                    instances = all_instances[(distance, p)]
                    trial_experiments: list[dict[str, Any]] = []
                    for t_idx, (inst, risk_t) in enumerate(
                        zip(instances, per_trial_risks),
                    ):
                        exp = run_tanner_graph_repair_experiment(
                            H,
                            inst["llr"],
                            inst["s"],
                            risk_t,
                            max_iters=max_iters,
                        )
                        trial_experiments.append(exp)
                    if trial_experiments:
                        n_exp = len(trial_experiments)
                        mean_delta_iters = sum(
                            t["delta_iterations"] for t in trial_experiments
                        ) / n_exp
                        mean_delta_success = sum(
                            t["delta_success"] for t in trial_experiments
                        ) / n_exp
                        mean_repair_improvement = sum(
                            t["repair_score_improvement"]
                            for t in trial_experiments
                        ) / n_exp
                        n_repaired = sum(
                            1 for t in trial_experiments
                            if t["best_swap"] is not None
                        )
                        tgr_results[mode_name][p][distance] = {
                            "mean_delta_iterations": round(mean_delta_iters, 12),
                            "mean_delta_success": round(mean_delta_success, 12),
                            "mean_repair_score_improvement": round(
                                mean_repair_improvement, 12,
                            ),
                            "num_repairs_applied": n_repaired,
                            "num_trials": n_exp,
                            "per_trial": trial_experiments,
                        }
        out["tanner_graph_repair_experiment"] = tgr_results

    # v6.7.0: Spectral graph optimization experiment.
    if enable_spectral_graph_optimization and "spectral_failure_risk" in out:
        from qec.experiments.tanner_graph_repair import (
            run_spectral_graph_optimization_experiment,
        )
        sgo_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        sfr_data_sgo = out["spectral_failure_risk"]
        for mode_name in MODE_ORDER:
            sgo_results[mode_name] = {}
            sfr_mode = sfr_data_sgo.get(mode_name, {})
            for p in p_values:
                sgo_results[mode_name][p] = {}
                sfr_p = sfr_mode.get(p, {})
                for distance in distances:
                    sfr_cell = sfr_p.get(distance)
                    if sfr_cell is None:
                        continue
                    per_trial_risks = sfr_cell.get("per_trial", [])
                    if not per_trial_risks:
                        continue
                    H = codes[distance]
                    instances = all_instances[(distance, p)]
                    trial_experiments: list[dict[str, Any]] = []
                    for t_idx, (inst, risk_t) in enumerate(
                        zip(instances, per_trial_risks),
                    ):
                        exp = run_spectral_graph_optimization_experiment(
                            H,
                            inst["llr"],
                            inst["s"],
                            risk_t,
                            max_iters=max_iters,
                        )
                        trial_experiments.append(exp)
                    if trial_experiments:
                        n_exp = len(trial_experiments)
                        mean_delta_iters = sum(
                            t["delta_iterations"] for t in trial_experiments
                        ) / n_exp
                        mean_delta_success = sum(
                            t["delta_success"] for t in trial_experiments
                        ) / n_exp
                        mean_spectral_improvement = sum(
                            t["spectral_improvement"]
                            for t in trial_experiments
                        ) / n_exp
                        n_optimized = sum(
                            1 for t in trial_experiments
                            if t["best_swap"] is not None
                        )
                        sgo_results[mode_name][p][distance] = {
                            "mean_delta_iterations": round(mean_delta_iters, 12),
                            "mean_delta_success": round(mean_delta_success, 12),
                            "mean_spectral_improvement": round(
                                mean_spectral_improvement, 12,
                            ),
                            "num_optimizations_applied": n_optimized,
                            "num_trials": n_exp,
                            "per_trial": trial_experiments,
                        }
        out["spectral_graph_optimization"] = sgo_results

    # v7.5.0: Spectral graph optimize (predictor-guided pipeline).
    if enable_spectral_graph_optimize:
        sgo2_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        for mode_name in MODE_ORDER:
            sgo2_results[mode_name] = {}
            for p in p_values:
                sgo2_results[mode_name][p] = {}
                for distance in distances:
                    H = codes[distance]
                    opt_result = run_spectral_graph_optimization(
                        H, max_iterations=10, max_candidates=10,
                    )
                    sgo2_results[mode_name][p][distance] = opt_result
        out["spectral_graph_optimize"] = sgo2_results

    # v7.3.0: Spectral graph repair loop experiment.
    if enable_spectral_graph_repair_loop and "spectral_failure_risk" in out and "bp_stability_predictor" in out and nb_localization_results:
        sgrl_results: dict[str, dict[float, dict[int, dict[str, Any]]]] = {}
        sfr_data_sgrl = out["spectral_failure_risk"]
        bsp_data_sgrl = out["bp_stability_predictor"]
        spm_data_sgrl = out.get("spectral_phase_map", {})
        for mode_name in MODE_ORDER:
            sgrl_results[mode_name] = {}
            sfr_mode_sgrl = sfr_data_sgrl.get(mode_name, {})
            bsp_mode_sgrl = bsp_data_sgrl.get(mode_name, {})
            for p in p_values:
                sgrl_results[mode_name][p] = {}
                sfr_p_sgrl = sfr_mode_sgrl.get(p, {})
                bsp_p_sgrl = bsp_mode_sgrl.get(p, {})
                for distance in distances:
                    sfr_cell_sgrl = sfr_p_sgrl.get(distance)
                    bsp_cell_sgrl = bsp_p_sgrl.get(distance)
                    if sfr_cell_sgrl is None or bsp_cell_sgrl is None:
                        continue
                    per_trial_risks_sgrl = sfr_cell_sgrl.get("per_trial", [])
                    per_trial_preds_sgrl = bsp_cell_sgrl.get("per_trial", [])
                    if not per_trial_risks_sgrl or not per_trial_preds_sgrl:
                        continue
                    H_sgrl = codes[distance]
                    m_sgrl, n_sgrl = H_sgrl.shape
                    num_edges_sgrl = int(np.count_nonzero(H_sgrl))
                    avg_var_deg_sgrl = (
                        num_edges_sgrl / n_sgrl if n_sgrl > 0 else 0.0
                    )
                    avg_chk_deg_sgrl = (
                        num_edges_sgrl / m_sgrl if m_sgrl > 0 else 0.0
                    )
                    loc_sgrl = nb_localization_results.get(distance, {})
                    nb_max_ipr_sgrl = loc_sgrl.get("nb_max_ipr", 0.0)
                    if nb_max_ipr_sgrl is None:
                        ipr_scores_sgrl = loc_sgrl.get("ipr_scores", [])
                        nb_max_ipr_sgrl = max(ipr_scores_sgrl) if ipr_scores_sgrl else 0.0
                    instances_sgrl = all_instances[(distance, p)]
                    trial_repair_results: list[dict[str, Any]] = []
                    n_sgrl_trials = min(
                        len(per_trial_risks_sgrl),
                        len(per_trial_preds_sgrl),
                        len(instances_sgrl),
                    )
                    for i_sgrl in range(n_sgrl_trials):
                        risk_t_sgrl = per_trial_risks_sgrl[i_sgrl]
                        pred_t_sgrl = per_trial_preds_sgrl[i_sgrl]
                        inst_sgrl = instances_sgrl[i_sgrl]
                        exp_sgrl = run_spectral_graph_repair_loop(
                            H_sgrl,
                            inst_sgrl["llr"],
                            inst_sgrl["s"],
                            risk_t_sgrl,
                            nb_spectral_radius=pred_t_sgrl.get(
                                "spectral_radius", 0.0,
                            ),
                            spectral_instability_ratio=pred_t_sgrl.get(
                                "spectral_instability_ratio", 0.0,
                            ),
                            ipr_localization_score=float(nb_max_ipr_sgrl),
                            avg_variable_degree=round(avg_var_deg_sgrl, 12),
                            avg_check_degree=round(avg_chk_deg_sgrl, 12),
                            max_candidates=10,
                            max_iters=max_iters,
                            enable_multistep_repair=enable_spectral_multistep_repair,
                            max_repair_depth=2 if enable_spectral_multistep_repair else 1,
                            enable_pruning=True,
                        )
                        trial_repair_results.append(exp_sgrl)
                    if trial_repair_results:
                        aggregate_sgrl = compute_repair_loop_aggregate_metrics(
                            trial_repair_results,
                        )
                        aggregate_sgrl["per_trial"] = trial_repair_results
                        sgrl_results[mode_name][p][distance] = aggregate_sgrl
        out["spectral_graph_repair_loop"] = sgrl_results

    return out


# ── Determinism check ────────────────────────────────────────────────

def run_determinism_check(
    seed: int = DEFAULT_SEED,
    distance: int = 3,
    p: float = 0.01,
    trials: int = 50,
    max_iters: int = DEFAULT_MAX_ITERS,
    bp_mode: str = DEFAULT_BP_MODE,
) -> dict[str, Any]:
    """Run a single configuration twice and compare results."""
    results = []
    for _ in range(2):
        rng = np.random.default_rng(seed)
        H = create_code(name="rate_0.50", lifting_size=distance, seed=seed).H_X
        instances = _pre_generate_instances(H, p, trials, rng)
        # Run baseline mode.
        r = run_mode("baseline", H, instances, max_iters=max_iters, bp_mode=bp_mode)
        results.append(r)

    r1, r2 = results
    fer_match = r1["fer"] == r2["fer"]
    audit_match = r1["audit_summary"] == r2["audit_summary"]
    passed = fer_match and audit_match

    return {
        "passed": passed,
        "fer_match": fer_match,
        "audit_match": audit_match,
        "fer_run1": r1["fer"],
        "fer_run2": r2["fer"],
    }


# ── Printing ─────────────────────────────────────────────────────────

def print_activation_report(eval_result: dict[str, Any]) -> None:
    """Print the activation audit summary for all modes."""
    audits = eval_result["audits"]
    config = eval_result["config"]
    distances = config["distances"]
    p_values = config["p_values"]

    print("\n" + "=" * 72)
    print("ACTIVATION AUDIT REPORT")
    print("=" * 72)

    for mode_name in MODE_ORDER:
        for p in p_values:
            for distance in distances:
                audit = audits[mode_name][p][distance]
                print(f"\n--- {mode_name} | p={p} | d={distance} ---")
                print(f"  added_rows:     min={audit['added_rows_min']}"
                      f"  mean={audit['added_rows_mean']:.2f}"
                      f"  max={audit['added_rows_max']}")
                print(f"  augmented_rows: min={audit['augmented_rows_min']}"
                      f"  mean={audit['augmented_rows_mean']:.2f}"
                      f"  max={audit['augmented_rows_max']}")
                print(f"  H_checksum:     first={audit['H_checksum_first']}"
                      f"  last={audit['H_checksum_last']}")
                print(f"  synd_checksum:  first={audit['syndrome_checksum_first']}"
                      f"  last={audit['syndrome_checksum_last']}")
                print(f"  mean_iters:     {audit['mean_iters']:.2f}")
                print(f"  max_iters:      {audit['max_iters_observed']}")
                print(f"  frac_iters==1:  {audit['fraction_iters_eq_1']:.4f}")
                print(f"  frac_zero_synd: {audit['fraction_zero_syndrome']:.4f}")

                # Warnings.
                rpc_enabled = MODES[mode_name]["structural"].rpc.enabled
                if rpc_enabled and audit["added_rows_mean"] == 0:
                    print("  WARNING: RPC enabled but added_rows_mean == 0")
                if audit["fraction_iters_eq_1"] > 0.95:
                    print("  WARNING: fraction_iters_eq_1 > 0.95")
                expected_schedule = MODES[mode_name]["schedule"]
                structural = MODES[mode_name]["structural"]
                # Schedule mismatch can't happen in this harness, but flag
                # if the mode definition is inconsistent.
                if expected_schedule not in ("flooding", "geom_v1"):
                    print(f"  WARNING: unexpected schedule {expected_schedule}")
                if structural.centered_field:
                    print(f"  centered_field: ON")
                if structural.pseudo_prior:
                    print(f"  pseudo_prior: ON (kappa={structural.pseudo_prior_strength})")


def print_dps_table(eval_result: dict[str, Any]) -> None:
    """Print the DPS slope table."""
    slopes = eval_result["slopes"]
    config = eval_result["config"]
    p_values = config["p_values"]

    print("\n" + "=" * 72)
    print("DPS SLOPE TABLE")
    print("=" * 72)

    # Header.
    p_headers = [f"p={p}" for p in p_values]
    header = f"{'Mode':<16}" + "".join(f"{h:>12}" for h in p_headers) + "  Inverted?"
    print(header)
    print("-" * len(header))

    for mode_name in MODE_ORDER:
        row = f"{mode_name:<16}"
        any_inverted = False
        for p in p_values:
            slope = slopes[mode_name][p]
            row += f"{slope:>12.4f}"
            if slope > 0:
                any_inverted = True
        inv_marker = "  INVERTED" if any_inverted else ""
        print(row + inv_marker)


def print_determinism_result(det_result: dict[str, Any]) -> None:
    """Print determinism check result."""
    print("\n" + "=" * 72)
    status = "PASS" if det_result["passed"] else "FAIL"
    print(f"Determinism: {status}")
    if not det_result["passed"]:
        print(f"  FER match:   {det_result['fer_match']}")
        print(f"  Audit match: {det_result['audit_match']}")
        print(f"  FER run1:    {det_result['fer_run1']}")
        print(f"  FER run2:    {det_result['fer_run2']}")
    print("=" * 72)


# ── Main ─────────────────────────────────────────────────────────────

def print_energy_trace(eval_result: dict[str, Any]) -> None:
    """Print energy trace summary for modes that have it."""
    results = eval_result["results"]
    config = eval_result["config"]
    p_values = config["p_values"]
    distances = config["distances"]

    has_any = False
    for mode_name in MODE_ORDER:
        for p in p_values:
            for distance in distances:
                r = results[mode_name][p][distance]
                if "energy_traces" in r and r["energy_traces"]:
                    if not has_any:
                        print("\n" + "=" * 72)
                        print("ENERGY TRACE SUMMARY")
                        print("=" * 72)
                        has_any = True
                    traces = r["energy_traces"]
                    # Show first trial's trace.
                    first = traces[0]
                    print(f"\n--- {mode_name} | p={p} | d={distance} ---")
                    print("[ENERGY TRACE]")
                    for i, e in enumerate(first):
                        print(f"iter {i} : {e:.2f}")


def print_basin_statistics(eval_result: dict[str, Any]) -> None:
    """Print aggregate basin switching statistics per mode."""
    results = eval_result["results"]
    config = eval_result["config"]
    p_values = config["p_values"]
    distances = config["distances"]

    has_any = False
    for mode_name in MODE_ORDER:
        for p in p_values:
            for distance in distances:
                r = results[mode_name][p][distance]
                if "basin_switch_fraction" not in r:
                    continue
                if not has_any:
                    print("\n" + "=" * 72)
                    print("BASIN SWITCHING STATISTICS")
                    print("=" * 72)
                    has_any = True
                frac = r["basin_switch_fraction"]
                delta = r["energy_delta"]
                print(f"  {mode_name:<30} p={p}  d={distance}"
                      f"  switch_frac={frac:.4f}"
                      f"  mean_delta={delta:.6f}")
                if "basin_class_counts" in r:
                    counts = r["basin_class_counts"]
                    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
                    print(f"    classification: {', '.join(parts)}")


def print_decoder_report(eval_result: dict[str, Any]) -> None:
    """Print a console FER comparison table from decoder comparison data."""
    results = eval_result["results"]
    config = eval_result["config"]
    distances = config["distances"]
    p_values = config["p_values"]

    print("\nDecoder Comparison Report")
    print("-----------------------------------------------------------")
    print(f"{'mode':<24} {'distance':>8}  {'p':>8}  {'FER_ref':>8}  {'FER_exp':>8}  {'ΔFER':>8}")
    print("-----------------------------------------------------------")

    for mode_name, mode_results in results.items():
        for p in p_values:
            if p not in mode_results:
                continue
            for d in distances:
                r = mode_results[p].get(d)
                if r is None or "decoder_comparison" not in r:
                    continue
                comps = r["decoder_comparison"]
                n = len(comps)
                if n == 0:
                    continue
                fer_ref = sum(1 for c in comps if not c["reference"]["success"]) / n
                fer_exp = sum(1 for c in comps if not c["experimental"]["success"]) / n
                delta = fer_exp - fer_ref
                print(f"{mode_name:<24} {d:>8}  {p:>8.4f}  {fer_ref:>8.4f}  {fer_exp:>8.4f}  {delta:>+8.4f}")
    print()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="v4.0.0 — BP Free-Energy Landscape Diagnostics",
    )
    parser.add_argument("--landscape", action="store_true",
                        help="Enable landscape diagnostics and basin switching")
    parser.add_argument("--iteration-diagnostics", action="store_true",
                        help="Enable iteration-trace diagnostics (PEI, BOI, OD, CIS, CVF)")
    parser.add_argument("--bp-dynamics", action="store_true",
                        help="Enable BP dynamics regime analysis (MSI, CPI, TSL, LEC, CVNE, GOS, EDS, BTI)")
    parser.add_argument("--bp-transitions", action="store_true",
                        help="Enable BP regime transition analysis (regime trace, dwell times, events)")
    parser.add_argument("--bp-phase-diagram", action="store_true",
                        help="Enable BP phase diagram aggregation (implicitly enables --bp-transitions)")
    parser.add_argument("--bp-freeze-detection", action="store_true",
                        help="Enable BP freeze detection (early metastability detection)")
    parser.add_argument("--bp-fixed-point-analysis", action="store_true",
                        help="Enable BP fixed-point trap analysis (correct/incorrect/degenerate classification)")
    parser.add_argument("--bp-basin-analysis", action="store_true",
                        help="Enable BP basin-of-attraction analysis (perturbation-based basin geometry)")
    parser.add_argument("--bp-landscape-map", action="store_true",
                        help="Enable BP attractor landscape mapping (attractor enumeration, pseudocodeword detection)")
    parser.add_argument("--bp-barrier-analysis", action="store_true",
                        help="Enable BP free-energy barrier estimation (escape perturbation measurement)")
    parser.add_argument("--bp-boundary-analysis", action="store_true",
                        help="Enable BP boundary analysis (attractor basin distance estimation)")
    parser.add_argument("--tanner-spectral-analysis", action="store_true",
                        help="Enable Tanner spectral fragility diagnostics (spectral gap, eigenmode localization)")
    parser.add_argument("--spectral-boundary-alignment", action="store_true",
                        help="Enable spectral-boundary alignment diagnostics (v5.5)")
    parser.add_argument("--spectral-trapping-sets", action="store_true",
                        help="Enable spectral trapping-set diagnostics (v5.6)")
    parser.add_argument("--bp-phase-space", action="store_true",
                        help="Enable BP phase-space exploration (v5.7)")
    parser.add_argument("--ternary-topology", action="store_true",
                        help="Enable ternary decoder topology classification (v5.7, implies --bp-phase-space)")
    parser.add_argument("--ternary-transition-metrics", action="store_true",
                        help="Add ternary transition metrics to output (v5.8, implies --ternary-topology)")
    parser.add_argument("--ternary-basin-probe", action="store_true",
                        help="Perform deterministic local basin probe around final LLR state (v5.8, implies --ternary-topology)")
    parser.add_argument("--phase-diagram", action="store_true",
                        help="Enable decoder phase diagram generation (v5.9, implies --ternary-topology --ternary-transition-metrics)")
    parser.add_argument("--nb-spectrum", action="store_true",
                        help="Enable non-backtracking spectrum diagnostics (v6.0)")
    parser.add_argument("--bethe-hessian", action="store_true",
                        help="Enable Bethe Hessian spectral diagnostics (v6.0)")
    parser.add_argument("--bp-stability", action="store_true",
                        help="Enable BP stability proxy diagnostics (v6.0, implies --nb-spectrum --bethe-hessian)")
    parser.add_argument("--bp-jacobian-estimator", action="store_true",
                        help="Enable BP Jacobian spectral radius estimator (v6.0)")
    parser.add_argument("--nb-localization", action="store_true",
                        help="Enable non-backtracking localization diagnostics (v6.1)")
    parser.add_argument("--nb-trapping-candidates", action="store_true",
                        help="Enable spectral trapping-set candidate detection (v6.2, implies --nb-localization)")
    parser.add_argument("--spectral-bp-alignment", action="store_true",
                        help="Enable spectral-BP attractor alignment diagnostics (v6.3, implies --nb-trapping-candidates --iteration-diagnostics)")
    parser.add_argument("--spectral-failure-risk", action="store_true",
                        help="Enable spectral failure risk scoring (v6.4, implies --spectral-bp-alignment)")
    parser.add_argument("--risk-aware-damping-experiment", action="store_true",
                        help="Enable risk-aware damping experiment (v6.5, implies --spectral-failure-risk)")
    parser.add_argument("--risk-guided-perturbation-experiment", action="store_true",
                        help="Enable risk-guided perturbation experiment (v6.5, implies --spectral-failure-risk)")
    parser.add_argument("--tanner-graph-repair-experiment", action="store_true",
                        help="Enable Tanner graph fragility repair experiment (v6.6, implies --spectral-failure-risk)")
    parser.add_argument("--spectral-graph-optimization", action="store_true",
                        help="Spectral Tanner graph optimization via non-backtracking spectral radius minimization (v6.7, implies --spectral-failure-risk)")
    parser.add_argument("--bp-stability-predictor", action="store_true",
                        help="Enable BP stability predictor (v6.8, implies --spectral-failure-risk)")
    parser.add_argument("--bp-prediction-validation", action="store_true",
                        help="Enable BP prediction validation (v6.9, implies --bp-stability-predictor)")
    parser.add_argument("--spectral-decoder-controller", action="store_true",
                        help="Enable spectral decoder controller experiment (v7.0, implies --bp-stability-predictor)")
    parser.add_argument("--spectral-cluster-control", action="store_true",
                        help="Enable cluster-aware decoder scheduling (v7.1, implies --spectral-decoder-controller --bp-stability-predictor)")
    parser.add_argument("--spectral-phase-map", action="store_true",
                        help="Enable spectral instability phase map (v7.2, implies --bp-stability-predictor)")
    parser.add_argument("--spectral-graph-repair-loop", action="store_true",
                        help="Enable spectral graph repair loop experiment (v7.3, implies --spectral-phase-map)")
    parser.add_argument("--spectral-multistep-repair", action="store_true",
                        help="Enable multi-step spectral graph repair with pruning (v7.3.x, implies --spectral-graph-repair-loop)")
    parser.add_argument("--spectral-design-analysis", action="store_true",
                        help="Enable spectral Tanner graph design analysis (v7.4, implies --spectral-failure-risk)")
    parser.add_argument("--spectral-graph-optimize", action="store_true",
                        help="Enable predictor-guided spectral graph optimization pipeline (v7.5, implies --spectral-failure-risk)")
    parser.add_argument("--spectral-optimizer-sanity", action="store_true",
                        help="Run spectral optimizer sanity experiment + predictor probe (v7.5)")
    parser.add_argument("--phase-grid-x", type=str, default="physical_error_rate",
                        help="Phase diagram x-axis parameter name (default: physical_error_rate)")
    parser.add_argument("--phase-grid-y", type=str, default="code_distance",
                        help="Phase diagram y-axis parameter name (default: code_distance)")
    parser.add_argument("--phase-grid-x-values", type=float, nargs="+", default=None,
                        help="Phase diagram x-axis values (default: uses --p-values)")
    parser.add_argument("--phase-grid-y-values", type=float, nargs="+", default=None,
                        help="Phase diagram y-axis values (default: uses --distances)")
    parser.add_argument("--phase-diagram-output", type=str, default=None,
                        help="Output path for phase diagram JSON artifact")
    parser.add_argument("--decoder", type=str, default="reference",
                        choices=["reference", "experimental"],
                        help="Decoder implementation to use (default: reference)")
    parser.add_argument("--compare-decoders", action="store_true",
                        help="Run both reference and experimental decoders on each instance")
    parser.add_argument("--paired-seed", action="store_true",
                        help="Ensure both decoders share the same deterministic seed per trial")
    parser.add_argument("--paired-errors", action="store_true",
                        help="Ensure both decoders receive identical error realizations")
    parser.add_argument("--decoder-report", action="store_true",
                        help="Print FER comparison table after evaluation")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--bp-mode", type=str, default=DEFAULT_BP_MODE)
    parser.add_argument("--distances", type=int, nargs="+",
                        default=DEFAULT_DISTANCES)
    parser.add_argument("--p-values", type=float, nargs="+",
                        default=DEFAULT_P_VALUES)
    return parser.parse_args()


def _run_phase_diagram(args: argparse.Namespace, eval_result: dict[str, Any]) -> None:
    """Build and print a decoder phase diagram from evaluation results.

    Extracts ternary topology results from the existing evaluation run
    and aggregates them into a phase-diagram JSON artifact.
    """
    import json as _json

    # Determine grid axes.
    x_name = args.phase_grid_x
    y_name = args.phase_grid_y
    x_values = args.phase_grid_x_values if args.phase_grid_x_values else [float(v) for v in args.p_values]
    y_values = args.phase_grid_y_values if args.phase_grid_y_values else [float(v) for v in args.distances]

    grid = make_phase_grid(x_name, x_values, y_name, y_values)

    results = eval_result["results"]

    # Use the first mode in the evaluation for phase diagram.
    mode_name = MODE_ORDER[0]

    def trial_runner(x_val, y_val):
        """Extract ternary topology results for grid point (x, y).

        Maps x to p-value and y to distance by finding the closest
        matching values in the evaluation results.
        """
        # Map x to p-value and y to distance.
        p = float(x_val)
        d = int(y_val)

        mode_results = results.get(mode_name, {})
        p_results = mode_results.get(p, {})
        d_result = p_results.get(d, {})

        ternary_list = d_result.get("ternary_topology", [])
        return ternary_list

    phase_diagram = build_decoder_phase_diagram(grid, trial_runner)
    boundary_analysis = analyze_phase_boundaries(phase_diagram)

    # Print phase-map summary table.
    print("\n" + "=" * 60)
    print("v5.9.0 — Decoder Phase Diagram Summary")
    print("=" * 60)
    print(f"Grid: {x_name} x {y_name}")
    print(f"  {x_name}: {x_values}")
    print(f"  {y_name}: {y_values}")
    print(f"  Mode: {mode_name}")

    # Dominant phase table.
    print(f"\nDominant Phase Map ({x_name} → columns, {y_name} → rows):")
    header = f"{'':>12s}"
    for x in x_values:
        header += f" {x:>10}"
    print(header)

    for y in y_values:
        row = f"{y:>12}"
        for x in x_values:
            cell = _find_cell(phase_diagram["cells"], x, y)
            if cell is not None:
                phase_label = {1: "+1", 0: " 0", -1: "-1"}.get(cell["dominant_phase"], " ?")
                row += f" {phase_label:>10s}"
            else:
                row += f" {'--':>10s}"
        print(row)

    # Summary counts.
    cells = phase_diagram["cells"]
    success_cells = sum(1 for c in cells if c["dominant_phase"] == 1)
    boundary_cells = sum(1 for c in cells if c["dominant_phase"] == 0)
    failure_cells = sum(1 for c in cells if c["dominant_phase"] == -1)
    total_cells = len(cells)

    print(f"\nPhase counts: {success_cells} success (+1), "
          f"{boundary_cells} boundary (0), {failure_cells} failure (-1) "
          f"out of {total_cells} cells")

    bs = boundary_analysis["boundary_summary"]
    print(f"Boundary analysis: {bs['num_boundary_cells']} boundary, "
          f"{bs['num_mixed_cells']} mixed, {bs['num_critical_cells']} critical")

    # v6.0: ASCII phase heatmap.
    print_phase_heatmap(phase_diagram)

    # Write JSON output if requested.
    if args.phase_diagram_output:
        output = {
            "phase_diagram": phase_diagram,
            "boundary_analysis": boundary_analysis,
        }
        with open(args.phase_diagram_output, "w") as f:
            _json.dump(output, f, indent=2)
        print(f"\nPhase diagram written to: {args.phase_diagram_output}")


def _find_cell(
    cells: list[dict[str, Any]],
    x: float | int,
    y: float | int,
) -> dict[str, Any] | None:
    """Find a cell by (x, y) coordinates."""
    for cell in cells:
        if cell["x"] == x and cell["y"] == y:
            return cell
    return None


def main() -> None:
    """Run full evaluation and print all reports."""
    args = _parse_args()

    print("v4.0.0 — BP Free-Energy Landscape Diagnostics")
    print(f"Seed: {args.seed}")
    print(f"Distances: {args.distances}")
    print(f"P values: {args.p_values}")
    print(f"Trials: {args.trials}")
    print(f"Max iters: {args.max_iters}")
    print(f"BP mode: {args.bp_mode}")
    if args.landscape:
        print("Landscape diagnostics: ENABLED")
    if args.iteration_diagnostics:
        print("Iteration-trace diagnostics: ENABLED")
    if args.bp_dynamics:
        print("BP dynamics regime analysis: ENABLED")
    if args.bp_transitions:
        print("BP regime transition analysis: ENABLED")
    if args.bp_phase_diagram:
        print("BP phase diagram analysis: ENABLED")
    if args.bp_freeze_detection:
        print("BP freeze detection: ENABLED")
    if args.bp_fixed_point_analysis:
        print("BP fixed-point trap analysis: ENABLED")
    if args.bp_basin_analysis:
        print("BP basin-of-attraction analysis: ENABLED")
    if args.bp_landscape_map:
        print("BP attractor landscape mapping: ENABLED")
    if args.bp_barrier_analysis:
        print("BP free-energy barrier estimation: ENABLED")
    if args.bp_boundary_analysis:
        print("BP boundary analysis: ENABLED")
    if args.tanner_spectral_analysis:
        print("Tanner spectral fragility diagnostics: ENABLED")
    if args.spectral_boundary_alignment:
        print("Spectral-boundary alignment diagnostics: ENABLED")
    if args.spectral_trapping_sets:
        print("Spectral trapping-set diagnostics: ENABLED")
    if args.bp_phase_space:
        print("BP phase-space exploration: ENABLED")
    if args.ternary_topology:
        print("Ternary decoder topology classification: ENABLED")
    if args.phase_diagram:
        print("Decoder phase diagram generation: ENABLED")
    if args.nb_spectrum:
        print("Non-backtracking spectrum diagnostics: ENABLED")
    if args.bethe_hessian:
        print("Bethe Hessian spectral diagnostics: ENABLED")
    if args.bp_stability:
        print("BP stability proxy diagnostics: ENABLED")
    if args.bp_jacobian_estimator:
        print("BP Jacobian spectral radius estimator: ENABLED")
    if args.nb_localization:
        print("Non-backtracking localization diagnostics: ENABLED")
    if args.nb_trapping_candidates:
        print("Spectral trapping-set candidate detection: ENABLED")
    if args.spectral_bp_alignment:
        print("Spectral-BP attractor alignment diagnostics: ENABLED")
    if args.spectral_failure_risk:
        print("Spectral failure risk scoring: ENABLED")
    if args.risk_aware_damping_experiment:
        print("Risk-aware damping experiment: ENABLED")
    if args.risk_guided_perturbation_experiment:
        print("Risk-guided perturbation experiment: ENABLED")
    if args.spectral_graph_optimization:
        print("Spectral graph optimization: ENABLED")
    if args.bp_stability_predictor:
        print("BP stability predictor: ENABLED")
    if args.bp_prediction_validation:
        print("BP prediction validation: ENABLED")
    if args.spectral_decoder_controller:
        print("Spectral decoder controller: ENABLED")
    if args.spectral_cluster_control:
        print("Spectral cluster control: ENABLED")
    if args.spectral_phase_map:
        print("Spectral instability phase map: ENABLED")
    if args.spectral_graph_repair_loop:
        print("Spectral graph repair loop: ENABLED")
    if args.spectral_multistep_repair:
        print("Spectral multi-step repair: ENABLED")
    if args.spectral_design_analysis:
        print("Spectral graph design analysis: ENABLED")
    if args.spectral_graph_optimize:
        print("Spectral graph optimize: ENABLED")
    if args.spectral_optimizer_sanity:
        print("Spectral optimizer sanity: ENABLED")
    print(f"Decoder: {args.decoder}")
    if args.compare_decoders:
        print("Decoder comparison mode: ENABLED")
    if args.paired_seed:
        print("Paired seed: ENABLED")
    if args.paired_errors:
        print("Paired errors: ENABLED")
    if args.decoder_report:
        print("Decoder report: ENABLED")

    # Resolve decoder callable.
    selected_decoder = get_decoder(args.decoder)

    # Full evaluation.
    eval_result = run_evaluation(
        seed=args.seed,
        distances=args.distances,
        p_values=args.p_values,
        trials=args.trials,
        max_iters=args.max_iters,
        bp_mode=args.bp_mode,
        enable_energy_trace=args.landscape or args.iteration_diagnostics or args.bp_dynamics or args.bp_transitions or args.bp_phase_diagram or args.bp_freeze_detection or args.bp_fixed_point_analysis or args.bp_basin_analysis or args.bp_landscape_map or args.bp_barrier_analysis or args.spectral_bp_alignment or args.spectral_cluster_control,
        enable_landscape=args.landscape,
        enable_iteration_diagnostics=args.iteration_diagnostics or args.bp_dynamics or args.bp_transitions or args.bp_phase_diagram or args.bp_freeze_detection or args.bp_fixed_point_analysis or args.bp_basin_analysis or args.bp_landscape_map or args.bp_barrier_analysis or args.spectral_bp_alignment or args.spectral_cluster_control,
        enable_bp_dynamics=args.bp_dynamics or args.bp_transitions or args.bp_phase_diagram or args.bp_freeze_detection,
        enable_bp_transitions=args.bp_transitions or args.bp_phase_diagram,
        enable_bp_phase_diagram=args.bp_phase_diagram,
        enable_bp_freeze_detection=args.bp_freeze_detection,
        enable_bp_fixed_point_analysis=args.bp_fixed_point_analysis or args.bp_basin_analysis or args.bp_landscape_map or args.bp_barrier_analysis,
        enable_bp_basin_analysis=args.bp_basin_analysis or args.bp_landscape_map or args.bp_barrier_analysis,
        enable_bp_landscape_map=args.bp_landscape_map or args.bp_barrier_analysis,
        enable_bp_barrier_analysis=args.bp_barrier_analysis or args.spectral_boundary_alignment,
        enable_bp_boundary_analysis=args.bp_boundary_analysis or args.spectral_boundary_alignment,
        enable_tanner_spectral_analysis=args.tanner_spectral_analysis or args.spectral_boundary_alignment or args.spectral_trapping_sets,
        enable_spectral_boundary_alignment=args.spectral_boundary_alignment,
        enable_spectral_trapping_sets=args.spectral_trapping_sets,
        enable_bp_phase_space=args.bp_phase_space or args.ternary_topology or args.ternary_transition_metrics or args.ternary_basin_probe or args.phase_diagram,
        enable_ternary_topology=args.ternary_topology or args.ternary_transition_metrics or args.ternary_basin_probe or args.phase_diagram,
        enable_ternary_transition_metrics=args.ternary_transition_metrics or args.phase_diagram,
        enable_ternary_basin_probe=args.ternary_basin_probe,
        enable_spectral_bp_alignment=args.spectral_bp_alignment or args.spectral_failure_risk or args.risk_aware_damping_experiment or args.risk_guided_perturbation_experiment or args.tanner_graph_repair_experiment or args.spectral_graph_optimization or args.bp_stability_predictor or args.bp_prediction_validation or args.spectral_decoder_controller or args.spectral_cluster_control or args.spectral_phase_map or args.spectral_graph_repair_loop or args.spectral_multistep_repair or args.spectral_design_analysis or args.spectral_graph_optimize,
        enable_spectral_failure_risk=args.spectral_failure_risk or args.risk_aware_damping_experiment or args.risk_guided_perturbation_experiment or args.tanner_graph_repair_experiment or args.spectral_graph_optimization or args.bp_stability_predictor or args.bp_prediction_validation or args.spectral_decoder_controller or args.spectral_cluster_control or args.spectral_phase_map or args.spectral_graph_repair_loop or args.spectral_multistep_repair or args.spectral_design_analysis or args.spectral_graph_optimize,
        enable_risk_aware_damping_experiment=args.risk_aware_damping_experiment,
        enable_risk_guided_perturbation_experiment=args.risk_guided_perturbation_experiment,
        enable_tanner_graph_repair_experiment=args.tanner_graph_repair_experiment,
        enable_spectral_graph_optimization=args.spectral_graph_optimization,
        enable_bp_stability_predictor=args.bp_stability_predictor or args.bp_prediction_validation or args.spectral_decoder_controller or args.spectral_cluster_control or args.spectral_phase_map or args.spectral_graph_repair_loop or args.spectral_multistep_repair,
        enable_bp_prediction_validation=args.bp_prediction_validation,
        enable_spectral_decoder_controller=args.spectral_decoder_controller or args.spectral_cluster_control,
        enable_spectral_cluster_control=args.spectral_cluster_control,
        enable_spectral_phase_map=args.spectral_phase_map or args.spectral_graph_repair_loop or args.spectral_multistep_repair,
        enable_spectral_graph_repair_loop=args.spectral_graph_repair_loop or args.spectral_multistep_repair,
        enable_spectral_multistep_repair=args.spectral_multistep_repair,
        enable_spectral_graph_design_analysis=args.spectral_design_analysis,
        enable_spectral_graph_optimize=args.spectral_graph_optimize,
        decoder_fn=selected_decoder,
        compare_decoders=args.compare_decoders,
        paired_seed=args.paired_seed,
        paired_errors=args.paired_errors,
    )

    # Print reports.
    print_activation_report(eval_result)
    print_dps_table(eval_result)
    print_energy_trace(eval_result)

    if args.landscape:
        print_basin_statistics(eval_result)

    if args.decoder_report and args.compare_decoders:
        print_decoder_report(eval_result)

    # v5.9.0: Phase diagram generation.
    if args.phase_diagram:
        _run_phase_diagram(args, eval_result)

    # v7.5.0: Spectral optimizer sanity experiment.
    if args.spectral_optimizer_sanity:
        _rng_sanity = np.random.default_rng(args.seed)
        _code_sanity = create_code("rate_0.50", lifting_size=32, seed=args.seed)
        _H_sanity = _code_sanity.H_X
        _instances_sanity = _pre_generate_instances(
            _H_sanity, args.p_values[0], min(args.trials, 20), _rng_sanity,
        )
        sanity_report = run_spectral_optimizer_sanity_experiment(
            _H_sanity,
            _instances_sanity,
            decode_fn=selected_decoder,
            syndrome_fn=syndrome,
            max_iters=args.max_iters,
            bp_mode=args.bp_mode,
        )
        print_sanity_report(sanity_report)

    # Determinism check.
    det_result = run_determinism_check(
        seed=args.seed,
        max_iters=args.max_iters,
        bp_mode=args.bp_mode,
    )
    print_determinism_result(det_result)

    # Exit code.
    if not det_result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
