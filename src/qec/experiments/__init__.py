"""
Layer 1a — Decoder behaviour experiments (opt-in, additive).

Provides risk-aware decoder experiments that use spectral structural
signals (v6.0–v6.4) to guide decoder behaviour.  Each experiment runs
two deterministic decodes (baseline and experimental) and returns
comparison metrics.  Includes Tanner graph fragility repair (v6.6),
spectral Tanner graph optimization (v6.7), BP prediction
validation (v6.9), spectral instability phase map (v7.2), and
spectral graph repair loop (v7.3), and sensitivity-based
preconditioned graph optimization (v7.6).

Does not modify decoder internals.  Fully deterministic.

Lazy-loading: submodule imports are deferred to attribute access so that
importing lightweight modules (e.g. metrics_probe) does not trigger
heavy transitive dependencies.
"""

import importlib as _importlib

# Mapping of public names → (submodule, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "run_risk_aware_damping_experiment": (".risk_aware_damping", "run_risk_aware_damping_experiment"),
    "run_risk_guided_perturbation": (".risk_guided_perturbation", "run_risk_guided_perturbation"),
    "run_tanner_graph_repair_experiment": (".tanner_graph_repair", "run_tanner_graph_repair_experiment"),
    "run_spectral_graph_optimization_experiment": (".tanner_graph_repair", "run_spectral_graph_optimization_experiment"),
    "run_bp_prediction_validation": (".bp_prediction_validation", "run_bp_prediction_validation"),
    "compute_spectral_instability_score": (".spectral_instability_phase_map", "compute_spectral_instability_score"),
    "run_spectral_phase_map_experiment": (".spectral_instability_phase_map", "run_spectral_phase_map_experiment"),
    "compute_phase_map_aggregate_metrics": (".spectral_instability_phase_map", "compute_phase_map_aggregate_metrics"),
    "run_spectral_graph_repair_loop": (".spectral_graph_repair_loop", "run_spectral_graph_repair_loop"),
    "compute_repair_loop_aggregate_metrics": (".spectral_graph_repair_loop", "compute_repair_loop_aggregate_metrics"),
    "run_sensitivity_preconditioned_optimization": (".sensitivity_preconditioner", "run_sensitivity_preconditioned_optimization"),
    "run_sensitivity_preconditioner_experiment": (".sensitivity_preconditioner", "run_sensitivity_preconditioner_experiment"),
    "run_spectral_validation_experiment": (".spectral_validation", "run_spectral_validation_experiment"),
    "serialize_artifact": (".spectral_validation", "serialize_artifact"),
    "detect_eeec_anomaly": (".eeec_anomaly_scan", "detect_eeec_anomaly"),
    "run_eeec_anomaly_scan": (".eeec_anomaly_scan", "run_eeec_anomaly_scan"),
    "run_spectral_heatmap_experiment": (".spectral_heatmap_experiment", "run_spectral_heatmap_experiment"),
    "serialize_heatmap_artifact": (".spectral_heatmap_experiment", "serialize_heatmap_artifact"),
    "run_incremental_spectral_benchmark": (".incremental_spectral_benchmark", "run_incremental_spectral_benchmark"),
    "serialize_benchmark_artifact": (".incremental_spectral_benchmark", "serialize_benchmark_artifact"),
    "run_stability_phase_diagram_experiment": (".stability_phase_diagram", "run_stability_phase_diagram_experiment"),
    "serialize_phase_diagram_artifact": (".stability_phase_diagram", "serialize_phase_diagram_artifact"),
    "detect_metastable_bp_oscillation": (".stability_phase_diagram", "detect_metastable_bp_oscillation"),
    "estimate_bp_stability_boundary": (".stability_phase_diagram", "estimate_bp_stability_boundary"),
    "predict_spectral_stability_boundary": (".stability_phase_diagram", "predict_spectral_stability_boundary"),
    "log_most_unstable_subgraph": (".stability_phase_diagram", "log_most_unstable_subgraph"),
    "track_repair_boundary_shift": (".stability_phase_diagram", "track_repair_boundary_shift"),
    "ExperimentHash": (".experiment_hash", "ExperimentHash"),
    "ExperimentMetadata": (".experiment_metadata", "ExperimentMetadata"),
    "git_commit": (".experiment_metadata", "git_commit"),
    "repo_version": (".experiment_metadata", "repo_version"),
    "ExperimentRunner": (".experiment_runner", "ExperimentRunner"),
    "BPThresholdEstimator": (".bp_threshold_estimator", "BPThresholdEstimator"),
    "PhaseDiagramOrchestrator": (".phase_diagram_orchestrator", "PhaseDiagramOrchestrator"),
    "generate_stability_heatmap": (".phase_diagram_orchestrator", "generate_stability_heatmap"),
    "render_ascii_heatmap": (".phase_diagram_orchestrator", "render_ascii_heatmap"),
    "ExperimentRegistry": (".registry", "ExperimentRegistry"),
    "register_experiment": (".registry", "register_experiment"),
    "run_sequence_landscape": (".sequence_landscape", "run_sequence_landscape"),
    "classify_sequence": (".sequence_landscape", "classify_sequence"),
    "run_target_sweep": (".hybrid_target_sweep", "run_target_sweep"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        submod, attr = _LAZY_IMPORTS[name]
        module = _importlib.import_module(submod, __name__)
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
