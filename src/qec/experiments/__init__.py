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
"""

from .risk_aware_damping import run_risk_aware_damping_experiment
from .risk_guided_perturbation import run_risk_guided_perturbation
from .tanner_graph_repair import run_tanner_graph_repair_experiment
from .tanner_graph_repair import run_spectral_graph_optimization_experiment
from .bp_prediction_validation import run_bp_prediction_validation
from .spectral_instability_phase_map import (
    compute_spectral_instability_score,
    run_spectral_phase_map_experiment,
    compute_phase_map_aggregate_metrics,
)
from .spectral_graph_repair_loop import (
    run_spectral_graph_repair_loop,
    compute_repair_loop_aggregate_metrics,
)
from .sensitivity_preconditioner import (
    run_sensitivity_preconditioned_optimization,
    run_sensitivity_preconditioner_experiment,
)
from .spectral_validation import (
    run_spectral_validation_experiment,
    serialize_artifact,
)
from .eeec_anomaly_scan import (
    detect_eeec_anomaly,
    run_eeec_anomaly_scan,
)
from .spectral_heatmap_experiment import (
    run_spectral_heatmap_experiment,
    serialize_heatmap_artifact,
)
from .incremental_spectral_benchmark import (
    run_incremental_spectral_benchmark,
    serialize_benchmark_artifact,
)
from .stability_phase_diagram import (
    run_stability_phase_diagram_experiment,
    serialize_phase_diagram_artifact,
    detect_metastable_bp_oscillation,
    estimate_bp_stability_boundary,
    predict_spectral_stability_boundary,
    log_most_unstable_subgraph,
    track_repair_boundary_shift,
)
from .experiment_hash import ExperimentHash, ExperimentRunner

from .experiment_metadata import ExperimentMetadata, git_commit, repo_version
from .experiment_runner import ExperimentRunner

from .phase_diagram_orchestrator import (
    PhaseDiagramOrchestrator,
    generate_stability_heatmap,
    render_ascii_heatmap,
)
from .registry import ExperimentRegistry, register_experiment
