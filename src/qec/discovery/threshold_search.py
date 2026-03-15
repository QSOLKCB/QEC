"""v23.0.0 — Deterministic spectral threshold search loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.qec.analysis.nonbacktracking_flow import NonBacktrackingEigenvectorFlowAnalyzer
from src.qec.analysis.bp_diagnostics import collect_bp_diagnostics
from src.qec.analysis.spectral_entropy import spectral_entropy
from src.qec.analysis.nb_threshold_predictor import predict_threshold_from_spectrum
from src.qec.analysis.spectral_regression import SpectralThresholdModel, load_training_dataset
from src.qec.analysis.threshold_predictor import predict_threshold_quality
from src.qec.analysis.spectral_frustration import SpectralFrustrationAnalyzer
from src.qec.analysis.trap_memory import TrapSubspaceMemory
from src.qec.diagnostics.spectral_nb import compute_nb_spectrum
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator
from src.qec.discovery.pareto_archive import ParetoArchive, ParetoMetrics
from src.qec.discovery.nonbacktracking_eigenvector_flow import NonBacktrackingEigenvectorFlowOptimizer
from src.qec.experiments.stability_phase_diagram import run_stability_phase_diagram_experiment
from src.utils.canonicalize import canonicalize

_ROUND = 12


@dataclass(frozen=True)
class SpectralSearchConfig:
    iterations: int = 10
    population: int = 3
    max_phase_diagram_size: int = 4
    seed: int = 0
    output_dir: str = "experiments/threshold_search"
    enable_beam_mutations: bool = True
    enable_nb_flow_mutation: bool = False
    enable_adaptive_mutation: bool = True
    trap_similarity_reject: float = 0.999
    min_entropy_reject: float = 0.0
    max_negative_modes_reject: int = 1_000_000
    enable_bp_diagnostics: bool = False
    enable_pareto: bool = False
    enable_nb_predictor: bool = False
    enable_learning: bool = False
    min_predicted_threshold: float = 0.0
    experiments_root: str = "experiments"
    min_predicted_threshold: float = 0.0


class PhaseDiagramOrchestrator:
    """Deterministic threshold evaluation via stability phase-diagram runs."""

    def evaluate(self, H: np.ndarray, *, max_phase_diagram_size: int, seed: int) -> dict[str, Any]:
        grid = max(1, int(max_phase_diagram_size))
        return run_stability_phase_diagram_experiment(
            H,
            grid_resolution=grid,
            perturbations_per_cell=1,
            base_seed=int(seed),
        )


class BPThresholdEstimator:
    """Deterministic scalar threshold estimate from phase-diagram outputs."""

    def estimate(self, phase_result: dict[str, Any]) -> float:
        boundary = phase_result.get("measured_boundary", {})
        score = float(boundary.get("mean_boundary_spectral_radius", 0.0))
        return float(np.round(np.float64(score), _ROUND))


def _write_canonical_json(path: Path, payload: Any) -> None:
    canonical = canonicalize(payload)
    text = json.dumps(canonical, sort_keys=True, indent=2)
    path.write_text(f"{text}\n", encoding="utf-8")


def _is_degree_preserving(H_ref: np.ndarray, H_new: np.ndarray) -> bool:
    return bool(
        np.array_equal(H_ref.sum(axis=0), H_new.sum(axis=0))
        and np.array_equal(H_ref.sum(axis=1), H_new.sum(axis=1))
        and np.all((H_new == 0.0) | (H_new == 1.0))
    )


def run_spectral_threshold_search(
    H0: np.ndarray,
    *,
    config: SpectralSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SpectralSearchConfig()
    H_current = np.asarray(H0, dtype=np.float64).copy()
    mutator = NBGradientMutator(enabled=True, enable_spectral_beam_search=False)
    beam_mutator = NBGradientMutator(enabled=True, enable_spectral_beam_search=True)
    flow_optimizer = NonBacktrackingEigenvectorFlowOptimizer(max_steps=1)
    flow_analyzer = NonBacktrackingEigenvectorFlowAnalyzer()
    frustration = SpectralFrustrationAnalyzer()
    trap_memory = TrapSubspaceMemory()
    orchestrator = PhaseDiagramOrchestrator()
    estimator = BPThresholdEstimator()
    pareto = ParetoArchive() if cfg.enable_pareto else None
    model = SpectralThresholdModel()
    if cfg.enable_learning:
        dataset = load_training_dataset(cfg.experiments_root)
        model.fit(dataset)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_threshold = -np.inf
    best_graph = H_current.copy()
    history: list[dict[str, Any]] = []
    candidate_metrics: list[dict[str, Any]] = []
    convergence_records: list[dict[str, Any]] = []

    for iteration in range(int(cfg.iterations)):
        baseline = frustration.compute_frustration(H_current)
        adaptive_steps = 1
        if cfg.enable_adaptive_mutation and baseline.max_ipr > 0.3:
            adaptive_steps += 1

        generated: list[tuple[np.ndarray, list[dict[str, Any]], str]] = []
        h_mut, ops = mutator.mutate(H_current, steps=adaptive_steps)
        generated.append((np.asarray(h_mut, dtype=np.float64), ops, "nb_gradient"))

        if cfg.enable_beam_mutations:
            h_beam, ops_beam = beam_mutator.mutate(H_current, steps=adaptive_steps)
            generated.append((np.asarray(h_beam, dtype=np.float64), ops_beam, "beam"))

        if cfg.enable_nb_flow_mutation:
            traj = flow_optimizer.optimize(H_current)
            generated.append((np.asarray(traj.H_final, dtype=np.float64), [], "nb_flow"))

        ranked: list[dict[str, Any]] = []
        for idx, (H_cand, ops_cand, source) in enumerate(generated):
            if not _is_degree_preserving(H_current, H_cand):
                continue

            nb = compute_nb_spectrum(H_cand)
            fr = frustration.compute_frustration(H_cand)
            eigvals, _ = flow_analyzer.compute_modes(H_cand)
            entropy = spectral_entropy(eigvals)
            trap_similarity = trap_memory.compute_similarity(fr.trap_modes)

            metrics = {
                "iteration": iteration,
                "candidate_index": idx,
                "source": source,
                "nb_spectral_radius": round(float(nb.get("spectral_radius", 0.0)), _ROUND),
                "bethe_hessian_negative_modes": int(fr.negative_modes),
                "bethe_negative_mass": round(float(fr.negative_modes), _ROUND),
                "ipr_localization": round(float(fr.max_ipr), _ROUND),
                "flow_ipr": round(float(fr.max_ipr), _ROUND),
                "spectral_entropy": round(float(entropy), _ROUND),
                "trap_similarity": round(float(trap_similarity), _ROUND),
                "mutations": ops_cand,
            }

            predicted_threshold = None
            if cfg.enable_nb_predictor:
                pred = predict_threshold_from_spectrum(metrics)
                metrics["nb_prediction_score"] = pred["prediction_score"]
                metrics["nb_predicted_threshold"] = pred["predicted_threshold"]
                predicted_threshold = float(pred["predicted_threshold"])

            if cfg.enable_learning:
                reg_pred = model.predict(metrics)
                metrics["regression_predicted_threshold"] = round(float(reg_pred), _ROUND)
                if predicted_threshold is None:
                    predicted_threshold = float(reg_pred)
                else:
                    predicted_threshold = round(float((predicted_threshold + reg_pred) / 2.0), _ROUND)

            if predicted_threshold is not None:
                metrics["predicted_threshold"] = round(float(predicted_threshold), _ROUND)
            prediction = predict_threshold_quality(
                spectral_radius=metrics["nb_spectral_radius"],
                bethe_negative_mass=float(metrics["bethe_hessian_negative_modes"]),
                flow_ipr=metrics["ipr_localization"],
                spectral_entropy_value=metrics["spectral_entropy"],
                trap_similarity=metrics["trap_similarity"],
            )
            metrics["predicted_threshold"] = round(float(prediction.predicted_threshold), _ROUND)
            metrics["prediction_score"] = round(float(prediction.score), _ROUND)
            rejected = (
                metrics["trap_similarity"] > cfg.trap_similarity_reject
                or metrics["spectral_entropy"] < cfg.min_entropy_reject
                or metrics["bethe_hessian_negative_modes"] > cfg.max_negative_modes_reject
                or metrics["predicted_threshold"] < float(cfg.min_predicted_threshold)
            )
            if predicted_threshold is not None and predicted_threshold < float(cfg.min_predicted_threshold):
                rejected = True
            metrics["rejected"] = bool(rejected)
            candidate_metrics.append(metrics)
            if rejected:
                continue

            phase = orchestrator.evaluate(
                H_cand,
                max_phase_diagram_size=cfg.max_phase_diagram_size,
                seed=cfg.seed + iteration * 1024 + idx,
            )
            threshold = estimator.estimate(phase)
            if cfg.enable_bp_diagnostics:
                diagnostics = collect_bp_diagnostics({
                    "residuals": [
                        cell.get("mean_residual_norm", 0.0)
                        for cell in phase.get("grid_results", [])
                    ],
                    "syndrome_weights": [],
                })
                metrics["bp_iterations"] = int(diagnostics.iterations_to_converge)
                metrics["bp_converged"] = bool(diagnostics.converged)
                metrics["bp_final_residual"] = round(float(diagnostics.final_residual), _ROUND)
                convergence_records.append({
                    "iterations": int(diagnostics.iterations_to_converge),
                    "final_residual": round(float(diagnostics.final_residual), _ROUND),
                    "converged": bool(diagnostics.converged),
                })
            ranked.append({
                "threshold": threshold,
                "H": H_cand,
                "metrics": metrics,
            })
            if pareto is not None:
                convergence_speed = 0.0
                if "bp_iterations" in metrics:
                    convergence_speed = round(1.0 / max(1.0, float(metrics["bp_iterations"])), _ROUND)
                pm = ParetoMetrics(
                    threshold=threshold,
                    spectral_stability=round(1.0 - float(metrics["nb_spectral_radius"]), _ROUND),
                    convergence_speed=convergence_speed,
                )
                pareto.add_candidate(pm, np.asarray(H_cand, dtype=np.float64))

        if ranked:
            ranked.sort(key=lambda r: (-r["threshold"], r["metrics"]["candidate_index"]))
            best_iter = ranked[0]
            H_current = best_iter["H"]
            if best_iter["threshold"] > best_threshold:
                best_threshold = float(best_iter["threshold"])
                best_graph = H_current.copy()
                _write_canonical_json(
                    output_dir / "best_graph.json",
                    {
                        "threshold": best_threshold,
                        "spectral_metrics": best_iter["metrics"],
                        "parity_check_matrix": best_graph,
                    },
                )
            history.append({
                "iteration": iteration,
                "threshold": float(best_iter["threshold"]),
                "spectral_metrics": best_iter["metrics"],
                "mutation_operations": best_iter["metrics"]["mutations"],
            })
        else:
            history.append({
                "iteration": iteration,
                "threshold": None,
                "spectral_metrics": None,
                "mutation_operations": [],
            })

    _write_canonical_json(output_dir / "search_history.json", {"history": history})
    _write_canonical_json(output_dir / "candidate_metrics.json", {"candidates": candidate_metrics})
    if pareto is not None:
        pareto.save_frontier(output_dir / "pareto_frontier.json")
    if cfg.enable_learning:
        model.save_model(output_dir / "spectral_threshold_model.json")
    if cfg.enable_bp_diagnostics:
        n_records = len(convergence_records)
        if n_records > 0:
            mean_bp_iterations = sum(r["iterations"] for r in convergence_records) / n_records
            max_bp_iterations = max(r["iterations"] for r in convergence_records)
            convergence_rate = sum(1 for r in convergence_records if r["converged"]) / n_records
            mean_final_residual = sum(r["final_residual"] for r in convergence_records) / n_records
        else:
            mean_bp_iterations = 0.0
            max_bp_iterations = 0
            convergence_rate = 0.0
            mean_final_residual = 0.0
        _write_canonical_json(
            output_dir / "convergence_summary.json",
            {
                "convergence_rate": round(float(convergence_rate), _ROUND),
                "max_bp_iterations": int(max_bp_iterations),
                "mean_bp_iterations": round(float(mean_bp_iterations), _ROUND),
                "mean_final_residual": round(float(mean_final_residual), _ROUND),
            },
        )

    result = {
        "best_threshold": None if not np.isfinite(best_threshold) else round(float(best_threshold), _ROUND),
        "history": history,
        "output_dir": str(output_dir),
    }
    return canonicalize(result)


class ThresholdSearchEngine:
    """Backward-compatible engine wrapper for threshold search."""

    @staticmethod
    def run(
        H_init: np.ndarray,
        config: SpectralSearchConfig,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        _ = output_dir
        return run_spectral_threshold_search(H_init, config=config)
