"""v23.0.0 — Deterministic spectral threshold search loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.qec.analysis.nonbacktracking_flow import NonBacktrackingEigenvectorFlowAnalyzer
from src.qec.analysis.spectral_entropy import spectral_entropy
from src.qec.analysis.spectral_frustration import SpectralFrustrationAnalyzer
from src.qec.analysis.trap_memory import TrapSubspaceMemory
from src.qec.diagnostics.spectral_nb import compute_nb_spectrum
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator
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
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
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

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_threshold = -np.inf
    best_graph = H_current.copy()
    history: list[dict[str, Any]] = []
    candidate_metrics: list[dict[str, Any]] = []

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
                "ipr_localization": round(float(fr.max_ipr), _ROUND),
                "spectral_entropy": round(float(entropy), _ROUND),
                "trap_similarity": round(float(trap_similarity), _ROUND),
                "mutations": ops_cand,
            }
            rejected = (
                metrics["trap_similarity"] > cfg.trap_similarity_reject
                or metrics["spectral_entropy"] < cfg.min_entropy_reject
                or metrics["bethe_hessian_negative_modes"] > cfg.max_negative_modes_reject
            )
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
            ranked.append({
                "threshold": threshold,
                "H": H_cand,
                "metrics": metrics,
            })

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

    result = {
        "best_threshold": None if not np.isfinite(best_threshold) else round(float(best_threshold), _ROUND),
        "history": history,
        "output_dir": str(output_dir),
    }
    return canonicalize(result)
