"""Deterministic Tanner-graph threshold search engine."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.qec.diagnostics.bethe_hessian import compute_bethe_hessian
from src.qec.diagnostics.phase_diagram import build_decoder_phase_diagram, make_phase_grid
from src.qec.diagnostics.spectral_nb import compute_nb_spectrum
from src.qec.diagnostics.tanner_spectral_analysis import compute_tanner_spectral_analysis
from src.qec.experiments.experiment_hash import ExperimentRunner
from src.qec.generation.tanner_graph_generator import generate_tanner_graph_candidates


_ROUND = 12


class BPThresholdEstimator:
    """Deterministic threshold estimator from phase-diagram cells."""

    def estimate(self, phase_diagram: dict[str, Any]) -> float:
        cells = sorted(
            list(phase_diagram.get("cells", [])),
            key=lambda c: (float(c.get("x", 0.0)), float(c.get("y", 0.0))),
        )
        passing = [float(c["x"]) for c in cells if float(c.get("success_fraction", 0.0)) >= 0.5]
        return round(max(passing), _ROUND) if passing else 0.0


def _derive_seed(base_seed: int, label: str) -> int:
    data = struct.pack(">Q", int(base_seed)) + label.encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _validate_graph(H: np.ndarray) -> bool:
    H_arr = np.asarray(H, dtype=np.float64)
    if H_arr.ndim != 2:
        return False
    if not np.all(np.isin(H_arr, [0.0, 1.0])):
        return False
    if np.any(np.sum(H_arr, axis=0) < 1.0):
        return False
    if np.any(np.sum(H_arr, axis=1) < 1.0):
        return False
    return True


def _collect_edges(H: np.ndarray, value: float) -> list[tuple[int, int]]:
    m, n = H.shape
    out: list[tuple[int, int]] = []
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] == value:
                out.append((ci, vi))
    return out


def _edge_swap(H: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    out = H.copy()
    edges = _collect_edges(out, 1.0)
    non_edges = _collect_edges(out, 0.0)
    if not edges or not non_edges:
        return out
    remove = edges[int(rng.randint(0, len(edges)))]
    add = non_edges[int(rng.randint(0, len(non_edges)))]
    if np.sum(out[remove[0]]) > 1.0 and np.sum(out[:, remove[1]]) > 1.0:
        out[remove[0], remove[1]] = 0.0
        out[add[0], add[1]] = 1.0
    return out


def _remove_add_edge(H: np.ndarray, seed: int) -> np.ndarray:
    return _edge_swap(H, seed)


def _check_rewire(H: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    out = H.copy()
    m, _ = out.shape
    edge_list = _collect_edges(out, 1.0)
    if not edge_list:
        return out
    ci_old, vi = edge_list[int(rng.randint(0, len(edge_list)))]
    available = [ci for ci in range(m) if ci != ci_old and out[ci, vi] == 0.0]
    if not available or np.sum(out[ci_old]) <= 1.0:
        return out
    ci_new = available[int(rng.randint(0, len(available)))]
    out[ci_old, vi] = 0.0
    out[ci_new, vi] = 1.0
    return out


def _degree_preserving_mutation(H: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    out = H.copy()
    m, n = out.shape
    if n < 2:
        return out
    vi_a = int(rng.randint(0, n))
    vi_b = int((vi_a + 1 + rng.randint(0, n - 1)) % n)
    checks_a = [ci for ci in range(m) if out[ci, vi_a] == 1.0 and out[ci, vi_b] == 0.0]
    checks_b = [ci for ci in range(m) if out[ci, vi_b] == 1.0 and out[ci, vi_a] == 0.0]
    if not checks_a or not checks_b:
        return out
    ci_a = checks_a[int(rng.randint(0, len(checks_a)))]
    ci_b = checks_b[int(rng.randint(0, len(checks_b)))]
    out[ci_a, vi_a] = 0.0
    out[ci_b, vi_b] = 0.0
    out[ci_a, vi_b] = 1.0
    out[ci_b, vi_a] = 1.0
    return out


class ThresholdSearchEngine:
    """Deterministic optimizer for Tanner graphs using BP threshold fitness."""

    def __init__(
        self,
        spec: dict[str, Any],
        *,
        iterations: int = 20,
        population: int = 6,
        seed: int = 0,
        max_graphs_evaluated: int = 200,
        max_phase_diagram_size: int = 6,
        artifacts_root: str = "experiments/threshold_search",
        spectral_instability_ratio_limit: float = 1.6,
        ipr_limit: float = 0.35,
        threshold_evaluator: Callable[[np.ndarray], float] | None = None,
        threshold_estimator: BPThresholdEstimator | None = None,
    ) -> None:
        self.spec = dict(spec)
        self.iterations = int(iterations)
        self.population = int(population)
        self.seed = int(seed)
        self.max_graphs_evaluated = int(max_graphs_evaluated)
        self.max_phase_diagram_size = int(max_phase_diagram_size)
        self.spectral_instability_ratio_limit = float(spectral_instability_ratio_limit)
        self.ipr_limit = float(ipr_limit)
        self.threshold_evaluator = threshold_evaluator
        self.threshold_estimator = threshold_estimator or BPThresholdEstimator()
        self._runner = ExperimentRunner(artifacts_root=artifacts_root)
        self._artifact_dir = Path(artifacts_root)
        self._evaluated = 0

    def _spectral_metrics(self, H: np.ndarray) -> dict[str, float]:
        nb = compute_nb_spectrum(H)
        bethe = compute_bethe_hessian(H)
        tanner = compute_tanner_spectral_analysis(H)
        avg_degree = float(np.mean(np.sum(np.asarray(H, dtype=np.float64), axis=0)))
        threshold = float(np.sqrt(avg_degree)) if avg_degree > 0.0 else 1.0
        return {
            "spectral_radius": round(float(nb["spectral_radius"]), _ROUND),
            "bethe_min_eigenvalue": round(float(bethe["min_eigenvalue"]), _ROUND),
            "ipr": round(float(tanner["max_variable_mode_ipr"]), _ROUND),
            "spectral_instability_ratio": round(float(nb["spectral_radius"]) / threshold, _ROUND),
        }

    def _passes_spectral_filter(self, metrics: dict[str, float]) -> bool:
        return (
            float(metrics["spectral_instability_ratio"]) <= self.spectral_instability_ratio_limit
            and float(metrics["ipr"]) <= self.ipr_limit
        )

    def _run_phase_diagram(self, H: np.ndarray) -> dict[str, Any]:
        grid_size = max(2, min(self.max_phase_diagram_size, 20))
        noise_values = [round(float(v), _ROUND) for v in np.linspace(0.01, 0.10, grid_size)]
        grid = make_phase_grid("x", noise_values, "y", [1.0])
        radius = float(compute_nb_spectrum(H)["spectral_radius"])

        def runner(x: float | int, y: float | int) -> list[dict[str, Any]]:
            raw_success = max(0.0, 1.0 - radius * float(x) / max(1.0, float(y) * 3.0))
            out = []
            for offset in (0.0, 0.02, 0.04):
                success = max(0.0, min(1.0, raw_success - offset))
                out.append({"final_ternary_state": 1 if success >= 0.5 else -1})
            return out

        phase_diagram = build_decoder_phase_diagram(grid, runner)
        return {
            "phase_diagram": phase_diagram,
            "threshold": round(float(self.threshold_estimator.estimate(phase_diagram)), _ROUND),
        }

    def _evaluate_threshold(self, H: np.ndarray, candidate_id: str) -> float:
        if self.threshold_evaluator is not None:
            return round(float(self.threshold_evaluator(H)), _ROUND)
        config = {
            "experiment": "threshold_search_candidate",
            "seed": self.seed,
            "candidate_id": candidate_id,
            "max_phase_diagram_size": self.max_phase_diagram_size,
            "graph": np.asarray(H, dtype=np.float64).astype(int).tolist(),
        }
        return round(float(self._runner.run(config, lambda _: self._run_phase_diagram(H)).get("threshold", 0.0)), _ROUND)

    @staticmethod
    def _sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
        metrics = entry["spectral_metrics"]
        return (-float(entry["threshold"]), float(metrics["spectral_instability_ratio"]), float(metrics["ipr"]), str(entry["candidate_id"]))

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def _record_artifacts(self, history: list[dict[str, Any]], best: dict[str, Any] | None, run_metadata: dict[str, Any]) -> None:
        self._write_json(self._artifact_dir / "run_metadata.json", run_metadata)
        self._write_json(self._artifact_dir / "search_history.json", history)
        self._write_json(self._artifact_dir / "candidates.json", history)
        if best is not None:
            self._write_json(self._artifact_dir / "best_graph.json", {"threshold": float(best["threshold"]), "graph_structure": best["graph_structure"]})

    def _mutate_population(self, ranked: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        elites = ranked[: max(1, self.population // 2)]
        next_population = [{"candidate_id": f"elite_{iteration}_{i}", "H": np.asarray(e["graph_structure"], dtype=np.float64)} for i, e in enumerate(elites)]
        operators = [_edge_swap, _remove_add_edge, _check_rewire, _degree_preserving_mutation]

        while len(next_population) < self.population:
            idx = len(next_population) - len(elites)
            parent = elites[idx % len(elites)]
            op = operators[(iteration + idx) % len(operators)]
            mut_seed = _derive_seed(self.seed, f"iter_{iteration}_child_{len(next_population)}")
            parent_H = np.asarray(parent["graph_structure"], dtype=np.float64)
            child_H = op(parent_H, mut_seed)
            if not _validate_graph(child_H):
                child_H = parent_H.copy()
            next_population.append({"candidate_id": f"cand_{iteration}_{len(next_population)}", "H": child_H})
        return next_population

    def search(self) -> dict[str, Any]:
        run_metadata = {
            "seed": self.seed,
            "iterations": self.iterations,
            "population": self.population,
            "max_graphs_evaluated": self.max_graphs_evaluated,
            "max_phase_diagram_size": self.max_phase_diagram_size,
        }
        population = [{"candidate_id": c["candidate_id"], "H": np.asarray(c["H"], dtype=np.float64)} for c in generate_tanner_graph_candidates(self.spec, self.population, base_seed=self.seed)]
        history: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for iteration in range(1, self.iterations + 1):
            scored: list[dict[str, Any]] = []
            for candidate in population:
                if self._evaluated >= self.max_graphs_evaluated:
                    break
                metrics = self._spectral_metrics(candidate["H"])
                accepted = self._passes_spectral_filter(metrics)
                threshold = self._evaluate_threshold(candidate["H"], candidate["candidate_id"]) if accepted else 0.0
                entry = {
                    "iteration": iteration,
                    "candidate_id": candidate["candidate_id"],
                    "threshold": round(float(threshold), _ROUND),
                    "spectral_metrics": metrics,
                    "accepted": accepted,
                    "graph_structure": np.asarray(candidate["H"], dtype=np.float64).astype(int).tolist(),
                }
                self._evaluated += 1
                scored.append(entry)
                history.append(entry)
                if best is None or self._sort_key(entry) < self._sort_key(best):
                    best = entry
            if not scored or self._evaluated >= self.max_graphs_evaluated:
                break
            population = self._mutate_population(sorted(scored, key=self._sort_key), iteration)
            self._record_artifacts(history, best, run_metadata)

        result = {"best": best, "history": history, "evaluated_graphs": self._evaluated, "artifact_dir": str(self._artifact_dir)}
        self._record_artifacts(history, best, run_metadata)
        return result


def run_threshold_search(
    spec: dict[str, Any],
    *,
    iterations: int = 20,
    population: int = 6,
    seed: int = 0,
    max_graphs_evaluated: int = 200,
    max_phase_diagram_size: int = 6,
    artifacts_root: str = "experiments/threshold_search",
) -> dict[str, Any]:
    return ThresholdSearchEngine(
        spec,
        iterations=iterations,
        population=population,
        seed=seed,
        max_graphs_evaluated=max_graphs_evaluated,
        max_phase_diagram_size=max_phase_diagram_size,
        artifacts_root=artifacts_root,
    ).search()
