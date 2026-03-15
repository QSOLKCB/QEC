from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.spectral_mutation_memory import SpectralMutationMemory
from src.qec.discovery.nb_eigenvector_flow_mutation import NBEigenvectorFlowMutator
from src.qec.discovery.threshold_search import PhaseDiagramOrchestrator, SpectralSearchConfig, run_spectral_threshold_search
from src.qec.generation.deterministic_construction import construct_deterministic_tanner_graph


def _small_graph() -> np.ndarray:
    spec = {
        "num_variables": 8,
        "num_checks": 4,
        "variable_degree": 2,
        "check_degree": 4,
    }
    return construct_deterministic_tanner_graph(spec)


def test_same_history_same_weights() -> None:
    mem_a = SpectralMutationMemory(max_records=8)
    mem_b = SpectralMutationMemory(max_records=8)
    history = [(0, 0.1), (2, 0.3), (1, 0.2), (2, 0.4)]
    for idx, imp in history:
        mem_a.record(idx, imp)
        mem_b.record(idx, imp)

    wa = mem_a.compute_weights(3)
    wb = mem_b.compute_weights(3)

    assert np.array_equal(wa, wb)


def test_record_only_positive_improvements() -> None:
    mem = SpectralMutationMemory(max_records=8)
    mem.record(0, -0.1)
    mem.record(1, 0.0)
    mem.record(2, 0.3)

    assert len(mem) == 1
    assert mem.records == [{"mode_index": 2, "improvement": 0.3}]


def test_fifo_eviction_deterministic_ordering() -> None:
    mem = SpectralMutationMemory(max_records=2)
    mem.record(0, 0.1)
    mem.record(1, 0.2)
    mem.record(2, 0.3)

    assert mem.records == [
        {"mode_index": 1, "improvement": 0.2},
        {"mode_index": 2, "improvement": 0.3},
    ]


def test_weights_normalize_to_one() -> None:
    mem = SpectralMutationMemory(max_records=8)
    mem.record(0, 0.1)
    mem.record(1, 0.2)
    mem.record(1, 0.3)

    weights = mem.compute_weights(3)

    assert float(np.round(np.sum(weights, dtype=np.float64), 12)) == 1.0


def test_multi_mode_flow_changes_with_memory_weights() -> None:
    mutator = NBEigenvectorFlowMutator()
    eigenvectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    flow_uniform = mutator.compute_multi_mode_flow(eigenvectors, np.asarray([0.5, 0.5], dtype=np.float64))
    flow_biased = mutator.compute_multi_mode_flow(eigenvectors, np.asarray([0.9, 0.1], dtype=np.float64))

    assert not np.array_equal(flow_uniform, flow_biased)


def test_threshold_search_writes_memory_artifact(tmp_path, monkeypatch) -> None:
    H0 = _small_graph()

    threshold_values = iter([0.11, 0.10, 0.12, 0.10])

    def _fake_eval(self, H, *, max_phase_diagram_size, seed):
        _ = (self, H, max_phase_diagram_size, seed)
        return {"measured_boundary": {"mean_boundary_spectral_radius": float(next(threshold_values))}}

    monkeypatch.setattr(PhaseDiagramOrchestrator, "evaluate", _fake_eval)

    cfg = SpectralSearchConfig(
        iterations=2,
        max_phase_diagram_size=1,
        output_dir=str(tmp_path),
        enable_nb_flow_mutation=True,
        enable_beam_mutations=False,
        enable_multi_mode_nb_mutation=True,
        enable_spectral_mutation_memory=True,
        multi_mode_k=2,
    )
    run_spectral_threshold_search(H0, config=cfg)

    artifact = json.loads((tmp_path / "spectral_mutation_memory.json").read_text(encoding="utf-8"))
    assert artifact["max_records"] == 1000
    assert "records" in artifact
    assert "weights" in artifact
    assert float(np.round(np.sum(np.asarray(artifact["weights"], dtype=np.float64), dtype=np.float64), 12)) == 1.0
