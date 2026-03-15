from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_mutation_memory import SpectralMutationMemory
from src.qec.discovery.mutation_interface import NBEigenvectorFlowMutation
from src.qec.discovery.nb_eigenvector_flow_mutation import NBEigenvectorFlowMutator


def test_multi_mode_operator_uses_memory_weights() -> None:
    operator = NBEigenvectorFlowMutation(NBEigenvectorFlowMutator())
    H = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    eigenvectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )

    memory = SpectralMutationMemory(max_records=10)
    memory.record(1, 0.9)
    memory.record(0, 0.1)

    _, _, source, meta = operator.mutate(
        H,
        {"eigenvectors": eigenvectors, "mutation_memory": memory},
        {
            "enable_multi_mode_nb_mutation": True,
            "enable_spectral_mutation_memory": True,
            "mode_count": 2,
        },
    )

    assert source == "nb_flow_multi_mode"
    assert meta["mode_index"] == 1
    assert meta["mode_weights"][1] > meta["mode_weights"][0]
import json

import numpy as np

from src.qec.discovery.nb_eigenvector_flow_mutation import NBEigenvectorFlowMutator, compute_multi_mode_flow
from src.qec.discovery.threshold_search import PhaseDiagramOrchestrator, SpectralSearchConfig, run_spectral_threshold_search
from src.qec.generation.deterministic_construction import construct_deterministic_tanner_graph
from src.qec.spectral.nb_spectrum import select_unstable_nb_modes


def _graph() -> np.ndarray:
    spec = {
        "num_variables": 8,
        "num_checks": 4,
        "variable_degree": 2,
        "check_degree": 4,
    }
    return construct_deterministic_tanner_graph(spec)


def test_select_unstable_nb_modes_deterministic_tie_break() -> None:
    eigvals = np.array([1.0 + 0.0j, -3.0 + 0.0j, 3.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    eigvecs = np.eye(4, dtype=np.complex128)

    selected_vecs, selected_vals = select_unstable_nb_modes(eigvals, eigvecs, k_modes=3)

    np.testing.assert_array_equal(selected_vals, np.array([-3.0 + 0.0j, 3.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128))
    np.testing.assert_array_equal(selected_vecs, eigvecs[:, [1, 2, 3]])


def test_compute_multi_mode_flow_normalization_is_deterministic() -> None:
    vecs = np.array([
        [1.0, -1.0, 0.0],
        [0.0, 2.0, -2.0],
        [3.0, 0.0, 4.0],
    ], dtype=np.float64)
    flow = compute_multi_mode_flow(vecs)
    expected = np.array([2.0, 4.0, 7.0], dtype=np.float64)
    expected = np.round(expected / np.linalg.norm(expected), 12)
    np.testing.assert_array_equal(flow, expected)


def test_multi_mode_mutation_is_deterministic() -> None:
    H = _graph()
    mut = NBEigenvectorFlowMutator()
    modes = np.array([
        [1.0, -0.2, 0.3],
        [-0.1, 0.9, 0.2],
        [0.4, -0.3, 0.7],
        [0.2, 0.1, -0.8],
        [0.0, 0.0, 0.0],
        [0.3, 0.2, -0.1],
    ], dtype=np.float64)

    out_a, meta_a = mut.mutate(H, modes)
    out_b, meta_b = mut.mutate(H, modes)

    np.testing.assert_array_equal(out_a, out_b)
    assert meta_a == meta_b


def test_multi_mode_compatibility_with_ipr_and_annealing_pipeline(tmp_path, monkeypatch) -> None:
    H0 = _graph()

    monkeypatch.setattr(
        PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {"measured_boundary": {"mean_boundary_spectral_radius": 0.1}},
    )

    cfg = SpectralSearchConfig(
        iterations=1,
        max_phase_diagram_size=1,
        output_dir=str(tmp_path),
        enable_nb_flow_mutation=True,
        enable_multi_mode_nb_mutation=True,
        nb_mutation_modes=3,
    )
    run_spectral_threshold_search(H0, config=cfg)

    payload = json.loads((tmp_path / "candidate_metrics.json").read_text(encoding="utf-8"))
    flow_entries = [m for m in payload["candidates"] if m.get("source") == "nb_flow"]
    assert len(flow_entries) == 1
    assert flow_entries[0]["nb_mutation_modes"] == 3
    assert isinstance(flow_entries[0]["multi_mode_flow_strength"], float)
    assert flow_entries[0]["multi_mode_flow_strength"] >= 0.0
