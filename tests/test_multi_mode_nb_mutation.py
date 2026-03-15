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
