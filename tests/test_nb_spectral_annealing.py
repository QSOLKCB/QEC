from __future__ import annotations

import numpy as np

from src.qec.discovery.mutation_interface import NBEigenvectorFlowMutation
from src.qec.discovery.nb_eigenvector_flow_mutation import NBEigenvectorFlowMutator


def test_single_mode_path_remains_deterministic() -> None:
    operator = NBEigenvectorFlowMutation(NBEigenvectorFlowMutator())
    H = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    leading = np.asarray([0.9, 0.1, 0.0, 0.0], dtype=np.float64)

    out1 = operator.mutate(
        H,
        {"leading_vector": leading},
        {"enable_multi_mode_nb_mutation": False},
    )
    out2 = operator.mutate(
        H,
        {"leading_vector": leading},
        {"enable_multi_mode_nb_mutation": False},
    )

    assert out1[2] == "nb_flow"
    assert np.array_equal(out1[0], out2[0])
    assert out1[3] == out2[3]
