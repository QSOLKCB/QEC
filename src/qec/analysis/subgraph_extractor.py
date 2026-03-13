from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse


def _to_dense(H_pc: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
    if scipy.sparse.issparse(H_pc):
        return np.asarray(H_pc.todense(), dtype=np.float64)
    return np.asarray(H_pc, dtype=np.float64)


def extract_induced_subgraph(
    H_pc: np.ndarray | scipy.sparse.spmatrix,
    support_nodes: list[int],
) -> dict[str, Any]:
    H = _to_dense(H_pc)
    m, _ = H.shape
    support = sorted({int(v) for v in support_nodes})
    a = len(support)
    if a == 0:
        return {"a": 0, "b": 0, "support_nodes": [], "subgraph": {"check_nodes": [], "H_sub": np.zeros((0, 0), dtype=np.float64)}}

    col = H[:, support]
    deg_in_support = np.sum(col != 0, axis=1)
    check_nodes = [int(ci) for ci in range(m) if int(deg_in_support[ci]) > 0]
    unsat = [int(ci) for ci in check_nodes if int(deg_in_support[ci]) % 2 == 1]
    b = len(unsat)

    H_sub = H[np.array(check_nodes, dtype=int)[:, None], np.array(support, dtype=int)] if check_nodes else np.zeros((0, a), dtype=np.float64)

    return {
        "a": a,
        "b": b,
        "support_nodes": support,
        "subgraph": {
            "check_nodes": check_nodes,
            "unsatisfied_check_nodes": unsat,
            "H_sub": np.asarray(H_sub, dtype=np.float64),
        },
    }
