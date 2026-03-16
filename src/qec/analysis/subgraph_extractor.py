"""Deterministic support-induced Tanner subgraph extraction utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse



def extract_support_subgraph(
    H_pc: np.ndarray | scipy.sparse.spmatrix,
    support_nodes: list[int] | tuple[int, ...] | np.ndarray,
) -> dict[str, Any]:
    H = scipy.sparse.csr_matrix(H_pc, dtype=np.float64)
    m, n = H.shape

    support = sorted({int(v) for v in support_nodes if 0 <= int(v) < n})
    if not support:
        return {
            "support_nodes": [],
            "check_nodes": [],
            "submatrix": scipy.sparse.csr_matrix((0, 0), dtype=np.float64),
        }

    sub = H[:, support].tocsr()
    check_degree = np.asarray(sub.astype(np.int64).sum(axis=1), dtype=np.int64).ravel()
    check_nodes = [int(ci) for ci in np.flatnonzero(check_degree > 0)]
    induced = sub[check_nodes, :].tocsr()

    return {
        "support_nodes": support,
        "check_nodes": check_nodes,
        "submatrix": induced,
    }



def compute_trapping_parameters(subgraph: dict[str, Any]) -> tuple[int, int]:
    support_nodes = [int(v) for v in subgraph.get("support_nodes", [])]
    a = int(len(support_nodes))

    sub = subgraph.get("submatrix")
    if sub is None:
        return a, 0

    sub_csr = scipy.sparse.csr_matrix(sub, dtype=np.int64)
    if sub_csr.shape[0] == 0:
        return a, 0

    deg = np.asarray(sub_csr.astype(np.int64).sum(axis=1), dtype=np.int64).ravel()
    odd_checks = np.count_nonzero(deg % 2)
    b = int(odd_checks)
    return a, b
