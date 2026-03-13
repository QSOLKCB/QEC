"""Deterministic Bethe-Hessian defect catalog."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from src.qec.analysis.subgraph_extractor import compute_trapping_parameters, extract_support_subgraph
from src.qec.analysis.trapping_set_classifier import classify_trapping_set


_ROUND = 12


@dataclass(frozen=True)
class SpectralDefect:
    eigenvalue: float
    severity: float
    support_nodes: tuple[int, ...]
    classification: str
    a: int
    b: int



def _build_bethe_hessian(H_pc: np.ndarray | scipy.sparse.spmatrix) -> tuple[scipy.sparse.csr_matrix, float]:
    H = scipy.sparse.csr_matrix(H_pc, dtype=np.float64)
    m, n = H.shape
    zero_v = scipy.sparse.csr_matrix((n, n), dtype=np.float64)
    zero_c = scipy.sparse.csr_matrix((m, m), dtype=np.float64)
    A = scipy.sparse.bmat([[zero_v, H.T], [H, zero_c]], format="csr", dtype=np.float64)

    degrees = np.asarray(A.sum(axis=1), dtype=np.float64).ravel()
    mean_degree = float(degrees.mean()) if degrees.size else 0.0
    r = 1.0 if mean_degree <= 1.0 else float(np.sqrt(mean_degree - 1.0))

    total = n + m
    I = scipy.sparse.eye(total, dtype=np.float64, format="csr")
    D = scipy.sparse.diags(degrees, dtype=np.float64, format="csr")
    HB = ((r * r - 1.0) * I - r * A + D).tocsr()
    return HB, r



def _extract_negative_modes(HB: scipy.sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    total = int(HB.shape[0])
    if total == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)

    if total == 1:
        val = float(HB[0, 0])
        return np.array([val], dtype=np.float64), np.array([[1.0]], dtype=np.float64)

    k = min(max(1, total - 1), 32)
    try:
        vals, vecs = scipy.sparse.linalg.eigsh(
            HB,
            k=k,
            which="SA",
            v0=np.ones(total, dtype=np.float64),
            tol=0.0,
            maxiter=max(8 * total, 200),
        )
    except (scipy.sparse.linalg.ArpackError, scipy.sparse.linalg.ArpackNoConvergence, RuntimeError, ValueError):
        dense = HB.toarray()
        vals_all, vecs_all = np.linalg.eigh(dense)
        vals = vals_all[:k]
        vecs = vecs_all[:, :k]

    order = np.argsort(vals, kind="mergesort")
    return np.asarray(vals[order], dtype=np.float64), np.asarray(vecs[:, order], dtype=np.float64)



def detect_spectral_defects(H_pc: np.ndarray | scipy.sparse.spmatrix) -> list[SpectralDefect]:
    H = scipy.sparse.csr_matrix(H_pc, dtype=np.float64)
    m, n = H.shape

    HB, _ = _build_bethe_hessian(H)
    eigenvalues, eigenvectors = _extract_negative_modes(HB)

    defects: list[SpectralDefect] = []
    for idx, eig in enumerate(eigenvalues):
        ev = float(eig)
        if ev >= 0.0:
            continue

        vec = np.asarray(eigenvectors[:, idx], dtype=np.float64)
        abs_vec = np.abs(vec)
        norm_sq = float(np.dot(abs_vec, abs_vec))
        if norm_sq <= 0.0:
            continue

        ipr = float(np.sum(abs_vec ** 4) / (norm_sq * norm_sq))
        threshold = float(np.max(abs_vec) * 0.5)
        support_mask = abs_vec >= threshold
        support_all = np.flatnonzero(support_mask)
        support_variables = [int(v) for v in support_all if int(v) < n]

        if not support_variables:
            support_variables = [int(np.argmax(abs_vec[:n]))] if n > 0 else []

        support_variables = sorted(set(support_variables))
        support_fraction = float(len(support_variables) / n) if n > 0 else 0.0

        topk = max(1, len(support_variables))
        top_idx = np.argsort(-abs_vec, kind="mergesort")[:topk]
        topk_mass_fraction = float(np.sum(abs_vec[top_idx] ** 2) / norm_sq)

        subgraph = extract_support_subgraph(H, support_variables)
        a, b = compute_trapping_parameters(subgraph)
        classification = classify_trapping_set(
            support_variables,
            a,
            b,
            {
                "ipr": ipr,
                "support_fraction": support_fraction,
                "topk_mass_fraction": topk_mass_fraction,
            },
        )

        severity = round(float(-ev * (1.0 + ipr)), _ROUND)
        defects.append(
            SpectralDefect(
                eigenvalue=round(ev, _ROUND),
                severity=severity,
                support_nodes=tuple(support_variables),
                classification=classification,
                a=int(a),
                b=int(b),
            )
        )

    defects.sort(
        key=lambda d: (
            -d.severity,
            d.eigenvalue,
            len(d.support_nodes),
            d.support_nodes,
            d.classification,
            d.a,
            d.b,
        )
    )
    return defects
