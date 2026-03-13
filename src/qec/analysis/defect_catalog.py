from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse

from src.qec.analysis.bethe_hessian import build_bethe_hessian, extract_negative_modes
from src.qec.analysis.localization_metrics import extract_support
from src.qec.analysis.subgraph_extractor import extract_induced_subgraph
from src.qec.analysis.trapping_set_classifier import classify_defect


@dataclass(frozen=True)
class SpectralDefect:
    eigenvalue: float
    defect_type: str
    participation_entropy: float
    ipr: float
    support_nodes: list[int]
    ab: tuple[int, int]
    severity: float


def _severity(eigenvalue: float, pe: float, n: int, a: int, b: int) -> float:
    ln_n = float(np.log(float(max(1, n))))
    pe_term = 1.0 - (pe / ln_n) if ln_n > 0.0 else 1.0
    ratio = float(b) / float(max(1, a))
    sev = (abs(float(eigenvalue)) / (1.0 + abs(float(eigenvalue)))) * (
        0.4 * pe_term + 0.4 * float(np.exp(-ratio)) + 0.2
    )
    return round(float(sev), 12)


def detect_spectral_defects(
    H_pc: np.ndarray | scipy.sparse.spmatrix,
    *,
    k_max: int = 8,
    use_multiresolution_scan: bool = False,
    scan_steps: int = 3,
) -> list[SpectralDefect]:
    if scipy.sparse.issparse(H_pc):
        H = H_pc.tocsr().astype(np.float64)
        n_vars = H.shape[1]
        avg_deg = float(np.asarray(H.sum(axis=1), dtype=np.float64).ravel().mean()) if H.shape[0] > 0 else 0.0
    else:
        H = np.asarray(H_pc, dtype=np.float64)
        n_vars = H.shape[1]
        avg_deg = float(np.mean(np.sum(H != 0, axis=1))) if H.shape[0] > 0 else 0.0

    r_base = float(np.sqrt(max(avg_deg - 1.0, 0.0))) if avg_deg > 1.0 else 1.0
    rs = [r_base]
    if use_multiresolution_scan and scan_steps > 0:
        rs = [1.0 + (float(k) * (r_base - 1.0) / float(scan_steps)) for k in range(scan_steps + 1)]

    defects: list[SpectralDefect] = []
    for r in rs:
        H_bh = build_bethe_hessian(H, r=r)
        evals, evecs = extract_negative_modes(H_bh, k_max=k_max)
        for i in range(evals.size):
            ev = float(evals[i])
            vec = np.asarray(evecs[:, i], dtype=np.float64)
            var_vec = vec[:n_vars]
            cls = classify_defect(var_vec)
            support = extract_support(var_vec)
            sg = extract_induced_subgraph(H, support)
            a = int(sg["a"])
            b = int(sg["b"])
            sev = _severity(ev, float(cls["participation_entropy"]), n_vars, a, b)
            defects.append(SpectralDefect(
                eigenvalue=round(ev, 12),
                defect_type=str(cls["defect_type"]),
                participation_entropy=float(cls["participation_entropy"]),
                ipr=float(cls["ipr"]),
                support_nodes=list(support),
                ab=(a, b),
                severity=sev,
            ))

    if use_multiresolution_scan and defects:
        merged: list[SpectralDefect] = []
        for d in defects:
            s = set(d.support_nodes)
            found = False
            for j, cur in enumerate(merged):
                c = set(cur.support_nodes)
                overlap = len(s & c)
                union = len(s | c)
                jacc = (float(overlap) / float(union)) if union > 0 else 0.0
                if jacc >= 0.5:
                    if (d.severity, abs(d.eigenvalue), d.support_nodes) > (cur.severity, abs(cur.eigenvalue), cur.support_nodes):
                        merged[j] = d
                    found = True
                    break
            if not found:
                merged.append(d)
        defects = merged

    defects.sort(key=lambda d: (-d.severity, d.defect_type, d.support_nodes, d.eigenvalue))
    return defects
