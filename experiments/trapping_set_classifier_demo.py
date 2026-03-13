from __future__ import annotations

import os
import sys

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.bethe_hessian import build_bethe_hessian, extract_negative_modes
from src.qec.analysis.defect_catalog import detect_spectral_defects


def _demo_matrix() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1],
    ], dtype=np.float64)


def main() -> None:
    H = _demo_matrix()
    H_bh = build_bethe_hessian(H)
    evals, _ = extract_negative_modes(H_bh, k_max=12)
    defects = detect_spectral_defects(H, k_max=12, use_multiresolution_scan=True, scan_steps=3)

    print(f"Negative modes: {len(evals)}")
    print(f"Detected {len(defects)} defects")
    for d in defects:
        print("-" * 32)
        print(f"Type: {d.defect_type}")
        print(f"(a,b): {d.ab}")
        print(f"Severity: {d.severity:.6f}")
        print(f"Support nodes: {d.support_nodes}")


if __name__ == "__main__":
    main()
