from __future__ import annotations

import os
import sys

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.analysis.api import detect_spectral_defects



def main() -> None:
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)

    defects = detect_spectral_defects(H)

    print("v13.3.0 trapping-set classifier demo")
    print(f"defects_detected={len(defects)}")
    for idx, defect in enumerate(defects):
        support = ",".join(str(v) for v in defect.support_nodes)
        print(
            f"#{idx}: eig={defect.eigenvalue:.12f} "
            f"severity={defect.severity:.12f} "
            f"class={defect.classification} "
            f"(a,b)=({defect.a},{defect.b}) "
            f"support=[{support}]"
        )


if __name__ == "__main__":
    main()
