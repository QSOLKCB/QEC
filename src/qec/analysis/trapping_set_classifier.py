from __future__ import annotations

import numpy as np

from src.qec.analysis.localization_metrics import (
    effective_support_size,
    inverse_participation_ratio,
    participation_entropy,
)


def classify_defect(v: np.ndarray) -> dict[str, float | str]:
    x = np.asarray(v, dtype=np.float64).ravel()
    n = max(1, x.size)
    pe = participation_entropy(x)
    ipr = inverse_participation_ratio(x)
    n_eff = effective_support_size(x)
    ln_n = float(np.log(float(n)))

    if pe < 0.15 * ln_n and ipr > 10.0 / float(n):
        defect_type = "SMALL_TRAPPING_SET"
    elif pe < 0.40 * ln_n:
        defect_type = "CYCLE_CLUSTER"
    elif pe < 0.55 * ln_n:
        defect_type = "PSEUDO_CODEWORD_BASIN"
    else:
        defect_type = "BENIGN_GLOBAL_MODE"

    return {
        "defect_type": defect_type,
        "participation_entropy": pe,
        "ipr": ipr,
        "n_eff": n_eff,
    }
