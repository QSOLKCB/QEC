"""Shared constants and utilities for pipeline modules.

Version: v71.0.0
"""

import numpy as np


# Keys excluded from pairwise delta computation — single source of truth
# shared by build_experiment_table (as reserved keys) and build_pairwise_comparison.
_EXCLUDED_KEYS = frozenset({
    "genome_id", "scenario", "version", "base_seed_label",
    "n_vars", "n_iters_base",
})


def _is_finite_numeric(val) -> bool:
    """Return True if val is a finite int or float (not NaN, not inf)."""
    if not isinstance(val, (int, float)):
        return False
    if isinstance(val, bool):
        return False
    try:
        return not (np.isnan(val) or np.isinf(val))
    except (TypeError, ValueError):
        return False
