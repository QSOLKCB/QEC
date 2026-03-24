"""v100.0.0 — Reproducibility metadata for experiment outputs.

Generates deterministic metadata dicts capturing version, seed,
and environment information sufficient for exact reproduction.

All functions are:
- deterministic (except optional timestamp)
- side-effect free
- pure analysis signals
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict, Optional


_VERSION = "v100.0.0"


def build_reproducibility_metadata(
    seed: int,
    *,
    include_timestamp: bool = False,
) -> Dict[str, Any]:
    """Build a reproducibility metadata dict.

    Parameters
    ----------
    seed : int
        The RNG seed used for the experiment.
    include_timestamp : bool
        If True, include an ISO-8601 timestamp.  Disabled by default
        to preserve determinism of the metadata dict itself.

    Returns
    -------
    dict
        Metadata sufficient for exact reproduction.
    """
    import numpy as np

    metadata: Dict[str, Any] = {
        "version": _VERSION,
        "seed": seed,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }

    try:
        import scipy
        metadata["scipy_version"] = scipy.__version__
    except ImportError:
        pass

    if include_timestamp:
        import datetime
        metadata["timestamp"] = datetime.datetime.now(
            datetime.timezone.utc,
        ).isoformat()

    return metadata
