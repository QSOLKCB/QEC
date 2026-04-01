"""Multi-scale field metrics layer.

Evaluates field structure at multiple resolutions to detect:
- local noise vs global structure
- scale-dependent stability
- hidden patterns missed at single scale

All functions are pure, deterministic, and never mutate inputs.
Dependencies: numpy only.

Note: Core implementations live in qec.core.metrics.
This module re-exports them for public API compatibility.
"""

from qec.core.metrics import (  # noqa: F401
    compute_multiscale_metrics,
    compute_multiscale_summary,
    compute_scale_consistency,
    compute_scale_divergence,
    downsample,
)

# Also re-export compute_field_metrics for any code that imported it via this module
from qec.core.metrics import compute_field_metrics  # noqa: F401
