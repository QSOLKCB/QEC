"""
Ternary decoder sandbox — deterministic decoder research module.

This package provides a ternary message-passing decoder for exploring
decoder dynamics using a three-state message alphabet (+1, 0, -1).

This is a research sandbox.  It does not modify the existing BP decoder.
All operations are fully deterministic with no hidden randomness.
"""

from .ternary_messages import encode_ternary, decode_ternary
from .ternary_update_rules import variable_node_update, check_node_update
from .ternary_decoder import run_ternary_decoder
from .ternary_metrics import (
    compute_ternary_stability,
    compute_ternary_entropy,
    compute_ternary_conflict_density,
)
from .ternary_trapping import (
    detect_zero_regions,
    compute_frustration_index,
    detect_persistent_zero_states,
    estimate_trapping_indicator,
)

__all__ = [
    "encode_ternary",
    "decode_ternary",
    "variable_node_update",
    "check_node_update",
    "run_ternary_decoder",
    "compute_ternary_stability",
    "compute_ternary_entropy",
    "compute_ternary_conflict_density",
    "detect_zero_regions",
    "compute_frustration_index",
    "detect_persistent_zero_states",
    "estimate_trapping_indicator",
]
