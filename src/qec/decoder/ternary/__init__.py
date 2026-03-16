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
from .ternary_rule_variants import (
    majority_rule,
    damped_majority_rule,
    conflict_averse_rule,
    parity_pressure_rule,
    RULE_REGISTRY,
    get_extended_rule_registry,
)
from .ternary_rule_mutations import (
    flip_zero_bias_rule,
    conservative_rule,
    inverted_majority_rule,
    generate_mutated_rules,
)
from .ternary_rule_evaluator import (
    run_decoder_with_rule,
    evaluate_decoder_rule,
)
from .ternary_coevolution import evaluate_rule_population

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
    "majority_rule",
    "damped_majority_rule",
    "conflict_averse_rule",
    "parity_pressure_rule",
    "RULE_REGISTRY",
    "get_extended_rule_registry",
    "flip_zero_bias_rule",
    "conservative_rule",
    "inverted_majority_rule",
    "generate_mutated_rules",
    "run_decoder_with_rule",
    "evaluate_decoder_rule",
    "evaluate_rule_population",
]
