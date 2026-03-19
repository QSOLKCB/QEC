"""
v9.2.0 — Discovery Public API.

Exposes all public discovery functions from a single module.
Import from here for stable access to the discovery layer.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
"""

from __future__ import annotations

from qec.discovery.discovery_engine import run_structure_discovery
from qec.discovery.objectives import compute_discovery_objectives
from qec.discovery.mutation_operators import mutate_tanner_graph
from qec.discovery.repair_operators import repair_tanner_graph
from qec.discovery.archive import update_discovery_archive
from qec.discovery.spectral_bad_edge import detect_bad_edges
from qec.discovery.cycle_pressure import compute_cycle_pressure
from qec.discovery.ace_filter import compute_local_ace_score
from qec.discovery.incremental_metrics import update_metrics_incrementally
from qec.discovery.basin_aware_flow import BasinAwareFlowConfig, BasinAwareSpectralFlow
from qec.discovery.adaptive_mutation_controller import (
    AdaptiveMutationConfig,
    AdaptiveMutationController,
    NonBacktrackingFlowMutator,
    NBGradientMutator,
)
from qec.discovery.nb_flow_mutation import NBFlowMutationConfig, NonBacktrackingFlowMutator

__all__ = [
    "run_structure_discovery",
    "compute_discovery_objectives",
    "mutate_tanner_graph",
    "repair_tanner_graph",
    "update_discovery_archive",
    "detect_bad_edges",
    "compute_cycle_pressure",
    "compute_local_ace_score",
    "update_metrics_incrementally",
    "BasinAwareFlowConfig",
    "BasinAwareSpectralFlow",
    "AdaptiveMutationConfig",
    "AdaptiveMutationController",
    "NonBacktrackingFlowMutator",
    "NBGradientMutator",
    "NBFlowMutationConfig",
    "NonBacktrackingFlowMutator",
]
