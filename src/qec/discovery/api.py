"""
v9.2.0 — Discovery Public API.

Exposes all public discovery functions from a single module.
Import from here for stable access to the discovery layer.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
"""

from __future__ import annotations

from src.qec.discovery.discovery_engine import run_structure_discovery
from src.qec.discovery.objectives import compute_discovery_objectives
from src.qec.discovery.mutation_operators import mutate_tanner_graph
from src.qec.discovery.repair_operators import repair_tanner_graph
from src.qec.discovery.archive import update_discovery_archive
from src.qec.discovery.spectral_bad_edge import detect_bad_edges
from src.qec.discovery.cycle_pressure import compute_cycle_pressure
from src.qec.discovery.ace_filter import compute_local_ace_score
from src.qec.discovery.incremental_metrics import update_metrics_incrementally
from src.qec.discovery.basin_aware_flow import BasinAwareFlowConfig, BasinAwareSpectralFlow
from src.qec.discovery.adaptive_mutation_controller import (
    AdaptiveMutationConfig,
    AdaptiveMutationController,
    NonBacktrackingFlowMutator,
    NBGradientMutator,
)

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
]
