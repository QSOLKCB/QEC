"""Stable public analysis API exports."""

from __future__ import annotations

from .trapping_sets import TrappingSetDetector
from .bp_residuals import BPResidualAnalyzer
from .bethe_hessian import BetheHessianAnalyzer, estimate_nishimori_temperature
from .absorbing_sets import AbsorbingSetPredictor
from .cycle_topology import CycleTopologyAnalyzer
from .residual_clusters import ResidualClusterAnalyzer
from .nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from .constraint_tension import ConstraintTensionAnalyzer
from .eigenvector_localization import EigenvectorLocalizationAnalyzer
from .localization_metrics import IPR, ParticipationEntropy, SpectralInstabilityScore
from .flow_alignment import FlowAlignmentAnalyzer
from .nb_instability_gradient import NBInstabilityGradientAnalyzer
from .nb_eigenmode_flow import NBEigenmodeFlowAnalyzer
from .nb_perturbation_scorer import NBPerturbationScorer
from .basin_switch_detector import detect_basin_switch

from .defect_catalog import SpectralDefect, detect_spectral_defects
from .trapping_set_classifier import classify_trapping_set
from .subgraph_extractor import extract_support_subgraph

__all__ = [
    "TrappingSetDetector",
    "BPResidualAnalyzer",
    "BetheHessianAnalyzer",
    "estimate_nishimori_temperature",
    "AbsorbingSetPredictor",
    "CycleTopologyAnalyzer",
    "ResidualClusterAnalyzer",
    "NonBacktrackingFlowAnalyzer",
    "ConstraintTensionAnalyzer",
    "EigenvectorLocalizationAnalyzer",
    "IPR",
    "ParticipationEntropy",
    "SpectralInstabilityScore",
    "FlowAlignmentAnalyzer",
    "NBInstabilityGradientAnalyzer",
    "NBEigenmodeFlowAnalyzer",
    "NBPerturbationScorer",
    "detect_basin_switch",
    "SpectralDefect",
    "detect_spectral_defects",
    "classify_trapping_set",
    "extract_support_subgraph",
]
