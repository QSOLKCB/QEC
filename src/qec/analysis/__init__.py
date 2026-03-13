"""
v12.0.0 — Analysis subsystem.

Provides trapping-set detection, BP residual analysis, Bethe Hessian
stability estimation, absorbing-set prediction, cycle topology
analysis, residual cluster detection, non-backtracking flow analysis,
constraint tension analysis, and eigenvector localization (IPR)
diagnostics for LDPC/QLDPC Tanner graphs.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from src.qec.analysis.trapping_sets import TrappingSetDetector
from src.qec.analysis.bp_residuals import BPResidualAnalyzer
from src.qec.analysis.bethe_hessian import BetheHessianAnalyzer
from src.qec.analysis.absorbing_sets import AbsorbingSetPredictor
from src.qec.analysis.cycle_topology import CycleTopologyAnalyzer
from src.qec.analysis.residual_clusters import ResidualClusterAnalyzer
from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from src.qec.analysis.constraint_tension import ConstraintTensionAnalyzer
from src.qec.analysis.eigenvector_localization import EigenvectorLocalizationAnalyzer
from src.qec.analysis.flow_alignment import FlowAlignmentAnalyzer
from src.qec.analysis.nb_instability_gradient import NBInstabilityGradientAnalyzer
from src.qec.analysis.nb_eigenmode_flow import NBEigenmodeFlowAnalyzer

__all__ = [
    "TrappingSetDetector",
    "BPResidualAnalyzer",
    "BetheHessianAnalyzer",
    "AbsorbingSetPredictor",
    "CycleTopologyAnalyzer",
    "ResidualClusterAnalyzer",
    "NonBacktrackingFlowAnalyzer",
    "ConstraintTensionAnalyzer",
    "EigenvectorLocalizationAnalyzer",
    "FlowAlignmentAnalyzer",
    "NBInstabilityGradientAnalyzer",
    "NBEigenmodeFlowAnalyzer",
]
