"""
v11.3.0 — Analysis subsystem.

Provides trapping-set detection, BP residual analysis, Bethe Hessian
stability estimation, absorbing-set prediction, and cycle topology
analysis for LDPC/QLDPC Tanner graphs.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from src.qec.analysis.trapping_sets import TrappingSetDetector
from src.qec.analysis.bp_residuals import BPResidualAnalyzer
from src.qec.analysis.bethe_hessian import BetheHessianAnalyzer
from src.qec.analysis.absorbing_sets import AbsorbingSetPredictor
from src.qec.analysis.cycle_topology import CycleTopologyAnalyzer

__all__ = [
    "TrappingSetDetector",
    "BPResidualAnalyzer",
    "BetheHessianAnalyzer",
    "AbsorbingSetPredictor",
    "CycleTopologyAnalyzer",
]
