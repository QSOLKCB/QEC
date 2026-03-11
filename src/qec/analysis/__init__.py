"""
v11.2.0 — Analysis subsystem.

Provides trapping-set detection, BP residual analysis, and Bethe Hessian
stability estimation for LDPC/QLDPC Tanner graphs.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from src.qec.analysis.trapping_sets import TrappingSetDetector
from src.qec.analysis.bp_residuals import BPResidualAnalyzer
from src.qec.analysis.bethe_hessian import BetheHessianAnalyzer

__all__ = [
    "TrappingSetDetector",
    "BPResidualAnalyzer",
    "BetheHessianAnalyzer",
]
