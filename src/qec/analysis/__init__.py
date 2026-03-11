"""
v11.0.0 — Analysis subsystem.

Provides trapping-set detection and structural analysis for LDPC/QLDPC
Tanner graphs.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from src.qec.analysis.trapping_sets import TrappingSetDetector

__all__ = [
    "TrappingSetDetector",
]
