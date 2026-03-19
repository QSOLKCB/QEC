"""
Reference BP decoder.

Frozen baseline implementation used for deterministic decoder experiments.
This file must never be modified once introduced.

Re-exports the canonical ``bp_decode`` from ``src.qec_qldpc_codes`` so that
the decoder experiment framework can refer to it by name without altering
the original module.
"""

from qec_qldpc_codes import bp_decode

__all__ = ["bp_decode"]
