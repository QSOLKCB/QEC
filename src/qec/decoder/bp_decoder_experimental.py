"""
Experimental BP decoder sandbox.

Safe location for testing decoding algorithm modifications.
Initially re-exports the same canonical ``bp_decode`` as the reference
decoder.  Future experiments modify this file only.
"""

from qec_qldpc_codes import bp_decode

__all__ = ["bp_decode"]
