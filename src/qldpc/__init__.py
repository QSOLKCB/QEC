"""
QLDPC CSS code construction package.

Protograph-based quantum LDPC codes with:
- GF(2^e) finite field arithmetic
- Deterministic shared-circulant lifting
- Structural CSS orthogonality (H_X @ H_Z^T = 0 mod 2)
- Joint X/Z belief propagation decoder

Submodules are imported lazily: only modules that exist are pulled in.
This allows incremental development (e.g. running field tests before
the decoder module is written).
"""

__all__ = []


def _try_import(name, attrs):
    """Import *attrs* from submodule *name*, silently skipping if missing."""
    try:
        mod = __import__(f"{__package__}.{name}", fromlist=attrs)
        for attr in attrs:
            globals()[attr] = getattr(mod, attr)
            __all__.append(attr)
    except (ImportError, ModuleNotFoundError):
        pass


_try_import("field", ["GF2e"])
_try_import("protograph", ["ProtographPair", "build_protograph_pair"])
_try_import("lift", ["LiftingTable", "generate_lifting_table"])
_try_import("css_code", ["CSSCode", "create_code", "PREDEFINED_CODES"])
_try_import("decoder_bp", ["JointBPDecoder", "depolarizing_channel", "hashing_bound"])
_try_import("invariants", ["ConstructionInvariantError"])
