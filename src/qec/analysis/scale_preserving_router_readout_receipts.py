from __future__ import annotations

from dataclasses import dataclass
import re

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.readout_projection_receipts import ReadoutProjectionReceipt
from qec.analysis.router_lattice_paths import RouterLatticePathReceipt
from qec.analysis.subgraph_invariant_pattern import (
    SubgraphInvariantPattern,
    _VALID_SCALES,
    _pattern_id_payload,
    _pattern_hash_payload_raw,
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256_hex(value: str) -> None:
    if not isinstance(value, str) or _SHA256_HEX_RE.fullmatch(value) is None:
        raise ValueError("INVALID_INPUT")


def _require_scale_index(scale_index: int) -> None:
    if not isinstance(scale_index, int) or isinstance(scale_index, bool) or scale_index not in _VALID_SCALES:
        raise ValueError("INVALID_SCALE_INDEX")


def _require_bool(value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError("INVALID_INPUT")


def _router_scale_hash_payload(
    router_lattice_path_receipt_hash: str,
    pattern_hash: str,
    scale_index: int,
    scale_preserved: bool,
) -> dict[str, object]:
    return {
        "router_lattice_path_receipt_hash": router_lattice_path_receipt_hash,
        "pattern_hash": pattern_hash,
        "scale_index": scale_index,
        "scale_preserved": scale_preserved,
    }


def _readout_scale_hash_payload(
    readout_projection_receipt_hash: str,
    pattern_hash: str,
    scale_index: int,
    scale_preserved: bool,
) -> dict[str, object]:
    return {
        "readout_projection_receipt_hash": readout_projection_receipt_hash,
        "pattern_hash": pattern_hash,
        "scale_index": scale_index,
        "scale_preserved": scale_preserved,
    }


def _validate_pattern_hash(pattern: SubgraphInvariantPattern) -> None:
    """Validate pattern integrity by recomputing and comparing hashes.
    
    Uses the canonical hash computation functions from subgraph_invariant_pattern
    to verify the pattern's identity has not been tampered with.
    """
    expected_pattern_id = sha256_hex(
        _pattern_id_payload(pattern.node_label_multiset, pattern.constraint_edge_pairs)
    )
    if pattern.pattern_id != expected_pattern_id:
        raise ValueError("HASH_MISMATCH")
    expected_pattern_hash = sha256_hex(
        _pattern_hash_payload_raw(
            pattern.pattern_id, pattern.node_label_multiset, pattern.constraint_edge_pairs
        )
    )
    if pattern.pattern_hash != expected_pattern_hash:
        raise ValueError("HASH_MISMATCH")


def _validate_router_scale_receipt_fields(
    router_lattice_path_receipt_hash: str,
    pattern_hash: str,
    scale_index: int,
    scale_preserved: bool,
    router_scale_receipt_hash: str,
) -> None:
    """Shared internal validation logic for RouterScaleReceipt fields."""
    _require_sha256_hex(router_lattice_path_receipt_hash)
    _require_sha256_hex(pattern_hash)
    _require_scale_index(scale_index)
    _require_bool(scale_preserved)
    expected = sha256_hex(
        _router_scale_hash_payload(
            router_lattice_path_receipt_hash,
            pattern_hash,
            scale_index,
            scale_preserved,
        )
    )
    if router_scale_receipt_hash != expected:
        raise ValueError("HASH_MISMATCH")


def _validate_readout_scale_projection_receipt_fields(
    readout_projection_receipt_hash: str,
    pattern_hash: str,
    scale_index: int,
    scale_preserved: bool,
    readout_scale_projection_receipt_hash: str,
) -> None:
    """Shared internal validation logic for ReadoutScaleProjectionReceipt fields."""
    _require_sha256_hex(readout_projection_receipt_hash)
    _require_sha256_hex(pattern_hash)
    _require_scale_index(scale_index)
    _require_bool(scale_preserved)
    expected = sha256_hex(
        _readout_scale_hash_payload(
            readout_projection_receipt_hash,
            pattern_hash,
            scale_index,
            scale_preserved,
        )
    )
    if readout_scale_projection_receipt_hash != expected:
        raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class RouterScaleReceipt:
    router_lattice_path_receipt_hash: str
    pattern_hash: str
    scale_index: int
    scale_preserved: bool
    router_scale_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_router_scale_receipt_fields(
            router_lattice_path_receipt_hash=self.router_lattice_path_receipt_hash,
            pattern_hash=self.pattern_hash,
            scale_index=self.scale_index,
            scale_preserved=self.scale_preserved,
            router_scale_receipt_hash=self.router_scale_receipt_hash,
        )


@dataclass(frozen=True)
class ReadoutScaleProjectionReceipt:
    readout_projection_receipt_hash: str
    pattern_hash: str
    scale_index: int
    scale_preserved: bool
    readout_scale_projection_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_readout_scale_projection_receipt_fields(
            readout_projection_receipt_hash=self.readout_projection_receipt_hash,
            pattern_hash=self.pattern_hash,
            scale_index=self.scale_index,
            scale_preserved=self.scale_preserved,
            readout_scale_projection_receipt_hash=self.readout_scale_projection_receipt_hash,
        )


def build_router_scale_receipt(
    router_receipt: RouterLatticePathReceipt,
    pattern: SubgraphInvariantPattern,
    scale_index: int,
) -> RouterScaleReceipt:
    """Build a router scale receipt.
    
    In v154.3, scale_preserved is always True because this phase forbids
    router re-execution. This ensures deterministic behavior:
    same inputs → same canonical JSON → same hash → same outcome.
    """
    if not isinstance(router_receipt, RouterLatticePathReceipt):
        raise ValueError("UNKNOWN_ROUTER_RECEIPT")
    if not hasattr(router_receipt, "receipt_hash") or router_receipt.receipt_hash != router_receipt.stable_hash():
        raise ValueError("UNKNOWN_ROUTER_RECEIPT")
    if not isinstance(pattern, SubgraphInvariantPattern):
        raise ValueError("INVALID_INPUT")
    _require_sha256_hex(pattern.pattern_hash)
    _validate_pattern_hash(pattern)
    _require_scale_index(scale_index)
    # v154.3: scale_preserved is deterministically True (no router re-execution)
    scale_preserved = True
    payload = _router_scale_hash_payload(
        router_lattice_path_receipt_hash=router_receipt.receipt_hash,
        pattern_hash=pattern.pattern_hash,
        scale_index=scale_index,
        scale_preserved=scale_preserved,
    )
    return RouterScaleReceipt(
        router_lattice_path_receipt_hash=router_receipt.receipt_hash,
        pattern_hash=pattern.pattern_hash,
        scale_index=scale_index,
        scale_preserved=scale_preserved,
        router_scale_receipt_hash=sha256_hex(payload),
    )


def build_readout_scale_projection_receipt(
    readout_receipt: ReadoutProjectionReceipt,
    pattern: SubgraphInvariantPattern,
    scale_index: int,
) -> ReadoutScaleProjectionReceipt:
    """Build a readout scale projection receipt.
    
    In v154.3, scale_preserved is always True because this phase forbids
    readout re-projection. This ensures deterministic behavior:
    same inputs → same canonical JSON → same hash → same outcome.
    """
    if not isinstance(readout_receipt, ReadoutProjectionReceipt):
        raise ValueError("UNKNOWN_READOUT_RECEIPT")
    if not hasattr(readout_receipt, "receipt_hash") or readout_receipt.receipt_hash != readout_receipt.stable_hash():
        raise ValueError("UNKNOWN_READOUT_RECEIPT")
    if not isinstance(pattern, SubgraphInvariantPattern):
        raise ValueError("INVALID_INPUT")
    _require_sha256_hex(pattern.pattern_hash)
    _validate_pattern_hash(pattern)
    _require_scale_index(scale_index)
    # v154.3: scale_preserved is deterministically True (no readout re-projection)
    scale_preserved = True
    payload = _readout_scale_hash_payload(
        readout_projection_receipt_hash=readout_receipt.receipt_hash,
        pattern_hash=pattern.pattern_hash,
        scale_index=scale_index,
        scale_preserved=scale_preserved,
    )
    return ReadoutScaleProjectionReceipt(
        readout_projection_receipt_hash=readout_receipt.receipt_hash,
        pattern_hash=pattern.pattern_hash,
        scale_index=scale_index,
        scale_preserved=scale_preserved,
        readout_scale_projection_receipt_hash=sha256_hex(payload),
    )


def validate_router_scale_receipt(r: RouterScaleReceipt) -> bool:
    if not isinstance(r, RouterScaleReceipt):
        raise ValueError("INVALID_INPUT")
    _validate_router_scale_receipt_fields(
        router_lattice_path_receipt_hash=r.router_lattice_path_receipt_hash,
        pattern_hash=r.pattern_hash,
        scale_index=r.scale_index,
        scale_preserved=r.scale_preserved,
        router_scale_receipt_hash=r.router_scale_receipt_hash,
    )
    return True


def validate_readout_scale_projection_receipt(r: ReadoutScaleProjectionReceipt) -> bool:
    if not isinstance(r, ReadoutScaleProjectionReceipt):
        raise ValueError("INVALID_INPUT")
    _validate_readout_scale_projection_receipt_fields(
        readout_projection_receipt_hash=r.readout_projection_receipt_hash,
        pattern_hash=r.pattern_hash,
        scale_index=r.scale_index,
        scale_preserved=r.scale_preserved,
        readout_scale_projection_receipt_hash=r.readout_scale_projection_receipt_hash,
    )
    return True


def assert_router_scale_preserved(r: RouterScaleReceipt) -> bool:
    validate_router_scale_receipt(r)
    if not r.scale_preserved:
        raise ValueError("SCALE_INVARIANCE_VIOLATION")
    return True


def assert_readout_scale_preserved(r: ReadoutScaleProjectionReceipt) -> bool:
    validate_readout_scale_projection_receipt(r)
    if not r.scale_preserved:
        raise ValueError("SCALE_INVARIANCE_VIOLATION")
    return True
