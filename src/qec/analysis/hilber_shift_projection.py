from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.layer_spec_contract import _ensure_json_safe

HILBER_SHIFT_VERSION = "v153.6"
SHIFT_ALGORITHM = "DISCRETE_CYCLIC_INDEX_SHIFT_V1"
MAX_SHIFT_ITEMS = 128
MAX_FILTER_BINDINGS = 128

_ALLOWED_DIRECTIONS = {"FORWARD", "REVERSE"}
_ALLOWED_WRAP_MODE = {"CYCLIC"}
_ALLOWED_ORDERING_RULE = {"EXPLICIT_INPUT_ORDER"}
_ALLOWED_COMPATIBILITY_STATUS = {"COMPATIBLE", "INCOMPATIBLE"}
_ALLOWED_COMPATIBILITY_REASON = {"IDENTITY_PRESERVED", "IDENTITY_MISMATCH"}
_ALLOWED_STABILITY_STATUS = {"STABLE", "UNSTABLE"}
_ALLOWED_STABILITY_REASON = {"ROUNDTRIP_MATCH", "ROUNDTRIP_MISMATCH"}


def _freeze_shift_pair(pair: dict[str, Any]) -> MappingProxyType[str, Any]:
    """Freeze a shift pair dict to prevent mutation."""
    return MappingProxyType(dict(pair))


def _scope_guard() -> None:
    forbidden = {
        "apply", "execute", "run",
        "traverse", "pathfind", "resolve",
        "search", "filter", "readout",
        "shell", "matrix", "markov",
    }
    for _cls in (
        HilberShiftSpec,
        ShiftProjectionReceipt,
        FilterCompatibilityReceipt,
        ShiftStabilityReceipt,
    ):
        for _name in forbidden:
            if hasattr(_cls, _name):
                raise RuntimeError("INVALID_STATE")


def _validate_filter_binding_hashes(hashes: tuple[str, ...]) -> tuple[str, ...]:
    """Validate filter binding hashes: type, uniqueness, and canonical ordering."""
    if not isinstance(hashes, tuple):
        raise ValueError("INVALID_INPUT")
    if len(hashes) > MAX_FILTER_BINDINGS:
        raise ValueError("INVALID_INPUT")
    if not all(isinstance(h, str) and h for h in hashes):
        raise ValueError("INVALID_INPUT")
    if len(set(hashes)) != len(hashes):
        raise ValueError("INVALID_INPUT")
    canonical = tuple(sorted(hashes))
    if canonical != hashes:
        raise ValueError("INVALID_INPUT")
    return hashes


def _validate_item_ids(item_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(item_ids, tuple) or not item_ids or len(item_ids) > MAX_SHIFT_ITEMS:
        raise ValueError("INVALID_INPUT")
    if not all(isinstance(x, str) and x for x in item_ids):
        raise ValueError("INVALID_INPUT")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("INVALID_INPUT")
    return item_ids


def _validate_shift_offset(offset: int, n: int) -> None:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0 or offset >= n:
        raise ValueError("INVALID_INPUT")


def _order_hash(items: tuple[str, ...]) -> str:
    return sha256_hex({"ordered_item_ids": list(items)})


@dataclass(frozen=True)
class HilberShiftSpec:
    shift_id: str
    shift_version: str
    source_type: str
    source_id: str
    source_hash: str
    ordered_item_ids: tuple[str, ...]
    shift_algorithm: str
    shift_direction: str
    shift_offset: int
    wrap_mode: str
    ordering_rule: str
    spec_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_item_ids", tuple(self.ordered_item_ids))
        if not all(isinstance(x, str) and x for x in (self.shift_id, self.shift_version, self.source_type, self.source_id, self.source_hash)):
            raise ValueError("INVALID_INPUT")
        _validate_item_ids(self.ordered_item_ids)
        _validate_shift_offset(self.shift_offset, len(self.ordered_item_ids))
        if self.shift_algorithm != SHIFT_ALGORITHM or self.shift_direction not in _ALLOWED_DIRECTIONS:
            raise ValueError("INVALID_INPUT")
        if self.wrap_mode not in _ALLOWED_WRAP_MODE or self.ordering_rule not in _ALLOWED_ORDERING_RULE:
            raise ValueError("INVALID_INPUT")
        if self.spec_hash and self.spec_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict[str, Any]:
        payload = {
            "shift_id": self.shift_id,
            "shift_version": self.shift_version,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_hash": self.source_hash,
            "ordered_item_ids": list(self.ordered_item_ids),
            "shift_algorithm": self.shift_algorithm,
            "shift_direction": self.shift_direction,
            "shift_offset": self.shift_offset,
            "wrap_mode": self.wrap_mode,
            "ordering_rule": self.ordering_rule,
        }
        _ensure_json_safe(payload)
        return payload

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), spec_hash=self.spec_hash)


@dataclass(frozen=True)
class ShiftProjectionReceipt:
    shift_id: str
    shift_spec_hash: str
    input_item_ids: tuple[str, ...]
    shifted_item_ids: tuple[str, ...]
    shift_pairs: tuple[MappingProxyType[str, Any], ...]
    input_order_hash: str
    shifted_order_hash: str
    shift_projection_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_item_ids", tuple(self.input_item_ids))
        object.__setattr__(self, "shifted_item_ids", tuple(self.shifted_item_ids))
        try:
            frozen_pairs = tuple(_freeze_shift_pair(x) for x in self.shift_pairs)
        except (KeyError, TypeError) as exc:
            raise ValueError("INVALID_INPUT") from exc
        object.__setattr__(self, "shift_pairs", frozen_pairs)
        _validate_item_ids(self.input_item_ids)
        _validate_item_ids(self.shifted_item_ids)
        if set(self.input_item_ids) != set(self.shifted_item_ids):
            raise ValueError("INVALID_INPUT")
        if len(self.shift_pairs) != len(self.input_item_ids):
            raise ValueError("INVALID_INPUT")
        try:
            for p in self.shift_pairs:
                _ = p["item_id"], p["source_index"], p["target_index"]
            expected_pairs = tuple(sorted(self.shift_pairs, key=lambda x: (x["target_index"], x["item_id"])))
            if expected_pairs != self.shift_pairs:
                raise ValueError("INVALID_INPUT")
            src = [p["source_index"] for p in self.shift_pairs]
            tgt = [p["target_index"] for p in self.shift_pairs]
            item_ids_in_pairs = [p["item_id"] for p in self.shift_pairs]
        except (KeyError, TypeError) as exc:
            raise ValueError("INVALID_INPUT") from exc
        if len(set(item_ids_in_pairs)) != len(self.input_item_ids):
            raise ValueError("INVALID_INPUT")
        if len(set(src)) != len(self.input_item_ids) or len(set(tgt)) != len(self.input_item_ids):
            raise ValueError("INVALID_INPUT")
        if set(src) != set(range(len(self.input_item_ids))) or set(tgt) != set(range(len(self.input_item_ids))):
            raise ValueError("INVALID_INPUT")
        if self.input_order_hash != _order_hash(self.input_item_ids) or self.shifted_order_hash != _order_hash(self.shifted_item_ids):
            raise ValueError("INVALID_INPUT")
        if self.shift_projection_hash and self.shift_projection_hash != self._projection_hash():
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _projection_payload(self) -> dict[str, Any]:
        return {
            "shift_id": self.shift_id,
            "shift_spec_hash": self.shift_spec_hash,
            "input_item_ids": list(self.input_item_ids),
            "shifted_item_ids": list(self.shifted_item_ids),
            "shift_pairs": [dict(p) for p in self.shift_pairs],
            "input_order_hash": self.input_order_hash,
            "shifted_order_hash": self.shifted_order_hash,
        }

    def _projection_hash(self) -> str:
        return sha256_hex(self._projection_payload())

    def _canonical_payload(self) -> dict[str, Any]:
        p = dict(self._projection_payload(), shift_projection_hash=self.shift_projection_hash)
        _ensure_json_safe(p)
        return p

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), receipt_hash=self.receipt_hash)


@dataclass(frozen=True)
class FilterCompatibilityReceipt:
    compatibility_id: str
    shift_projection_receipt_hash: str
    input_order_hash: str
    shifted_order_hash: str
    filter_binding_hashes: tuple[str, ...]
    compatibility_status: str
    compatibility_reason: str
    compatibility_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        validated = _validate_filter_binding_hashes(self.filter_binding_hashes)
        object.__setattr__(self, "filter_binding_hashes", validated)
        if self.compatibility_status not in _ALLOWED_COMPATIBILITY_STATUS or self.compatibility_reason not in _ALLOWED_COMPATIBILITY_REASON:
            raise ValueError("INVALID_INPUT")
        if self.compatibility_status == "COMPATIBLE" and self.compatibility_reason != "IDENTITY_PRESERVED":
            raise ValueError("INVALID_INPUT")
        if self.compatibility_status == "INCOMPATIBLE" and self.compatibility_reason != "IDENTITY_MISMATCH":
            raise ValueError("INVALID_INPUT")
        if self.compatibility_hash and self.compatibility_hash != self._compatibility_hash():
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _compatibility_payload(self) -> dict[str, Any]:
        return {
            "compatibility_id": self.compatibility_id,
            "shift_projection_receipt_hash": self.shift_projection_receipt_hash,
            "input_order_hash": self.input_order_hash,
            "shifted_order_hash": self.shifted_order_hash,
            "filter_binding_hashes": list(self.filter_binding_hashes),
            "compatibility_status": self.compatibility_status,
            "compatibility_reason": self.compatibility_reason,
        }

    def _compatibility_hash(self) -> str:
        return sha256_hex(self._compatibility_payload())

    def _canonical_payload(self) -> dict[str, Any]:
        p = dict(self._compatibility_payload(), compatibility_hash=self.compatibility_hash)
        _ensure_json_safe(p)
        return p

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), receipt_hash=self.receipt_hash)


@dataclass(frozen=True)
class ShiftStabilityReceipt:
    stability_id: str
    shift_projection_receipt_hash: str
    filter_compatibility_receipt_hash: str
    input_order_hash: str
    shifted_order_hash: str
    inverse_order_hash: str
    roundtrip_stable: bool
    stability_status: str
    stability_reason: str
    stability_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        if self.stability_status not in _ALLOWED_STABILITY_STATUS or self.stability_reason not in _ALLOWED_STABILITY_REASON:
            raise ValueError("INVALID_INPUT")
        if self.roundtrip_stable != (self.input_order_hash == self.inverse_order_hash):
            raise ValueError("INVALID_INPUT")
        exp_status = "STABLE" if self.roundtrip_stable else "UNSTABLE"
        exp_reason = "ROUNDTRIP_MATCH" if self.roundtrip_stable else "ROUNDTRIP_MISMATCH"
        if self.stability_status != exp_status or self.stability_reason != exp_reason:
            raise ValueError("INVALID_INPUT")
        if self.stability_hash and self.stability_hash != self._stability_hash():
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _stability_payload(self) -> dict[str, Any]:
        return {
            "stability_id": self.stability_id,
            "shift_projection_receipt_hash": self.shift_projection_receipt_hash,
            "filter_compatibility_receipt_hash": self.filter_compatibility_receipt_hash,
            "input_order_hash": self.input_order_hash,
            "shifted_order_hash": self.shifted_order_hash,
            "inverse_order_hash": self.inverse_order_hash,
            "roundtrip_stable": self.roundtrip_stable,
            "stability_status": self.stability_status,
            "stability_reason": self.stability_reason,
        }

    def _stability_hash(self) -> str:
        return sha256_hex(self._stability_payload())

    def _canonical_payload(self) -> dict[str, Any]:
        p = dict(self._stability_payload(), stability_hash=self.stability_hash)
        _ensure_json_safe(p)
        return p

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), receipt_hash=self.receipt_hash)


def build_hilber_shift_spec(shift_id: str, source_type: str, source_id: str, source_hash: str, ordered_item_ids: tuple[str, ...], shift_direction: str, shift_offset: int) -> HilberShiftSpec:
    spec = HilberShiftSpec(shift_id, HILBER_SHIFT_VERSION, source_type, source_id, source_hash, tuple(ordered_item_ids), SHIFT_ALGORITHM, shift_direction, shift_offset, "CYCLIC", "EXPLICIT_INPUT_ORDER", "")
    return HilberShiftSpec(**{**spec.__dict__, "spec_hash": spec.stable_hash()})


def build_shift_projection_receipt(spec: HilberShiftSpec) -> ShiftProjectionReceipt:
    if spec.spec_hash != spec.stable_hash():
        raise ValueError("INVALID_INPUT")
    if spec.shift_version != HILBER_SHIFT_VERSION or spec.shift_algorithm != SHIFT_ALGORITHM:
        raise ValueError("INVALID_INPUT")
    n = len(spec.ordered_item_ids)
    pairs: list[dict[str, Any]] = []
    targets: set[int] = set()
    shifted = [""] * n
    for i, item_id in enumerate(spec.ordered_item_ids):
        t = (i + spec.shift_offset) % n if spec.shift_direction == "FORWARD" else (i - spec.shift_offset) % n
        if t in targets:
            raise ValueError("INVALID_INPUT")
        targets.add(t)
        shifted[t] = item_id
        pairs.append({"item_id": item_id, "source_index": i, "target_index": t})
    if targets != set(range(n)):
        raise ValueError("INVALID_INPUT")
    sp = tuple(sorted(pairs, key=lambda x: (x["target_index"], x["item_id"])))
    payload = {
        "shift_id": spec.shift_id,
        "shift_spec_hash": spec.spec_hash,
        "input_item_ids": tuple(spec.ordered_item_ids),
        "shifted_item_ids": tuple(shifted),
        "shift_pairs": sp,
        "input_order_hash": _order_hash(tuple(spec.ordered_item_ids)),
        "shifted_order_hash": _order_hash(tuple(shifted)),
    }
    projection_hash = sha256_hex({
        "shift_id": payload["shift_id"],
        "shift_spec_hash": payload["shift_spec_hash"],
        "input_item_ids": list(payload["input_item_ids"]),
        "shifted_item_ids": list(payload["shifted_item_ids"]),
        "shift_pairs": list(payload["shift_pairs"]),
        "input_order_hash": payload["input_order_hash"],
        "shifted_order_hash": payload["shifted_order_hash"],
    })
    payload["shift_projection_hash"] = projection_hash
    receipt_hash = sha256_hex({
        "shift_id": payload["shift_id"],
        "shift_spec_hash": payload["shift_spec_hash"],
        "input_item_ids": list(payload["input_item_ids"]),
        "shifted_item_ids": list(payload["shifted_item_ids"]),
        "shift_pairs": list(payload["shift_pairs"]),
        "input_order_hash": payload["input_order_hash"],
        "shifted_order_hash": payload["shifted_order_hash"],
        "shift_projection_hash": payload["shift_projection_hash"],
    })
    payload["receipt_hash"] = receipt_hash
    return ShiftProjectionReceipt(**payload)


def build_filter_compatibility_receipt(compatibility_id: str, shift_projection_receipt: ShiftProjectionReceipt, filter_binding_hashes: tuple[str, ...]) -> FilterCompatibilityReceipt:
    canonical_bindings = tuple(sorted(filter_binding_hashes))
    _validate_filter_binding_hashes(canonical_bindings)
    order_preserved = shift_projection_receipt.input_order_hash == shift_projection_receipt.shifted_order_hash
    status = "COMPATIBLE" if order_preserved else "INCOMPATIBLE"
    reason = "IDENTITY_PRESERVED" if status == "COMPATIBLE" else "IDENTITY_MISMATCH"
    payload = {
        "compatibility_id": compatibility_id,
        "shift_projection_receipt_hash": shift_projection_receipt.receipt_hash,
        "input_order_hash": shift_projection_receipt.input_order_hash,
        "shifted_order_hash": shift_projection_receipt.shifted_order_hash,
        "filter_binding_hashes": canonical_bindings,
        "compatibility_status": status,
        "compatibility_reason": reason,
    }
    compatibility_hash = sha256_hex({
        "compatibility_id": payload["compatibility_id"],
        "shift_projection_receipt_hash": payload["shift_projection_receipt_hash"],
        "input_order_hash": payload["input_order_hash"],
        "shifted_order_hash": payload["shifted_order_hash"],
        "filter_binding_hashes": list(payload["filter_binding_hashes"]),
        "compatibility_status": payload["compatibility_status"],
        "compatibility_reason": payload["compatibility_reason"],
    })
    payload["compatibility_hash"] = compatibility_hash
    receipt_hash = sha256_hex({
        "compatibility_id": payload["compatibility_id"],
        "shift_projection_receipt_hash": payload["shift_projection_receipt_hash"],
        "input_order_hash": payload["input_order_hash"],
        "shifted_order_hash": payload["shifted_order_hash"],
        "filter_binding_hashes": list(payload["filter_binding_hashes"]),
        "compatibility_status": payload["compatibility_status"],
        "compatibility_reason": payload["compatibility_reason"],
        "compatibility_hash": payload["compatibility_hash"],
    })
    payload["receipt_hash"] = receipt_hash
    return FilterCompatibilityReceipt(**payload)


def build_shift_stability_receipt(stability_id: str, shift_projection_receipt: ShiftProjectionReceipt, filter_compatibility_receipt: FilterCompatibilityReceipt) -> ShiftStabilityReceipt:
    if filter_compatibility_receipt.shift_projection_receipt_hash != shift_projection_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    if filter_compatibility_receipt.input_order_hash != shift_projection_receipt.input_order_hash:
        raise ValueError("INVALID_INPUT")
    if filter_compatibility_receipt.shifted_order_hash != shift_projection_receipt.shifted_order_hash:
        raise ValueError("INVALID_INPUT")
    n = len(shift_projection_receipt.shifted_item_ids)
    inverse_items = [""] * n
    for p in shift_projection_receipt.shift_pairs:
        src = p["source_index"]
        tgt = p["target_index"]
        inverse_items[src] = shift_projection_receipt.shifted_item_ids[tgt]
    inverse_order_hash = _order_hash(tuple(inverse_items))
    roundtrip = inverse_order_hash == shift_projection_receipt.input_order_hash
    status = "STABLE" if roundtrip else "UNSTABLE"
    reason = "ROUNDTRIP_MATCH" if roundtrip else "ROUNDTRIP_MISMATCH"
    payload = {
        "stability_id": stability_id,
        "shift_projection_receipt_hash": shift_projection_receipt.receipt_hash,
        "filter_compatibility_receipt_hash": filter_compatibility_receipt.receipt_hash,
        "input_order_hash": shift_projection_receipt.input_order_hash,
        "shifted_order_hash": shift_projection_receipt.shifted_order_hash,
        "inverse_order_hash": inverse_order_hash,
        "roundtrip_stable": roundtrip,
        "stability_status": status,
        "stability_reason": reason,
    }
    stability_hash = sha256_hex(dict(payload))
    payload["stability_hash"] = stability_hash
    receipt_hash = sha256_hex(dict(payload))
    payload["receipt_hash"] = receipt_hash
    return ShiftStabilityReceipt(**payload)


def validate_shift_stability_receipt(receipt: ShiftStabilityReceipt) -> None:
    if receipt.stability_hash != receipt._stability_hash() or receipt.receipt_hash != receipt.stable_hash():
        raise ValueError("INVALID_INPUT")


_scope_guard()
