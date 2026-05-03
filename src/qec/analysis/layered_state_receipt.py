from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.layer_spec_contract import LayerSpec, _deep_freeze, _ensure_json_safe


def _canonical_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze a mapping using canonical key ordering."""
    ordered: dict[str, Any] = {}
    for key in sorted(mapping, key=_canonical_key):
        ordered[key] = _deep_freeze(mapping[key])
    return MappingProxyType(ordered)


@dataclass(frozen=True)
class BaseStateReference:
    base_hash: str
    base_type: str
    base_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.base_hash or not self.base_type:
            raise ValueError("INVALID_INPUT")
        metadata = {} if self.base_metadata is None else dict(self.base_metadata)
        _ensure_json_safe(metadata)
        object.__setattr__(self, "base_metadata", _freeze_mapping(metadata))

    def _canonical_payload(self) -> dict:
        payload = {
            "base_hash": self.base_hash,
            "base_type": self.base_type,
            "base_metadata": dict(self.base_metadata or {}),
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        return dict(self._canonical_payload())

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class LayeredState:
    base_hash: str
    layer_spec_hash: str
    layer_payload_hash: str
    layered_hash: str

    def __post_init__(self) -> None:
        if not self.base_hash or not self.layer_spec_hash or not self.layer_payload_hash:
            raise ValueError("INVALID_INPUT")
        expected = sha256_hex(
            {
                "base_hash": self.base_hash,
                "layer_spec_hash": self.layer_spec_hash,
                "layer_payload_hash": self.layer_payload_hash,
            }
        )
        if expected != self.layered_hash:
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        payload = {
            "base_hash": self.base_hash,
            "layer_spec_hash": self.layer_spec_hash,
            "layer_payload_hash": self.layer_payload_hash,
            "layered_hash": self.layered_hash,
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        return dict(self._canonical_payload())

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class LayeredReceipt:
    base_hash: str
    layer_spec_hash: str
    layer_payload_hash: str
    layered_hash: str
    receipt_hash: str

    def _canonical_payload(self) -> dict:
        payload = {
            "base_hash": self.base_hash,
            "layer_spec_hash": self.layer_spec_hash,
            "layer_payload_hash": self.layer_payload_hash,
            "layered_hash": self.layered_hash,
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["receipt_hash"] = self.receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


def _canonical_layer_payload(layer_payload: dict[str, Any]) -> Mapping[str, Any]:
    """Canonicalize + freeze payload with a single ordering rule."""
    _ensure_json_safe(layer_payload)
    return _freeze_mapping(layer_payload)


def build_layered_receipt(
    base_state_ref: BaseStateReference,
    layer_spec: LayerSpec,
    layer_payload: dict[str, Any],
) -> LayeredReceipt:
    payload = _canonical_layer_payload(dict(layer_payload))
    layer_payload_hash = sha256_hex(dict(payload))
    layer_spec_hash = layer_spec.stable_hash()
    layered_hash = sha256_hex(
        {
            "base_hash": base_state_ref.base_hash,
            "layer_spec_hash": layer_spec_hash,
            "layer_payload_hash": layer_payload_hash,
        }
    )
    receipt = LayeredReceipt(
        base_hash=base_state_ref.base_hash,
        layer_spec_hash=layer_spec_hash,
        layer_payload_hash=layer_payload_hash,
        layered_hash=layered_hash,
        receipt_hash="",
    )
    receipt_hash = receipt.stable_hash()
    return LayeredReceipt(
        base_hash=receipt.base_hash,
        layer_spec_hash=receipt.layer_spec_hash,
        layer_payload_hash=receipt.layer_payload_hash,
        layered_hash=receipt.layered_hash,
        receipt_hash=receipt_hash,
    )


def validate_layered_receipt(
    receipt: LayeredReceipt,
    base_state_ref: BaseStateReference,
    layer_spec: LayerSpec,
    layer_payload: dict[str, Any],
) -> None:
    if receipt.base_hash != base_state_ref.base_hash:
        raise ValueError("INVALID_INPUT")
    if receipt.layer_spec_hash != layer_spec.stable_hash():
        raise ValueError("INVALID_INPUT")

    payload = _canonical_layer_payload(dict(layer_payload))
    if receipt.layer_payload_hash != sha256_hex(dict(payload)):
        raise ValueError("INVALID_INPUT")

    expected_layered_hash = sha256_hex(
        {
            "base_hash": receipt.base_hash,
            "layer_spec_hash": receipt.layer_spec_hash,
            "layer_payload_hash": receipt.layer_payload_hash,
        }
    )
    if expected_layered_hash != receipt.layered_hash:
        raise ValueError("INVALID_INPUT")

    recomputed_receipt_hash = LayeredReceipt(
        base_hash=receipt.base_hash,
        layer_spec_hash=receipt.layer_spec_hash,
        layer_payload_hash=receipt.layer_payload_hash,
        layered_hash=receipt.layered_hash,
        receipt_hash="",
    ).stable_hash()
    if recomputed_receipt_hash != receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")


if hasattr(LayeredState, "apply"):
    raise RuntimeError("INVALID_STATE")
if hasattr(LayeredState, "execute"):
    raise RuntimeError("INVALID_STATE")
