from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from qec.analysis.canonical_hashing import canonicalize_json, sha256_hex


def _ensure_json_safe(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("INVALID_INPUT")
    elif isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str) or k == "":
                raise ValueError("INVALID_INPUT")
            _ensure_json_safe(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _ensure_json_safe(v)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError("INVALID_INPUT")


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    ordered = {}
    for key in sorted(mapping, key=lambda x: (type(x).__name__, str(x))):
        value = mapping[key]
        if isinstance(value, dict):
            ordered[key] = _freeze_mapping(value)
        elif isinstance(value, list):
            ordered[key] = tuple(value)
        else:
            ordered[key] = value
    return MappingProxyType(ordered)


@dataclass(frozen=True)
class LayerInvariantSet:
    invariants: Tuple[str, ...]

    def __post_init__(self) -> None:
        if any((not isinstance(v, str) or v == "") for v in self.invariants):
            raise ValueError("INVALID_INPUT")
        if len(set(self.invariants)) != len(self.invariants):
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        payload = {
            "invariants": list(
                sorted(self.invariants, key=lambda x: (type(x).__name__, str(x)))
            )
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        return dict(self._canonical_payload())

    def to_canonical_json(self) -> str:
        return canonicalize_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_canonical_json())

    def __deepcopy__(self, memo: dict[int, Any]) -> "LayerInvariantSet":
        memo[id(self)] = self
        return self


@dataclass(frozen=True)
class LayerCompatibilityConstraint:
    constraint_id: str
    constraint_type: str
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.constraint_id or not self.constraint_type:
            raise ValueError("INVALID_INPUT")
        allowed = {"router_path", "readout_shell", "mask64", "hilbert_shift_label"}
        if self.constraint_type not in allowed:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "params", _freeze_mapping(self.params))

    def _canonical_payload(self) -> dict:
        payload = {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type,
            "params": dict(self.params),
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        return dict(self._canonical_payload())

    def to_canonical_json(self) -> str:
        return canonicalize_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_canonical_json())

    def __deepcopy__(self, memo: dict[int, Any]) -> "LayerCompatibilityConstraint":
        memo[id(self)] = self
        return self


@dataclass(frozen=True)
class LayerSpec:
    layer_id: str
    layer_version: str
    invariant_set: LayerInvariantSet
    activation_rules: Mapping[str, Any]
    removal_rules: Mapping[str, Any]
    compatibility_constraints: Tuple[LayerCompatibilityConstraint, ...]

    def __post_init__(self) -> None:
        if not self.layer_id or not self.layer_version:
            raise ValueError("INVALID_INPUT")
        if len({c.constraint_id for c in self.compatibility_constraints}) != len(
            self.compatibility_constraints
        ):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "activation_rules", _freeze_mapping(self.activation_rules))
        object.__setattr__(self, "removal_rules", _freeze_mapping(self.removal_rules))

    def _canonical_payload(self) -> dict:
        ordered_constraints = sorted(
            self.compatibility_constraints, key=lambda x: (type(x).__name__, str(x))
        )
        payload = {
            "layer_id": self.layer_id,
            "layer_version": self.layer_version,
            "invariant_set": self.invariant_set.to_dict(),
            "activation_rules": dict(self.activation_rules),
            "removal_rules": dict(self.removal_rules),
            "compatibility_constraints": [c.to_dict() for c in ordered_constraints],
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        return dict(self._canonical_payload())

    def to_canonical_json(self) -> str:
        return canonicalize_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_canonical_json())

    def __deepcopy__(self, memo: dict[int, Any]) -> "LayerSpec":
        memo[id(self)] = self
        return self


@dataclass(frozen=True)
class LayerSpecReceipt:
    layer_spec_hash: str
    receipt_hash: str

    def _canonical_payload(self) -> dict:
        payload = {"layer_spec_hash": self.layer_spec_hash}
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["receipt_hash"] = self.receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonicalize_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_canonical_json())


def build_layer_spec_receipt(layer_spec: LayerSpec) -> LayerSpecReceipt:
    layer_spec_hash = layer_spec.stable_hash()
    receipt = LayerSpecReceipt(layer_spec_hash=layer_spec_hash, receipt_hash="")
    receipt_hash = receipt.stable_hash()
    return LayerSpecReceipt(layer_spec_hash=layer_spec_hash, receipt_hash=receipt_hash)


def validate_layer_spec_receipt(layer_spec: LayerSpec, receipt: LayerSpecReceipt) -> None:
    if receipt.layer_spec_hash != layer_spec.stable_hash():
        raise ValueError("INVALID_INPUT")
    recomputed = LayerSpecReceipt(layer_spec_hash=receipt.layer_spec_hash, receipt_hash="").stable_hash()
    if recomputed != receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")


assert not hasattr(LayerSpec, "apply")
assert not hasattr(LayerSpec, "execute")
