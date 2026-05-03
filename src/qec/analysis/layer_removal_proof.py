from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.layer_spec_contract import _deep_freeze, _ensure_json_safe
from qec.analysis.layered_state_receipt import LayeredReceipt


def _canonical_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    ordered: dict[str, Any] = {}
    for key in sorted(mapping, key=_canonical_key):
        ordered[key] = _deep_freeze(mapping[key])
    return MappingProxyType(ordered)


@dataclass(frozen=True)
class ReturnPathProof:
    base_hash: str
    layered_hash: str
    layer_spec_hash: str
    return_path_hash: str

    def __post_init__(self) -> None:
        expected = sha256_hex(
            {
                "base_hash": self.base_hash,
                "layered_hash": self.layered_hash,
                "layer_spec_hash": self.layer_spec_hash,
            }
        )
        if expected != self.return_path_hash:
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        payload = {
            "base_hash": self.base_hash,
            "layered_hash": self.layered_hash,
            "layer_spec_hash": self.layer_spec_hash,
            "return_path_hash": self.return_path_hash,
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
class BoundaryIntegrityReceipt:
    base_hash: str
    layered_hash: str
    layer_spec_hash: str
    layer_payload_hash: str
    boundary_integrity_hash: str

    def __post_init__(self) -> None:
        expected = sha256_hex(
            {
                "base_hash": self.base_hash,
                "layer_payload_hash": self.layer_payload_hash,
                "layer_spec_hash": self.layer_spec_hash,
                "layered_hash": self.layered_hash,
            }
        )
        if expected != self.boundary_integrity_hash:
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        payload = {
            "base_hash": self.base_hash,
            "layered_hash": self.layered_hash,
            "layer_spec_hash": self.layer_spec_hash,
            "layer_payload_hash": self.layer_payload_hash,
            "boundary_integrity_hash": self.boundary_integrity_hash,
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
class LayerRemovalReceipt:
    base_hash: str
    layered_hash: str
    layer_spec_hash: str
    layer_payload_hash: str
    return_path_proof: ReturnPathProof
    boundary_integrity_receipt: BoundaryIntegrityReceipt
    removal_metadata: Mapping[str, Any]
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "removal_metadata", _freeze_mapping(dict(self.removal_metadata)))
        _ensure_json_safe(self._canonical_payload())

    def _canonical_payload(self) -> dict:
        payload = {
            "base_hash": self.base_hash,
            "layered_hash": self.layered_hash,
            "layer_spec_hash": self.layer_spec_hash,
            "layer_payload_hash": self.layer_payload_hash,
            "return_path_proof": self.return_path_proof.to_dict(),
            "boundary_integrity_receipt": self.boundary_integrity_receipt.to_dict(),
            "removal_metadata": dict(self.removal_metadata),
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


def build_layer_removal_receipt(layered_receipt: LayeredReceipt) -> LayerRemovalReceipt:
    return_path_hash = sha256_hex(
        {
            "base_hash": layered_receipt.base_hash,
            "layered_hash": layered_receipt.layered_hash,
            "layer_spec_hash": layered_receipt.layer_spec_hash,
        }
    )
    return_path = ReturnPathProof(
        base_hash=layered_receipt.base_hash,
        layered_hash=layered_receipt.layered_hash,
        layer_spec_hash=layered_receipt.layer_spec_hash,
        return_path_hash=return_path_hash,
    )

    boundary_integrity_hash = sha256_hex(
        {
            "base_hash": layered_receipt.base_hash,
            "layer_payload_hash": layered_receipt.layer_payload_hash,
            "layer_spec_hash": layered_receipt.layer_spec_hash,
            "layered_hash": layered_receipt.layered_hash,
        }
    )
    boundary = BoundaryIntegrityReceipt(
        base_hash=layered_receipt.base_hash,
        layered_hash=layered_receipt.layered_hash,
        layer_spec_hash=layered_receipt.layer_spec_hash,
        layer_payload_hash=layered_receipt.layer_payload_hash,
        boundary_integrity_hash=boundary_integrity_hash,
    )

    receipt = LayerRemovalReceipt(
        base_hash=layered_receipt.base_hash,
        layered_hash=layered_receipt.layered_hash,
        layer_spec_hash=layered_receipt.layer_spec_hash,
        layer_payload_hash=layered_receipt.layer_payload_hash,
        return_path_proof=return_path,
        boundary_integrity_receipt=boundary,
        removal_metadata={"model": "identity_only", "reversible": True},
        receipt_hash="",
    )
    return LayerRemovalReceipt(
        base_hash=receipt.base_hash,
        layered_hash=receipt.layered_hash,
        layer_spec_hash=receipt.layer_spec_hash,
        layer_payload_hash=receipt.layer_payload_hash,
        return_path_proof=receipt.return_path_proof,
        boundary_integrity_receipt=receipt.boundary_integrity_receipt,
        removal_metadata=receipt.removal_metadata,
        receipt_hash=receipt.stable_hash(),
    )


def validate_layer_removal_receipt(
    removal_receipt: LayerRemovalReceipt,
    layered_receipt: LayeredReceipt,
) -> None:
    if layered_receipt.stable_hash() != layered_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")

    if removal_receipt.base_hash != layered_receipt.base_hash:
        raise ValueError("INVALID_INPUT")
    if removal_receipt.layered_hash != layered_receipt.layered_hash:
        raise ValueError("INVALID_INPUT")
    if removal_receipt.layer_spec_hash != layered_receipt.layer_spec_hash:
        raise ValueError("INVALID_INPUT")
    if removal_receipt.layer_payload_hash != layered_receipt.layer_payload_hash:
        raise ValueError("INVALID_INPUT")

    expected_return_path_hash = sha256_hex(
        {
            "base_hash": removal_receipt.base_hash,
            "layered_hash": removal_receipt.layered_hash,
            "layer_spec_hash": removal_receipt.layer_spec_hash,
        }
    )
    if removal_receipt.return_path_proof.return_path_hash != expected_return_path_hash:
        raise ValueError("INVALID_INPUT")

    expected_boundary_hash = sha256_hex(
        {
            "base_hash": removal_receipt.base_hash,
            "layer_payload_hash": removal_receipt.layer_payload_hash,
            "layer_spec_hash": removal_receipt.layer_spec_hash,
            "layered_hash": removal_receipt.layered_hash,
        }
    )
    if removal_receipt.boundary_integrity_receipt.boundary_integrity_hash != expected_boundary_hash:
        raise ValueError("INVALID_INPUT")

    recomputed_receipt_hash = LayerRemovalReceipt(
        base_hash=removal_receipt.base_hash,
        layered_hash=removal_receipt.layered_hash,
        layer_spec_hash=removal_receipt.layer_spec_hash,
        layer_payload_hash=removal_receipt.layer_payload_hash,
        return_path_proof=removal_receipt.return_path_proof,
        boundary_integrity_receipt=removal_receipt.boundary_integrity_receipt,
        removal_metadata=removal_receipt.removal_metadata,
        receipt_hash="",
    ).stable_hash()
    if recomputed_receipt_hash != removal_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")


if hasattr(LayerRemovalReceipt, "apply"):
    raise RuntimeError("INVALID_STATE")
if hasattr(LayerRemovalReceipt, "execute"):
    raise RuntimeError("INVALID_STATE")
