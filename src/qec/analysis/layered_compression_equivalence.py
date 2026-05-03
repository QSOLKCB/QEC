from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.layer_removal_proof import LayerRemovalReceipt, validate_layer_removal_receipt
from qec.analysis.layer_spec_contract import _deep_freeze, _ensure_json_safe
from qec.analysis.layered_state_receipt import LayeredReceipt


_REQUIRED_IDENTITY_FIELDS: tuple[str, ...] = (
    "base_hash",
    "layer_spec_hash",
    "layer_payload_hash",
    "layered_hash",
    "removal_receipt_hash",
    "return_path_hash",
    "boundary_integrity_hash",
    "compression_contract_hash",
)


def _canonical_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    ordered: dict[str, Any] = {}
    for key in sorted(mapping, key=_canonical_key):
        ordered[key] = _deep_freeze(mapping[key])
    return MappingProxyType(ordered)


@dataclass(frozen=True)
class LayeredCompressionContract:
    compression_id: str
    compression_version: str
    compression_rules: Mapping[str, Any]
    preserved_fields: tuple[str, ...]
    equivalence_rules: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.compression_id or not self.compression_version:
            raise ValueError("INVALID_INPUT")
        if len(set(self.preserved_fields)) != len(self.preserved_fields):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "compression_rules", _freeze_mapping(dict(self.compression_rules)))
        object.__setattr__(self, "equivalence_rules", _freeze_mapping(dict(self.equivalence_rules)))
        object.__setattr__(self, "preserved_fields", tuple(sorted(self.preserved_fields, key=_canonical_key)))
        _ensure_json_safe(self._canonical_payload())

    def _canonical_payload(self) -> dict:
        payload = {
            "compression_id": self.compression_id,
            "compression_version": self.compression_version,
            "compression_rules": dict(self.compression_rules),
            "preserved_fields": list(self.preserved_fields),
            "equivalence_rules": dict(self.equivalence_rules),
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
class CompressedLayeredProof:
    compression_contract_hash: str
    source_layered_receipt_hash: str
    source_removal_receipt_hash: str
    preserved_identity_hashes: Mapping[str, str]
    compressed_proof_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "preserved_identity_hashes", _freeze_mapping(dict(self.preserved_identity_hashes)))
        for required in _REQUIRED_IDENTITY_FIELDS:
            if required not in self.preserved_identity_hashes:
                raise ValueError("INVALID_INPUT")
        if self.compressed_proof_hash and self.stable_hash() != self.compressed_proof_hash:
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        payload = {
            "compression_contract_hash": self.compression_contract_hash,
            "source_layered_receipt_hash": self.source_layered_receipt_hash,
            "source_removal_receipt_hash": self.source_removal_receipt_hash,
            "preserved_identity_hashes": dict(self.preserved_identity_hashes),
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["compressed_proof_hash"] = self.compressed_proof_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class LayerEquivalenceReceipt:
    compressed_proof_hash: str
    layered_receipt_hash: str
    removal_receipt_hash: str
    base_hash: str
    layered_hash: str
    layer_spec_hash: str
    layer_payload_hash: str
    equivalence_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        if self.equivalence_hash and self._equivalence_hash() != self.equivalence_hash:
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.stable_hash() != self.receipt_hash:
            raise ValueError("INVALID_INPUT")

    def _equivalence_hash(self) -> str:
        return sha256_hex(
            {
                "compressed_proof_hash": self.compressed_proof_hash,
                "layered_receipt_hash": self.layered_receipt_hash,
                "removal_receipt_hash": self.removal_receipt_hash,
                "base_hash": self.base_hash,
                "layered_hash": self.layered_hash,
                "layer_spec_hash": self.layer_spec_hash,
                "layer_payload_hash": self.layer_payload_hash,
            }
        )

    def _canonical_payload(self) -> dict:
        payload = {
            "compressed_proof_hash": self.compressed_proof_hash,
            "layered_receipt_hash": self.layered_receipt_hash,
            "removal_receipt_hash": self.removal_receipt_hash,
            "base_hash": self.base_hash,
            "layered_hash": self.layered_hash,
            "layer_spec_hash": self.layer_spec_hash,
            "layer_payload_hash": self.layer_payload_hash,
            "equivalence_hash": self.equivalence_hash,
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


def build_compressed_layered_proof(
    layered_receipt: LayeredReceipt,
    removal_receipt: LayerRemovalReceipt,
    compression_contract: LayeredCompressionContract,
) -> CompressedLayeredProof:
    if layered_receipt.stable_hash() != layered_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    if removal_receipt.stable_hash() != removal_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    validate_layer_removal_receipt(removal_receipt, layered_receipt)

    if not all(field in compression_contract.preserved_fields for field in _REQUIRED_IDENTITY_FIELDS):
        raise ValueError("INVALID_INPUT")

    contract_hash = compression_contract.stable_hash()
    identities = _freeze_mapping(
        {
            "base_hash": layered_receipt.base_hash,
            "layer_spec_hash": layered_receipt.layer_spec_hash,
            "layer_payload_hash": layered_receipt.layer_payload_hash,
            "layered_hash": layered_receipt.layered_hash,
            "removal_receipt_hash": removal_receipt.receipt_hash,
            "return_path_hash": removal_receipt.return_path_proof.return_path_hash,
            "boundary_integrity_hash": removal_receipt.boundary_integrity_receipt.boundary_integrity_hash,
            "compression_contract_hash": contract_hash,
        }
    )
    proof = CompressedLayeredProof(
        compression_contract_hash=contract_hash,
        source_layered_receipt_hash=layered_receipt.receipt_hash,
        source_removal_receipt_hash=removal_receipt.receipt_hash,
        preserved_identity_hashes=identities,
        compressed_proof_hash="",
    )
    return CompressedLayeredProof(
        compression_contract_hash=proof.compression_contract_hash,
        source_layered_receipt_hash=proof.source_layered_receipt_hash,
        source_removal_receipt_hash=proof.source_removal_receipt_hash,
        preserved_identity_hashes=proof.preserved_identity_hashes,
        compressed_proof_hash=proof.stable_hash(),
    )


def build_layer_equivalence_receipt(
    compressed_proof: CompressedLayeredProof,
    layered_receipt: LayeredReceipt,
    removal_receipt: LayerRemovalReceipt,
) -> LayerEquivalenceReceipt:
    equivalence_hash = sha256_hex(
        {
            "compressed_proof_hash": compressed_proof.compressed_proof_hash,
            "layered_receipt_hash": layered_receipt.receipt_hash,
            "removal_receipt_hash": removal_receipt.receipt_hash,
            "base_hash": layered_receipt.base_hash,
            "layered_hash": layered_receipt.layered_hash,
            "layer_spec_hash": layered_receipt.layer_spec_hash,
            "layer_payload_hash": layered_receipt.layer_payload_hash,
        }
    )
    receipt = LayerEquivalenceReceipt(
        compressed_proof_hash=compressed_proof.compressed_proof_hash,
        layered_receipt_hash=layered_receipt.receipt_hash,
        removal_receipt_hash=removal_receipt.receipt_hash,
        base_hash=layered_receipt.base_hash,
        layered_hash=layered_receipt.layered_hash,
        layer_spec_hash=layered_receipt.layer_spec_hash,
        layer_payload_hash=layered_receipt.layer_payload_hash,
        equivalence_hash=equivalence_hash,
        receipt_hash="",
    )
    finalized = LayerEquivalenceReceipt(
        compressed_proof_hash=receipt.compressed_proof_hash,
        layered_receipt_hash=receipt.layered_receipt_hash,
        removal_receipt_hash=receipt.removal_receipt_hash,
        base_hash=receipt.base_hash,
        layered_hash=receipt.layered_hash,
        layer_spec_hash=receipt.layer_spec_hash,
        layer_payload_hash=receipt.layer_payload_hash,
        equivalence_hash=receipt.equivalence_hash,
        receipt_hash=receipt.stable_hash(),
    )
    validate_layer_equivalence_receipt(finalized, compressed_proof, layered_receipt, removal_receipt)
    return finalized


def validate_layer_equivalence_receipt(
    equivalence_receipt: LayerEquivalenceReceipt,
    compressed_proof: CompressedLayeredProof,
    layered_receipt: LayeredReceipt,
    removal_receipt: LayerRemovalReceipt,
) -> None:
    if layered_receipt.stable_hash() != layered_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    if removal_receipt.stable_hash() != removal_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    validate_layer_removal_receipt(removal_receipt, layered_receipt)
    if compressed_proof.stable_hash() != compressed_proof.compressed_proof_hash:
        raise ValueError("INVALID_INPUT")

    preserved = compressed_proof.preserved_identity_hashes
    expected = {
        "base_hash": layered_receipt.base_hash,
        "layer_spec_hash": layered_receipt.layer_spec_hash,
        "layer_payload_hash": layered_receipt.layer_payload_hash,
        "layered_hash": layered_receipt.layered_hash,
        "removal_receipt_hash": removal_receipt.receipt_hash,
        "return_path_hash": removal_receipt.return_path_proof.return_path_hash,
        "boundary_integrity_hash": removal_receipt.boundary_integrity_receipt.boundary_integrity_hash,
        "compression_contract_hash": compressed_proof.compression_contract_hash,
    }
    for key, value in expected.items():
        if preserved.get(key) != value:
            raise ValueError("INVALID_INPUT")

    if equivalence_receipt.compressed_proof_hash != compressed_proof.compressed_proof_hash:
        raise ValueError("INVALID_INPUT")
    if equivalence_receipt.layered_receipt_hash != layered_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    if equivalence_receipt.removal_receipt_hash != removal_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    if equivalence_receipt.base_hash != layered_receipt.base_hash:
        raise ValueError("INVALID_INPUT")
    if equivalence_receipt.layered_hash != layered_receipt.layered_hash:
        raise ValueError("INVALID_INPUT")
    if equivalence_receipt.layer_spec_hash != layered_receipt.layer_spec_hash:
        raise ValueError("INVALID_INPUT")
    if equivalence_receipt.layer_payload_hash != layered_receipt.layer_payload_hash:
        raise ValueError("INVALID_INPUT")

    recomputed_receipt_hash = LayerEquivalenceReceipt(
        compressed_proof_hash=equivalence_receipt.compressed_proof_hash,
        layered_receipt_hash=equivalence_receipt.layered_receipt_hash,
        removal_receipt_hash=equivalence_receipt.removal_receipt_hash,
        base_hash=equivalence_receipt.base_hash,
        layered_hash=equivalence_receipt.layered_hash,
        layer_spec_hash=equivalence_receipt.layer_spec_hash,
        layer_payload_hash=equivalence_receipt.layer_payload_hash,
        equivalence_hash=equivalence_receipt.equivalence_hash,
        receipt_hash="",
    ).stable_hash()
    if recomputed_receipt_hash != equivalence_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")


for _forbidden_name in ("apply", "execute", "compress", "decompress", "run"):
    if hasattr(CompressedLayeredProof, _forbidden_name):
        raise RuntimeError("INVALID_STATE")
    if hasattr(LayerEquivalenceReceipt, _forbidden_name):
        raise RuntimeError("INVALID_STATE")
