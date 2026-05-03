from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.layer_spec_contract import _deep_freeze, _ensure_json_safe
from qec.analysis.layered_compression_equivalence import (
    CompressedLayeredProof,
    LayerEquivalenceReceipt,
)

_REQUIRED_PRESERVED_FIELDS: tuple[str, ...] = (
    "base_hash",
    "layer_spec_hash",
    "layer_payload_hash",
    "layered_hash",
    "removal_receipt_hash",
    "return_path_hash",
    "boundary_integrity_hash",
    "compression_contract_hash",
    "compressed_proof_hash",
    "equivalence_hash",
    "equivalence_receipt_hash",
)


def _canonical_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    ordered: dict[str, Any] = {}
    for key in sorted(mapping, key=_canonical_key):
        ordered[key] = _deep_freeze(mapping[key])
    return MappingProxyType(ordered)


@dataclass(frozen=True)
class FractalInvariantContract:
    fractal_id: str
    fractal_version: str
    scale_rules: Mapping[str, Any]
    pattern_rules: Mapping[str, Any]
    preserved_fields: tuple[str, ...]
    equivalence_rules: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.fractal_id or not self.fractal_version:
            raise ValueError("INVALID_INPUT")
        if len(set(self.preserved_fields)) != len(self.preserved_fields):
            raise ValueError("INVALID_INPUT")
        if set(self.preserved_fields) != set(_REQUIRED_PRESERVED_FIELDS):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "scale_rules", _freeze_mapping(dict(self.scale_rules)))
        object.__setattr__(self, "pattern_rules", _freeze_mapping(dict(self.pattern_rules)))
        object.__setattr__(self, "equivalence_rules", _freeze_mapping(dict(self.equivalence_rules)))
        object.__setattr__(self, "preserved_fields", tuple(sorted(self.preserved_fields, key=_canonical_key)))
        _ensure_json_safe(self._canonical_payload())

    def _canonical_payload(self) -> dict:
        payload = {
            "fractal_id": self.fractal_id,
            "fractal_version": self.fractal_version,
            "scale_rules": dict(self.scale_rules),
            "pattern_rules": dict(self.pattern_rules),
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
class InvariantPatternNode:
    pattern_id: str
    scale_index: int
    identity_fields: Mapping[str, str]
    pattern_hash: str

    def __post_init__(self) -> None:
        if not self.pattern_id or not isinstance(self.scale_index, int) or self.scale_index < 0:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "identity_fields", _freeze_mapping(dict(self.identity_fields)))
        if set(self.identity_fields.keys()) != set(_REQUIRED_PRESERVED_FIELDS):
            raise ValueError("INVALID_INPUT")
        if self.pattern_hash and self.stable_hash() != self.pattern_hash:
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        payload = {
            "pattern_id": self.pattern_id,
            "scale_index": self.scale_index,
            "identity_fields": dict(self.identity_fields),
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["pattern_hash"] = self.pattern_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class FractalInvariantCompressionReceipt:
    fractal_contract_hash: str
    source_compressed_proof_hash: str
    source_equivalence_receipt_hash: str
    pattern_nodes: tuple[InvariantPatternNode, ...]
    fractal_pattern_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        canonical_nodes = tuple(sorted(self.pattern_nodes, key=lambda n: (n.scale_index, n.pattern_id, n.pattern_hash)))
        seen: set[tuple[int, str]] = set()
        for node in canonical_nodes:
            key = (node.scale_index, node.pattern_id)
            if key in seen:
                raise ValueError("INVALID_INPUT")
            seen.add(key)
        object.__setattr__(self, "pattern_nodes", canonical_nodes)
        if self._pattern_hash() != self.fractal_pattern_hash:
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.stable_hash() != self.receipt_hash:
            raise ValueError("INVALID_INPUT")

    def _pattern_hash(self) -> str:
        return sha256_hex([n._canonical_payload() for n in self.pattern_nodes])

    def _canonical_payload(self) -> dict:
        payload = {
            "fractal_contract_hash": self.fractal_contract_hash,
            "source_compressed_proof_hash": self.source_compressed_proof_hash,
            "source_equivalence_receipt_hash": self.source_equivalence_receipt_hash,
            "pattern_nodes": [n._canonical_payload() for n in self.pattern_nodes],
            "fractal_pattern_hash": self.fractal_pattern_hash,
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["pattern_nodes"] = [n.to_dict() for n in self.pattern_nodes]
        payload["receipt_hash"] = self.receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class FractalInvariantEquivalenceReceipt:
    fractal_receipt_hash: str
    compressed_proof_hash: str
    layer_equivalence_receipt_hash: str
    base_hash: str
    layered_hash: str
    layer_spec_hash: str
    layer_payload_hash: str
    removal_receipt_hash: str
    return_path_hash: str
    boundary_integrity_hash: str
    compression_contract_hash: str
    fractal_pattern_hash: str
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
                "fractal_receipt_hash": self.fractal_receipt_hash,
                "compressed_proof_hash": self.compressed_proof_hash,
                "layer_equivalence_receipt_hash": self.layer_equivalence_receipt_hash,
                "base_hash": self.base_hash,
                "layered_hash": self.layered_hash,
                "layer_spec_hash": self.layer_spec_hash,
                "layer_payload_hash": self.layer_payload_hash,
                "removal_receipt_hash": self.removal_receipt_hash,
                "return_path_hash": self.return_path_hash,
                "boundary_integrity_hash": self.boundary_integrity_hash,
                "compression_contract_hash": self.compression_contract_hash,
                "fractal_pattern_hash": self.fractal_pattern_hash,
            }
        )

    def _canonical_payload(self) -> dict:
        payload = {
            "fractal_receipt_hash": self.fractal_receipt_hash,
            "compressed_proof_hash": self.compressed_proof_hash,
            "layer_equivalence_receipt_hash": self.layer_equivalence_receipt_hash,
            "base_hash": self.base_hash,
            "layered_hash": self.layered_hash,
            "layer_spec_hash": self.layer_spec_hash,
            "layer_payload_hash": self.layer_payload_hash,
            "removal_receipt_hash": self.removal_receipt_hash,
            "return_path_hash": self.return_path_hash,
            "boundary_integrity_hash": self.boundary_integrity_hash,
            "compression_contract_hash": self.compression_contract_hash,
            "fractal_pattern_hash": self.fractal_pattern_hash,
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




def _validated_identity_fields(
    compressed_proof: CompressedLayeredProof,
    equivalence_receipt: LayerEquivalenceReceipt,
) -> Mapping[str, str]:
    # Proof linkage: equivalence receipt must reference the same compressed proof
    if equivalence_receipt.compressed_proof_hash != compressed_proof.compressed_proof_hash:
        raise ValueError("INVALID_INPUT")
    # Source receipt linkage: proof's source hashes must match equivalence receipt
    if compressed_proof.source_layered_receipt_hash != equivalence_receipt.layered_receipt_hash:
        raise ValueError("INVALID_INPUT")
    if compressed_proof.source_removal_receipt_hash != equivalence_receipt.removal_receipt_hash:
        raise ValueError("INVALID_INPUT")
    # Equivalence hash must be non-empty (rejects receipts built without a valid equivalence hash)
    if not equivalence_receipt.equivalence_hash:
        raise ValueError("INVALID_INPUT")
    preserved = compressed_proof.preserved_identity_hashes
    if preserved["base_hash"] != equivalence_receipt.base_hash:
        raise ValueError("INVALID_INPUT")
    if preserved["layered_hash"] != equivalence_receipt.layered_hash:
        raise ValueError("INVALID_INPUT")
    if preserved["layer_spec_hash"] != equivalence_receipt.layer_spec_hash:
        raise ValueError("INVALID_INPUT")
    if preserved["layer_payload_hash"] != equivalence_receipt.layer_payload_hash:
        raise ValueError("INVALID_INPUT")
    if preserved["removal_receipt_hash"] != equivalence_receipt.removal_receipt_hash:
        raise ValueError("INVALID_INPUT")
    if preserved["compression_contract_hash"] != compressed_proof.compression_contract_hash:
        raise ValueError("INVALID_INPUT")

    validated_identity = {
        "base_hash": equivalence_receipt.base_hash,
        "layer_spec_hash": equivalence_receipt.layer_spec_hash,
        "layer_payload_hash": equivalence_receipt.layer_payload_hash,
        "layered_hash": equivalence_receipt.layered_hash,
        "removal_receipt_hash": equivalence_receipt.removal_receipt_hash,
        "return_path_hash": preserved["return_path_hash"],
        "boundary_integrity_hash": preserved["boundary_integrity_hash"],
        "compression_contract_hash": compressed_proof.compression_contract_hash,
        "compressed_proof_hash": compressed_proof.compressed_proof_hash,
        "equivalence_hash": equivalence_receipt.equivalence_hash,
        "equivalence_receipt_hash": equivalence_receipt.receipt_hash,
    }
    return _freeze_mapping(validated_identity)


def build_fractal_invariant_compression_receipt(compressed_proof: CompressedLayeredProof, equivalence_receipt: LayerEquivalenceReceipt, fractal_contract: FractalInvariantContract) -> FractalInvariantCompressionReceipt:
    if compressed_proof.stable_hash() != compressed_proof.compressed_proof_hash:
        raise ValueError("INVALID_INPUT")
    if equivalence_receipt.stable_hash() != equivalence_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    if equivalence_receipt.compressed_proof_hash != compressed_proof.compressed_proof_hash:
        raise ValueError("INVALID_INPUT")
    identity_fields = _validated_identity_fields(compressed_proof, equivalence_receipt)
    node_base = InvariantPatternNode(pattern_id="layered-equivalence", scale_index=0, identity_fields=identity_fields, pattern_hash="")
    node = InvariantPatternNode(pattern_id=node_base.pattern_id, scale_index=node_base.scale_index, identity_fields=node_base.identity_fields, pattern_hash=node_base.stable_hash())
    empty = FractalInvariantCompressionReceipt(
        fractal_contract_hash=fractal_contract.stable_hash(),
        source_compressed_proof_hash=compressed_proof.compressed_proof_hash,
        source_equivalence_receipt_hash=equivalence_receipt.receipt_hash,
        pattern_nodes=(node,),
        fractal_pattern_hash=sha256_hex([node._canonical_payload()]),
        receipt_hash="",
    )
    return FractalInvariantCompressionReceipt(
        fractal_contract_hash=empty.fractal_contract_hash,
        source_compressed_proof_hash=empty.source_compressed_proof_hash,
        source_equivalence_receipt_hash=empty.source_equivalence_receipt_hash,
        pattern_nodes=empty.pattern_nodes,
        fractal_pattern_hash=empty.fractal_pattern_hash,
        receipt_hash=empty.stable_hash(),
    )


def build_fractal_invariant_equivalence_receipt(fractal_receipt: FractalInvariantCompressionReceipt, compressed_proof: CompressedLayeredProof, equivalence_receipt: LayerEquivalenceReceipt) -> FractalInvariantEquivalenceReceipt:
    if fractal_receipt.stable_hash() != fractal_receipt.receipt_hash or compressed_proof.stable_hash() != compressed_proof.compressed_proof_hash or equivalence_receipt.stable_hash() != equivalence_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    identity = _validated_identity_fields(compressed_proof, equivalence_receipt)
    temp = FractalInvariantEquivalenceReceipt(
        fractal_receipt_hash=fractal_receipt.receipt_hash,
        compressed_proof_hash=compressed_proof.compressed_proof_hash,
        layer_equivalence_receipt_hash=equivalence_receipt.receipt_hash,
        base_hash=identity["base_hash"],
        layered_hash=identity["layered_hash"],
        layer_spec_hash=identity["layer_spec_hash"],
        layer_payload_hash=identity["layer_payload_hash"],
        removal_receipt_hash=identity["removal_receipt_hash"],
        return_path_hash=identity["return_path_hash"],
        boundary_integrity_hash=identity["boundary_integrity_hash"],
        compression_contract_hash=identity["compression_contract_hash"],
        fractal_pattern_hash=fractal_receipt.fractal_pattern_hash,
        equivalence_hash="",
        receipt_hash="",
    )
    temp = replace(temp, equivalence_hash=temp._equivalence_hash())
    final = replace(temp, receipt_hash=temp.stable_hash())
    validate_fractal_invariant_equivalence_receipt(final, fractal_receipt, compressed_proof, equivalence_receipt)
    return final


def validate_fractal_invariant_equivalence_receipt(fractal_equivalence_receipt: FractalInvariantEquivalenceReceipt, fractal_receipt: FractalInvariantCompressionReceipt, compressed_proof: CompressedLayeredProof, equivalence_receipt: LayerEquivalenceReceipt) -> None:
    if compressed_proof.stable_hash() != compressed_proof.compressed_proof_hash or equivalence_receipt.stable_hash() != equivalence_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    if fractal_receipt.stable_hash() != fractal_receipt.receipt_hash or fractal_equivalence_receipt.stable_hash() != fractal_equivalence_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    if fractal_receipt.source_compressed_proof_hash != compressed_proof.compressed_proof_hash or fractal_receipt.source_equivalence_receipt_hash != equivalence_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    identity = _validated_identity_fields(compressed_proof, equivalence_receipt)
    if fractal_equivalence_receipt.base_hash != identity["base_hash"] or fractal_equivalence_receipt.layered_hash != identity["layered_hash"]:
        raise ValueError("INVALID_INPUT")
    if fractal_equivalence_receipt.layer_spec_hash != identity["layer_spec_hash"] or fractal_equivalence_receipt.layer_payload_hash != identity["layer_payload_hash"]:
        raise ValueError("INVALID_INPUT")
    if fractal_equivalence_receipt.removal_receipt_hash != identity["removal_receipt_hash"] or fractal_equivalence_receipt.return_path_hash != identity["return_path_hash"]:
        raise ValueError("INVALID_INPUT")
    if fractal_equivalence_receipt.boundary_integrity_hash != identity["boundary_integrity_hash"] or fractal_equivalence_receipt.compression_contract_hash != identity["compression_contract_hash"]:
        raise ValueError("INVALID_INPUT")
    if fractal_equivalence_receipt.compressed_proof_hash != compressed_proof.compressed_proof_hash:
        raise ValueError("INVALID_INPUT")
    if fractal_equivalence_receipt.layer_equivalence_receipt_hash != equivalence_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    if fractal_equivalence_receipt.fractal_receipt_hash != fractal_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    # Independently recompute expected fractal_pattern_hash from upstream inputs and verify
    # both the fractal_receipt and the fractal_equivalence_receipt carry the correct value
    node_base = InvariantPatternNode(
        pattern_id="layered-equivalence",
        scale_index=0,
        identity_fields=identity,
        pattern_hash="",
    )
    expected_fractal_pattern_hash = sha256_hex([node_base._canonical_payload()])
    if fractal_equivalence_receipt.fractal_pattern_hash != expected_fractal_pattern_hash:
        raise ValueError("INVALID_INPUT")
    if fractal_receipt.fractal_pattern_hash != expected_fractal_pattern_hash:
        raise ValueError("INVALID_INPUT")
    if fractal_equivalence_receipt.equivalence_hash != fractal_equivalence_receipt._equivalence_hash():
        raise ValueError("INVALID_INPUT")


for _forbidden_name in ("apply", "execute", "compress", "decompress", "render", "traverse", "run"):
    if hasattr(FractalInvariantCompressionReceipt, _forbidden_name):
        raise RuntimeError("INVALID_STATE")
    if hasattr(FractalInvariantEquivalenceReceipt, _forbidden_name):
        raise RuntimeError("INVALID_STATE")
