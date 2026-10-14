# SPDX-License-Identifier: MPL-2.0
"""Bounded acoustic-artifact bytes over unchanged native Digital/Packet models.

The wrapper binds a caller-labelled stage, a separately pinned source identity,
and the actual canonical JSON bytes. It never promotes a stage to verified
acoustic or decoder evidence. Each ordered chunk is a complete native transport
transaction. Validation reconstructs its expected input and replays it in full.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

from qec.routing.digital import (
    DigitalRequest, DigitalSwitchInput, FrameFault, FrameSchema,
    execute_digital_switch, validate_digital_receipt,
)
from qec.routing.packet import (
    FaultPlan, PacketEndpoint, PacketFault, PacketInput, PacketRequest,
    PacketTopology, QueuePolicy, Route, execute_packet_fabric,
    validate_packet_receipt,
)
from qec.sonify.canonical import canonical_bytes, canonical_sha256

from ._common import digest, exact, sequence

CONTRACT_VERSION = "177.4"
CHUNK_BYTES = 1024
MAX_CHUNKS = 64
MAX_ARTIFACT_BYTES = CHUNK_BYTES * MAX_CHUNKS
ARTIFACT_ROLES = ("original", "observed", "filtered", "candidate", "rejected", "verified")
TRANSPORTS = ("digital", "packet")
FAULT_MODES = ("none", "corrupt", "drop")
CLAIM_BOUNDARY = MappingProxyType({
    "implemented_unreleased": True,
    "bounded_classical_software_transport_only": True,
    "role_is_caller_label": True,
    "source_identity_is_declared_provenance": True,
    "source_hash_authenticates_origin": False,
    "acoustic_semantics_verified": False,
    "decoder_output_verified": False,
    "role_promoted_by_transport": False,
    "unserialized_pcm_transported": False,
    "physical_network_or_quantum_evidence": False,
    "native_transport_contracts_modified": False,
    "logical_ticks_are_physical_latency": False,
})


def _seal(name: str, **fields) -> dict:
    unsigned = {"schema": "qec.acoustic-" + name + ".v1",
                "contract_version": CONTRACT_VERSION, **fields}
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _unhex(value: object) -> bytes:
    if type(value) is not str or len(value) > 2 * MAX_ARTIFACT_BYTES or len(value) % 2:
        raise ValueError("INVALID_INPUT")
    data = bytes.fromhex(value)
    if data.hex() != value:
        raise ValueError("INVALID_INPUT")
    return data


@dataclass(frozen=True)
class AcousticTransportArtifact:
    """Source-bound byte identity; ``role`` is explicitly a caller label.

    Payload is the *entire* canonical UTF-8 JSON object being transported,
    including PCM bytes when the selected artifact contains them. Acoustic
    validity and stage/source relationships require their own native validators.
    """

    role: str
    source_sha256: str
    payload: bytes

    def __post_init__(self):
        try:
            if type(self.role) is not str or self.role not in ARTIFACT_ROLES:
                raise ValueError("INVALID_INPUT")
            digest(self.source_sha256)
            if type(self.payload) is not bytes or not 1 <= len(self.payload) <= MAX_ARTIFACT_BYTES:
                raise ValueError("INVALID_INPUT")
            value = json.loads(self.payload.decode("utf-8"))
            if type(value) is not dict or not value or canonical_bytes(value) != self.payload:
                raise ValueError("INVALID_INPUT")
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("INVALID_INPUT") from exc

    @classmethod
    def from_artifact(cls, artifact: object, *, role: str, source_sha256: str):
        try:
            if type(artifact) is not dict:
                raise ValueError("INVALID_INPUT")
            return cls(role, source_sha256, canonical_bytes(artifact))
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("INVALID_INPUT") from exc

    def as_dict(self) -> dict:
        return _seal("transport-artifact", role=self.role, source_sha256=self.source_sha256,
                     payload_encoding="canonical_json_utf8", payload_hex=self.payload.hex(),
                     payload_bytes=len(self.payload), payload_sha256=hashlib.sha256(self.payload).hexdigest(),
                     role_authority="caller_label_only")

    @classmethod
    def from_dict(cls, value: object):
        try:
            if type(value) is not dict:
                raise ValueError("INVALID_INPUT")
            result = cls(value["role"], value["source_sha256"], _unhex(value["payload_hex"]))
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def _input(artifact: AcousticTransportArtifact, transport: str, fault_mode: str) -> dict:
    if type(artifact) is not AcousticTransportArtifact:
        raise ValueError("INVALID_INPUT")
    if type(transport) is not str or transport not in TRANSPORTS:
        raise ValueError("INVALID_INPUT")
    if type(fault_mode) is not str or fault_mode not in FAULT_MODES:
        raise ValueError("INVALID_INPUT")
    return _seal("transport-input", artifact=artifact.as_dict(), transport=transport,
                 chunk_bytes=CHUNK_BYTES, max_chunks=MAX_CHUNKS, fault_mode=fault_mode,
                 fault_scope="first_chunk_only", transaction_scope="one_native_transaction_per_chunk")


def _chunks(artifact: AcousticTransportArtifact) -> tuple[dict, ...]:
    snapshot = artifact.as_dict()
    count = (len(artifact.payload) + CHUNK_BYTES - 1) // CHUNK_BYTES
    return tuple(_seal("transport-chunk", artifact_sha256=snapshot["sha256"],
                       source_sha256=artifact.source_sha256, role=artifact.role,
                       sequence=index, chunk_count=count, offset=index * CHUNK_BYTES,
                       artifact_payload_bytes=len(artifact.payload),
                       artifact_payload_sha256=snapshot["payload_sha256"],
                       payload_hex=artifact.payload[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES].hex(),
                       payload_sha256=hashlib.sha256(
                           artifact.payload[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES]).hexdigest())
                 for index in range(count))


def _native_input(chunk: dict, transport: str, fault_mode: str):
    payload = bytes.fromhex(chunk["payload_hex"])
    request_id = f"acoustic-{chunk['sequence']:02d}-of-{chunk['chunk_count']:02d}"
    fault = fault_mode if chunk["sequence"] == 0 else "none"
    # Native provenance fields bind the complete chunk manifest. They remain
    # declared hashes, not claims that this is a verified decoder output.
    if transport == "digital":
        faults = (() if fault == "none" else
                  (FrameFault("bit_flip", 0, slot=0),) if fault == "corrupt" else
                  (FrameFault("frame_loss", 0),))
        return DigitalSwitchInput(
            DigitalRequest(request_id, "source", "destination", payload, chunk["sha256"]),
            FrameSchema(slot_count=2, max_frames=CHUNK_BYTES), output_slot=1, faults=faults,
        )
    topology = PacketTopology(
        ("ingress", "egress"), (PacketEndpoint("1", "egress"),),
        (Route("ingress", "", ("egress",), route_id="acoustic-route"),),
    )
    faults = () if fault == "none" else (PacketFault(0, "ingress", "egress", fault, duration=1),)
    return PacketInput(
        (PacketRequest(request_id, "ingress", "1", payload, chunk["sha256"]),),
        topology, QueuePolicy(capacity=4, max_hops=2), FaultPlan(faults), fragment_bytes=256,
    )


def _build_receipt(artifact: AcousticTransportArtifact, transport: str, fault_mode: str,
                   native_receipts: tuple | None = None) -> dict:
    source = _input(artifact, transport, fault_mode)
    manifests = _chunks(artifact)
    if native_receipts is not None and len(native_receipts) != len(manifests):
        raise ValueError("INVALID_INPUT")
    rows, received = [], []
    for index, manifest in enumerate(manifests):
        native = _native_input(manifest, transport, fault_mode)
        snapshot = native.as_dict()
        if transport == "digital":
            receipt = execute_digital_switch(native) if native_receipts is None else native_receipts[index]
            if native_receipts is not None:
                validate_digital_receipt(receipt, expected_input_sha256=snapshot["sha256"],
                                         expected_request_sha256=snapshot["request"]["sha256"],
                                         expected_frame_schema_sha256=snapshot["frame_schema"]["sha256"])
            accepted = receipt["accepted"]
            observed_hex = receipt["reassembled_payload_hex"]
        else:
            receipt = execute_packet_fabric(native) if native_receipts is None else native_receipts[index]
            if native_receipts is not None:
                validate_packet_receipt(receipt, expected_input_sha256=snapshot["sha256"],
                                        expected_topology_sha256=snapshot["topology"]["sha256"],
                                        expected_queue_policy_sha256=snapshot["queue_policy"]["sha256"],
                                        expected_fault_plan_sha256=snapshot["fault_plan"]["sha256"])
            result = receipt["results"][0]
            accepted = result["outcome"] == "accepted"
            observed_hex = result["reassembled_payload_hex"]
        observed = None if observed_hex is None else bytes.fromhex(observed_hex)
        verified = accepted and observed == bytes.fromhex(manifest["payload_hex"])
        if verified:
            received.append(observed)
        rows.append(_seal("transport-chunk-result", manifest=manifest, native_receipt=receipt,
                          accepted=verified, observed_payload_hex=observed_hex,
                          observed_payload_sha256=None if observed is None else hashlib.sha256(observed).hexdigest()))
    joined = b"".join(received)
    complete = len(received) == len(manifests)
    accepted = complete and joined == artifact.payload and hashlib.sha256(joined).hexdigest() == source["artifact"]["payload_sha256"]
    return _seal("transport-receipt", input=source, chunks=rows, chunk_count=len(manifests),
                 accepted_chunk_count=len(received), accepted=accepted,
                 outcome="delivered" if accepted else "rejected", role=artifact.role,
                 artifact_sha256=source["artifact"]["sha256"], source_sha256=artifact.source_sha256,
                 reassembled_payload_hex=joined.hex() if accepted else None,
                 reassembled_payload_sha256=hashlib.sha256(joined).hexdigest() if accepted else None,
                 payload_identity_verified=accepted, partial_delivery_permitted=False,
                 claim_boundary=dict(CLAIM_BOUNDARY))


def execute_acoustic_transport(artifact: AcousticTransportArtifact, *, transport: str = "digital",
                               fault_mode: str = "none") -> dict:
    """Execute a fixed bounded native model; faults target only the first chunk.

    ``corrupt`` flips the first Digital symbol bit or first Packet fragment byte;
    ``drop`` removes the first Digital frame or first Packet fragment. A rejected
    native transaction suppresses the complete adapter output, including prefixes.
    """
    return _build_receipt(artifact, transport, fault_mode)


def validate_acoustic_transport_receipt(receipt: object, *, expected_artifact_sha256: str,
                                        expected_source_sha256: str,
                                        expected_input_sha256: str | None = None) -> dict:
    """Require external artifact/source pins and replay *every* native receipt.

    The optional input pin additionally fixes transport and fault mode. Valid
    rejection evidence has ``all_passed=True`` and ``accepted=False``. No result
    here authenticates source provenance or verifies acoustic stage semantics.
    """
    try:
        digest(expected_artifact_sha256)
        digest(expected_source_sha256)
        if type(receipt) is not dict or type(receipt["input"]) is not dict:
            raise ValueError("INVALID_INPUT")
        source = receipt["input"]
        artifact = AcousticTransportArtifact.from_dict(source["artifact"])
        if artifact.as_dict()["sha256"] != expected_artifact_sha256 or artifact.source_sha256 != expected_source_sha256:
            raise ValueError("INVALID_INPUT")
        expected_input = _input(artifact, source["transport"], source["fault_mode"])
        exact(source, expected_input)
        if expected_input_sha256 is not None:
            digest(expected_input_sha256)
            if expected_input["sha256"] != expected_input_sha256:
                raise ValueError("INVALID_INPUT")
        rows = sequence(receipt["chunks"], MAX_CHUNKS)
        native_receipts = tuple(row["native_receipt"] for row in rows)
        replay = _build_receipt(artifact, source["transport"], source["fault_mode"], native_receipts)
        exact(receipt, replay)
        return _seal("transport-validation", receipt_sha256=replay["sha256"],
                     artifact_sha256=expected_artifact_sha256, source_sha256=expected_source_sha256,
                     native_replay="FULL", complete_native_replay_verified=True,
                     expected_native_inputs_verified=True, chunk_sequence_verified=True,
                     accepted=replay["accepted"], payload_identity_verified=replay["payload_identity_verified"],
                     external_bindings_verified={"artifact": True, "source": True,
                                                 "transport_input": expected_input_sha256 is not None},
                     acoustic_semantics_verified=False, all_passed=True)
    except (KeyError, TypeError, ValueError, AttributeError, UnicodeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc
