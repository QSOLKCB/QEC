# SPDX-License-Identifier: MPL-2.0
"""Actual-byte transport, externally pinned replay, and fail-closed mutations."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qec.experiments.acoustic.channel import AcousticSource, ChannelConfig, PCM
from qec.experiments.acoustic.transport import (
    ARTIFACT_ROLES, CHUNK_BYTES, MAX_ARTIFACT_BYTES, MAX_CHUNKS,
    AcousticTransportArtifact, execute_acoustic_transport,
    validate_acoustic_transport_receipt,
)
from qec.routing.digital import DigitalRequest, DigitalSwitchInput, execute_digital_switch
from qec.routing.packet import PacketInput, PacketRequest, execute_packet_fabric
from qec.sonify.canonical import canonical_bytes, canonical_sha256


def _rehash(value):
    value["sha256"] = canonical_sha256({key: item for key, item in value.items() if key != "sha256"})
    return value


def _artifact(size=2 * CHUNK_BYTES + 19, *, role="observed", source="a" * 64):
    overhead = len(canonical_bytes({"data": ""}))
    return AcousticTransportArtifact.from_artifact(
        {"data": "x" * (size - overhead)}, role=role, source_sha256=source,
    )


def _validate(receipt, artifact, **kwargs):
    return validate_acoustic_transport_receipt(
        receipt, expected_artifact_sha256=artifact.as_dict()["sha256"],
        expected_source_sha256=artifact.source_sha256, **kwargs,
    )


@pytest.fixture(scope="module", params=("digital", "packet"))
def transported(request):
    artifact = _artifact()
    return artifact, execute_acoustic_transport(artifact, transport=request.param)


def test_actual_multichunk_bytes_replay_and_determinism(transported):
    artifact, receipt = transported
    validation = _validate(receipt, artifact, expected_input_sha256=receipt["input"]["sha256"])
    assert validation["native_replay"] == "FULL"
    assert validation["expected_native_inputs_verified"]
    assert validation["external_bindings_verified"] == {"artifact": True, "source": True, "transport_input": True}
    assert validation["all_passed"] and validation["accepted"]
    assert not validation["acoustic_semantics_verified"]
    assert bytes.fromhex(receipt["reassembled_payload_hex"]) == artifact.payload
    assert receipt["reassembled_payload_sha256"] == hashlib.sha256(artifact.payload).hexdigest()
    assert len(receipt["chunks"]) == 3
    chunks = [row["manifest"] for row in receipt["chunks"]]
    assert [chunk["sequence"] for chunk in chunks] == [0, 1, 2]
    assert [chunk["offset"] for chunk in chunks] == [0, 1024, 2048]
    assert all(chunk["chunk_count"] == 3 for chunk in chunks)
    assert b"".join(bytes.fromhex(chunk["payload_hex"]) for chunk in chunks) == artifact.payload
    for row in receipt["chunks"]:
        native = row["native_receipt"]["input"]
        request = native["request"] if receipt["input"]["transport"] == "digital" else native["requests"][0]
        assert request["payload_hex"] == row["manifest"]["payload_hex"]
        assert request["payload_sha256"] == hashlib.sha256(bytes.fromhex(request["payload_hex"])).hexdigest()
    assert execute_acoustic_transport(artifact, transport=receipt["input"]["transport"]) == receipt
    assert _validate(json.loads(json.dumps(receipt)), artifact) == _validate(receipt, artifact)


def test_all_roles_remain_distinct_even_for_identical_pcm_bytes():
    pcm = PCM((0, 1, -1, 32767, -32768)).as_dict()
    artifacts = [AcousticTransportArtifact.from_artifact(pcm, role=role, source_sha256="a" * 64)
                 for role in ARTIFACT_ROLES]
    assert len({artifact.as_dict()["sha256"] for artifact in artifacts}) == 6
    assert len({artifact.as_dict()["payload_sha256"] for artifact in artifacts}) == 1
    for artifact in artifacts:
        receipt = execute_acoustic_transport(artifact, transport="packet")
        assert _validate(receipt, artifact)["accepted"]
        assert receipt["role"] == artifact.role
        assert receipt["claim_boundary"]["role_is_caller_label"]
        assert not receipt["claim_boundary"]["decoder_output_verified"]
        assert not receipt["claim_boundary"]["role_promoted_by_transport"]
        assert json.loads(bytes.fromhex(receipt["reassembled_payload_hex"])) == pcm


def test_original_source_transports_actual_encoded_pcm():
    source = AcousticSource(ChannelConfig(cycles=1, repetitions=1), (0, 1, 2), "b" * 64).as_dict()
    artifact = AcousticTransportArtifact.from_artifact(source, role="original", source_sha256=source["sha256"])
    receipt = execute_acoustic_transport(artifact, transport="packet")
    assert _validate(receipt, artifact)["payload_identity_verified"]
    recovered = json.loads(bytes.fromhex(receipt["reassembled_payload_hex"]))
    assert recovered == source
    assert recovered["encoded_sound"]["pcm_hex"] == source["encoded_sound"]["pcm_hex"]
    assert len(bytes.fromhex(recovered["encoded_sound"]["pcm_hex"])) == 288


def test_actual_six_stage_artifacts_keep_separate_source_bound_identities():
    from qec.experiments.acoustic.bridge import MeasuredSyndromeReference, run_pcm_bridge, validate_bridge_receipt
    from qec.experiments.acoustic.noise import (
        CancellationConfig, NoiseFault, NoisePlan, apply_noise,
        cancel_reference, independent_noise_reference,
    )
    from qec.experiments.acoustic.receiver import receive_pcm

    config = ChannelConfig(cycles=1, repetitions=1)
    source = AcousticSource(config, (0, 0, 0, 0), "b" * 64).as_dict()
    original = PCM.from_dict(source["encoded_sound"])
    plan = NoisePlan((NoiseFault.make("broadband", amplitude=8),), seed=37)
    noise = apply_noise(original, plan, source_sha256=source["sha256"])
    observed = PCM.from_dict(noise["observed"])
    noise_reference = independent_noise_reference(len(original.samples), plan)
    cancellation = cancel_reference(
        observed, noise_reference,
        CancellationConfig("c" * 64, "privileged_synthetic_reference", "noise_cancellation", True, True),
        observed_source_sha256=source["sha256"],
    )
    filtered = PCM.from_dict(cancellation["filtered"])
    assert observed != original and filtered == original
    candidate = receive_pcm(filtered, config, 4)
    reference = MeasuredSyndromeReference(3, (0, 0, 0, 0), "independent-fixture", "transport-test")
    bridge_inputs = dict(pcm=filtered, config=config, observation=candidate, reference=reference,
                         expected_reference_sha256=reference.as_dict()["sha256"])
    verified = run_pcm_bridge(**bridge_inputs)
    assert validate_bridge_receipt(verified, **bridge_inputs) == verified
    rejected_pcm = PCM((0,) * len(original.samples), config.sample_rate)
    rejected_candidate = receive_pcm(rejected_pcm, config, 4)
    rejected = run_pcm_bridge(rejected_pcm, config, rejected_candidate, reference,
                              expected_reference_sha256=reference.as_dict()["sha256"])
    assert verified["verified_recovery"] and not rejected["verified_recovery"]
    stages = dict(original=source, observed=noise, filtered=cancellation,
                  candidate=candidate, rejected=rejected, verified=verified)
    identities = []
    for role, stage in stages.items():
        artifact = AcousticTransportArtifact.from_artifact(stage, role=role, source_sha256=source["sha256"])
        receipt = execute_acoustic_transport(artifact, transport="packet")
        assert _validate(receipt, artifact)["accepted"]
        assert receipt["role"] == role and receipt["source_sha256"] == source["sha256"]
        assert json.loads(bytes.fromhex(receipt["reassembled_payload_hex"])) == stage
        identities.append(receipt["artifact_sha256"])
    assert len(set(identities)) == 6


@pytest.mark.parametrize("transport", ("digital", "packet"))
@pytest.mark.parametrize("fault_mode", ("corrupt", "drop"))
def test_native_faults_replay_valid_rejection_without_partial_output(transport, fault_mode):
    artifact = _artifact(size=CHUNK_BYTES + 1, role="candidate")
    receipt = execute_acoustic_transport(artifact, transport=transport, fault_mode=fault_mode)
    validation = _validate(receipt, artifact)
    assert validation["all_passed"] and not validation["accepted"]
    assert not validation["payload_identity_verified"]
    assert receipt["outcome"] == "rejected"
    assert receipt["role"] == "candidate"
    assert receipt["accepted_chunk_count"] == 1
    assert receipt["chunks"][1]["accepted"]
    assert receipt["reassembled_payload_hex"] is None
    assert receipt["reassembled_payload_sha256"] is None
    if transport == "packet":
        assert receipt["chunks"][0]["observed_payload_hex"] is not None
        assert receipt["chunks"][0]["observed_payload_sha256"] != receipt["chunks"][0]["manifest"]["payload_sha256"]


@pytest.mark.parametrize("mutation", ("reverse", "omit", "duplicate", "manifest_sequence", "manifest_offset", "manifest_digest"))
def test_rehashed_chunk_mutations_fail_closed(transported, mutation):
    artifact, original = transported
    receipt = deepcopy(original)
    if mutation == "reverse":
        receipt["chunks"].reverse()
    elif mutation == "omit":
        receipt["chunks"].pop()
        receipt["chunk_count"] -= 1
    elif mutation == "duplicate":
        receipt["chunks"][1] = deepcopy(receipt["chunks"][0])
    else:
        manifest = receipt["chunks"][0]["manifest"]
        key = {"manifest_sequence": "sequence", "manifest_offset": "offset", "manifest_digest": "artifact_payload_sha256"}[mutation]
        manifest[key] = "f" * 64 if mutation == "manifest_digest" else 1
        _rehash(manifest)
        _rehash(receipt["chunks"][0])
    _rehash(receipt)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        _validate(receipt, artifact)


@pytest.mark.parametrize("mutation", ("accepted", "outer_bytes", "native_bytes", "claim", "empty"))
def test_rehashed_receipt_mutations_fail_closed(transported, mutation):
    artifact, original = transported
    receipt = deepcopy(original)
    if mutation == "accepted":
        receipt["accepted"] = False
    elif mutation == "outer_bytes":
        receipt["reassembled_payload_hex"] = b"substitute".hex()
        receipt["reassembled_payload_sha256"] = hashlib.sha256(b"substitute").hexdigest()
    elif mutation == "native_bytes":
        native = receipt["chunks"][0]["native_receipt"]
        result = native if receipt["input"]["transport"] == "digital" else native["results"][0]
        result["reassembled_payload_hex"] = b"substitute".hex()
        result["reassembled_payload_sha256"] = hashlib.sha256(b"substitute").hexdigest()
        _rehash(result)
        _rehash(native)
        _rehash(receipt["chunks"][0])
    elif mutation == "claim":
        receipt["claim_boundary"]["decoder_output_verified"] = True
    else:
        receipt["chunks"] = []
        receipt["chunk_count"] = 0
    _rehash(receipt)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        _validate(receipt, artifact)


def test_self_consistent_substituted_native_input_is_not_expected_input(transported):
    artifact, original = transported
    receipt = deepcopy(original)
    native = receipt["chunks"][0]["native_receipt"]["input"]
    if receipt["input"]["transport"] == "digital":
        source = DigitalSwitchInput.from_dict(native)
        request = source.request
        replacement = DigitalRequest(request.request_id, request.source, request.destination,
                                     b"z" * len(request.payload), request.decoder_output_sha256)
        other = execute_digital_switch(replace(source, request=replacement))
    else:
        source = PacketInput.from_dict(native)
        request = source.requests[0]
        replacement = PacketRequest(request.request_id, request.source, request.destination,
                                    b"z" * len(request.payload), request.provenance_sha256)
        other = execute_packet_fabric(replace(source, requests=(replacement,)))
    receipt["chunks"][0]["native_receipt"] = other
    _rehash(receipt["chunks"][0])
    _rehash(receipt)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        _validate(receipt, artifact)


def test_external_pins_reject_self_consistent_artifact_source_or_role_substitution(transported):
    artifact, receipt = transported
    for replacement in (
        AcousticTransportArtifact(artifact.role, "b" * 64, artifact.payload),
        AcousticTransportArtifact("verified", artifact.source_sha256, artifact.payload),
        _artifact(len(artifact.payload) + 1),
    ):
        other = execute_acoustic_transport(replacement, transport="packet")
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            _validate(other, artifact)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_acoustic_transport_receipt(receipt, expected_artifact_sha256=artifact.as_dict()["sha256"],
                                            expected_source_sha256="b" * 64)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        _validate(receipt, artifact, expected_input_sha256="f" * 64)


@pytest.mark.parametrize("pins", ({}, {"expected_artifact_sha256": None, "expected_source_sha256": "a" * 64},
                                  {"expected_artifact_sha256": "a" * 64, "expected_source_sha256": None}))
def test_expected_artifact_and_source_pins_are_required(pins):
    with pytest.raises((TypeError, ValueError)):
        validate_acoustic_transport_receipt({}, **pins)


@pytest.mark.parametrize("payload", (b"", b"{}", b"[]", b'{"x":1.0}', b'{"x":1,"x":1}',
                                    b'{"x": 1}', b'{"x":NaN}', b'{"x":"\xff"}', b'{"x":1}\n'))
def test_noncanonical_or_empty_artifact_bytes_rejected(payload):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        AcousticTransportArtifact("original", "a" * 64, payload)


@pytest.mark.parametrize("size", (CHUNK_BYTES - 1, CHUNK_BYTES, CHUNK_BYTES + 1, MAX_ARTIFACT_BYTES))
def test_exact_chunk_and_artifact_bounds(size):
    artifact = _artifact(size)
    assert len(artifact.payload) == size
    receipt = execute_acoustic_transport(artifact, transport="packet")
    assert receipt["chunk_count"] == (size + CHUNK_BYTES - 1) // CHUNK_BYTES
    assert receipt["chunk_count"] <= MAX_CHUNKS
    assert _validate(receipt, artifact)["accepted"]
    assert bytes.fromhex(receipt["reassembled_payload_hex"]) == artifact.payload


def test_immutable_snapshot_and_strict_serialization():
    original = {"data": [0, 1, 2]}
    artifact = AcousticTransportArtifact.from_artifact(original, role="filtered", source_sha256="a" * 64)
    original["data"].append(3)
    assert artifact.payload == canonical_bytes({"data": [0, 1, 2]})
    with pytest.raises(FrozenInstanceError):
        artifact.role = "verified"
    assert AcousticTransportArtifact.from_dict(artifact.as_dict()) == artifact
    for mutate in (lambda d: d.update(extra=True), lambda d: d.update(payload_bytes=True),
                   lambda d: d.update(payload_hex=d["payload_hex"].upper()),
                   lambda d: d.update(payload_sha256="f" * 64)):
        value = artifact.as_dict()
        mutate(value)
        _rehash(value)
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            AcousticTransportArtifact.from_dict(value)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        _artifact(MAX_ARTIFACT_BYTES + 1)
    for role in ("", "decoded", True):
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            AcousticTransportArtifact(role, "a" * 64, artifact.payload)
    for transport, fault_mode in ((True, "none"), ("digital", False), ("unknown", "none"), ("packet", "repair")):
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            execute_acoustic_transport(artifact, transport=transport, fault_mode=fault_mode)


def test_hash_seed_independence():
    script = """
from qec.experiments.acoustic.transport import AcousticTransportArtifact, execute_acoustic_transport
a = AcousticTransportArtifact.from_artifact({'pcm_hex': '00000100ffff'}, role='observed', source_sha256='a'*64)
print(execute_acoustic_transport(a, transport='packet')['sha256'])
"""
    src = str(Path(__file__).resolve().parents[2] / "src")
    hashes = [subprocess.check_output([sys.executable, "-c", script],
                                     env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": src}, text=True)
              for seed in ("1", "97")]
    assert hashes[0] == hashes[1]
