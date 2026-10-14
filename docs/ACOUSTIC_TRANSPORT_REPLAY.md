<!-- SPDX-License-Identifier: MPL-2.0 -->
# Acoustic artifact transport and replay

Status: **implemented, unreleased**, planned contract **177.4**. This additive
experiment uses the unchanged native Digital v174 and Packet v175 contracts.
It does not change the released package version, decoder, transport schemas,
or frozen fixtures. No merge or release tag is implied.

## What this boundary proves

`qec.experiments.acoustic.transport` transports the complete canonical UTF-8
JSON bytes of a selected artifact through a bounded native software model.
The artifact can contain actual PCM bytes, a source manifest, an observation,
a candidate, or a rejected/verified bridge receipt. An artifact that contains
only a waveform hash transports that hash and its other serialized fields;
it does **not** transport the absent waveform.

Successful validation proves that the exact selected bytes survived the
declared native model, with the required caller artifact/source bindings. It
does not establish acoustic correctness, decoder correctness, physical source
provenance, quantum protection, network fidelity, measured throughput, or
physical latency. All timing in native receipts is logical.

The `original`, `observed`, `filtered`, `candidate`, `rejected`, and `verified`
roles are explicit **caller labels**. They keep artifact identities separate,
even when two stages happen to serialize to identical bytes. Transport cannot
promote a candidate or rejection to verified syndrome evidence. A label of
`verified` alone has no authority: callers must retain and validate the native
acoustic observation and independently pinned bridge/reference evidence.

## Source, artifact, and actual-byte identities

`AcousticTransportArtifact(role, source_sha256, payload)` is frozen. Its payload
must be a nonempty canonical JSON object encoded as UTF-8, with sorted keys,
compact separators, and no floats. Duplicate keys, alternate whitespace,
nonfinite values, noncanonical escaping/hex, malformed UTF-8, and empty objects
are rejected. `from_artifact()` snapshots an ordinary JSON dictionary into
these exact bytes.

Three identities have different jobs:

- `source_sha256`: caller-declared source identity. Prefer the complete
  `AcousticSource.as_dict()["sha256"]`, which binds configuration and encoded
  samples, rather than its nested externally declared provenance digest.
- Artifact `sha256`: canonical wrapper identity binding source, role, actual
  bytes, encoding, byte length, and byte digest. Different roles or sources
  have different artifact identities even with identical payloads.
- `payload_sha256`: SHA-256 of actual canonical artifact bytes, including any
  included self-hash field. This differs from an inner artifact's own self-hash,
  which normally excludes its own `sha256` field.

All wrapper self-hashes exclude only their own `sha256`. Nested self-hashes and
identities remain covered. Source hashes bind declarations; they do not
authenticate origin or authorize decoder actions.

## Bounded chunk contract

The maximum selected artifact is **65,536 bytes**, partitioned into at most
**64 chunks of 1,024 bytes**. The last chunk may be shorter; chunks are never
empty. Oversized artifacts reject before native execution; they are not silently
truncated, reduced to hashes, or summarized. This transport cap is deliberately
smaller than some maximum-sized acoustic PCM/source artifacts.

Each sealed chunk manifest binds:

- Whole wrapper identity, source identity, and unchanged caller role
- Zero-based sequence, total chunk count, and exact byte offset
- Whole artifact byte length and actual-byte digest
- Exact contiguous chunk bytes and their independently recomputed digest

The native payload is the exact chunk bytes. Its request ID explicitly carries
the chunk ordinal/count; its native provenance field carries the entire sealed
chunk manifest identity. Digital calls that field `decoder_output_sha256`:
here it is a declared chunk-manifest identity, not a claim that these are
verified decoder output bytes. Packet uses `provenance_sha256`.

Each chunk uses one complete, independent native transaction:

- Digital: two slots, declared ingress 0 and egress 1, exact source/destination
  catalogues, frame limit explicitly raised from the native default to 1,024
- Packet: declared ingress-to-egress route, binary destination `1`, 256-byte
  fragments, queue capacity 4, and maximum two hops

This is bounded batch transport, not a persistent network connection. Packet's
larger 4,096-byte native limit is not used to change the common chunk contract.

`fault_mode="none"`, `"corrupt"`, or `"drop"` selects a declared first-chunk
fault. Corruption flips the first Digital symbol bit or first Packet fragment
byte. Drop removes the first Digital frame or first Packet fragment. Later
chunks still execute and remain available as evidence. Any rejected chunk
suppresses the **entire** adapter reassembled output; successful suffixes are
never reported as complete delivery. Packet's rejected native observations may
contain partial/corrupted bytes; these remain explicitly rejected evidence.

## Native API

```python
from qec.experiments.acoustic.channel import AcousticSource, ChannelConfig
from qec.experiments.acoustic.transport import (
    AcousticTransportArtifact,
    execute_acoustic_transport,
    validate_acoustic_transport_receipt,
)

source = AcousticSource(
    ChannelConfig(cycles=1, repetitions=1),
    measured_symbols=(0, 1, 2),
    source_sha256="a" * 64,  # Caller-declared synthetic provenance.
).as_dict()

artifact = AcousticTransportArtifact.from_artifact(
    source, role="original", source_sha256=source["sha256"],
)
expected_artifact = artifact.as_dict()["sha256"]
expected_source = source["sha256"]
receipt = execute_acoustic_transport(artifact, transport="packet")

validation = validate_acoustic_transport_receipt(
    receipt,
    expected_artifact_sha256=expected_artifact,
    expected_source_sha256=expected_source,
    expected_input_sha256=receipt["input"]["sha256"],
)
assert validation["native_replay"] == "FULL"
assert validation["accepted"]
assert bytes.fromhex(receipt["reassembled_payload_hex"]) == artifact.payload
```

The example captures input identities during trusted local construction.
Production callers must retain pins separately from an untrusted received
receipt; copying expected pins out of that receipt provides no origin binding.

Create separate wrappers for actual stage artifacts as follows, always keeping
the original source pin and the separate native acoustic evidence:

| Role | Bytes to select |
|---|---|
| `original` | Complete source manifest, including `encoded_sound.pcm_hex` |
| `observed` | Complete impaired `PCM.as_dict()` or impairment receipt containing it |
| `filtered` | Complete filtered `PCM.as_dict()` or filter receipt containing it |
| `candidate` | Actual harmonic-observation receipt, before independent verification |
| `rejected` | Actual failed observation or failed syndrome-bridge receipt |
| `verified` | Actual independently validated syndrome-bridge receipt |

Observation/bridge receipts can reference PCM by hash rather than include it.
Transport the corresponding PCM artifact separately when waveform bytes are
needed. Store one wrapper/receipt per selected role; do not overwrite the
original or use an observed/filtered hash as a replacement source identity.

## FULL replay and fail-closed validation

`validate_acoustic_transport_receipt` requires both
`expected_artifact_sha256` and `expected_source_sha256`. The optional
`expected_input_sha256` additionally fixes the transport choice and fault mode.
Without that optional pin, a different correctly declared transport/fault model
for the same pinned artifact remains a different valid execution.

The validator performs all of these checks:

1. Strictly reconstruct the source-bound byte wrapper and check both mandatory
   external pins. Canonicalize nothing on the caller's behalf.
2. Reconstruct the exact ordered chunk partition and expected native inputs
   from the selected bytes and declared fixed transport profile.
3. Invoke `validate_digital_receipt` or `validate_packet_receipt` on **every**
   native receipt, pinning expected input and relevant native component hashes.
   This invokes each unchanged native executor and compares its full evidence.
4. Read actual native reassembled bytes. Check exact per-chunk bytes, sequence,
   count, offsets, complete length, and recompute the whole artifact byte hash.
5. Compare the complete reconstructed adapter receipt, including every nested
   native receipt, manifest, outcome, observation, and claim boundary.

A self-consistent native execution using substituted bytes fails against the
expected native input. A rehashed false event/result is rejected by native
replay. Reordered, omitted, duplicated, shortened, or relabelled chunks fail
full adapter comparison. A fully rehashed replacement artifact or source fails
the external pins. Nonempty original artifact bytes and a nonempty complete
chunk sequence are required, preventing empty/prefix success.

Valid rejection evidence returns `all_passed=True`, `accepted=False`, and
`payload_identity_verified=False`. Validation success must not be confused
with successful transport or acoustic recovery.

## Verification

Focused regression command:

```sh
python -m pytest -q tests/acoustic/test_transport.py
```

Tests exercise exact carried PCM, all six distinct role identities, deterministic
JSON/hash-seed replay, chunk and artifact bounds, immutable snapshots, native
corruption/drop rejection, complete source-pinned native replay, rehashed false
outputs/claims, alternate self-consistent native inputs, and substituted
artifact/source/role identities. The full repository suite remains mandatory
before publication; focused tests alone are not release evidence.
