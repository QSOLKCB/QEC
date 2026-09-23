<!-- SPDX-License-Identifier: MPL-2.0 -->
# QEC v172.1 — Marker/Common-Control Contract

## Scope and authority

The marker is a pure, deterministic common-control computation over one
published v172.0 matrix snapshot. It seals a complete request and programme,
looks up the exact requested horizontal/vertical endpoints, and returns either
a one-coordinate closure **plan** or an explicit rejection. It then releases
control. A plan is not an actuated or committed connection.

The supported policy is `exact-coordinate-idle-only.v1`. There is no fallback,
operator override, external lookup, fault injection, implicit reservation,
shared marker state, wall-clock scheduling or decoder invocation. The marker
cannot modify the matrix, payload, or caller-declared decoder-output identity.
All policy and authority fields are fixed and replay-validated. The default
marker name `marker-0` is an identifier, not a shared lock or hardware identity.

Multi-stage path search belongs to v172.2; reservation, contention and release
of fabric resources belong to v172.3; end-to-end continuity proof belongs to
v172.4. Releasing this computation's marker does not release any fabric resource.

## Inputs and bounds

`CrossbarRequest` is frozen and contains:

- `request_id`;
- exact `horizontal_link_id` and `vertical_link_id`;
- immutable `payload` bytes (0–1,048,576 bytes);
- `decoder_output_sha256`, a full lowercase SHA-256 reference declared by the caller.

Request text fields and marker IDs must be nonempty UTF-8 text of at most 4,096
Unicode code points. The decoder reference is opaque: binding it does not
verify the referenced output's existence, provenance or mathematical validity.
The payload hash is independently computed from the supplied bytes. Payload
bytes are retained as lowercase hexadecimal in the sealed input register.

The matrix retains v172.0 bounds: 4,096 links per axis and 65,536 intersections.
Validation reconstructs the complete manifest. Endpoint lookup is bounded by
the two axis lengths, while selection evaluates zero or one coordinates.
`coordinate_evaluations` counts selection decisions, not validation work.

Changing any request field, payload byte, declared decoder reference or matrix
state changes the sealed programme identity. Even off-route link-state changes
invalidate reuse of an old programme. A programme must be compiled before it
can be executed, and execution recomputes the exact expected programme.

## Deterministic decision order

1. Replay-validate the matrix and seal the complete request.
2. Verify programme identity, matrix/request bindings, policy and authority.
3. Observe both declared endpoints and record both link states.
4. Reject an unknown horizontal endpoint first, then an unknown vertical endpoint.
5. For known endpoints, evaluate their one canonical intersection.
6. Reject a non-idle horizontal link first, then a non-idle vertical link.
7. Otherwise return that exact coordinate as the sole closure-plan entry.
8. Record marker release for both selection and rejection.

Rejection reasons are `unknown_horizontal_link`, `unknown_vertical_link`, or
`horizontal_` / `vertical_` followed by `busy`, `quarantined` or `unavailable`.
Successful selection uses outcome `plan_selected` and reason
`exact_coordinate_idle`. A valid rejected receipt has outcome `rejected`, an
empty closure plan, and can pass replay validation; validation success is not
successful route selection. Malformed inputs or authority/identity mismatches
raise `ValueError("INVALID_INPUT")` before a receipt is returned.

## Canonical artifacts

All new schemas use contract version `172.1`, sorted-key compact UTF-8 JSON,
and full SHA-256 identities excluding each artifact's own `sha256` field.
The existing matrix schemas and contract version `172.0` are unchanged.

| Artifact | Schema | Contents |
|---|---|---|
| `crossbar_marker_input_register.json` | `qec.crossbar-marker-input-register.v1` | Sealed request, exact endpoints, payload bytes/hash/length and declared decoder identity |
| `crossbar_marker_program.json` | `qec.crossbar-marker-program.v1` | Marker ID, matrix/register hashes, exact policy, evaluation bound and claim boundary |
| `crossbar_common_control_receipt.json` | `qec.crossbar-common-control-receipt.v1` | Full matrix/register/programme, event chain, outcome, reason, plan and release |
| `crossbar_common_control_validation.json` | `qec.crossbar-common-control-validation.v1` | Recomputed receipt identity, replay checks and external-binding check coverage |

The event-chain root binds the matrix, register and programme hashes. Each
event binds its logical sequence index, kind, details and preceding hash.
Normal selection has six events; unknown-endpoint rejection has five.
The final event is always `marker_released`. No timing is inferred from indices.
The validation artifact exposes the receipt's hash as
`crossbar_common_control_receipt_hash`.

## Replay and trust boundary

`validate_common_control_receipt()` reconstructs the request from its retained
bytes, validates the matrix, recompiles the programme, reruns the decision, and
compares the entire canonical receipt. Recomputing an outer hash cannot make a
fabricated decision, partial event chain, altered authority, different
coordinate or numeric stand-in for a boolean pass validation.

Self-contained replay proves internal consistency. A coherently replaced set
of inputs describes a different valid run; it is not evidence of the intended
run. Supply `expected_matrix_sha256`, `expected_request_sha256` and
`expected_program_sha256` from trusted records to pin that run. The validation
artifact explicitly records which external bindings were checked. Its flags
are derived output; consumers should run validation rather than trust a
standalone validation JSON file. Receipt dictionaries are detached JSON
representations; callers may edit them, but edits must pass complete replay.

## Python API

```python
from qec.routing.crossbar import (
    CrossbarRequest, compile_marker_program, demo_matrix,
    execute_marker_program, validate_common_control_receipt,
)

manifest = demo_matrix("example", horizontal_count=2, vertical_count=2).as_dict()
request = CrossbarRequest("request-1", "H000", "V001", b"opaque-correction", "a" * 64)
program = compile_marker_program(manifest, request, marker_id="marker-a")
receipt = execute_marker_program(manifest, request, program)
validation = validate_common_control_receipt(
    receipt,
    expected_matrix_sha256=manifest["sha256"],
    expected_request_sha256=request.as_dict()["sha256"],
    expected_program_sha256=program["sha256"],
)
```

The example decoder reference is a synthetic caller-declared fixture.

## CLI

Use an existing matrix manifest and payload file:

```bash
qec-crossbar marker \
  --manifest artifacts/crossbar/crossbar_matrix_manifest.json \
  --request-id request-1 \
  --horizontal-link-id H000 --vertical-link-id V001 \
  --payload-file correction.bin \
  --decoder-output-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --output-dir artifacts/crossbar-marker

qec-crossbar marker-validate \
  --receipt artifacts/crossbar-marker/crossbar_common_control_receipt.json
```

`marker-validate` also accepts `--expected-matrix-sha256`,
`--expected-request-sha256` and `--expected-program-sha256`. Marker JSON readers
reject duplicate keys. `marker` exits 0 for a selected plan, 2 for a valid
rejection (artifacts are still emitted), and 1 for malformed input or I/O errors.
`marker-validate` exits 0 for either replay-valid outcome and 1 for failure.
The receipt embeds all replay inputs; the companion files are convenience views.

## Validation and limitations

Tests cover exact coordinate identity, every non-idle link state, precedence,
unknown endpoints, no fallback, immutable inputs, sealed programme bindings,
recomputed-hash tampering, event removal/reordering, canonical types, optional
trusted bindings, payload limits, subprocess/hash-seed determinism and CLI
round-trips. Crossbar CI also runs the full QEC regression suite.

The contract establishes deterministic classical software computation under
its declared inputs. It does not establish a physical connection, end-to-end
continuity, authenticated provenance, decoder correctness, hardware behaviour,
carrier-grade reliability, cross-era equivalence or quantum advantage.
