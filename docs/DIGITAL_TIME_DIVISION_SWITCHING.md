<!-- SPDX-License-Identifier: MPL-2.0 -->
# Deterministic Digital Time-Division Switching

**Status: implemented, unreleased.** This additive v174.0–v174.5 work does not
change the released v173.0 package/source version, decoder, or prior switching
schemas. A version in a new artifact identifies its planned contract, not a
published package or release tag. Repository-wide tests and separately authorized
release evidence remain mandatory before a tag can describe this work as released.

## Contract/version map

| Planned version | Native implementation | Principal artifacts |
|---|---|---|
| 174.0 | `FrameSchema`, `DigitalFrame`, `DigitalRequest`, `DigitalSwitchInput`, `execute_digital_switch` | `digital_frame_schema.json`, `digital_switch_input.json`, `digital_frame_switch_receipt.json` |
| 174.1 | `SlotState`, `SlotCommand`, `AllocationInput`, `execute_slot_allocation` | `timeslot_allocation_receipt.json` |
| 174.2 | `FrameObservation`, `synchronize_frames` | `frame_sync_receipt.json` |
| 174.3 | `interchange_frames` | `timeslot_interchange_receipt.json` |
| 174.4 | `FrameFault`, `run_digital_fault_battery` | `digital_fault_battery.json` |
| 174.5 | `build_digital_report`, native SVG/WAV demo export, offline browser lab | `digital_switch_report.json`, separate demo-only exports |

All canonical artifacts have a `qec.digital-*.v1` schema, explicit
`contract_version`, and full lowercase SHA-256 `sha256`. Each self-hash excludes
only that artifact's own `sha256`; nested artifacts and their identities remain
bound. Serialization is the existing QEC canonical JSON implementation: sorted
keys, compact separators, UTF-8, no floats. Payload hashes use SHA-256 over the
actual bytes. Public parsers reject unknown fields, alternate encodings, wrong
hashes, booleans in integer fields, unsupported states and noncanonical ordering.

## Software model and bounds

This is a closed, single-request transport transaction and a separate bounded
multi-command allocation model. It is not a concurrent switch server. Endpoint
catalogues are declared, sorted unique strings; each known destination is reachable
through any available output slot under the exact-catalogue policy. There is no
hidden topology inference or fallback. Unknown source/destination is a native
rejection. Cross-era callers must bind the actual endpoint catalogues and slot
state they intend to compare.

Bounds are 16 slots, 1,024 frames/payload bytes, 64 allocation commands, 16
nonoverlapping fault operations, 64 source and destination labels, and logical
ticks through `2**31 - 1`. The default frame count limit is 256. The observation
stream is capped at 2,048 records. Identifier length is at most 128 Unicode
characters; UTF-8 encoding must succeed. Inputs are frozen dataclasses with
validated detached tuple snapshots. Returned JSON dictionaries are portable
artifacts, not mutable execution state; changing them invalidates replay.

A switch input declares:

- Original opaque payload bytes and a caller-declared decoder output identity
- Frame schema, ingress slot, optional requested output slot and initial tick
- Sorted disjoint busy and quarantined output-slot inventories
- Exact source and destination catalogues
- A sorted, bounded, nonoverlapping channel fault plan

Initial busy slots receive explicit deterministic synthetic occupant labels.
The synthetic labels are slot-qualified and deterministically avoid the active
request ID; no caller namespace is reserved. These labels describe caller-declared
occupancy and do not authenticate an actual owner. The general allocator accepts explicit owner IDs and request hashes.

## v174.0 fixed-width frames

Each frame has exactly `14 + 2 * slot_count` bytes:

| Bytes | Meaning |
|---|---|
| 0–1 | Fixed synchronization marker `d4 7d` |
| 2–5 | Unsigned big-endian frame index |
| 6–13 | Unsigned big-endian logical tick |
| 14 onward | One unsigned big-endian 16-bit symbol per zero-based slot |

Data symbols are 0–255, including byte zero. Symbol 256 is explicit IDLE.
257–65,535 are invalid. Missing data is absence, never an IDLE replacement.
One payload byte occupies the declared ingress slot in each frame; all other
slots are IDLE. Empty payloads have exactly one all-IDLE frame and an explicitly
false `has_payload` observation. They do not claim nonempty payload evidence.

Allocation occurs at `start_tick`; frame index `i` uses `start_tick + 1 + i`;
release occurs at `start_tick + frame_count + 1`. These are logical ordering
units, not seconds or measured latency. Overflow rejects before any execution.

## v174.1 actual allocation and release

The allocator accepts a complete initial slot inventory and a closed command
sequence ordered by `(tick, release-before-allocate, request_id, command_id)`.
Caller order is validated, never silently repaired. Command IDs are unique.
Each declared owner ID has at most one initial slot; one active reservation per
request ID is allowed.

An unspecified output slot selects the lowest idle slot. An explicit occupied
slot yields `slot_collision`; an explicit quarantined slot yields
`slot_quarantined`; no idle capacity yields `no_capacity`. Duplicate reservation
and conflicting owner identity are explicit rejections. Releases require both
owner ID and exact request hash. A failed release changes no slot.

The native switch executes this allocator before emitting any frames, uses the
actually selected output slot in interchange, and releases only its own
reservation after success or channel failure. Each allocation event carries full
before/after state. Initial busy/quarantined state is preserved. Endpoint or
capacity rejection has no frames, no sync/interchange receipt, no output payload
and `transport_attempted=false`. A complete replay of rejection evidence is
valid evidence of rejection, never delivery.

## v174.2 synchronization and fail-closed handling

Receive observations contain bounded raw wire bytes, a declared byte-boundary
offset, and any explicitly modelled colliding slots. Malformed wire is valid
*input to a rejection receipt*, not a valid frame. Sync independently derives:

- Frame width and synchronization-marker errors
- Missing indices, duplicate indices and arrival reordering
- Out-of-range indices, logical-tick shifts and boundary slips
- Invalid symbols and declared slot collisions

No sorting, repair, substitution or partial acceptance occurs. Failed sync has
an empty `decoded_frames` result while retaining full raw observations and
per-arrival diagnostics. The native switch reconstructs these observations from
its bound encoder output and fault plan; standalone sync receipts only describe
their caller-declared observations.

## v174.3 interchange and reassembly

A mapping is a complete bijection: `mapping[input_slot] = output_slot`.
The native single-request switch derives a swap permutation between ingress
and allocated egress. The general interchange API accepts any full permutation.
Each frame resets memory to IDLE, records every input-slot write, and preserves
the complete output-memory snapshot. Duplicate addresses, omissions, wrong
ranges or booleans are invalid input.

Interchange requires complete sync replay. It cannot accept a failed sync or
invent memory on rejection. The switch then checks all unused output slots are
still IDLE, reconstructs bytes from the selected slot, and recomputes both length
and the original byte SHA-256. A carried payload hash is never sufficient.
Every channel failure suppresses the entire reassembled payload, including any
valid prefix. The decoder identity remains declared and unchanged; this layer
does not certify decoder correctness.

## Replay and trust pins

Validators reconstruct native execution and compare the *entire* canonical
artifact. Rehashing a false event, substituted memory, removed evidence, changed
claim, or forged outcome cannot bypass replay.

- `validate_digital_receipt`: `expected_input_sha256`,
  `expected_request_sha256`, `expected_frame_schema_sha256`
- `validate_allocation_receipt`: `expected_input_sha256`
- `validate_sync_receipt`: `expected_input_sha256` binds schema, observations,
  expected count and start tick; narrower optional schema/observation pins exist
- `validate_interchange_receipt`: `expected_sync_receipt_sha256`,
  `expected_mapping_sha256`
- `validate_digital_report`: `expected_receipt_sha256`

Pins must come from a separately trusted caller. A self-consistent alternative
input remains a valid *different* execution when no pin is supplied. Validation
reports explicitly enumerate which external bindings were checked; an unpinned
receipt is not authenticated provenance. Sync `input_sha256` hashes the sealed
`qec.digital-sync-input.v1` snapshot. Mapping identity hashes the sealed
`qec.digital-interchange-mapping.v1` permutation and semantics. Observation-only
identity uses `qec.digital-observation-stream.v1` with the ordered observations.

## v174.4 fault battery

`run_digital_fault_battery()` executes a frozen default corpus with real native
receipts for intact delivery, payload bit corruption, idle-slot corruption,
invalid symbols, missing frames, duplicates, reordering, alignment slip, slot
collision, initial saturation, requested-slot collision/quarantine, and unknown
destination. An additional allocation sequence exercises saturation, rejected
wrong-owner release, valid release and same-tick capacity reuse. Every case
compares exact expected failure classes and accepted status. Battery validation
regenerates the complete default corpus; omitted cases or edited expectations
cannot claim coverage. Custom switch inputs make no default-corpus claim.

Faults use original logical frame indices. Reorder consumes the next original
frame too; overlapping or out-of-range operations reject before execution.
Bit faults select an explicit symbol bit (0–15); collision creates the declared
invalid symbol 257 plus a collision observation. This is a declared software
channel model, not measured physical noise or collision behaviour.

## CLI and offline laboratory

Run from a configured checkout:

```sh
python -m qec.routing.digital demo --output-dir artifacts/digital
python -m qec.routing.digital run --input artifacts/digital/digital_switch_input.json --output-dir artifacts/digital
python -m qec.routing.digital validate --receipt artifacts/digital/digital_frame_switch_receipt.json
python -m qec.routing.digital fault-battery --output-dir artifacts/digital
python -m qec.routing.digital lab --receipt artifacts/digital/digital_frame_switch_receipt.json --output-dir artifacts/digital-demo
```

`run` emits native source, schema, allocation, switch receipt, report and
validation; sync/interchange artifacts exist only when transport was attempted.
A subsequent pre-transport rejection removes those two optional filenames from
that output directory rather than leaving stale successful evidence. JSON input
is bounded to 16 MiB and rejects duplicate keys and nonfinite constants.

`lab` first replays the supplied native receipt, then exports an explicitly
non-evidence `qec.digital-lab-demo-export.v1` JSON, SVG and deterministic mono
8 kHz WAV. These presentations show encoded source frames, with acceptance and
failure labels in demo JSON; they are not a rendering of repaired channel data.
The presentation limit is 64 frames and truncation is explicit. Audio is an
integer-only symbolic square tone per source frame, with silence for IDLE; its
sample timing is a display convention, never measured switching latency.

Open [the offline browser lab](labs/digital/index.html) directly from disk. It
has selectable payloads, slot mappings, fault controls, logical-frame inspection,
read-only native receipt inspection, and user-triggered WebAudio. No asset,
network, package service, socket, or cloud dependency is required. Its separate
teaching wire format is documented on the page and must not be mistaken for the
native frame specification above. Browser imports inspect shape only and never
claim native hash or replay validation. Browser exports use
`qec.digital-lab-demo.v1`, explicitly `canonical_evidence=false`.

## Evidence, tests and claim boundaries

Native tests cover malformed/ambiguous inputs, capacity and release, every
channel fault, empty/NUL/255 payloads, exact frame layout, bad permutations,
recomputed payload identity, trusted-source substitution, rehashed nested
mutations, frozen fixtures, hash-seed independence, reports, demo artifacts and
CLI replay. Frozen fixtures are in `tests/digital/fixtures/`. Focused command:

```sh
python -m pytest -q tests/digital
```

The repository-wide `python -m pytest -q` is the release gate, not this focused
suite. No changed source is a release until source commit, package version,
environment, canonical fixtures, test receipt, generated artifact manifest,
claim validation, clean-tree declaration and limitations are recorded.

The machine-readable claim boundary explicitly limits all native results to
bounded classical software behaviour under declared inputs. There are no
physical-network fidelity, carrier-grade reliability/capacity/latency,
quantum-hardware, fault-tolerance, threshold, advantage, provenance or decoder
correctness claims. Reports are derived views and cannot override receipts.

### Historical context vs design vs generated evidence

The [reference companion](roadmap/TELECOM_SWITCHING_REFERENCES.md) supplies the
architectural inspiration. This document makes no historical assertion about a
particular telephone system or implementation. Fixed frame widths, symbol values,
limits, allocation rules and permutations above are **QEC design choices**.
Native artifacts and reproducible tests are **generated software-model evidence**.
A historical analogy is never a proof input.
