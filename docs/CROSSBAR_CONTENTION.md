<!-- SPDX-License-Identifier: MPL-2.0 -->
# QEC v172.3 — Contention and Busy-Link Receipts

## Contract and boundary

v172.3 executes a bounded canonical sequence of reservation, release and
quarantine commands over an immutable v172.2 fabric. It maintains an owned
resource ledger and reuses the published multi-stage selector against each
current effective snapshot. Published v172.0, v172.1 and v172.2 contracts are
unchanged; new artifact schemas use contract version `172.3`.

This is a **closed classical software-model batch**. The ledger starts with no
owned reservations. Initial busy, quarantined and unavailable links remain
blocked without invented owners. The receipt records active reservations at
the end; it does not automatically release them or import state from a previous
receipt. A longer supported lifecycle must appear in the same bounded input.
This API is not a persistent reservation server, concurrency primitive or
physical actuator.

## Commands and exact ordering

`ContentionBatch(fabric, commands)` owns immutable copies. Each
`ContentionCommand` has a nonnegative logical `tick`, unique `command_id`, and
one kind with exactly its applicable arguments:

| Kind | Required arguments | Meaning |
|---|---|---|
| `reserve` | `request: MultiStageRequest` | Find and atomically reserve the first complete idle path |
| `release` | `reservation_id`, `expected_request_sha256` | Release only the active reservation with the matching sealed request identity |
| `quarantine` | `resource: CrossbarResource` | Permanently quarantine a resource for the remainder of this batch |

Unused arguments must be `None`; caller-supplied paths, ownership ledgers,
priorities and connection-commit directives are not accepted.

The command tuple must already be in this canonical order:

1. logical tick ascending;
2. kind: release, then quarantine, then reserve;
3. for reserves, the complete v172.2 request-register SHA-256 ascending;
4. command ID ascending.

The full request hash includes endpoint intent, payload and declared decoder
identity. Equal request hashes use command ID as the final tie-break. There is
no wall clock, set traversal, caller arrival order, fairness or authentication
claim. Noncanonical input order and duplicate command IDs are rejected, not
silently repaired. The `order_key` property supports explicit caller sorting.

A release at the same tick as its target's creation runs first and is rejected
because that reservation is not active yet. A release of an earlier reservation
can free capacity for a later reserve at the same tick. Quarantine runs before
all reservation requests at that tick.

## Resource identity and atomicity

`CrossbarResource(kind, matrix_id, link_id)` names a horizontal or vertical link
qualified by matrix ID, or an interstage link with `matrix_id=None`. Its
`resource_id` is the canonical SHA-256 of the versioned resource identity record.
The existing intersection identity is unchanged.

Resource catalogue order is stage, matrix, horizontal ordinal, vertical ordinal,
then the fabric's canonical interstage-link order. Explicit catalogue order
controls resource serialization; active reservations are sorted by reservation
ID. Maps provide lookup and sets membership only.

Each reserve invokes the unchanged v172.2 selector on the effective fabric:
owned resources appear busy, quarantined resources appear quarantined, and
other resources retain their declared initial state. If selection succeeds,
all selected horizontal, vertical and interstage resources are acquired in one
logical transition. A path through N stages holds exactly `3*N - 1` resources.
No resource can have two owners and no rejected request can leave a partial
reservation. Disjoint routes can coexist; later requests may select an
alternate route around a held path.

The reservation ID is the successful reserve command's ID. Its sealed record
binds the request hash, command hash, path-search receipt hash, full path plan
and canonical resource IDs. Request payload and decoder reference retain their
published meanings and bytes.

Release checks both reservation ID and request hash. It clears the complete
owned path, never an unrelated or initially busy resource. Repeated or stale
release attempts are explicit rejections. The owner hash is an identity check,
not a secret, credential, signature or proof of caller authorization.

Quarantining an owned resource first revokes its **entire reservation**, releasing
all other held resources, then marks the target quarantined. A quarantine of an
unowned idle or initially busy resource changes only that resource. Unknown,
already-quarantined and unavailable resources produce rejection without state
change. There is no unquarantine operation or silent restoration to idle.

Marker release occurs after every command, including rejection. Marker release
is separate from path release: a successful reservation remains held until an
explicit release, quarantine revocation or the end of this recorded batch.

## Bounds and outcomes

| Bound | Limit |
|---|---|
| Commands | 1–32 |
| Logical tick | Exact integer 0 through 2^31−1 |
| Total matrix intersections | 4,096 |
| Total axis and interstage resources | 4,096 |
| Sum of payload bytes across reserve commands | 1,048,576 |
| Shared search-evaluation budget | Exact integer 1–65,536; default 65,536 |

Fabric stage/matrix/wire bounds remain those of v172.2. Resource names and
command IDs also use the marker's nonempty UTF-8 text bound of 4,096 code points.
Boolean and floating-point values cannot impersonate integer ticks or budgets.
The stricter contention caps bound repeated snapshots and lifecycle evidence.

Every path search receives the remaining **batch-wide** evaluation budget.
Evaluation accounting is exactly v172.2 wire/vertical/horizontal accounting.
A completed search consumes its actual cost; a partially completed search
returns `search_budget_exhausted` and reserves nothing. Once no budget remains,
a reserve returns `batch_search_budget_exhausted` without starting a search.
Release and quarantine commands still execute with zero search budget.
Input validation, snapshot construction, ledger hashing and command processing
are additional structurally bounded work, not search-counter units.

| Outcome | Reason |
|---|---|
| `reserved` | `complete_path_reserved` |
| `released` | `owned_path_released` |
| `quarantined` | `resource_quarantined` |
| `rejected` | Published v172.2 endpoint/no-path/search-budget reasons |
| `rejected` | `batch_search_budget_exhausted` |
| `rejected` | `reservation_not_active` or `reservation_owner_mismatch` |
| `rejected` | `unknown_resource`, `resource_already_quarantined`, or `resource_unavailable` |

Each reserve result includes the pre-search inventory of **all** non-idle
resources, their exact state, reservation owner and owner-request hash when
owned. This inventory can include unrelated blocked resources. It is not a
minimal cut, causal attribution or independent proof that busy links alone
caused rejection. The embedded search receipt establishes the actual outcome
under the effective snapshot; exhaustion never establishes absence of a route.

## Evidence and replay

| Artifact | Schema |
|---|---|
| `crossbar_contention_input.json` | `qec.crossbar-contention-input.v1` |
| `crossbar_contention_program.json` | `qec.crossbar-contention-program.v1` |
| `crossbar_contention_receipt.json` | `qec.crossbar-contention-receipt.v1` |
| `crossbar_contention_validation.json` | `qec.crossbar-contention-validation.v1` |

Primary identity: `crossbar_contention_receipt_hash`.

The programme seals the complete batch, fixed policy, marker ID and shared
budget. The receipt embeds the initial fabric and commands, programme, initial
and final ledger states, per-command outcomes, complete nested path-search
receipts, busy-resource inventories, reservations and revocations. Each result
binds the ledger hashes before and after its transition. Hash-chained events
record programme verification, commands, resource transitions and marker
release. Completion events bind the canonical hash of the complete result.
All artifacts use canonical compact sorted-key UTF-8 JSON and SHA-256 excluding
their own hash field.

`validate_contention_receipt()` reconstructs the batch from its initial fabric,
runs every command again and requires exact canonical receipt equality. It
rejects partial reservation, altered ownership, missing busy evidence, reordered
commands, forged release, changed search snapshots, rehashed event mutations,
resurrected reservations and altered claim boundaries.

Optional `expected_input_sha256`, `expected_program_sha256` and
`expected_fabric_sha256` pin replay to trusted inputs. Without pins, replay
establishes internal consistency only. `all_passed` means replay-valid; a valid
batch can contain rejected commands and active reservations.

## API and CLI

```python
from qec.routing.crossbar import (
    demo_contention_batch, compile_contention_program,
    execute_contention_program, validate_contention_receipt,
)
batch = demo_contention_batch()
program = compile_contention_program(batch)
receipt = execute_contention_program(batch, program)
validation = validate_contention_receipt(receipt,
    expected_input_sha256=batch.as_dict()["sha256"],
    expected_program_sha256=program["sha256"],
    expected_fabric_sha256=batch.fabric.as_dict()["sha256"])
```

The demo records reservation, contention rejection, release, successful retry,
quarantine revocation and rejection of the quarantined endpoint.

```bash
qec-crossbar contention-demo --output-dir artifacts/crossbar-contention
qec-crossbar contend \
  --input artifacts/crossbar-contention/crossbar_contention_input.json \
  --output-dir artifacts/crossbar-contention
qec-crossbar contention-validate \
  --receipt artifacts/crossbar-contention/crossbar_contention_receipt.json
```

`contend` accepts `--marker-id` and `--max-search-evaluations`. A valid executed
batch exits 0 even if individual commands reject; inspect its results for those
outcomes. Invalid input or I/O failure exits 1. `contention-validate` accepts the
three trusted-hash options and exits 0 only for replay-valid evidence. Duplicate
JSON object keys are rejected. Existing commands remain unchanged.

## Validation and remaining milestones

Tests check whole-path ownership against an independent two-branch ledger
oracle across 64 lifecycle scenarios and every command prefix. Coverage also
includes exact tie-breaking, disjoint reservations, alternate routing, stale
and wrong-owner release, quarantine on all resource kinds, initial blocked
states, shared-budget boundaries, malformed contracts, immutable inputs,
rehashed tampering, hash-seed determinism, a frozen fixture and CLI round-trips.

Reservation is not connection commit. A separate continuity receipt remains
v172.4; cross-era equivalence remains v172.5. This contract does not establish
physical switching fidelity, authenticated provenance, decoder correctness,
carrier-grade reliability or quantum advantage.
