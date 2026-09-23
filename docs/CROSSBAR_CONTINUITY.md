<!-- SPDX-License-Identifier: MPL-2.0 -->
# QEC v172.4 — Crossbar Continuity Verification

## Contract

v172.4 adds separate continuity evidence over published v172.2 path-search and
v172.3 contention receipts. It replays the entire source, then independently
walks every selected route forwards against its exact pre-selection fabric.
The published v172.0–v172.3 schemas, selectors, ledgers and fixtures are unchanged.

A successful witness proves that **all** supplied selected coordinates and wires
form one continuous, idle, endpoint-bound route in the declared snapshot. It is
not a search, a new reservation, a connection commit or a physical actuation.
The route may subsequently be released or revoked by quarantine; the historical
witness remains true for the snapshot in which selection occurred.

## Independent forward verification

`verify_path_continuity(fabric_manifest, request, path_plan)` validates the fabric
and sealed `MultiStageRequest`, then applies this fixed rule:

1. Require exactly N coordinates and N−1 interstage wire IDs for N stages.
2. Start at the request's qualified source matrix and horizontal link.
3. In each stage, require the next coordinate to belong to that exact matrix and
   horizontal. Recompute its complete intersection identity and ordinals from
   the published matrix; compare every field with canonical type-sensitive JSON.
4. Require both axis links to be idle. Record the coordinate and both states.
5. For each boundary, require the named idle wire to leave that exact matrix and
   vertical link. Use its declared target matrix and horizontal for the next step.
6. Require the final coordinate's matrix and vertical to equal the requested
   destination exactly. No selected coordinate or wire may remain unused.

One coordinate per ordered stage and adjacent-stage wiring exclude branches,
cycles, duplicate coordinates, skipped stages and disconnected fragments. Matrix
qualification prevents identical local link names in different matrices from
being conflated. No selector choices or suffix-reachability flags are used by
the walk. A different complete idle route can pass this low-level check; it is
not proof of first-path selection. Receipt creation additionally replays the
published source selector to establish that the checked route was selected.

The witness records the reconstructed plan, alternating coordinate/wire steps,
counts, snapshot and request hashes, payload hash and declared decoder identity.
It is a derived component of the receipt, not independently authenticated input.

## Source coverage and lifecycle

`create_continuity_receipt(source_receipt)` accepts exactly:

- a complete replay-valid v172.2 `qec.crossbar-path-search-receipt.v1`; or
- a complete replay-valid v172.3 `qec.crossbar-contention-receipt.v1`.

For a path-search source, there is one route row. For a contention source, there
is one row per **reserve attempt**, in canonical command order, including rejected
attempts and exhausted batches that never started a search. Release and quarantine
commands remain covered by full source replay and the embedded source receipt.

A selected route gets outcome `continuous` and a witness. A rejected attempt gets
`not_selected`, its original reason, and a null witness. Rejection never becomes a
claim that a route exists. Malformed, substituted or inconsistent evidence raises
`ValueError("INVALID_INPUT")` and produces no receipt, even if hashes were recomputed.

Contention rows bind the command ID, request, nested search receipt, pre-command
ledger hash and reservation hash. `reservation_active_at_batch_end` is derived
from the replayed final ledger: true for an active reservation, false for one
released or revoked, and null when no reservation was created. A standalone path
plan also uses null. This field says nothing about availability after the batch.
Continuity uses the effective **pre-selection** snapshot in each nested search,
not the initial fabric or final busy/quarantined ledger.

Receipt outcome is `continuity_verified` when at least one route was selected;
otherwise it is `no_selected_route`. A valid empty selection set, including a
batch containing only releases/quarantines, is never labelled a continuity proof.
All selected routes must pass; there is no partial-success mode.

## Bounds and identities

No new search or Cartesian route enumeration is introduced. The walk checks
exactly N coordinates and N−1 wires, with 2 ≤ N ≤ 16. A source contains one path
attempt or at most 32 contention attempts: at most 512 coordinate and 480 wire
checks. Fabric parsing, lookup construction, canonical hashing and full source
replay are additional bounded work under the unchanged upstream contracts.

All new artifacts use contract version `172.4`, sorted-key compact UTF-8 canonical
JSON and SHA-256 excluding their own `sha256` field:

| Artifact | Schema |
|---|---|
| `crossbar_continuity_receipt.json` | `qec.crossbar-continuity-receipt.v1` |
| `crossbar_continuity_validation.json` | `qec.crossbar-continuity-validation.v1` |
| Embedded route witness | `qec.crossbar-continuity-witness.v1` |

Primary identity: `crossbar_continuity_receipt_hash`.

The receipt embeds a detached copy of the complete source and its full hash,
every route row, counts and the fixed claim boundary. Hash-chained logical events
bind source replay, every row, and completion. These events have no timing claim.
`validate_continuity_receipt()` replays the embedded source, repeats every forward
walk and requires exact canonical equality of the entire derived receipt.
Validation reports are derived output; consumers must run the validator.

Creation and validation both accept optional trusted pins:

| Argument / CLI option | Binding |
|---|---|
| `expected_source_sha256` / `--expected-source-sha256` | Entire upstream receipt |
| `expected_fabric_sha256` / `--expected-fabric-sha256` | Path snapshot, or contention **initial** fabric |
| `expected_input_sha256` / `--expected-input-sha256` | Path request register, or whole contention input batch |
| `expected_program_sha256` / `--expected-program-sha256` | Upstream search or contention programme |

Pins do not change receipt bytes. Without them, replay proves internal consistency,
not external provenance. `all_passed` means replay-valid evidence;
`continuity_verified` additionally requires a nonempty set of selected routes.

## API and CLI

```python
from qec.routing.crossbar import (
    demo_contention_batch, compile_contention_program, execute_contention_program,
    create_continuity_receipt, validate_continuity_receipt,
)

batch = demo_contention_batch()
source = execute_contention_program(batch, compile_contention_program(batch))
receipt = create_continuity_receipt(source, expected_source_sha256=source["sha256"])
validation = validate_continuity_receipt(receipt,
    expected_source_sha256=source["sha256"])
```

```bash
qec-crossbar contention-demo --output-dir artifacts/crossbar-contention
qec-crossbar contend \
  --input artifacts/crossbar-contention/crossbar_contention_input.json \
  --output-dir artifacts/crossbar-contention
qec-crossbar continuity \
  --receipt artifacts/crossbar-contention/crossbar_contention_receipt.json \
  --output-dir artifacts/crossbar-continuity
qec-crossbar continuity-validate \
  --receipt artifacts/crossbar-continuity/crossbar_continuity_receipt.json
```

`continuity` also accepts a path-search receipt. It writes receipt and validation
artifacts, exiting 0 when routes are verified, 2 for valid `no_selected_route`,
and 1 for invalid input or I/O failure. `continuity-validate` exits 0 for any
replay-valid receipt, including no selection, and 1 otherwise. Both commands
reject duplicate JSON keys and accept all four trusted pins above.

## Validation and claim boundary

Tests exercise an independent two-branch membership oracle, connected alternate
routes versus first-path selection, all blocked states, exact endpoints, broken
wires, partial/extra/reordered coordinates, identity and type substitutions,
2/16-stage limits, 32-attempt budget exhaustion, disjoint active reservations,
release/quarantine history, full source coverage, rehashed tampering, trusted
bindings, event hashes, a frozen fixture, hash-seed determinism and CLI round-trips.

These are executable software-model checks, not a machine-checked mathematical
proof in a theorem prover. They establish no physical switching fidelity,
authenticated provenance, current liveness, decoder correctness, carrier-grade
reliability or quantum advantage. Cross-era equivalence remains v172.5.
