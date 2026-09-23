<!-- SPDX-License-Identifier: MPL-2.0 -->
# v173.0 — Stored-Program Switch Skeleton

The ESS skeleton separates a fixed programme configuration, sealed call data,
a logical input queue and a switching-fabric adapter. It produces a deterministic
command stream and independently validated native planning evidence. It lives in
`src/qec/routing/ess/`; existing Crossbar, Panel, Strowger and decoder contracts
are unchanged.

## Components

| Component | v173.0 contract |
|---|---|
| `ProgramStore` | One built-in `logical-queue-independent-plan-once.v1` programme, with ID, version, marker ID and per-call search budget |
| `CallStore` | Immutable `MultiStageRequest` records sorted by unique native request ID; that request ID is the ESS call ID |
| `InputEvent` / `EventQueue` | One `plan_route` event per call, nondecreasing logical ticks and contiguous zero-based sequence numbers |
| `CrossbarFabricAdapter` | The sole registered adapter; delegates to v172.2 path selection over a detached immutable fabric snapshot |
| `SwitchInput` | Binds all four components and requires complete one-to-one call/event coverage |
| `execute_switch` | Dispatches in queue order, validates each native result, returns commands, results and complete input snapshots |
| `validate_switch_receipt` | Reconstructs every command and result, compares canonical bytes and optionally checks trusted hashes |

Programme identity is separate from call data. Changing programme version,
marker ID or budget changes the programme and execution identities without
changing the call-store identity. Changing payload or endpoints changes the call
and execution identities without changing the programme. Version and programme
ID are labels, not code-loading instructions. The fixed operation and adapter
identity cannot be replaced through JSON.

The programme store is a configuration snapshot. The complete source and
translation-table manifest belongs to v173.1. The call store holds sealed inputs;
it does not yet implement the intake/analyse/select/connect/verify/release/reject
state machine planned for v173.2. Queue traversal here is the minimum deterministic
dispatch mechanism; interrupt classes and timing receipts remain v173.3.

## Queue and bounds

Calls must arrive sorted by request ID. Event order is separately declared:
ascending `(logical_tick, sequence)`, with `sequence == 0, 1, …` in array order.
Equal ticks preserve explicit sequence order. A call can therefore be dispatched
before another call whose ID sorts earlier. Invalid ordering is rejected, never
silently sorted or deduplicated.

The input is a closed batch. Every stored call must have exactly one event; unknown
calls, duplicates and missing events are invalid. There is no incremental enqueue,
retry, cancellation, release or persistent processing state in this contract.

| Bound | Limit |
|---|---|
| Calls / events | 0–16, with exactly matching coverage |
| Programme labels and call IDs | 1–128 Unicode code points, valid UTF-8 |
| Payload | 0–4,096 bytes per call; at most 65,536 bytes per batch |
| Logical ticks | Exact integers from 0 through `2**63 - 1` |
| Per-call search budget | Exact integer from 1 through 131,072 |
| Total native search allowance | At most 16 × 131,072 evaluations per execution |
| Fabric | Existing v172.2 bounds: 2–16 stages, up to 64 matrices, 65,536 intersections and 4,096 wires |

Booleans and floats cannot stand in for integer fields. Collections are copied
into immutable tuples; serialization returns detached JSON data. Payloads remain
opaque bytes, and decoder output hashes remain caller-declared references.

## Adapter validation boundary

For each event, the controller compiles the existing native path programme from
the exact fabric, call and stored budget. It seals a command containing the full
input, programme, event, adapter, request and native-programme hashes before
calling the adapter.

The adapter returns a native search receipt. The controller invokes v172.4
continuity verification with trusted fabric, request and native-programme hash
pins. That verifier first replays the v172.2 receipt and then checks every selected
coordinate and wire. Even a valid native receipt from another request, fabric or
programme is rejected. An adapter cannot certify its own success, substitute a
payload, alter the compiled programme or force a connection.

Results are read from the detached validated evidence. Each result embeds the
complete continuity receipt, which contains the complete native search receipt.
Native `plan_selected` and `rejected` outcomes and their exact reasons are retained.
Unknown endpoints, blocked capacity and exhausted search budgets produce native
rejections. Malformed inputs or evidence raise `ValueError("INVALID_INPUT")`.

All calls see **the same immutable fabric snapshot**. Multiple calls can select
the same path. This establishes independent planning, not simultaneous admission,
owned reservations, physical actuation or a committed connection. Stateful
contention is not routed through this adapter.

The adapter is an explicit registered boundary, not an arbitrary callable or a
dynamic plug-in loader. Additional adapters require new implementation and
validation contracts. No ESS/Fabric equivalence claim is made before v173.5.

## Artifacts and replay

| Artifact | Contents |
|---|---|
| `ess_switch_input.json` | Separate sealed programme, call, queue and adapter snapshots |
| `ess_command_stream.json` | Ordered commands bound to the complete input |
| `ess_switch_skeleton_receipt.json` | Input, stream, per-call native evidence, derived counts and claim boundary |
| `ess_switch_skeleton_validation.json` | Replay result, counts and trusted-pin coverage |

All v173.0 schemas use the `qec.ess-*.v1` namespace and `contract_version: "173.0"`.
Hashes use sorted-key compact UTF-8 JSON, excluding each artifact's own `sha256`.
Primary identities are `ess_switch_skeleton_receipt_hash` and
`ess_command_stream_hash`, exposed by the validation artifact.

Validation reconstructs the entire receipt. Re-signing an altered command,
omitted result, substituted receipt, count or claim boundary does not make it
valid. A different valid input can legitimately produce a different valid receipt;
use `expected_input_sha256` to require the precise trusted batch. Programme-store
and adapter pins support narrower external bindings. Without pins, replay proves
internal consistency, not provenance or correspondence to an external request.

An empty batch is valid: no commands or routes, zero counts and a drained queue.
`all_passed` means replay succeeded, including legitimate native rejections.
`all_adapter_results_verified` is vacuously true for an empty batch; use
`dispatched_count` and `selected_count` when requiring nonempty route evidence.

## Python and CLI

```python
from qec.routing.ess import demo_switch_input, execute_switch, validate_switch_receipt

source = demo_switch_input()
receipt = execute_switch(source)
validation = validate_switch_receipt(
    receipt, expected_input_sha256=source.as_dict()["sha256"],
)
assert validation["all_passed"]
assert validation["selected_count"] == 1
assert validation["rejected_count"] == 1
```

```bash
qec-ess demo --output-dir artifacts/ess
qec-ess run --input artifacts/ess/ess_switch_input.json --output-dir artifacts/ess
qec-ess validate --receipt artifacts/ess/ess_switch_skeleton_receipt.json
```

`validate` also accepts `--expected-input-sha256`,
`--expected-program-store-sha256` and `--expected-adapter-sha256`.
`python -m qec.routing.ess.cli` provides the same commands. Exit 0 means valid
execution/replay, including native rejections; exit 1 means invalid input, evidence
or file I/O. Argument syntax errors use argparse's exit 2. Duplicate JSON object
keys are rejected.

The demo deliberately stores calls in ID order and dispatches them in the opposite
order at tick 0: an unknown-endpoint rejection followed by a continuous selected
route. Tests cover full native replay, immutable stores, input bounds, same-tick
ordering, empty and maximum batches, independent capacity semantics, hash seeds,
frozen identities, CLI round-trips and adapter evidence substitution.

These software checks do not establish physical switching fidelity, authenticated
provenance, decoder correctness or quantum advantage. No decoder is invoked.
