<!-- SPDX-License-Identifier: MPL-2.0 -->
# Canonical packet fabric and offline softswitch boundary

Status: **implemented, unreleased**, v175.0–v175.6. The source package remains
released at **173.0.0**. These additions do not change any published Strowger,
Panel, Crossbar, ESS skeleton, or decoder contract. Python's standard library is
sufficient for this era's runtime and tests need only pytest.

## Phase-to-implementation map

| Phase | Native implementation | Evidence and validation |
|---|---|---|
| v175.0 | `PacketRequest`, `PacketEnvelope`, `packetize`, `canonical_packet_schema` | Original bytes, fragment bytes, sequence identity and declared provenance are independently hash-bound |
| v175.1 | `PacketTopology`, `PacketEndpoint`, `Route`, `lookup_route` | Exact binary longest-prefix lookup; canonical table and stable metric/ID tie breaks |
| v175.2 | `QueuePolicy`, tick phases in `execute_packet_fabric` | Bounded per-node queues, priority scheduling, one explicit overflow rule |
| v175.3 | Native global and per-packet event chains | Every admission, dequeue, lookup, attempted alternative, transformation, drop, forwarding and reassembly decision |
| v175.4 | `PacketFault`, `FaultPlan`, native failover receipt | Fixed fault intervals, ordered alternatives, recovery after an interval, no random selection |
| v175.5 | `SoftswitchObservation`, adapter receipt and validator | Offline SIP/RTP/SDP observation identity; external and non-authoritative |
| v175.6 | `create_packet_report`, `create_packet_demo_export`, offline HTML lab | Replayed native report plus separate, explicitly non-proof visualization/sonification |

All canonical objects carry a versioned `qec.packet-*.v1` schema,
`contract_version: "175.6"`, and a complete lowercase 64-character SHA-256.
Hashes cover sorted-key, compact, UTF-8 JSON with no floats, excluding the
object's own `sha256` field. Raw payload hashes cover bytes, not hex text.
Every `from_dict` reconstructs the canonical object and compares its complete
serialization. Extra, missing, reordered noncanonical, and rehashed false
fields are rejected with `ValueError("INVALID_INPUT")`.

The phase names and design inspiration follow the roadmap and its
[acceptance rules](roadmap/SWITCHING_SAFETY_AND_ACCEPTANCE.md),
[artifact registry](roadmap/SWITCHING_ARTIFACT_REGISTRY.md), and
[historical reference boundary](roadmap/TELECOM_SWITCHING_REFERENCES.md).
This document makes no new claims about historical exchange designs. Binary
addresses, per-node queues, service rates, fault intervals and tie breaks below
are QEC software design choices, not claims of physical protocol fidelity.

## Native API

```python
from qec.routing.packet import (
    PacketEndpoint, PacketRequest, PacketTopology, Route, PacketInput,
    QueuePolicy, PacketFault, FaultPlan, execute_packet_fabric,
    validate_packet_receipt,
)

fabric = PacketTopology(
    nodes=("ingress", "lane0", "lane1", "egress"),
    destinations=(PacketEndpoint("10", "egress"),),
    routes=(
        Route("ingress", "", ("lane0", "lane1"), route_id="ingress-route"),
        Route("lane0", "", ("egress",), route_id="lane0-route"),
        Route("lane1", "", ("egress",), route_id="lane1-route"),
    ),
)
source = PacketInput(
    requests=(PacketRequest("correction-1", "ingress", "10", b"opaque", "a" * 64),),
    topology=fabric,
    queue_policy=QueuePolicy(capacity=16, max_hops=8),
    fault_plan=FaultPlan((PacketFault(
        0, "ingress", "lane0", duration=1024, source_state="busy",
    ),)),
    fragment_bytes=256,
)
receipt = execute_packet_fabric(source)
validation = validate_packet_receipt(
    receipt, expected_input_sha256=source.as_dict()["sha256"],
)
assert validation["all_passed"]
assert receipt["results"][0]["packet_paths"][0]["nodes"] == [
    "ingress", "lane1", "egress",
]
```

The provenance digest is caller-declared. It can bind a correction/decoder
identity supplied by a caller; this layer neither reads nor modifies decoder
internals. The digest is not authentication and cannot establish correctness.

### Immutable packet identity

A request binds its ID, source, binary destination, original payload bytes and
hash, provenance hash, immutable priority, and declared injection tick.
Packetization is deterministic: contiguous zero-based fragments with offsets,
an explicit count, and one empty fragment for an empty payload. Each envelope
binds the request hash, original payload hash, exact fragment bytes/hash,
sequence-identity artifact, source, destination, provenance and priority.

Mutable hop count, queue position, forwarding tick and wire bytes are excluded
from envelope identity. They are bound into the event chain instead. Wire
corruption changes transported bytes but does not rewrite an envelope or the
original request. Input containers are copied to immutable tuples and native
objects are frozen. JSON artifacts are detached serializable dictionaries;
mutating one invalidates its hash/replay and cannot mutate the native input.

### Exact route semantics

Addresses are binary strings, 1–32 bits; route prefixes are 0–32 bits. An empty
prefix is the declared default. These are not IPv4/IPv6 addresses. A prefix
matches exactly when `destination.startswith(prefix)`. The most-specific match
wins; equal-length matches use lowest metric, then lexicographically smallest
unique route ID. The manifest orders routes by node, descending prefix length,
metric, route ID and prefix. Input route/node collections may be unordered;
constructors canonicalize them, while deserialization requires canonical order.

Every destination maps to exactly one named node. Arrival at that node is the
only delivery operation. Arrival at an unknown address rejects explicitly.
All route edges and fault references must name registered nodes. Alternate
next hops form an explicitly ordered, unique tuple. Only the chosen route's
alternatives are considered: no silent use of another equal-length route or
less-specific default is allowed. Self-edges are forbidden; multi-node loops
terminate at the declared hop limit. Destination delivery is checked before
hop-limit rejection, so a destination reached in exactly the permitted hop
count remains valid.

### Scheduling and queues

Initial state: every node queue is empty and every declared edge is usable
unless an explicit fault is active. There are no hidden reservations, clocks,
callbacks, live adapters or mutable process globals.

Each logical tick has two phases:

1. Admit all original injections and preceding forwards. Group by canonical
   node name; within a node order by priority, request ID, fragment sequence.
2. Service at most one packet per nonempty node, in canonical node-name order.
   Queue service order is priority, enqueue tick, request ID, fragment sequence.
   Lower numeric priority wins. Every forward arrives at the following tick.

A full queue has one outcome: **drop the arriving packet without eviction**.
An arriving high-priority packet cannot evict an earlier queued low-priority
packet. Admission occurs before that tick's service. Dropped packets are never
silently retried. All admission/overflow/dequeue events include queue snapshots
or the declared drop decision. Every operation changes global hash lineage and
that packet's own lineage. No claim about physical latency follows from ticks.

### Fault and failover semantics

A `PacketFault(logical_tick, node, next_hop, kind="link_down", duration=1,
source_state="declared_fault")` applies on its directed edge over
`[logical_tick, logical_tick + duration)`. Kinds, in precedence order:

- `link_down`: record an unavailable attempt; try the next declared alternative.
- `drop`: after selecting the link, record an explicit loss; do not fail over.
- `corrupt`: XOR the first wire byte with 1, or append `01` to an empty fragment;
  forward the changed wire bytes with a transformation event.

Overlapping same-kind intervals for one edge are rejected. Different kinds may
overlap and the fixed precedence above applies. All active fault hashes are
listed on route attempts; applied drop/corrupt events identify their faults.
A link becomes usable again after its interval; later packets can use it, with
no retransmission or background recovery service. The `recovery` demo proves
this on a subsequent request. The optional `source_state` retains a migration
annotation (`busy`, `quarantined`, `unavailable`, or `declared_fault`) in native
evidence; it cannot change the fault's execution semantics.

Native drop reasons remain visible: `unknown_destination`, `no_route`,
`all_next_hops_unavailable`, `hop_limit`, `queue_overflow`, and
`declared_link_drop`. Reassembly reports `missing_fragments` or
`payload_mismatch`, preserving every source drop and traversed packet path.

### Destination reassembly and complete replay

Every result binds the request and immutable packet hashes, received fragments,
source drops, packet paths, terminal event, and original/recomputed payload
identities. Reassembly requires exactly the expected contiguous sequences and
offsets, recomputes SHA-256 over the assembled received bytes, and compares both
hash and bytes to the original request. An incomplete or corrupt request never
receives an `accepted_payload_sha256`. An empty batch validates its empty
execution, with `nonempty_evidence: false`; it cannot claim route evidence.

`validate_packet_receipt` reconstructs the full input and reruns every event,
queue state, route attempt, fault, per-packet chain and reassembly. It compares
the entire receipt, not merely its self-hash or summary. Trusted pins are
optional keyword arguments:

- `expected_input_sha256`
- `expected_topology_sha256`
- `expected_queue_policy_sha256`
- `expected_fault_plan_sha256`
- `expected_receipt_sha256`

Without independent trusted pins, validation proves internal replay consistency,
not that the user intended a substituted input. The returned validation records
which external bindings were actually checked.

Separated schema/table/queue/chain/failover/reassembly artifacts must use
`validate_packet_artifact(artifact, fabric_receipt=..., expected_input_sha256=...)`.
The complete source receipt is mandatory, because a hash-only view cannot
reconstruct missing inputs. `create_packet_report` and
`validate_packet_report` likewise retain and replay the entire source receipt.

## Softswitch boundary

`SoftswitchObservation(logical_index, protocol, source_label, capture_sha256,
observation)` accepts only bounded offline `sip`, `rtp` or `sdp` observations.
`create_softswitch_adapter_receipt` can reference an independently validated
software receipt. `validate_softswitch_adapter_receipt` requires that full
software receipt when a reference is present, and optionally pins the adapter
receipt hash. Observations have contiguous logical indices.

The adapter records are external, non-authoritative, and unauthenticated. Their
hashes establish only record integrity. A reported SIP success cannot force an
accepted packet result. `PacketInput` has no observation field. No sockets,
network stack, protocol daemon, login, carrier service or cloud provider is
started. No observation is fabricated in the default CLI adapter export: its
capture list is explicitly empty.

## CLI and registered artifacts

```sh
python -m qec.routing.packet demo --scenario failover --output-dir /tmp/packet
python -m qec.routing.packet run --input /tmp/packet/packet_fabric_input.json --output-dir /tmp/packet
python -m qec.routing.packet validate --receipt /tmp/packet/packet_fabric_receipt.json
```

Available demo scenarios: `success`, `failover`, `overflow`, `corrupt`, `drop`,
`recovery`. Validation CLI accepts every trusted-pin argument above with hyphens
instead of underscores. JSON input rejects duplicate keys and has a 256 MiB
read bound. No wall-clock data or generated randomness enters artifacts.

The `run` command produces:

- `packet_fabric_input.json`: complete immutable input snapshot
- `canonical_packet_schema.json`: v175.0 envelope/identity contract
- `route_table_manifest.json`: v175.1 topology, destinations and ordered routes
- `queue_policy_manifest.json`: v175.2 fixed scheduling/overflow contract
- `packet_hop_receipt_chain.json`: v175.3 global/per-packet lineage
- `packet_failover_receipt.json`: v175.4 declared attempts, fault and failover bindings
- `softswitch_adapter_receipt.json`: v175.5 external offline observation boundary
- `packet_fabric_receipt.json`: full source, chain, failover and reassembly evidence
- `packet_fabric_validation.json`: complete replay and binding status
- `packet_fabric_report.json`: v175.6 derived report with replayable source
- `packet_demo.json`: noncanonical demonstration namespace, without proof hashes

Primary registry identities are exposed as
`canonical_packet_fabric_receipt_hash` and `deterministic_failover_receipt_hash`
in validation artifacts. The corresponding full canonical artifact's `sha256`
is authoritative for software replay; filenames alone confer no authority.

## Offline lab

Open [the local HTML lab](labs/packet/index.html). It contains all six native
example traces inline, with no CDN, fonts, requests, sockets or dependencies.
Controls select a case, step/scrub logical events, inspect topology and queue
occupancy, and enable user-triggered Web Audio sonification. Browser timing only
controls presentation; it does not decide model outcomes.

Exports use `qec.packet-demo.v1`, `demo_only: true`, `cryptographic_proof: false`
and `hardware_evidence: false`. They deliberately omit canonical receipt hashes
and cannot be loaded as native receipts. The browser importer accepts bounded
demonstration observations, not canonical software proof. Native CLI replay is
the evidence authority. Sonification frequencies convey event classes only.

## Explicit bounds and limitations

| Quantity | Bound |
|---|---|
| Nodes / routes / registered endpoints | 16 / 128 / 128 |
| Requests / original payload bytes per request | 16 / 4,096 |
| Fragment bytes / fragments per request / total packets | 256 / 64 / 256 |
| Per-node queue capacity / hop limit | 64 / 16 |
| Route metric / numeric priorities | 0–65,535 / 0–7 |
| Injection or fault-start tick / fault interval end | 0–1,023 / at most 1,024 |
| Fault records / observations | 128 / 256 |
| Name / observation text length | 64 ASCII identifier characters / 256 Unicode characters |
| Event count / CLI JSON file bytes | 131,072 / 268,435,456 |

Fragment size must satisfy both per-request and global packet limits; a 4,096
byte request cannot request one-byte fragments. Runtime ticks may exceed the
last injection/fault tick while finite queues drain. An explicit termination
bound is last injection tick + packet count × (hop limit + 1) + 1. Queue service
and hop consumption prove termination independently of wall time. All counts
require exact integers; booleans and floats are rejected. Surrogates, uppercase
hex and oversized buffers are invalid. Initial arbitrary queue snapshots,
retransmission, congestion control, variable MTUs, packet-header corruption,
real SIP/RTP parsing and protocol-compliance claims are outside this contract.

## Verification

Focused validation:

```sh
python -m pytest -q tests/packet
python -m pytest -q tests/ess/test_ess_skeleton.py tests/crossbar/test_crossbar_contract_types.py
# Optional standalone lab DOM smoke check (requires Node.js):
node tests/packet/lab_dom_check.cjs
```

Tests cover canonical fixture round trips, pinned hashes, PYTHONHASHSEED 0/42/12345
byte identity, native fault controls and recovery, strict/boundary types,
maximum fragments, provenance separation, priority/overflow, exact-prefix tie
breaks, loops, zero-hop/empty payloads, frozen snapshots, full replay,
rehashed/omitted/duplicated/substituted evidence, claims, trust pins, adapter
non-authority, CLI artifacts and lab namespace/network isolation.

The repository-wide `python -m pytest -q` is the integration/release gate.
The standalone DOM check exercises control logic and safety boundaries; it is not
a rendered-browser or acoustic assessment. Focused passes are not a release
declaration. No physical telephone-network,
hardware fault-tolerance, threshold, quantum-advantage, authenticated-provenance,
carrier reliability, or decoder-correctness claim follows from these artifacts.
