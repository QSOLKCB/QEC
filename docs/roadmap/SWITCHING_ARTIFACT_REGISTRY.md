<!-- SPDX-License-Identifier: MPL-2.0 -->
# QEC Switching Artifact Registry

This registry centralises the planned canonical artifacts introduced by the
telecommunications switching programme in [`ROADMAP.md`](../../ROADMAP.md).

Artifact names are planning contracts until their release lands. Published
schemas and release tags become authoritative once implemented.

## Programme-lineage overview

The table records architectural sequence and each artifact family's intended
role. It does not assert that one row cryptographically binds the preceding row.
A binding exists only when the downstream schema explicitly includes and
validates the upstream artifact's full hash. The planned v176.x migration and
cross-era receipts are where explicit multi-architecture binding is introduced.

| Phase | Architecture | Primary model artifact | Primary receipt or report | Programme-lineage role |
|---|---|---|---|---|
| v170.3.0 | Strowger | route topology and pre-route state in the canonical route receipt | `strowger_route_receipt_hash` | Mechanical step-by-step routing baseline |
| v171.x | Panel | `panel_topology.json`, `panel_sender_program.json` | `panel_route_receipt.json` | Separates compiled control intent from payload and path actuation |
| v172.x | Crossbar | `crossbar_matrix_manifest.json`, `crossbar_marker_program.json` | `crossbar_continuity_receipt.json` | Proves bounded coordinate selection and continuous-path establishment |
| v173.x | ESS | `ess_program_store_manifest.json` | `ess_call_processing_receipt.json` | Binds immutable stored programmes, logical scheduling and fabric commands |
| v174.x | Digital TDM | `digital_frame_schema.json` | `timeslot_interchange_receipt.json` | Proves framed symbols, deterministic slot allocation and reassembly |
| v175.x | Packet fabric | `canonical_packet_schema.json`, `route_table_manifest.json` | `packet_hop_receipt_chain.json` | Preserves payload identity through queues, hops and deterministic failover |
| v176.x | Cross-era | `switch_migration_contract.json` | `cross_era_equivalence_receipt.json` | Explicitly binds declared invariants across architecture migration |

## v171.x — Panel separated-control artifacts

| Artifact | Role |
|---|---|
| `panel_topology.json` | Canonical motor groups, banks, selector limits and path inventory |
| `panel_digit_register.json` | Sealed representation of the complete collected request |
| `panel_sender_program.json` | Deterministically compiled route-control programme |
| `panel_route_receipt.json` | Binds request, sender programme, actuation and verified outcome |
| `panel_fault_battery.json` | Reproducible busy-bank, stall, corruption and disagreement cases |
| `panel_claim_validation.json` | Machine validation of Panel report claims |

Primary hashes:

```text
panel_sender_register_receipt_hash
panel_separated_control_receipt_hash
```

## v172.x — Crossbar artifacts

| Artifact | Role |
|---|---|
| `crossbar_matrix_manifest.json` | Immutable horizontal/vertical matrix and link-state ordering |
| `crossbar_marker_input_register.json` | v172.1 sealed request, endpoint intent, payload bytes and declared decoder identity |
| `crossbar_marker_program.json` | v172.1 matrix/request-bound exact-coordinate policy and authority limits |
| `crossbar_common_control_receipt.json` | v172.1 complete input snapshots, logical event chain, plan/rejection and marker release |
| `crossbar_common_control_validation.json` | v172.1 replay result and optional trusted-input binding checks |
| `crossbar_fabric_manifest.json` | v172.2 immutable ordered matrix stages and canonical adjacent-stage wires |
| `crossbar_path_input_register.json` | v172.2 qualified endpoints and embedded v172.1 request |
| `crossbar_path_search_program.json` | v172.2 fabric/request-bound policy and evaluation budget |
| `crossbar_path_search_receipt.json` | v172.2 bounded look-ahead trace and first complete plan or rejection |
| `crossbar_path_search_validation.json` | v172.2 replay and optional trusted-input bindings |
| `crossbar_contention_input.json` | v172.3 immutable fabric and canonical logical command batch |
| `crossbar_contention_program.json` | v172.3 input-bound ordering policy, marker and shared search budget |
| `crossbar_contention_receipt.json` | v172.3 owned reservations, busy inventories, release, quarantine and full replay evidence |
| `crossbar_contention_validation.json` | v172.3 reconstructed lifecycle and optional trusted input bindings |
| `crossbar_continuity_receipt.json` | v172.4 full source replay and independent forward witnesses for every selected route |
| `crossbar_continuity_validation.json` | v172.4 complete replay, nonempty continuity status and trusted source bindings |
| `switch_equivalence_corpus.json` | v172.5 bounded shared cases and native fault controls |
| `switch_equivalence_adapter_manifest.json` | v172.5 fixed native topologies and explicit comparison mappings |
| `switch_equivalence_matrix.json` | v172.5 source-bound Strowger/Panel/Crossbar route-decision comparisons |
| `switch_equivalence_validation.json` | v172.5 native replay, corpus coverage and expected-difference validation |

Primary hashes:

```text
crossbar_matrix_receipt_hash
crossbar_common_control_receipt_hash
crossbar_path_search_receipt_hash
crossbar_contention_receipt_hash
crossbar_continuity_receipt_hash
switch_equivalence_matrix_hash
```

v172.0–v172.5 artifacts retain their published schemas; the v172.x arc is released.
It compares only declared route decisions for a bounded corpus. Native commit,
payload capabilities and reservation lifecycles remain explicitly distinct.
Broader cross-era migration stays assigned to v176.x.

## v173.x — ESS stored-program artifacts

v173.0 implements the following skeleton artifacts; the programme/source manifest,
full lifecycle, scheduling and equivalence artifacts below remain planned.

| Artifact | Role |
|---|---|
| `ess_switch_input.json` | v173.0 separate programme, call, queue and adapter snapshots |
| `ess_command_stream.json` | v173.0 input-bound ordered native planning commands |
| `ess_switch_skeleton_receipt.json` | v173.0 dispatch results with complete validated native evidence |
| `ess_switch_skeleton_validation.json` | v173.0 complete replay and optional trusted input bindings |

Primary v173.0 hashes: `ess_switch_skeleton_receipt_hash`, `ess_command_stream_hash`.
See [the skeleton contract](../ESS_STORED_PROGRAM_SKELETON.md).

Planned v173.1–v173.5 artifacts:

| Artifact | Role |
|---|---|
| `ess_program_store_manifest.json` | Version and hash identity of stored switch logic |
| `ess_translation_table_manifest.json` | Immutable routing and feature translation data |
| `ess_event_schedule_receipt.json` | Logical-tick ordering and fixed-priority scheduling evidence |
| `ess_call_processing_receipt.json` | Complete call-processing state-machine execution |
| `ess_feature_module_manifest.json` | Bounded optional feature modules and authority limits |
| `ess_fabric_equivalence_receipt.json` | Equivalence of programme command streams across fabric adapters |

Primary hashes:

```text
ess_program_store_manifest_hash
ess_call_processing_receipt_hash
```

## v174.x — Digital time-division artifacts

| Artifact | Role |
|---|---|
| `digital_frame_schema.json` | Fixed-width frame, slot, byte-order and idle-symbol contract |
| `timeslot_allocation_receipt.json` | Stable allocation, release and collision decisions |
| `frame_sync_receipt.json` | Missing, duplicate, reordered and slipped frame evidence |
| `timeslot_interchange_receipt.json` | Canonical input-slot to output-slot mapping and memory state |
| `digital_fault_battery.json` | Reproducible corruption, loss, collision and alignment cases |
| `digital_switch_report.json` | Validated summary of digital-switch observations |

Primary hashes:

```text
digital_frame_switch_receipt_hash
deterministic_timeslot_allocation_receipt_hash
```

## v175.x — Packet and softswitch artifacts

| Artifact | Role |
|---|---|
| `canonical_packet_schema.json` | Versioned packet envelope, payload hash and sequence identity |
| `route_table_manifest.json` | Canonically ordered routes and exact lookup semantics |
| `queue_policy_manifest.json` | Bounded capacity, priority, overflow and scheduling rules |
| `packet_hop_receipt_chain.json` | Every transformation, queue decision and forwarding action |
| `packet_failover_receipt.json` | Deterministic alternate-route and failure-recovery evidence |
| `softswitch_adapter_receipt.json` | External-protocol observations kept outside proof authority |
| `packet_fabric_report.json` | Validated end-to-end packet-fabric summary |

Primary hashes:

```text
canonical_packet_fabric_receipt_hash
deterministic_failover_receipt_hash
```

## v176.x — Cross-era artifacts

| Artifact | Role |
|---|---|
| `switch_compatibility_corpus_manifest.json` | Shared requests, topologies, contention and fault fixtures |
| `cross_era_replay_receipt.json` | Execution of the shared corpus across every switch model |
| `switch_migration_contract.json` | Canonical translation of routes, capacities and policies |
| `cross_era_equivalence_receipt.json` | Destination, payload, outcome and failure-class equivalence |
| `architecture_selection_policy.json` | Deterministic ranking from declared requirements |
| `switching_evolution_report.json` | Reproducible architecture comparison without hardware claims |

Primary hashes:

```text
cross_era_switch_equivalence_receipt_hash
switch_migration_receipt_hash
```

## Registry rules

- every artifact must declare a schema and version;
- canonical JSON ordering and encoding must be explicit;
- receipts recompute identities rather than trusting embedded hashes;
- an upstream binding may be claimed only when the full upstream hash is part of the validated downstream schema;
- reports are derived views and cannot override source evidence;
- renamed or superseded artifacts require an explicit migration contract;
- browser-only demonstration exports must not reuse canonical receipt schemas;
- published release artifacts and schemas override roadmap planning names.
