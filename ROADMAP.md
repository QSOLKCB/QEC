<!-- SPDX-License-Identifier: MPL-2.0 -->
# QSOLKCB / QEC — ROADMAP

## Deterministic switching • canonical routing • replay-safe proof artifacts • cross-era equivalence

---

# 🧭 Stable Tip Metadata

```text
latest completed release → v170.3.0
stable commit            → dada8b7a20a75753db43acc01a6a9e723ebaa6b6
current frontier         → v171.0
next work                → PanelSeparatedControlExchange
active programme         → v171.x–v176.x — Deterministic Telecommunications Switching Lineage
completed baseline       → v170.3.0 — Deterministic Strowger Syndrome Exchange
deferred runtime arc     → v193.x — QEC OS Runtime & Benchmark Reset
```

Published tags are authoritative. If this roadmap conflicts with published
release history, release history wins and the roadmap must be corrected.

Stable compatibility remains explicitly anchored to the `v137.*` contract
lineage. Later roadmap programmes may extend those contracts, but must not
silently reinterpret or remove them.

The pre-v170 planning document is preserved through
[`docs/archive/ROADMAP_PRE_V170.md`](docs/archive/ROADMAP_PRE_V170.md).
The former extension documents are retained as superseded historical pointers:
[`ROADMAP_EXTENSION.md`](ROADMAP_EXTENSION.md) and
[`ROADMAP_EXTENSION_v2.md`](ROADMAP_EXTENSION_v2.md).

## Companion documents

| Document | Purpose |
|---|---|
| [Switching Safety and Acceptance Rules](docs/roadmap/SWITCHING_SAFETY_AND_ACCEPTANCE.md) | Normative deterministic law, release gates, forbidden behaviours, claim boundaries and phase acceptance criteria |
| [Switching Artifact Registry](docs/roadmap/SWITCHING_ARTIFACT_REGISTRY.md) | Central inventory of planned canonical artifacts, receipts and programme-lineage roles |
| [Telecommunications Switching Design References](docs/roadmap/TELECOM_SWITCHING_REFERENCES.md) | Historical architecture context and citation boundaries |

---

# 🧠 Programme Law

QEC does not make the physical world deterministic.

QEC makes the declared boundary deterministic.

```text
same declared input
+ same topology
+ same switch programme
+ same initial state
+ same operator command sequence
+ same fault plan
→ same ordered event stream
→ same canonical JSON
→ same SHA-256 identity
→ same validated outcome
```

Every phase must produce versioned contracts, canonical artifacts, validation,
replay tests, explicit failures and claim boundaries. The complete normative
rules live in the
[Switching Safety and Acceptance Rules](docs/roadmap/SWITCHING_SAFETY_AND_ACCEPTANCE.md).

---

# ☎️ Telecommunications Switching Lineage

```text
Strowger step-by-step
→ Panel separated control
→ Crossbar coordinate switching
→ ESS stored-program control
→ Digital time-division switching
→ Packet and softswitch fabrics
→ cross-era replay and migration proof
```

The sequence is a design grammar rather than a claim that QEC is telephone
hardware. Each era contributes a deterministic systems idea that can be
expressed as a QEC contract.

Historical context and source boundaries are maintained separately in
[Telecommunications Switching Design References](docs/roadmap/TELECOM_SWITCHING_REFERENCES.md).

---

# ✅ Completed Foundation

## v166.x — Decoder Governance

Completed contracts established the canonical decoder baseline, candidate
manifests, replay and fast-path equivalence, implementation boundaries,
benchmark ladders, rollback readiness and promotion receipts.

Governance approval is not silent runtime replacement.

## v167.0 — Symbolic Sonification Runtime Skeleton

Only the v167.0 deterministic symbolic-event runtime skeleton is implemented.
The former v167.1–v167.9 mapping-pack, graph-mapping, MIDI-export, prompt,
telemetry, CLI and demo-benchmark assignments remain unimplemented historical
planning boundaries. They are not completed capabilities and are not the active
frontier. Any future sonification continuation must receive non-conflicting
release numbers and preserve the v167.0 schema boundary.

Sonification remains an interpretation surface rather than proof authority.

## v170.x — Exact Ququart Evidence, NEXUS Replication and Strowger Routing

The v170 line established:

- the exact packed `[[5,1,3]]_4` ququart decoder and bounded correction model;
- exact finite FER enumeration and deterministic Monte Carlo batteries;
- harmonic receiver fault injection and disjoint receiver/decoder telemetry;
- replication receipts and report-claim validation;
- NEXUS execution and replication evidence boundaries;
- the deterministic Strowger Syndrome Exchange;
- mixed-radix pulse routing and first-free trunk hunting;
- route, check and state-dark tone verification;
- canonical Operator Desk events;
- replayable pre-route state and hash-chained route receipts;
- strict separation between browser demonstrations and canonical evidence.

The Strowger exchange is the baseline implementation against which later
switching architectures will be compared.

---

# 🔗 Switching Programme Artifact Chronology

The entries below are a chronological registry of released and planned hash
identities. They are not a transitive cryptographic proof chain. Existing v170
artifact families validate their own declared inputs, but do not automatically
bind the preceding entry unless that upstream hash is explicitly present in the
artifact schema. Explicit cross-era binding is a planned v176.x deliverable.

```text
RELEASED
  canonical_decoder_baseline_receipt_hash             (v166.0)
  ququart_exact_fer_manifest_hash                      (v170.1.x)
  replication_receipt_hash                            (v170.1.1)
  report_claim_validation_hash                        (v170.1.1)
  nexus_replication_receipt_hash                       (v170.2.x)
  strowger_route_receipt_hash                          (v170.3.0)

PLANNED
  panel_sender_register_receipt_hash                   (v171.x)
  panel_separated_control_receipt_hash                 (v171.x)
  crossbar_matrix_receipt_hash                         (v172.x)
  crossbar_common_control_receipt_hash                 (v172.x)
  ess_program_store_manifest_hash                      (v173.x)
  ess_call_processing_receipt_hash                     (v173.x)
  digital_frame_switch_receipt_hash                    (v174.x)
  deterministic_timeslot_allocation_receipt_hash       (v174.x)
  canonical_packet_fabric_receipt_hash                 (v175.x)
  deterministic_failover_receipt_hash                  (v175.x)
  cross_era_switch_equivalence_receipt_hash            (v176.x)
  switch_migration_receipt_hash                        (v176.x)
  qec_os_runtime_contract_hash                         (deferred v193.x)
```

## Artifact overview

| Phase | Primary model | Primary evidence | Programme-lineage role |
|---|---|---|---|
| v170.3.0 | Strowger route topology and pre-route state | Canonical Strowger route receipt | Mechanical step-by-step baseline |
| v171.x | `panel_topology.json`, `panel_sender_program.json` | `panel_route_receipt.json` | Separate compiled control intent from payload and actuation |
| v172.x | `crossbar_matrix_manifest.json` | `crossbar_continuity_receipt.json` | Prove bounded coordinate selection and complete-path continuity |
| v173.x | `ess_program_store_manifest.json` | `ess_call_processing_receipt.json` | Bind immutable programmes, logical scheduling and fabric commands |
| v174.x | `digital_frame_schema.json` | `timeslot_interchange_receipt.json` | Prove frames, deterministic slot allocation and reassembly |
| v175.x | `canonical_packet_schema.json`, `route_table_manifest.json` | `packet_hop_receipt_chain.json` | Preserve payload identity through queues, hops and failover |
| v176.x | `switch_migration_contract.json` | `cross_era_equivalence_receipt.json` | Explicitly bind declared invariants across architecture migration |

The complete artifact inventory and role definitions live in the
[Switching Artifact Registry](docs/roadmap/SWITCHING_ARTIFACT_REGISTRY.md).

---

# Phase v171.x — Panel Separated-Control Exchange

**Status:** PLANNED

## Goal

Translate the Panel sender/register concept into a QEC architecture where the
complete request is collected, compiled and sealed before path actuation.
Control intent and transported payload receive separate canonical identities.

## Architectural rule

```text
request intake
→ canonical digit register
→ sender route programme
→ motor/path actuation plan
→ independent route verification
→ commit or fail closed
```

## Planned releases

- **v171.0 — Panel Exchange Skeleton**  
  Canonical panel topology, motor-group abstraction, banks and bounded selector movement.
- **v171.1 — Sender and Register Contract**  
  Capture the complete request before route actuation and emit a sender programme receipt.
- **v171.2 — Separated Control-Path Receipt**  
  Bind control intent, payload identity and selected route without conflating them.
- **v171.3 — Deterministic Panel Translation Tables**  
  Versioned digit-to-route translation with exact ordering and no ambient lookup.
- **v171.4 — Panel Fault and Capacity Battery**  
  Busy banks, motor stalls, translation corruption, sender disagreement and fail-closed recovery.
- **v171.5 — Offline Panel Laboratory**  
  Dependency-free visualisation and sonification labelled as demonstration evidence.

Evidence names are registered under
[v171.x Panel artifacts](docs/roadmap/SWITCHING_ARTIFACT_REGISTRY.md#v171x--panel-separated-control-artifacts).
Acceptance criteria are maintained under
[v171.x Panel separated control](docs/roadmap/SWITCHING_SAFETY_AND_ACCEPTANCE.md#v171x--panel-separated-control).

---

# Phase v172.x — Crossbar Coordinate Matrix and Common Control

**Status:** PLANNED

## Goal

Replace sequential route progression with bounded coordinate selection over an
immutable horizontal/vertical matrix. A central marker computes and reserves a
complete admissible path, but has no authority beyond the route contract.

## Architectural rule

```text
canonical request
→ marker input register
→ deterministic path search
→ coordinate closure plan
→ path continuity verification
→ connection receipt
→ marker release
```

## Planned releases

- **v172.0 — Crossbar Matrix Core**  
  Immutable coordinate matrix, link states and canonical intersection identity.
- **v172.1 — Marker/Common-Control Contract**  
  Central route computation with bounded authority and complete receipts.
- **v172.2 — Multi-Stage Link Selection**  
  Deterministic look-ahead and first admissible complete path.
- **v172.3 — Contention and Busy-Link Receipts**  
  Exact tie-breaking, reservation, release, quarantine and replay-safe contention tests.
- **v172.4 — Crossbar Continuity Verification**  
  Prove that every selected coordinate belongs to one continuous route.
- **v172.5 — Strowger/Panel/Crossbar Equivalence Battery**  
  Shared corpus proving declared route and outcome equivalence without identical traces.

Evidence names are registered under
[v172.x Crossbar artifacts](docs/roadmap/SWITCHING_ARTIFACT_REGISTRY.md#v172x--crossbar-artifacts).
Acceptance criteria are maintained under
[v172.x Crossbar coordinate switching](docs/roadmap/SWITCHING_SAFETY_AND_ACCEPTANCE.md#v172x--crossbar-coordinate-switching).

---

# Phase v173.x — Electronic Switching System / Stored-Program Control

**Status:** PLANNED

## Goal

Separate versioned stored-program control from the switching fabric. Every
programme, translation table, schedule and optional feature module is hash-bound
and unable to override validation.

## Architectural rule

```text
immutable programme store
+ immutable translation tables
+ canonical input event queue
+ deterministic scheduler
→ call-processing state machine
→ fabric command stream
→ validated switching receipt
```

## Planned releases

- **v173.0 — Stored-Program Switch Skeleton**  
  Programme store, call store, event queue and switching-fabric adapter boundary.
- **v173.1 — Immutable Programme Store Manifest**  
  Full source, version and hash identity for routing logic and translation tables.
- **v173.2 — Deterministic Call-Processing State Machine**  
  Exact intake, analyse, select, connect, verify, release and reject transitions.
- **v173.3 — Interrupt and Timing Schedule Receipt**  
  Logical ticks and fixed priority classes replace wall-clock scheduling.
- **v173.4 — Feature Module Boundary**  
  Optional forwarding, retry and fan-out modules remain bounded and explicit.
- **v173.5 — ESS/Fabric Equivalence Receipt**  
  Prove deterministic command streams across supported fabric adapters.

Evidence names are registered under
[v173.x ESS artifacts](docs/roadmap/SWITCHING_ARTIFACT_REGISTRY.md#v173x--ess-stored-program-artifacts).
Acceptance criteria are maintained under
[v173.x ESS stored-program control](docs/roadmap/SWITCHING_SAFETY_AND_ACCEPTANCE.md#v173x--ess-stored-program-control).

---

# Phase v174.x — Deterministic Digital Time-Division Switching

**Status:** PLANNED

## Goal

Replace dedicated path state with canonical fixed-width frames assigned to
logical time slots. Missing, duplicated, reordered and shifted frames become
explicit fail-closed evidence.

## Architectural rule

```text
canonical correction symbols
→ fixed-width frame encoder
→ deterministic time-slot allocator
→ frame alignment and continuity checks
→ time-slot interchange
→ frame receipt
```

## Planned releases

- **v174.0 — Digital Frame Switch Core**  
  Fixed-width frames, canonical slot numbering and explicit idle symbols.
- **v174.1 — Deterministic Time-Slot Allocator**  
  Stable allocation, release and collision handling under a declared policy.
- **v174.2 — Frame Synchronisation and Slip Receipt**  
  Detect missing, duplicated, reordered or shifted frames and fail closed.
- **v174.3 — Time-Slot Interchange Matrix**  
  Canonical input-slot to output-slot mapping with replayable memory state.
- **v174.4 — Digital Channel Fault Battery**  
  Bit corruption, slot collision, frame loss, duplicate frames and alignment faults.
- **v174.5 — Digital Switch Evidence Lab**  
  Offline frame visualisation and sonification separated from canonical evidence.

Evidence names are registered under
[v174.x Digital artifacts](docs/roadmap/SWITCHING_ARTIFACT_REGISTRY.md#v174x--digital-time-division-artifacts).
Acceptance criteria are maintained under
[v174.x Digital time-division switching](docs/roadmap/SWITCHING_SAFETY_AND_ACCEPTANCE.md#v174x--digital-time-division-switching).

---

# Phase v175.x — Canonical Packet Fabric and Softswitch Boundary

**Status:** PLANNED

## Goal

Move from fixed circuits to independently routed canonical envelopes without
importing ordinary network nondeterminism into the proof core. The core model is
offline and deterministic; live networks remain receipt-bound adapters.

## Architectural rule

```text
canonical payload
→ canonical packet envelope
→ deterministic route-table lookup
→ bounded queue discipline
→ hop receipts
→ destination reassembly
→ end-to-end validation
```

## Planned releases

- **v175.0 — Canonical Packet Envelope**  
  Versioned header schema, payload hash, sequence identity and provenance.
- **v175.1 — Deterministic Route Table**  
  Canonically ordered routes, exact longest-match semantics and stable tie-breaking.
- **v175.2 — Bounded Queue and Scheduling Contract**  
  Fixed capacity, declared priority policy, explicit drops and no ambient timing.
- **v175.3 — Hop-by-Hop Receipt Chain**  
  Every transformation, queue decision and forwarding action extends lineage.
- **v175.4 — Replay-Safe Failover**  
  Declared alternate routes, fixed failure injection and deterministic recovery.
- **v175.5 — Softswitch/VoIP Adapter Boundary**  
  Real protocols may be observed, but live networks cannot become proof authority.
- **v175.6 — Packet Fabric Evidence Lab**  
  Offline topology, queue, route and fault visualisation with demonstration exports.

Evidence names are registered under
[v175.x Packet artifacts](docs/roadmap/SWITCHING_ARTIFACT_REGISTRY.md#v175x--packet-and-softswitch-artifacts).
Acceptance criteria are maintained under
[v175.x Canonical packet fabric](docs/roadmap/SWITCHING_SAFETY_AND_ACCEPTANCE.md#v175x--canonical-packet-fabric).

---

# Phase v176.x — Cross-Era Equivalence, Migration and Architecture Selection

**Status:** PLANNED

## Goal

Prove which payload, destination, outcome and failure invariants survive when the
switching substrate changes. Architecture-specific traces remain visible; only
declared semantic and proof-boundary equivalence is required.

## Equivalence law

```text
same canonical correction request
+ era-specific topology derived from one migration contract
+ declared equivalent availability and fault assumptions
→ same semantic destination
→ same accepted/rejected outcome class
→ same payload identity
→ architecture-specific event streams
→ one cross-era equivalence receipt
```

## Planned releases

- **v176.0 — Historical Switch Compatibility Corpus**  
  Shared request, topology, contention and fault fixtures for every switching era.
- **v176.1 — Cross-Era Replay Harness**  
  Execute the corpus against Strowger, Panel, Crossbar, ESS, Digital and Packet models.
- **v176.2 — Switch Migration Contract**  
  Canonical translation of routes, capacities, policies and claim boundaries.
- **v176.3 — Cross-Era Equivalence Receipt**  
  Machine-validated destination, payload, outcome and failure-class equivalence.
- **v176.4 — Deterministic Architecture Selection Policy**  
  Rank architectures from declared requirements without assuming newer is better.
- **v176.5 — Switching Evolution Report**  
  Compare complexity, capacity model, fault surfaces, event counts and receipt size
  without making hardware-performance claims.

Evidence names are registered under
[v176.x Cross-era artifacts](docs/roadmap/SWITCHING_ARTIFACT_REGISTRY.md#v176x--cross-era-artifacts).
Acceptance criteria are maintained under
[v176.x Cross-era equivalence and migration](docs/roadmap/SWITCHING_SAFETY_AND_ACCEPTANCE.md#v176x--cross-era-equivalence-and-migration).

---

# ⏸️ Deferred Phase v193.x — QEC OS Runtime & Benchmark Reset

The v193.x runtime programme remains deferred.

The switching lineage prepares reusable routing, scheduling, queueing, failover,
migration and equivalence contracts. It does not activate a QEC operating system,
silently replace the decoder or turn historical switch models into production
network infrastructure.

Expected v193 concerns remain:

- explicit runtime APIs;
- golden corpus fixtures;
- decoder router activation under tested governance;
- GF(2)/stabilizer runtime core;
- QLDPC construction and syndrome/noise harnesses;
- benchmark reset and logical error-rate reporting;
- cross-backend differential tests;
- Odin readiness and parity specifications.

Activation requires separate release contracts and cannot be inferred from
completion of v171.x–v176.x.

---

# Final Direction

QEC's switching programme is a deterministic systems experiment in architectural
evolution:

```text
movement becomes compiled control
→ compiled control becomes coordinate selection
→ coordinate selection becomes stored programme
→ stored programme becomes framed digital time
→ framed time becomes canonical packets
→ every migration remains replayable and provable
```

The objective is to change the switching substrate without losing payload
identity, failure semantics, claim discipline or replay truth.
