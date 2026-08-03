<!-- SPDX-License-Identifier: MPL-2.0 -->
# QSOLKCB / QEC — ROADMAP

## Deterministic switching • canonical routing • replay-safe proof artifacts • bounded operator control • cross-era equivalence

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

Published tags are authoritative.

If this roadmap disagrees with published release history:

- published release history wins;
- the roadmap must be corrected;
- no future release may rewrite the meaning of an existing receipt.

Stable compatibility remains anchored to the established QEC canonical-identity,
SHA-256, replay, claim-boundary, and decoder-governance contracts.

---

# 🧠 Core Identity

QEC is a deterministic, replay-safe proof system for:

- quantum error-correction research;
- exact and sampled finite-code evidence;
- canonical JSON and SHA-256 proof artifacts;
- decoder governance and immutable decoder boundaries;
- deterministic routing and switching;
- bounded operator control;
- external-adapter receipts;
- cross-environment and cross-architecture replay;
- source-bound scientific claims;
- reproducible reports and replication evidence.

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

Violation:

```text
SYSTEM INVALID
```

Every roadmap item must produce:

- a contract;
- a canonical artifact;
- a SHA-256 identity;
- a validation rule;
- an explicit failure mode;
- a deterministic replay test;
- a claim boundary.

If an idea cannot produce those, it remains inspiration rather than QEC.

---

# ☎️ Telecommunications Lineage as an Engineering Programme

The post-v170 roadmap follows the development of automatic telecommunications
switching as an architectural sequence:

```text
Strowger step-by-step
→ Panel separated control
→ Crossbar coordinate switching
→ ESS stored-program control
→ Digital time-division switching
→ Packet and softswitch fabrics
→ cross-era replay and migration proof
```

This sequence is a design grammar, not a claim that QEC is telephone hardware.
It is also not a claim that every historical network migrated through one
identical sequence. The Bell System lineage is used because each era introduced
a useful deterministic systems idea that can be expressed as a QEC contract.

| Historical era | Systems idea | QEC interpretation |
|---|---|---|
| Strowger | Direct step-by-step route progression | Explicit state transitions, first-free selection, pulse decoding and route receipts |
| Panel | Sender/register separates control from the talking path | Route intent is compiled before path actuation; control and payload paths receive separate identities |
| Crossbar | Coordinate matrix with common control | Central deterministic path computation over a bounded switching matrix |
| ESS | Stored-program control | Versioned immutable switch programmes, deterministic scheduling and software-defined features |
| Digital circuit switching | Time-division multiplexing | Canonical frames, fixed time slots, frame alignment and deterministic multiplexing |
| Packet switching / softswitch | Software-routed packet envelopes | Canonical packets, deterministic route tables, bounded queues, replay-safe failover and adapter-only networking |

## Historical claim boundary

The roadmap may describe historical design concepts, but implementation receipts
prove only QEC software behaviour under declared models.

No roadmap release may claim:

- physical telephone-network fidelity;
- quantum hardware behaviour;
- hardware fault tolerance;
- real-world carrier reliability;
- network latency guarantees;
- universal superiority of one switching architecture;
- that a historical analogy validates decoder correctness.

The decoder remains responsible for mathematical correction validity. Switching
layers route declared requests and preserve evidence; they do not invent truth.

---

# ✅ Completed Foundation

## v166.x — Decoder Governance

Completed contracts include:

- canonical decoder baseline;
- candidate manifests;
- replay equivalence;
- optimization contracts;
- fast-path equivalence;
- implementation boundaries;
- benchmark ladders;
- rollback readiness;
- promotion receipts.

Governance approval is not silent runtime replacement.

## v167.x — Symbolic Sonification Runtime

Completed work established deterministic symbolic event schemas, mapping packs,
ternary/fuzzy music state, graph mapping, φ/Fibonacci rhythm and pitch, MIDI-like
export, lyric prompt compilation, proof telemetry sonification, CLI fixtures and
demo benchmarks.

Sonification remains an interpretation surface, not proof authority.

## v170.x — Exact Ququart Evidence, NEXUS Replication and Strowger Routing

The v170 line established:

- the exact packed `[[5,1,3]]_4` ququart decoder and bounded correction model;
- exact finite FER enumeration and deterministic Monte Carlo batteries;
- harmonic receiver fault injection and disjoint receiver/decoder telemetry;
- replication receipts and report-claim validation;
- NEXUS execution and replication evidence boundaries;
- the deterministic Strowger Syndrome Exchange;
- mixed-radix pulse routing, selector stages, first-free trunk hunting and
  vertical/rotary connector resolution;
- route, check and state-dark tone verification;
- canonical Operator Desk events;
- replayable pre-route state and hash-chained route receipts;
- strict separation between browser demonstrations and canonical evidence.

The Strowger exchange is now the baseline switching implementation against which
later architectures will be compared.

---

# 🔗 Switching Proof Chain

```text
canonical_decoder_baseline_receipt_hash
→ ququart_exact_fer_manifest_hash
→ replication_receipt_hash
→ report_claim_validation_hash
→ nexus_replication_receipt_hash
→ strowger_route_receipt_hash                         (v170.3.0)
→ panel_sender_register_receipt_hash                  (v171.x)
→ panel_separated_control_receipt_hash                (v171.x)
→ crossbar_matrix_receipt_hash                        (v172.x)
→ crossbar_common_control_receipt_hash                (v172.x)
→ ess_program_store_manifest_hash                     (v173.x)
→ ess_call_processing_receipt_hash                    (v173.x)
→ digital_frame_switch_receipt_hash                   (v174.x)
→ deterministic_timeslot_allocation_receipt_hash      (v174.x)
→ canonical_packet_fabric_receipt_hash                (v175.x)
→ deterministic_failover_receipt_hash                 (v175.x)
→ cross_era_switch_equivalence_receipt_hash           (v176.x)
→ switch_migration_receipt_hash                       (v176.x)
→ qec_os_runtime_contract_hash                        (deferred v193.x)
```

Each arrow extends lineage. It does not erase the preceding architecture.

---

# Phase v171.x — Panel Separated-Control Exchange

**Status:** PLANNED

## Motivation

Panel switching introduced a useful separation between collected dialled intent
and the machinery that established the path. QEC will translate that idea into
a sender/register architecture where routing logic is compiled and sealed before
switch actuation.

## Architectural rule

```text
request intake
→ canonical digit register
→ sender route programme
→ motor/path actuation plan
→ independent route verification
→ commit or fail closed
```

The control path and transported correction request must have separate hashes.
Neither may silently mutate the other.

## Planned releases

- **v171.0 — Panel Exchange Skeleton**  
  Canonical panel topology, motor-group abstraction, banks and bounded selector movement.
- **v171.1 — Sender and Register Contract**  
  Capture the complete request before route actuation; emit a sender programme receipt.
- **v171.2 — Separated Control-Path Receipt**  
  Bind control intent, payload identity and selected route without conflating them.
- **v171.3 — Deterministic Panel Translation Tables**  
  Versioned digit-to-route translation with exact ordering and no ambient lookup.
- **v171.4 — Panel Fault and Capacity Battery**  
  Busy banks, motor stalls, translation corruption, sender disagreement and fail-closed recovery.
- **v171.5 — Offline Panel Laboratory**  
  Dependency-free visualisation and sonification clearly marked as demonstration evidence.

## Expected artifacts

```text
panel_topology.json
panel_digit_register.json
panel_sender_program.json
panel_route_receipt.json
panel_fault_battery.json
panel_claim_validation.json
```

## Acceptance gates

- same request and translation table produce the same sender programme;
- route actuation cannot begin before sender sealing;
- payload bytes remain unchanged across route compilation;
- a changed translation table changes the programme hash;
- motor/path faults are explicit and reproducible;
- no fallback route is chosen without a declared deterministic rule;
- Strowger and Panel may differ in event sequence while preserving declared
  destination and outcome equivalence.

---

# Phase v172.x — Crossbar Coordinate Matrix and Common Control

**Status:** PLANNED

## Motivation

Crossbar systems replace sequential wiper movement with coordinate selection over
horizontal and vertical bars. Common control computes a path, establishes it,
and becomes available for the next request.

QEC will use this model to introduce a bounded coordinate switching matrix and a
central marker that has no authority beyond the declared route contract.

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
  Central route computation with bounded authority and complete input/output receipts.
- **v172.2 — Multi-Stage Link Selection**  
  Deterministic look-ahead over declared stages and first admissible complete path.
- **v172.3 — Contention and Busy-Link Receipts**  
  Exact tie-breaking, reservation, release, quarantine and replay-safe contention tests.
- **v172.4 — Crossbar Continuity Verification**  
  End-to-end proof that every selected coordinate belongs to one continuous route.
- **v172.5 — Strowger/Panel/Crossbar Equivalence Battery**  
  Shared corpus proving declared route/outcome equivalence without requiring identical traces.

## Expected artifacts

```text
crossbar_matrix_manifest.json
crossbar_marker_program.json
crossbar_path_search_receipt.json
crossbar_contention_receipt.json
crossbar_continuity_receipt.json
switch_equivalence_matrix.json
```

## Acceptance gates

- matrix ordering is canonical;
- path search is bounded and deterministic;
- tie-breaking never depends on set or dictionary iteration order;
- a marker cannot alter decoder output or payload identity;
- reservations and releases are included in the event chain;
- no partial path may be committed;
- equivalent outcomes across switch types are machine validated.

---

# Phase v173.x — Electronic Switching System / Stored-Program Control

**Status:** PLANNED

## Motivation

Electronic Switching Systems moved call control from hardwired relay logic into
stored programmes while early networks could still use physical switching
matrices. QEC will separate a versioned switch programme from its switching
fabric and require every programme, table and feature module to be hash-bound.

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
  Full source/version/hash identity for executable routing logic and translation tables.
- **v173.2 — Deterministic Call-Processing State Machine**  
  Exact transitions for intake, analyse, select, connect, verify, release and reject.
- **v173.3 — Interrupt and Timing Schedule Receipt**  
  Logical ticks and fixed priority classes replace wall-clock-dependent scheduling.
- **v173.4 — Feature Module Boundary**  
  Optional forwarding, retry, conference-like fan-out and policy modules remain bounded,
  explicit and unable to override validation.
- **v173.5 — ESS/Fabric Equivalence Receipt**  
  Prove that the stored programme issues a deterministic command stream to each
  supported fabric adapter.

## Expected artifacts

```text
ess_program_store_manifest.json
ess_translation_table_manifest.json
ess_event_schedule_receipt.json
ess_call_processing_receipt.json
ess_feature_module_manifest.json
ess_fabric_equivalence_receipt.json
```

## Acceptance gates

- programme and data identities are separate;
- modifying a programme or translation table changes the execution identity;
- scheduling uses logical time only;
- feature modules cannot force acceptance;
- external or historical hardware models remain adapters, never authorities;
- the same programme/input corpus produces byte-identical command streams.

---

# Phase v174.x — Deterministic Digital Time-Division Switching

**Status:** PLANNED

## Motivation

Digital switching replaces a dedicated physical path with framed symbols assigned
to time slots. QEC will model deterministic multiplexing without claiming to
simulate a carrier-grade voice network.

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
  Offline frame visualisation and optional sonification separated from canonical evidence.

## Expected artifacts

```text
digital_frame_schema.json
timeslot_allocation_receipt.json
frame_sync_receipt.json
timeslot_interchange_receipt.json
digital_fault_battery.json
digital_switch_report.json
```

## Acceptance gates

- frame width and byte order are explicit;
- slot allocation is deterministic under saturation;
- frame sequence uses logical indices rather than wall-clock timestamps;
- idle, missing and invalid symbols are distinct;
- reassembly reproduces the original payload identity or rejects;
- no frame-level result is described as physical quantum evidence.

---

# Phase v175.x — Canonical Packet Fabric and Softswitch Boundary

**Status:** PLANNED

## Motivation

Packet switching replaces fixed circuits with independently routed envelopes.
Modern softswitches and VoIP systems move increasing control into software, but
QEC must not import ordinary network nondeterminism into its proof core.

The packet fabric will therefore be an offline deterministic model first. Any
real network integration remains an adapter with separately captured evidence.

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
  Versioned header schema, payload hash, sequence identity and explicit provenance.
- **v175.1 — Deterministic Route Table**  
  Canonically ordered routes, exact longest-match semantics and stable tie-breaking.
- **v175.2 — Bounded Queue and Scheduling Contract**  
  Fixed capacity, declared priority policy, explicit drops and no ambient timing.
- **v175.3 — Hop-by-Hop Receipt Chain**  
  Every transformation, queue decision and forwarding action extends packet lineage.
- **v175.4 — Replay-Safe Failover**  
  Declared alternate routes, fixed failure injection and deterministic recovery.
- **v175.5 — Softswitch/VoIP Adapter Boundary**  
  Real protocols may be observed through adapters, but live networks cannot become proof authority.
- **v175.6 — Packet Fabric Evidence Lab**  
  Offline topology, queue, route and fault visualisation with exportable demonstration records.

## Expected artifacts

```text
canonical_packet_schema.json
route_table_manifest.json
queue_policy_manifest.json
packet_hop_receipt_chain.json
packet_failover_receipt.json
softswitch_adapter_receipt.json
packet_fabric_report.json
```

## Acceptance gates

- packet identity excludes mutable transport metadata unless explicitly declared;
- routing and queue ordering are canonical;
- queue overflow has one declared outcome;
- failover never uses random route selection;
- every accepted reassembly recomputes the original payload hash;
- live network observations are labelled external and non-authoritative;
- no cloud service is required for core tests.

---

# Phase v176.x — Cross-Era Equivalence, Migration and Architecture Selection

**Status:** PLANNED

## Motivation

The value of the switching lineage is not five unrelated demonstrations. QEC must
prove what remains invariant while the switching substrate changes.

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

Byte-identical traces are not required across architectures. Declared semantic
and proof-boundary equivalence is required.

## Planned releases

- **v176.0 — Historical Switch Compatibility Corpus**  
  Shared request, topology, contention and fault fixtures for every switching era.
- **v176.1 — Cross-Era Replay Harness**  
  Execute the corpus against Strowger, Panel, Crossbar, ESS, Digital and Packet models.
- **v176.2 — Switch Migration Contract**  
  Canonical translation of routes, capacities, policies and claim boundaries between eras.
- **v176.3 — Cross-Era Equivalence Receipt**  
  Machine-validated destination, payload, outcome and failure-class equivalence.
- **v176.4 — Deterministic Architecture Selection Policy**  
  Select an architecture from declared requirements using a canonical ranking tuple;
  never infer that newer automatically means better.
- **v176.5 — Switching Evolution Report**  
  Reproducible report comparing complexity, capacity model, fault surfaces, event counts
  and receipt size without making hardware-performance claims.

## Expected artifacts

```text
switch_compatibility_corpus_manifest.json
cross_era_replay_receipt.json
switch_migration_contract.json
cross_era_equivalence_receipt.json
architecture_selection_policy.json
switching_evolution_report.json
```

## Acceptance gates

- all architectures consume one canonical request schema;
- migration is explicit and hash-bound;
- outcome equivalence is machine checked;
- architecture-specific failures remain visible rather than normalized away;
- comparison metrics are observations, not correctness proofs;
- architecture selection uses declared requirements and deterministic ordering;
- no historical switch is deleted after a newer model lands.

---

# ⏸️ Deferred Phase v193.x — QEC OS Runtime & Benchmark Reset

The v193.x runtime programme remains deferred.

The telecommunications switching lineage prepares reusable routing, scheduling,
queueing, failover, migration and equivalence contracts for that future work. It
does not activate a QEC operating system, silently replace the decoder, or turn
historical switch models into production network infrastructure.

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

# 🧪 Global Acceptance Law

Every switching-era release must pass:

```text
python -m pytest -q
```

A focused suite may be used during development, but the full repository suite is
the release gate when routing, receipts, canonicalisation, claims, operator state,
shared schemas or proof-chain identities change.

Required release evidence:

- source commit SHA;
- package version;
- environment manifest;
- canonical fixtures;
- deterministic test receipt;
- generated artifact manifest;
- full 64-character SHA-256 identities;
- claim-validation result;
- clean working-tree declaration;
- explicit limitations.

## Global forbidden behaviours

- randomness without a declared deterministic seed and receipt;
- wall-clock-dependent outcomes;
- hidden mutable state;
- unordered traversal affecting results;
- silent route fallback;
- force-accept operator commands;
- external adapters acting as authorities;
- browser demonstration digests presented as cryptographic evidence;
- decoder mutation by a switching layer;
- hardware, threshold or quantum-advantage claims from routing simulations;
- deleting historical implementations when a later architecture is added.

---

# 📚 Historical Design References

These sources motivate the architectural sequence; they are not QEC proof inputs.

- *A History of Engineering and Science in the Bell System: Switching Technology* — Panel, coordinate and common-control development.
- Western Electric, *Fundamentals of Telephone Communication Systems* — Panel senders and crossbar common control.
- Bell System Technical Journal, *Organization of the No. 1 ESS Stored Program* — stored-program switching control.
- ITU historical material on digital circuit-switched data networks, X.25 and the transition toward packet networks.

Any historical fact used in an implementation document should be cited there and
kept separate from canonical benchmark evidence.

---

# Final Direction

QEC's switching programme is not nostalgia pasted onto a decoder.

It is a deterministic systems experiment in architectural evolution:

```text
movement becomes compiled control
→ compiled control becomes coordinate selection
→ coordinate selection becomes stored programme
→ stored programme becomes framed digital time
→ framed time becomes canonical packets
→ every migration remains replayable and provable
```

The design objective is not to imitate an old exchange room.

The objective is to prove that QEC can change its switching substrate without
losing payload identity, failure semantics, claim discipline or replay truth.
