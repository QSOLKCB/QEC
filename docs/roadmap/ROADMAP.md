<!-- SPDX-License-Identifier: MPL-2.0 -->
# QSOLKCB / QEC — ROADMAP

## Switching and evidence foundations → acoustic/phononic quantum-information protection

---

# 🧭 Stable Tip Metadata

```text
latest published combined tag → v176.5 (14 October 2026)
immutable switching baseline → c514a4ecd2a9bd815558010e1b3f7821f3bce588
package metadata at tag     → 173.0.0 (independent of combined tag version)
completed programme         → v171.x–v176.x telecommunications switching lineage
next programme              → v177.x acoustic syndrome observation and recovery
long-term research mission  → v178.x–v182.x quantum/phononic protection
next work                   → v177.0 source-bound acoustic channel contract
```

Published tags are authoritative. If this roadmap conflicts with published
release history, release history wins and the roadmap must be corrected.

The v173.1–v176.5 labels below identify **implementation/contract milestones**
included in the owner's [combined v176.5 release](https://github.com/QSOLKCB/QEC/releases/tag/v176.5),
not separately published tags. Package metadata at that tag remains `173.0.0`.
The v177.x–v182.x assignments below are **research proposals**, not implemented
features, quantum-device measurements, release guarantees or performance claims.

Stable compatibility remains explicitly anchored to the `v137.*` contract
lineage. Later roadmap programmes may extend those contracts, but must not
silently reinterpret or remove them.

The pre-v170 planning document is preserved through
[`docs/archive/ROADMAP_PRE_V170.md`](../archive/ROADMAP_PRE_V170.md).
The former extension documents are retained as superseded historical pointers:
[`ROADMAP_EXTENSION.md`](../../archive/docs/ROADMAP_EXTENSION.md) and
[`ROADMAP_EXTENSION_v2.md`](../../archive/docs/ROADMAP_EXTENSION_v2.md).

## Companion documents

| Document | Purpose |
|---|---|
| [Switching Safety and Acceptance Rules](SWITCHING_SAFETY_AND_ACCEPTANCE.md) | Normative deterministic law, release gates, forbidden behaviours, claim boundaries and phase acceptance criteria |
| [Switching Artifact Registry](SWITCHING_ARTIFACT_REGISTRY.md) | Central inventory of planned canonical artifacts, receipts and programme-lineage roles |
| [Telecommunications Switching Design References](TELECOM_SWITCHING_REFERENCES.md) | Historical architecture context and citation boundaries |

## Visual overview

[![QEC deterministic switching roadmap mind map](QEC_DETERMINISTIC_SWITCHING_MIND_MAP.jpg)](QEC_DETERMINISTIC_SWITCHING_MIND_MAP.jpg)

*NotebookLM-generated visual overview. `ROADMAP.md` and its companion documents
remain authoritative for release assignments, contracts and acceptance rules.*

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
[Switching Safety and Acceptance Rules](SWITCHING_SAFETY_AND_ACCEPTANCE.md).

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
[Telecommunications Switching Design References](TELECOM_SWITCHING_REFERENCES.md).

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

The preserved symbolic sonification skeleton remains demonstration-only. A
future verified acoustic syndrome receiver or physically justified phononic
controller must establish its **own** separate mathematical/experimental authority.

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
artifact schema. The combined v176.5 migration/replay contracts bind the complete declared
native inputs and resulting six-era evidence explicitly.

```text
RELEASED
  canonical_decoder_baseline_receipt_hash             (v166.0)
  ququart_exact_fer_manifest_hash                      (v170.1.x)
  replication_receipt_hash                            (v170.1.1)
  report_claim_validation_hash                        (v170.1.1)
  nexus_replication_receipt_hash                       (v170.2.x)
  strowger_route_receipt_hash                          (v170.3.0)
  panel_sender_register_receipt_hash                   (v171.x)
  panel_separated_control_receipt_hash                 (v171.x)
  crossbar_matrix_receipt_hash                         (v172.0)
  crossbar_common_control_receipt_hash                 (v172.1)
  crossbar_path_search_receipt_hash                    (v172.2)
  crossbar_contention_receipt_hash                     (v172.3)
  crossbar_continuity_receipt_hash                     (v172.4)

  switch_equivalence_matrix_hash                      (v172.5)
  ess_switch_skeleton_receipt_hash                     (v173.0)

INCLUDED IN COMBINED v176.5 TAG (contract milestones, not individual tags)
  ess_program_store_manifest_hash                      (v173.1–v173.5)
  ess_call_processing_receipt_hash                     (v173.1–v173.5)
  digital_frame_switch_receipt_hash                    (v174.x)
  deterministic_timeslot_allocation_receipt_hash       (v174.x)
  canonical_packet_fabric_receipt_hash                 (v175.x)
  deterministic_failover_receipt_hash                  (v175.x)
  cross_era_switch_equivalence_receipt_hash            (v176.x)
  switch_migration_receipt_hash                        (v176.x)

PROPOSED / NOT IMPLEMENTED OR REGISTERED
  acoustic_channel_contract_hash                       (v177.x)
  harmonic_observation_receipt_hash                    (v177.x)
  acoustic_recovery_trial_receipt_hash                 (v177.x)
  quantum_noise_model_receipt_hash                     (v178.x)
  phononic_interaction_model_receipt_hash              (v179.x)
  logical_protection_comparison_receipt_hash           (v180.x–v182.x)
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
[Switching Artifact Registry](SWITCHING_ARTIFACT_REGISTRY.md).

---

# Phase v171.x — Panel Separated-Control Exchange

**Status:** RELEASED through v171.5.0

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

## Phase milestones

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
[v171.x Panel artifacts](SWITCHING_ARTIFACT_REGISTRY.md#v171x--panel-separated-control-artifacts).
Acceptance criteria are maintained under
[v171.x Panel separated control](SWITCHING_SAFETY_AND_ACCEPTANCE.md#v171x--panel-separated-control).

---

# Phase v172.x — Crossbar Coordinate Matrix and Common Control

**Status:** v172.0–v172.5 RELEASED; v172.x complete

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

## Phase milestones

- **v172.0 — Crossbar Matrix Core** — RELEASED
  Immutable coordinate matrix, link states and canonical intersection identity.
- **v172.1 — Marker/Common-Control Contract** — RELEASED
  Exact single-matrix coordinate planning, sealed request/programme binding,
  explicit rejection, marker release, complete event receipts and replay validation.
  See [the v172.1 contract](../CROSSBAR_COMMON_CONTROL.md).
- **v172.2 — Multi-Stage Link Selection** — RELEASED
  Immutable layered fabrics, bounded reverse reachability and first admissible
  complete path selection with replayable search receipts.
  See [the v172.2 contract](../CROSSBAR_MULTISTAGE_SELECTION.md).
- **v172.3 — Contention and Busy-Link Receipts** — RELEASED
  Exact logical ordering, atomic owned reservations, release, quarantine revocation
  and replay-safe contention evidence. See [the v172.3 contract](../CROSSBAR_CONTENTION.md).
- **v172.4 — Crossbar Continuity Verification** — RELEASED
  Independent forward walks prove complete coordinate/wire continuity over replayed
  selections, with reservation lifecycle scope and trusted source bindings.
  See [the v172.4 contract](../CROSSBAR_CONTINUITY.md).
- **v172.5 — Strowger/Panel/Crossbar Equivalence Battery** — RELEASED
  Shared bounded corpus comparing destination, route admission and selected lane
  with native receipts, continuity evidence and explicit capability differences.
  Thirty-six availability cases agree; five architecture-specific fault controls
  remain different. See [the v172.5 contract](../SWITCH_EQUIVALENCE_BATTERY.md).

Evidence names are registered under
[v172.x Crossbar artifacts](SWITCHING_ARTIFACT_REGISTRY.md#v172x--crossbar-artifacts).
Acceptance criteria are maintained under
[v172.x Crossbar coordinate switching](SWITCHING_SAFETY_AND_ACCEPTANCE.md#v172x--crossbar-coordinate-switching).

---

# Phase v173.x — Electronic Switching System / Stored-Program Control

**Status:** v173.0 separately released; v173.1–v173.5 shipped in combined v176.5.

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

## Phase milestones

- **v173.0 — Stored-Program Switch Skeleton** — RELEASED
  Separate immutable programme/call stores, a bounded logical event queue and
  Crossbar adapter boundary with complete native replay and continuity evidence.
  Independent planning only; no full call lifecycle or resource mutation.
  See [the v173.0 contract](../ESS_STORED_PROGRAM_SKELETON.md).
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

The additive v173.1–v173.5 implementation has source-bound programmes, real native
reservations/releases, logical scheduling, bounded features and explicitly scoped
two-fabric equivalence. See [the lifecycle contract](../ESS_CALL_PROCESSING.md).
It preserves the v173.0 schemas and package metadata. These additive
milestones were reviewed, merged and shipped in the combined v176.5 tag.

Evidence names are registered under
[v173.x ESS artifacts](SWITCHING_ARTIFACT_REGISTRY.md#v173x--ess-stored-program-artifacts).
Acceptance criteria are maintained under
[v173.x ESS stored-program control](SWITCHING_SAFETY_AND_ACCEPTANCE.md#v173x--ess-stored-program-control).

---

# Phase v174.x — Deterministic Digital Time-Division Switching

**Status:** Implemented and shipped within combined v176.5 (not separately tagged).

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

## Phase milestones

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

Native frames, deterministic owned slot allocation/release, synchronization,
interchange, verified payload reassembly and fault evidence are implemented.
The offline lab and SVG/WAV exports are separately labelled demonstrations.
See [the Digital contract](../DIGITAL_TIME_DIVISION_SWITCHING.md). Native
replay and Node interaction tests passed for the combined release; rendered-
browser and acoustic QA were not claimed.

Evidence names are registered under
[v174.x Digital artifacts](SWITCHING_ARTIFACT_REGISTRY.md#v174x--digital-time-division-artifacts).
Acceptance criteria are maintained under
[v174.x Digital time-division switching](SWITCHING_SAFETY_AND_ACCEPTANCE.md#v174x--digital-time-division-switching).

---

# Phase v175.x — Canonical Packet Fabric and Softswitch Boundary

**Status:** Implemented and shipped within combined v176.5 (not separately tagged).

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

## Phase milestones

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

Immutable envelopes, exact binary-prefix routes, bounded priority queues, hop
chains, declared failover and verified reassembly are implemented. Softswitch
observations remain external, offline and non-authoritative. The packet lab is
a separate demonstration. See [the Packet contract](../PACKET_FABRIC_AND_SOFTSWITCH.md).
Native replay and offline Node interaction checks accompanied the combined
release; rendered-browser and acoustic QA were not claimed.

Evidence names are registered under
[v175.x Packet artifacts](SWITCHING_ARTIFACT_REGISTRY.md#v175x--packet-and-softswitch-artifacts).
Acceptance criteria are maintained under
[v175.x Canonical packet fabric](SWITCHING_SAFETY_AND_ACCEPTANCE.md#v175x--canonical-packet-fabric).

---

# Phase v176.x — Cross-Era Equivalence, Migration and Architecture Selection

**Status:** Implemented and shipped within combined v176.5 (not separately tagged).

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
→ same payload identity where natively supported (explicit Strowger capability gap)
→ architecture-specific event streams
→ one cross-era equivalence receipt
```

## Phase milestones

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

The released corpus executes all six native architectures: 36 shared
availability cases and 15 architecture-specific fault controls (306 executions).
Migration binds native inputs and evidence, exact rejection classes remain visible,
and architecture selection follows declared capabilities/preferences. Strowger’s
missing native payload capability and all lifecycle differences remain explicit.
See [the cross-era contract](../CROSS_ERA_MIGRATION_AND_SELECTION.md).

Evidence names are registered under
[v176.x Cross-era artifacts](SWITCHING_ARTIFACT_REGISTRY.md#v176x--cross-era-artifacts).
Acceptance criteria are maintained under
[v176.x Cross-era equivalence and migration](SWITCHING_SAFETY_AND_ACCEPTANCE.md#v176x--cross-era-equivalence-and-migration).

---

# 🎯 Final Mission — Protect Quantum Information Using Sound

**Long-term research objective:** determine whether engineered acoustic/phononic
modes, tones, harmonics, subharmonics, spectral interference, distortion and
controlled feedback can help **detect, suppress and correct errors affecting
encoded logical quantum information**, preserving logical coherence and
entanglement more effectively than appropriately matched reference methods.

The success condition is **reduced logical error or improved logical entanglement
fidelity**, demonstrated under declared noise and resource assumptions and,
ultimately, verified on a suitable physical quantum system. Sound is a candidate
instrument, not evidence of a quantum effect by itself. Negative results are
valid, retained outcomes.

## Four authority boundaries

1. **Audible observation:** translating existing QEC faults, syndromes or
   receipts into sound, without influencing decoder correctness.
2. **Classical acoustic syndrome channel:** representing *measured classical
   syndromes* as tones/phase/frequency structures, receiving actual noisy PCM,
   and passing only trusted symbols into an independently validated decoder.
   Improving this readout may improve end-to-end recovery, but does not
   directly correct an unknown quantum state.
3. **Quantum-compatible feedback:** driving device-specific control actions
   from trustworthy syndrome measurements, explicitly modelling measurement
   back-action, leakage, timing, control error and induced decoherence.
4. **Physical phononic/bosonic protection:** testing actual vibrational modes,
   quantum acoustic resonators or other calibrated transducers coupled to
   quantum devices, including coherent control or engineered dissipation.
   Quantum claims require quantum-state evidence; a WAV/spectrogram does not
   measure unknown logical amplitudes.

Ordinary air pressure waves, PCM files, microwave controls and **quantized
phonons** are not interchangeable. Any bridge between them must state its
Hamiltonian or transduction mechanism, coupling assumptions, observable
quantities and limitations. Stabilizer measurements may reveal syndromes without
revealing logical amplitudes; a sonification pipeline may not bypass no-cloning
or pretend direct access to an unknown logical state.

### Core protection law

For a declared encoded state, physical noise `N`, reference recovery
`R_ref` and experimental candidate `R_candidate`, compare **logical**
outcomes under the same code, disturbance, measurement assumptions and
comparable resources. Require independent reference states or justified
logical observables, not just self-consistent report hashes. Classical SHA-256
agreement proves only recovered *classical byte* identity relative to an
independently trusted commitment.

```text
encoded logical quantum information + declared noise
            → actual/measured syndrome and control observations
            ├─ classical acoustic path: redundant tones → interference
            │    → spectral receiver → trusted measured syndrome
            └─ physical candidate: phononic coupling → controlled evolution
                     ↓
           separately validated QEC decoder/recovery
                     ↓
          logical error / entanglement fidelity
          vs non-acoustic, same-budget reference
                     ↓
           verified gain / loss / inconclusive
          + complete evidence and claim boundary
```

### Donor interfaces and preservation

- [QEC](https://github.com/QSOLKCB/QEC) contributes exact
  [qutrit](../QUTRIT_HARMONIC_QEC.md)/ququart harmonic readout, correction,
  fault-injection batteries, Digital/Packet switching and canonical receipts.
  The harmonic phase character `r_h(s) = exp(2πihs/3)` already has informative
  and state-dark orders; it has **not** been validated as microphone-based
  noise recovery or a physical qubit-protection mechanism.
- [SPECTRAL](https://github.com/QSOLKCB/SPECTRAL) contributes deterministic
  PCM/WAV, waveform and spectral analysis, controlled signal generation,
  QEC Sonification Laboratory and observation-bound render receipts.
- [SONIFICATION](https://github.com/QSOLKCB/SONIFICATION) contributes
  ETQ-303 symbolic events, D4-triality and Tanner-graph tuning-fork research.
  **Do not render PCM in its root ETQ runtime**, which explicitly prohibits
  sampled audio, including in-memory buffers; audio laboratories must remain
  separate and explicitly authorised.
- The v176.5 six-era switching architecture is a **frozen input/transport
  baseline**. Native payload gaps (especially Strowger), switch reservations,
  Digital frames and Packet queues remain distinct. New experiments use
  additive adapters, not retroactive modifications of published contracts.

The existing intermittent legacy spectral determinism defect in
`tests/test_spectral_threshold_search.py::test_artifact_determinism` remains
unfixed by v176.5. New canonical evidence must isolate that legacy path or
resolve it through a separate compatibility-reviewed change; a passing rerun
is not proof of repair.

---

# Phase v177.x — Acoustic Syndrome Encoding, Interference and Recovery

**Status: PROPOSED.** Classical sound-signal processing plus existing exact QEC
**software** correction; no physical quantum-protection claim.

## Goal

Move beyond hearing an error: turn actual noisy waveforms into an independently
checked *redundant measurement channel* for QEC syndromes. Demonstrate whether
accurately recovered syndromes improve a declared correction result, with
untrusted observations rejected before any decoder action.

## Phase milestones

- **v177.0 — Source-Bound Acoustic Contract:** immutable input symbols, PCM
  layout/sample clock, source and fixture hashes, bounded deterministic noise
  seeds, independent reference permissions, known-answer clean baselines,
  serialized receipts and scientific claim classes.
- **v177.1 — Harmonic/Subharmonic Signal Alphabet:** investigate tones,
  multitone phase/amplitude coding, frequency shifts, overtones, subharmonics,
  qutrit/ququart harmonic orders and distortion-sensitive dark/reference
  components. Declare what redundancy actually exists and which harmonics
  identify state versus reveal only corruption.
- **v177.2 — Noise and Cancellation Battery:** additive speech-like overlap,
  broadband/coloured noise, correlated interference, impulses, clipping,
  timing slips, missing samples, phase drift, nonlinear distortion, echo and
  aliasing. Compare matched filters, reference-based subtraction, phase-aligned
  cancellation and robust spectral estimators; an arbitrary second voice
  does **not** automatically cancel the first.
- **v177.3 — Trusted Symbol and Quantum Syndrome Bridge:** receive symbols
  **from the noisy PCM**, implement classical redundancy/FEC baselines and
  confidence/reject rules, and connect independently validated recoveries to
  existing exact qutrit/ququart correction cycles. Distinguish correcting a
  damaged classical syndrome transmission from correcting quantum errors
  inside the reference code model.
- **v177.4 — Additive Digital/Packet Replay:** integrate acoustic channel
  inputs/outputs without changing v176.5 schemas. Retain original, observed,
  filtered, candidate, rejected and verified data as separate source-bound
  artifacts; recompute classical payload hashes and complete transport replay.
- **v177.5 — Comparative Evidence and SPECTRAL Laboratory:** compare
  unfiltered, cancellation-only, FEC-only, direct-syndrome, exact-reference
  and combined receivers using matched channel noise/bandwidth/latency
  budgets. Export sonified experiments but measure BER/SER, erasure/false
  accept, verified recovery, runtime, memory and independent receipt validity.

**Acceptance:** real noisy PCM-to-syndrome end-to-end fixtures, unchanged
v176.5 replay, no untrusted correction, independent byte/syndrome validation,
cross-platform reproducibility and explicit negative controls. No physical
logical-qubit benefit is claimed at this stage.

---

# Phase v178.x — Quantum Syndrome Measurement and Noise Models

**Status: PROPOSED; quantum simulation only.**

Build an explicitly quantum model of encoded logical states, noise, ancillas,
syndrome measurements, readout faults and feedback latency. Use valid unitary,
Kraus/CPTP or equivalently justified evolution; test trace/positivity and
measurement back-action. Prevent oracle access to exact hidden errors in
candidate receivers.

**Milestones:** v178.0 quantum code/state/channel contracts;
v178.1 repeated noisy stabilizer extraction, leakage and faulty ancillas;
v178.2 measured syndrome → acoustic transducer/receiver boundary;
v178.3 compare end-to-end logical error and entanglement fidelity against
direct classical electronic readout at matched resources;
v178.4 independently reproduce exact small-code cases and retain
model failures/negative results.

**Gate:** a classical improvement qualifies as *simulated* logical-protection
gain only when full quantum-state evolution, observation cost and decoder
effects support that conclusion.

---

# Phase v179.x — Physical Phononic and Spectral Control Feasibility

**Status: PROPOSED; device-specific theoretical/numerical study.**

Compare candidate acoustic/phononic platforms: mechanical resonators,
surface/bulk acoustic modes, superconducting-circuit acoustic couplers and
ion-trap vibrations. Specify the actual coupling Hamiltonian, calibration
parameters, drive frequency/power, temperatures, mode occupation, quality
factors and noise spectral densities; distinguish a classical driving signal
from a quantum excitation.

**Milestones:** v179.0 select measurable candidate platforms;
v179.1 Hamiltonian/transducer and decoherence budgets;
v179.2 harmonic/subharmonic, parametric, phase-interference and distortion
control experiments; v179.3 compare coherent drive, engineered dissipation,
spectral noise tailoring and non-acoustic equal-power controls;
v179.4 feasibility/no-go report with leakage and heating penalties.

**Gate:** a physically consistent interaction model and bounded safe control
with a measurable quantum-error mechanism. No assumed benefits from
destructive interference, prettier tones or engineered subharmonics.

---

# Phase v180.x — Bosonic Codes and Hybrid Error Protection

**Status: PROPOSED; theory and qualified quantum simulation.**

Investigate cat, binomial and GKP-style oscillator encodings where supported
by the platform/model, together with existing qubit/qutrit/ququart reference
decoders. Characterize photon/phonon loss, thermal excitation, displacement,
phase diffusion, dephasing, leakage and correlated readout errors.

**Milestones:** v180.0 code/reference selection and exact obligations;
v180.1 oscillator channel and syndrome model;
v180.2 candidate feedback, stabilization or recovery operations;
v180.3 logical lifetime/entanglement fidelity under matched resources;
v180.4 adverse cases including wrong-phase or distortion-induced errors.

**Gate:** reproducible *simulated* logical improvement beyond appropriate
baseline, or a documented lack of improvement. An audio recording alone never
satisfies this phase.

---

# Phase v181.x — Conditional Hardware-in-the-Loop and Quantum Pilot

**Status: PROPOSED; activation depends on suitable actual hardware.**

Use read-only instrument observation first, then explicitly authorised bounded
control interfaces with safety limits, calibration, stop conditions, sample
clock and firmware receipts. Hardware-in-the-loop with emulated qubits must
remain labelled *emulation*. A quantum-device claim requires actual quantum
hardware and raw independently retained measurements.

**Milestones:** v181.0 calibrated hardware identity and observation;
v181.1 energy/temperature/rate-limited controlled feedback;
v181.2 emulator versus device boundary/latency validation;
v181.3 repeated physical quantum syndrome/protection trials if available;
v181.4 divergence, failure and thermal/dephasing impact report.

**Gate:** credible raw device evidence and independent controls. If equipment
is unavailable, retain an explicit blocked outcome rather than inventing
experimental success.

---

# Phase v182.x — Evidence-Governed Logical Quantum Protection

**Status: PROPOSED; ultimate research target, not a promised result.**

Predeclare noise/control hypotheses and compare acoustic/phononic-assisted
versus conventional recovery on encoded logical information. Report logical
error per round, memory lifetime, fidelity (including entanglement fidelity
where measurable), syndrome misidentification, induced noise, cost and
confidence intervals. Threshold or break-even claims require the relevant
code families, scaling evidence and statistical support.

**Milestones:** v182.0 frozen hypotheses and blind/negative controls;
v182.1 repeated logical information-preservation battery;
v182.2 fault-tolerance/resource/assumption audit;
v182.3 independent reproduction and source-bound raw-data chain;
v182.4 final result: `verified_simulated_gain`,
`verified_physical_gain`, `no_detectable_gain`,
`worse_than_baseline`, or `inconclusive` with explicit uncertainty
and limitations.

**Gate:** only calibrated measurements of **quantum** logical performance
may establish a physical protection claim. Classical waveform hashes and
software-model results remain separately labelled.

---

# Universal Acceptance and Claim Rules

- Preserve v137/v166/v167/v170–v176 contracts and fixtures; every new
  connector/decoder pathway is additive and independently reviewed.
- Keep `ideal_syndrome`, `measured_syndrome`, `encoded_sound`,
  `received_waveform`, `reference_noise`, `estimated_syndrome`,
  `decoder_decision` and `logical_outcome` separate. No hidden ideal-state
  oracle or false provenance inferred from an embedded hash.
- Reject unsupported, ambiguous, discordant and state-dark-only observations.
  Distortion may reveal faults, but not necessarily identify their source.
- Test no-tone, no-cancellation, conventional error-correction, same-power
  control and state-of-the-art relevant non-acoustic baselines.
- Validate exact small-code cases, independent signal recovery, controlled
  physical noise and fair trial budgets; publish adverse and negative cases.
- Software evidence uses canonical identities, fixed declared seeds,
  dependency/environment manifests, complete replay and CI; physical
  observations retain timestamps, raw measurements and uncertainties and
  are never falsely promised to reproduce as byte-identical measurements.
- New integrated code must pass focused tests and the full
  `python -m pytest -q` gate when shared decoder/routing/evidence surfaces
  change; capture test baselines and claimed coverage.
- Promote evidence only through explicit levels:
  `audible_demonstration` → `classical_channel_verified` →
  `quantum_simulation_verified` → `physical_quantum_verified`.
  Higher levels do not follow automatically from lower ones.

## Scientific references (inspiration, not QEC validation)

- [Building a Fault-Tolerant Quantum Computer Using Concatenated Cat Codes
  (PRX Quantum, 2022)](https://doi.org/10.1103/PRXQuantum.3.010329):
  architecture proposal involving acoustic resonators and superconducting
  circuits; not proof of QEC's implementation.
- [Hardware-efficient quantum error correction via concatenated bosonic qubits
  (Nature, 2025)](https://doi.org/10.1038/s41586-025-08642-7):
  relevant physical logical-qubit protection experiments; not an acoustic
  demonstration of this QEC roadmap.
- [Quantum Error Correction Using the Phononic DoF
  (JST Moonshot, September 2026)](https://www.jst.go.jp/moonshot/en/program/goal6/appeal/63_takahashi_ap03.html):
  ion-vibrational-mode research and bosonic-code context.

---

# Final Direction

```text
six-era deterministic switching and provenance (v176.5: released)
→ acoustic channel error suppression and recovery (v177: proposed)
→ quantum-state/noise/syndrome modelling (v178: proposed)
→ physical phononic coupling and spectral control (v179: proposed)
→ bosonic/hybrid logical information protection (v180: proposed)
→ measured feedback and physical pilot if available (v181: proposed)
→ independently verified logical quantum protection (v182: target)
```

**Success is preservation of logical quantum information, not musical
consonance.** Tones, harmonics, subharmonics and distortion earn a correction
role only when a reproducible, physically justified test shows that they
improve that preservation compared with a fair alternative.
