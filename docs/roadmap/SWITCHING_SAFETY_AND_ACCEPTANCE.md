<!-- SPDX-License-Identifier: MPL-2.0 -->
# QEC Switching Safety and Acceptance Rules

This document is the normative safety, validation and release-gate companion to
[`ROADMAP.md`](../../ROADMAP.md).

The roadmap defines the switching phases and deliverables. This document defines
the rules every phase must obey.

## Core deterministic law

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

A violation is:

```text
SYSTEM INVALID
```

Every switching release must produce:

- a versioned contract;
- one or more canonical artifacts;
- full 64-character SHA-256 identities;
- a validation rule;
- an explicit failure mode;
- a deterministic replay test;
- a machine-readable claim boundary.

If an idea cannot produce those, it remains inspiration rather than QEC.

## Global release gate

Every switching-era release must pass:

```text
python -m pytest -q
```

Focused suites may be used during development. The full repository suite is the
release gate whenever routing, receipts, canonicalisation, claims, operator
state, shared schemas or proof-chain identities change.

Required release evidence:

- source commit SHA;
- package version;
- environment manifest;
- canonical fixtures;
- deterministic test receipt;
- generated artifact manifest;
- full SHA-256 identities;
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

## Claim boundary

Switching receipts prove only declared QEC software behaviour under the model,
inputs, topology, state and faults bound into the receipt.

They do not prove:

- physical telephone-network fidelity;
- quantum hardware behaviour;
- hardware fault tolerance;
- carrier-grade reliability or latency;
- universal superiority of one switching architecture;
- decoder correctness merely because a route was valid;
- that a historical analogy is scientific evidence.

The decoder remains responsible for mathematical correction validity. Switching
layers route declared requests and preserve evidence; they do not invent truth.

## Browser and adapter boundary

Offline browser laboratories may visualise, sonify and export demonstrations,
but their local digests are not canonical cryptographic evidence unless they use
the same validated canonical implementation and explicitly declare that fact.

Real protocols, networks, cloud systems and historical hardware models remain
adapters. Their observations must be captured in separate receipts and may not
become proof authority.

## Phase acceptance matrix

### v171.x — Panel separated control

- the same request and translation table produce the same sender programme;
- route actuation cannot begin before sender sealing;
- payload bytes remain unchanged across route compilation;
- a changed translation table changes the programme hash;
- motor and path faults are explicit and reproducible;
- no fallback route is chosen without a declared deterministic rule;
- Strowger and Panel may differ in event sequence while preserving declared
  destination and outcome equivalence.

### v172.x — Crossbar coordinate switching

- matrix ordering is canonical;
- path search is bounded and deterministic;
- tie-breaking never depends on set or dictionary iteration order;
- a marker cannot alter decoder output or payload identity;
- reservations and releases are included in the event chain;
- no partial path may be committed;
- equivalent outcomes across switch types are machine validated.

### v173.x — ESS stored-program control

- programme and data identities are separate;
- modifying a programme or translation table changes execution identity;
- scheduling uses logical time only;
- feature modules cannot force acceptance;
- external or historical hardware models remain adapters;
- the same programme and input corpus produce byte-identical command streams.

### v174.x — Digital time-division switching

- frame width and byte order are explicit;
- slot allocation is deterministic under saturation;
- frame sequence uses logical indices rather than wall-clock timestamps;
- idle, missing and invalid symbols are distinct;
- reassembly reproduces the original payload identity or rejects;
- no frame-level result is described as physical quantum evidence.

### v175.x — Canonical packet fabric

- packet identity excludes mutable transport metadata unless declared;
- routing and queue ordering are canonical;
- queue overflow has one declared outcome;
- failover never uses random route selection;
- every accepted reassembly recomputes the original payload hash;
- live network observations are labelled external and non-authoritative;
- no cloud service is required for core tests.

### v176.x — Cross-era equivalence and migration

- all architectures consume one canonical request schema;
- migration is explicit and hash-bound;
- outcome equivalence is machine checked;
- architecture-specific failures remain visible rather than normalized away;
- comparison metrics are observations, not correctness proofs;
- architecture selection uses declared requirements and deterministic ordering;
- no historical switch implementation is removed when a newer model lands.
