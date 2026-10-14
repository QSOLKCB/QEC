<!-- SPDX-License-Identifier: MPL-2.0 -->
# v176.0–v176.5: cross-era migration, replay and architecture selection

This additive implementation is **unreleased**. Package version 173.0.0 and every
published v137–v173.0 implementation/fixture remain unchanged. It does not activate
v193, any network adapter, or the deferred sonification programme.

## Contract and scope

`qec.routing.cross_era` executes the actual Strowger, Panel, Crossbar, ESS, Digital
and Packet implementations. One canonical request supplies opaque payload bytes,
a caller-declared decoder-output hash, one of four semantic destinations, two
lane availability states and an explicit fault control. The migration contract
binds the full corpus, published historical adapter manifest, destination mapping,
policy, all new native input snapshots, ESS source/programme identity, and initial
state/fault plans. Every downstream artifact binds the upstream full SHA-256.

The v176.5 schema family assembles the six planned v176.0–.5 deliverables; it is a
single development contract, not six released versions. Hashing uses compact
sorted-key UTF-8 JSON, with each artifact's own `sha256` excluded. Serialization
is detached. Replay reconstructs native inputs and executes every source again;
re-signing altered evidence, substitutions, mappings or summary flags cannot pass.
Trust pins bind external expectations. Without pins, validation establishes
internal consistency, not authenticated provenance.

### Native mappings

- Strowger: published mixed-radix route, selector trunks and pre-route state.
- Panel: published sender/register, translation, motor/bank topology and faults.
- Crossbar: published layered path planning plus independent continuity evidence.
- ESS: source-bound programme, native whole-path reservation, verification and
  owner-bound release. This is a real lifecycle, rather than independent planning.
- Digital: native allocator picks a free output slot, frames are synchronized and
  interchanged, the payload is reassembled, and the reservation is released.
- Packet: native envelopes, exact binary-prefix routes, queues, explicit link
  availability/fault windows, hop chain and destination payload-hash recomputation.

`idle`, `busy` and `quarantined` remain explicit in each translated input. Packet
link-down annotations retain the original availability cause. Capacity fixtures
are shared **initial contention snapshots**. Each case starts independently; the
harness does not assert universal concurrent capacity or transfer a live ledger.
ESS dynamic simultaneous admission/retry/release, Digital allocator contention
and Packet overflow have their own native batteries. Those different lifecycle
semantics are intentionally not flattened into universal equivalence.

## Corpus and independently stated oracle

The default corpus contains 51 cases and runs 306 native executions:

- 36 destination × two-lane availability combinations;
- five published Strowger/Panel/Crossbar fault controls;
- one ESS bounded-search control;
- six Digital corruption/loss/duplicate/reorder/alignment/collision controls;
- three Packet corruption/drop/all-alternates-down controls.

The small reference oracle derives admission and the first available lane directly
from declared availability and the affected architecture. It does not consult a
native event trace. It also requires the exact independently declared native reason
and failure class. Only each model’s explicit capacity reason maps to the common
capacity class; another rejection cannot be hidden by a saturated topology. Native
reasons and receipts remain visible. The 36 shared
availability cases should agree; the 15 controls must expose differences. A custom
corpus is always marked separately and cannot claim default-corpus coverage.

Bounds: 1–64 canonically ID-ordered unique cases, 1–64 ASCII identifier characters,
0–256 opaque payload bytes per request and 8,192 total payload bytes. Booleans or
floats cannot substitute for integer destinations. Native per-model bounds also
apply. Fault controls require otherwise idle lanes and at least two payload bytes;
arbitrary mixed-fault/capacity combinations are not claimed by this corpus.

## What equivalence means

Each of the 15 architecture pairs is compared on requested and achieved semantic
destination, route admission/rejection class, selected lane and failure class.
Capacity rejections have a common comparison class but retain their exact native
reason and input cause. Other faults remain architecture-qualified.

Payload identities are compared only where both native models support them and
both executions succeed. Strowger's released API has no payload bytes, so its
payload comparison is explicitly unsupported: the common case envelope binds
those bytes without inventing a native Strowger capability. Crossbar preserves
input identity while planning; it does not deliver a physical payload. Native
commits, reservations, transmitted frames and packet queues are not interchangeable.
There is no six-way native payload or lifecycle equivalence claim.

## Selection and report

`ArchitectureSelectionPolicy` declares required implemented capabilities and an
explicit six-architecture preference permutation. Selection filters incompatible
models and preserves that declared order; impossible combinations return
`no_eligible_architecture`. It never ranks by historical age or assumed superiority.
Crossbar reservation capability refers to its separate released contention API,
not the independent planning adapter used by the cross-era availability corpus.

The report derives receipt byte counts, native event counts, Digital frame counts,
capacity/complexity descriptions and observed fault controls from validated native
evidence. Digital event counts cover allocator events; frames are separate. These
are model observations, not hardware performance, latency, reliability or correctness
proofs. No wall clock or cloud service enters the canonical core.

## Artifacts and commands

- `switch_compatibility_corpus_manifest.json`: shared closed request corpus.
- `switch_migration_contract.json`: full source-bound native translation.
- `cross_era_replay_receipt.json`: all native evidence and reference comparisons.
- `cross_era_equivalence_receipt.json`: validated pairwise declared invariants.
- `architecture_selection_policy.json`: requirements and deterministic ordering.
- `switching_evolution_report.json`: bounded observations with explicit limits.

```sh
python -m qec.routing.cross_era_cli demo --output-dir artifacts/cross-era
python -m qec.routing.cross_era_cli run \
  --input artifacts/cross-era/switch_compatibility_corpus_manifest.json \
  --output-dir artifacts/cross-era
python -m qec.routing.cross_era_cli validate --kind report \
  --receipt artifacts/cross-era/switching_evolution_report.json
python -m qec.routing.cross_era_cli select --required bounded_queues
```

`validate --expected-sha256` can pin any complete supplied artifact. Python
validators additionally provide corpus, migration, replay or policy pins. Invalid
JSON, duplicate keys, oversized files, mutated sources or failed invariants exit
nonzero. CLI input is bounded to 64 MB; in-memory validation also bounds shape,
depth and total content before canonical comparison. Generated demonstration labs
use different schemas and are never accepted as canonical receipts.

## Historical and scientific boundary

The roadmap's historical sources motivate this architecture vocabulary; this
contract makes no new historical fidelity claims. All generated evidence concerns
only declared QEC software. It does not prove decoder correctness, physical
telephony, quantum hardware, thresholds, quantum advantage or carrier-grade service.
Published contracts are retained rather than replaced.
