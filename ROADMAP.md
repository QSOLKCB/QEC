# QSOLKCB / QEC — ROADMAP (Post-v146.0)

---

## 0. Core Law (Global)

```text
same input
→ same ordering
→ same canonical JSON
→ same stable hash
→ same bytes

All modules MUST:

be deterministic
be replay-safe
be bounded
fail fast on invalid input
produce canonical artifacts
avoid randomness, wall-clock dependence, and implicit async
preserve decoder immutability
remain analysis-layer only unless explicitly proven otherwise
1. Current State

QEC has completed its original proof-to-governance trajectory.

distributed proof
→ hardware-aware bounded control research
→ IRIS invariant runtime
→ SPHAERA formal invariant geometry runtime
→ SCOL orchestration research
→ governance + policy benchmarking
→ proof-carrying action representation
Completed Milestones
v143.5 — SPHAERA Proof Artifact
end-to-end execution proof
canonical outputs
reproducible evidence
v144.x — SCOL Foundation Arc
recurrence detection
stress lattices
state-conditioned filter mesh
deterministic transition policy
bounded refinement
consensus bridge
evaluation pack
v145.x — Promotion / Policy Lab Arc
governed orchestration
governed simulation
policy sensitivity
policy family benchmarking
v146.0 — Execution Bridge
proof-carrying, non-executing action capsules
deterministic action representation
replay-safe certification artifact

This means the old roadmap is complete enough that continuing to append to it would blur phases.

The next roadmap is therefore a new one.

2. New Strategic Position

QEC is now a:

deterministic proof system
+ invariant-driven analysis engine
+ geometry-aware runtime
+ policy-evaluable orchestration substrate
+ proof-carrying action representation system

The next direction is no longer:

build more abstract architecture

The next direction is:

prove under constraint
→ benchmark under scarcity
→ compare across hostile substrates
→ extract lawful application patterns
3. v147.x — RETRO ARC
Replay Evaluation Through Retro-Oriented Constraints
Purpose

Test QEC, invariants, action capsules, and “quantumy” structure under historically hostile compute constraints.

These constraints include:

tiny word sizes
tiny RAM / ROM budgets
no-FPU arithmetic
fixed-point or emulated floating point
scanline / palette / sprite / tracker timing limits
narrow buses
slow storage or cartridge-style loading
sparse or joystick-like control inputs
emulator vs hardware drift
offline control trajectories rather than live stochastic training

This arc is not nostalgia for nostalgia’s sake.

It is a formal proving ground for:

what survives when compute becomes small,
state becomes expensive,
timing becomes visible,
and arithmetic becomes cruel
RETRO Law

All retro-oriented modules must satisfy:

target profile
+ constraint budget
+ canonical trace
→ bounded reduction
→ invariant analysis
→ replay-safe comparison
→ canonical receipt

Additional retro rules:

no live stochastic RL in core QEC
no “agent training loop” inside the main system
no hidden emulator state dependence
no non-canonical host shortcuts
no architecture-specific silent fallback
no reliance on modern abundant floating point unless explicitly modeled as a comparison surface
Constraint Families
A. 68k / Amiga Constraint Family

Focus:

68000 / 68020 / 68030 / 68040 / 68060 style environments
PiStorm-style replacement / extension constraints
Amiga emulator and graphics-card constraints
no-FPU or emulated-FPU paths
module-tracker / pattern-audio timing surfaces
Neo Geo / emulator port stress under classic hardware budgets
B. 4-bit / 8-bit Arithmetic Scarcity Family

Focus:

Intel 4004 and 8008-like austerity
home-brew SBC surfaces
tiny instruction sets
tiny memory maps
painfully explicit arithmetic
floating-point approximation as a precious resource, not a default
C. Z80 / 6502 Minimal Systems Family

Focus:

Z80 operating systems and kernels
calculator-class and hobby OS environments
Z80 / i8080 emulation
tiny-model / quantized inference surfaces
BASIC-era program structure
small-memory control and state compression
D. Atari / Offline Policy Family

Focus:

Atari 2600 trajectories
cartridge / frame / joystick constraint surfaces
emulator-driven canonical play traces
offline policy comparison datasets
policy evaluation without live stochastic training
Implementation Principle

The retro arc proceeds by:

define targets
→ normalize constraints
→ canonicalize traces
→ run invariant analysis under budget
→ compare cross-target stability
→ certify replay
→ publish evidence

That means:

profile first
benchmark second
compare third
only then bind action capsules to retro experiment surfaces
4. v147.x — RETRO FOUNDATION ARC
v147.0 — Retro Target Registry
Purpose

Create the canonical registry of retro target classes and constraint surfaces.

Model
target family
→ explicit hardware / emulator profile
→ budget declaration
→ canonical target receipt
Primary Module
src/qec/analysis/retro_target_registry.py
Inputs
target identifier
ISA family
word size
address width
RAM budget
ROM budget
storage budget
cycle / timing budget
display budget
audio budget
input budget
FPU availability / emulation mode
host / emulator / hardware provenance
Outputs
RetroTargetReceipt
Metrics [0,1]
arithmetic_constraint_pressure
memory_constraint_pressure
timing_constraint_pressure
display_constraint_pressure
audio_constraint_pressure
replay_surface_clarity
Classification
68k_amiga
intel_4004
intel_8008
z80_minimal
z80_os
m6502_basic
atari_2600
hybrid_retro
Role

Defines the legal target space.
Nothing later is allowed to be fuzzy about the machine it claims to test.

v147.1 — Retro Trace Intake Bridge
Purpose

Normalize traces from emulators, SBCs, and benchmark playback into canonical QEC-compatible receipts.

Model
raw target trace
→ canonical field extraction
→ timing normalization
→ event ordering
→ trace receipt
Primary Module
src/qec/analysis/retro_trace_intake_bridge.py
Inputs
CPU traces
memory traces
bus / cycle traces
display events
audio / tracker events
controller / input events
cartridge / disk / ROM metadata
target receipt
Outputs
RetroTraceReceipt
Metrics [0,1]
trace_completeness
event_order_integrity
timing_observability
input_sparsity
capture_replay_integrity
Role

Creates one lawful trace language across wildly different retro systems.

v147.2 — Constraint Profile Kernel
Purpose

Derive a canonical retro constraint profile from target + trace.

Model
target receipt
+ trace receipt
→ budget stress characterization
→ scarcity classification
→ constraint profile receipt
Primary Module
src/qec/analysis/retro_constraint_profile_kernel.py
Inputs
RetroTargetReceipt
RetroTraceReceipt
policy thresholds
Outputs
RetroConstraintProfileReceipt
Metrics [0,1]
word_size_pressure
address_space_pressure
memory_pressure
storage_pressure
timing_pressure
arithmetic_pressure
display_pressure
audio_pressure
Classification
word_starved
memory_starved
no_fpu
fixed_point_bound
scanline_bound
tracker_bound
input_sparse
balanced_retro
Role

Makes the constraint surface explicit before any “quantumy” or invariant analysis begins.

v147.3 — Arithmetic Scarcity Kernel
Purpose

Test invariant retention and QEC behavior under brutal arithmetic limits.

Model
state / invariant data
+ retro constraint profile
→ reduced arithmetic lane
→ bounded approximation
→ scarcity receipt
Primary Module
src/qec/analysis/arithmetic_scarcity_kernel.py
Inputs
invariant structures
phase / spectral / geometry summaries
retro constraint profile
quantization / fixed-point policy
optional float-emulation policy
Outputs
ArithmeticScarcityReceipt
Metrics [0,1]
quantization_drift
fixed_point_stability
approximation_pressure
invariant_retention
float_emulation_cost
replay_stability
Classification
exact_survives
bounded_degradation
emulation_required
instability_exposed
invalid_under_budget
Role

This is where QEC learns whether its mathematics survives when arithmetic becomes small, ugly, and explicit.

v147.4 — Retro Projection Kernel
Purpose

Project QEC state, invariant, and action-capsule structure into retro-native representational forms.

Model
qec artifact
+ target profile
→ retro-native projection
→ pattern / tile / tracker / scanline / register form
→ projection receipt
Primary Module
src/qec/analysis/retro_projection_kernel.py
Inputs
invariant receipts
transition / refinement / governance receipts
action capsules
retro target receipt
projection policy
Outputs
RetroProjectionReceipt
Projection Families
tile / character-map projection
scanline / raster projection
sprite / object projection
palette-constrained geometry projection
tracker-pattern / note-grid projection
fixed-register / nibble / byte projection
BASIC-visible state projection
Metrics [0,1]
projection_loss
motif_preservation
palette_pressure
channel_pressure
timing_coherence
action_readability
Role

Turns “quantumy things” into forms retro systems can actually host, display, or sonify.

v147.5 — Retro Invariant Stress Battery
Purpose

Benchmark QEC invariants and action representations across retro targets.

Model
canonical retro targets
+ canonical traces
+ qec analyses
→ cross-target replay battery
→ benchmark receipt
Primary Module
src/qec/analysis/retro_invariant_stress_battery.py
Benchmark Families
68k / Amiga workload surfaces
no-FPU / float-emulation surfaces
4-bit / 8-bit arithmetic austerity surfaces
Z80 OS / kernel / tiny-model surfaces
6502 / BASIC expression surfaces
Atari trajectory surfaces
display / audio / timing projection surfaces
Outputs
RetroBenchmarkReceipt
Required Assertions
byte-identical replay
invariant retention under declared budget
stable action capsule reconstruction
deterministic projection output
canonical tie-breaking under reduced arithmetic
no hidden host dependence
emulator / hardware comparison receipts when available
Role

This is the new proof wall.

v147.6 — Offline Atari Policy Audit
Purpose

Use Atari trajectories and offline-control corpora as a static policy stress surface for QEC.

Model
offline trajectory corpus
+ qec policy / governance / action logic
→ deterministic policy audit
→ audit receipt
Primary Module
src/qec/analysis/offline_atari_policy_audit.py
Inputs
offline trajectory bundles
target metadata
transition / refinement / governance receipts
action capsules
audit policy
Outputs
OfflineAtariPolicyAuditReceipt
Metrics [0,1]
corpus_coverage
action_consistency
counterfactual_stability
replay_alignment
control_entropy_proxy
governance_repeatability
Role

Brings “game-like policy stress” into QEC without violating the law against live stochastic training.

v147.7 — RETRO Evaluation Pack
Purpose

Package the first full research-grade evaluation of QEC under retro constraints.

Deliverables
comparative design summary across retro target families
arithmetic austerity report
projection fidelity report
replay certification tables
emulator / SBC comparison appendix
offline policy audit summary
integration notes for optional promotion
Outputs
RETROEvaluationPackReceipt
markdown research summary
benchmark tables
paper-ready figures / diagrams (optional)
Role

Marks the end of the retro foundation arc.

5. v148.x — OPTIONAL PROMOTION ARC

This arc is conditional.

It only proceeds if v147.x proves that retro-constrained testing gives genuinely useful evidence without any violation of determinism, replay identity, boundedness, or architecture cleanliness.

v148.0 — Proof-Carrying Retro Experiment Capsule
Purpose

Bind retro experiments to proof-carrying experiment descriptors while staying non-executing.

Result
retro target
+ invariant test
+ policy audit
→ certified experiment capsule
Constraint

Still analysis-only.
Still not execution.
Still not emulator control logic.

v148.1 — Cross-Target Invariant Consensus Bridge
Purpose

Compare the same QEC analysis across multiple retro families and produce a deterministic cross-target consensus view.

Result
same analysis
across 68k / 4004 / 8008 / z80 / 6502 / atari
→ stability map
→ consensus / divergence receipt
v148.2 — Retro Publication / Documentation Release
Purpose

Finalize the public-facing research and demo material for the retro arc.

Possible Outputs
paper draft
README updates
docs / diagrams
benchmark appendix
sonification / visualization demos
replay certification summary
6. Guardrails for the New Roadmap
Never Allowed
stochastic sampling in live QEC execution
online RL training inside the core system
PRNG-driven routing
wall-clock dependent behavior
hidden emulator-state assumptions
decoder-core modification
architecture overlap that mutates SPHAERA / SCOL semantics
using retro targets as an excuse for non-canonical shortcuts
Required Mitigations
quantization rules must be explicit and canonical
float emulation policies must be explicit and auditable
tie-breaking must be canonical and published
projection losses must be measured, not hand-waved
target profiles must be finite and reviewable
emulator vs hardware provenance must be explicit
all receipts must serialize canonically and hash stably
7. Final Strategic Direction
v143.5
→ completed deterministic runtime proof

v144.x – v146.0
→ orchestration, governance, benchmarking, and proof-carrying action representation

v147.x
→ retro-constrained proving ground for invariants, action capsules, and “quantumy” structure

v148.x
→ optional promotion to certified retro experiment layer and publication

This keeps QEC faithful to its law:

prove first
→ constrain harder
→ benchmark again
→ promote only if lawful
