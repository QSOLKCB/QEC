# QSOLKCB / QEC — ROADMAP.md
## Post-v146.0

---

## Core Law (Global, Non-Negotiable)

```text
same input
→ same ordering
→ same canonical JSON
→ same stable hash
→ same bytes

These five lines are not defaults. They are architectural invariants. They define the system.

Every module, pipeline, receipt, benchmark, and metric in this roadmap must satisfy all five simultaneously under all conditions, including retro constraint, hostile arithmetic, adversarial input, cross-platform replay, and real-world operationalization.

All modules must:

be deterministic
be replay-safe
produce bounded outputs
fail fast on invalid input
produce canonical, stably hashed artifacts
exclude randomness, wall-clock dependence, and implicit async
preserve decoder immutability without exception
remain analysis-layer only unless an explicitly proven mathematical bridge is established and documented

This law is never relaxed. If a module cannot satisfy it, the module is invalid.

Current State

QEC has completed its original proof-to-governance trajectory.

distributed proof
→ hardware-aware bounded control research
→ IRIS invariant runtime
→ SPHAERA formal invariant geometry runtime
→ SCOL orchestration research
→ governance + policy benchmarking
→ proof-carrying action representation
Completed Milestones
Release Arc	Key Deliverable	Status
v143.5	SPHAERA Proof Artifact — end-to-end execution proof, canonical outputs, reproducible geometric evidence	Complete
v144.x	SCOL Foundation Arc — recurrence detection, stress lattices, state-conditioned filter mesh, deterministic transition policy, bounded refinement, consensus bridge, evaluation pack	Complete
v145.x	Promotion / Policy Lab Arc — governed orchestration, governed simulation, policy sensitivity, policy family benchmarking	Complete
v146.0	Execution Bridge — proof-carrying, non-executing action capsules, deterministic action representation, replay-safe certification artifact	Complete
v146.1	Action Capsule Integrity & Replay Certification Hardening — strengthened lineage integrity, replay identity validation, canonical round-trip enforcement, mutation protection	Complete / Hardening Line

The v143–v146 program had a defined terminus: a system capable of representing, carrying, and certifying actions as proof-bearing artifacts without executing them.

That terminus was reached at v146.0 and hardened at v146.1.

The architecture is now stable. The decoder is frozen. The evidence trail is canonical.

Continuing to append abstract modules to the v143–v146 arc would dilute it without producing new evidence. The next phase is not architectural elaboration. It is empirical pressure.

Strategic Position

QEC is now a:

deterministic proof system
+ invariant-driven analysis engine
+ geometry-aware runtime
+ policy-evaluable orchestration substrate
+ proof-carrying action representation system

This is a substantial, complete foundation. The next question is not what the architecture can express, but what it can survive.

Modern compute environments — abundant memory, pipelined FPUs, massive bandwidth, speculative execution, oversized caches — mask redundancy and structural bloat. An invariant tested only under abundance is not yet an invariant. It is a claim.

The previous arc earned the right to make claims. The next arc must justify those claims through evidence produced:

under retro constraint
under arithmetic scarcity
under adversarial pressure
under cross-emulator replay comparison
under real-world operationalization demand

The next direction is therefore not:

build more abstract architecture

The next direction is:

prove under constraint
→ benchmark under scarcity
→ compare across hostile substrates
→ stress-test adversarially
→ extract lawful, measurable application patterns for real systems
Transition: Why the Previous Roadmap Is Complete

The v143–v146 arc was primarily constructive:

it built the runtime
it built the geometry layer
it built the orchestration substrate
it built the policy evaluation framework
it built the action capsule representation

That construction program terminated at v146.0 and was integrity-hardened at v146.1.

The transition to v147.x is not a continuation of the same program. It is a shift in method: from construction to empirical pressure.

The retro and adversarial arcs below do not exist to decorate the architecture. They exist to find cracks in it, or prove that the cracks do not exist.

A framework that survives only under modern abundance is not yet trusted.
A framework that survives under historical hostility, reduced arithmetic, adversarial perturbation, and measurable operational deployment begins to earn that trust.

ROADMAP SPINE

v146 ARC

v147.x
→ retro-constrained proving ground for invariants, action capsules, and “quantumy” structure

v148.x
→ optional promotion to certified retro experiment layer and publication

v147.x — RETRO-CONSTRAINED PROVING GROUND

The v147.x release cycle implements a cross-domain empirical research program that maps QEC proof-carrying action capsules onto constrained target substrates.

This arc is designed explicitly to:

falsify weak invariants
expose pathological convergence
eliminate compute redundancy through measurable analysis
test action capsules under hostile arithmetic
produce canonical receipts under all conditions

Every claim in this arc must be supported by a canonical receipt resulting from deterministic reduction.

v147.0 — Retro Target Registry
Purpose

Define the canonical registry of retro target classes and constraint surfaces with sufficient precision that any downstream module can reference a target by identifier and receive a fully specified, auditable budget declaration.

The registry must be finite, reviewable, and non-expandable without explicit release versioning.

A target that cannot be fully declared in the registry schema is not a legal retro target.

Model
target identifier
+ ISA family
+ hardware / emulator provenance
→ fully specified constraint budget
→ RetroTargetReceipt
Primary Module
src/qec/analysis/retro_target_registry.py
Inputs
Field	Type	Description
target_id	str	Canonical identifier
isa_family	enum	68k, intel_4004, intel_8008, z80, m6502, tracker_audio, display_raster
word_size	int	4, 8, 16, or 32
address_width	int	Declared address bus width
ram_budget	int	Working RAM ceiling
rom_budget	int	ROM / cartridge storage ceiling
cycle_budget	int	Cycles per frame or operation
display_budget	struct	Scanlines, palette depth, sprite count
audio_budget	struct	Channels, sample rate, pattern columns
input_budget	struct	Input bits per frame / scan budget
fpu_policy	enum	none, emulated, coprocessor, native
provenance	enum	hardware, emulator, hybrid
Output
RetroTargetReceipt
Bounded Metrics [0,1]
arithmetic_constraint_pressure
memory_constraint_pressure
timing_constraint_pressure
display_constraint_pressure
audio_constraint_pressure
replay_surface_clarity
Classification Labels
68k_amiga
intel_4004
intel_8008
z80_minimal
z80_os
m6502_basic
atari_2600
tracker_surface
hybrid_retro
Role

Defines the legal target space. Nothing in the retro arc is permitted to be vague about the machine it claims to test.

v147.1 — Retro Trace Intake Bridge
Purpose

Normalize traces from emulators, single-board computers, and recorded benchmark playback into canonical QEC-compatible trace receipts.

The goal is one lawful trace language across wildly different retro systems.

Without this module, “retro constraint” is a description, not a data type.

Model
raw target trace
+ RetroTargetReceipt
→ canonical field extraction
→ timing normalization
→ event ordering
→ RetroTraceReceipt
Primary Module
src/qec/analysis/retro_trace_intake_bridge.py
Inputs
CPU state traces
memory read/write traces
bus and cycle traces
display event logs
audio / tracker events
controller and input event logs
cartridge / ROM / disk metadata
RetroTargetReceipt
Output
RetroTraceReceipt
Bounded Metrics [0,1]
trace_completeness
event_order_integrity
timing_observability
input_sparsity
capture_replay_integrity
Role

The trace bridge is the boundary layer between the heterogeneous outside world and the uniform inside world of QEC analysis.

v147.2 — Retro Constraint Profile Kernel
Purpose

Derive a canonical retro constraint profile from a target receipt and its normalized trace.

The constraint profile characterizes the pressure landscape before any invariant analysis begins.

Every downstream module — arithmetic scarcity, projection, capsule audit, redundancy collapse — must reference this profile rather than re-deriving constraint pressure independently.

Model
RetroTargetReceipt
+ RetroTraceReceipt
→ budget stress characterization
→ scarcity classification
→ RetroConstraintProfileReceipt
Primary Module
src/qec/analysis/retro_constraint_profile_kernel.py
Inputs
RetroTargetReceipt
RetroTraceReceipt
declared policy thresholds
Output
RetroConstraintProfileReceipt
Bounded Metrics [0,1]
word_size_pressure
address_space_pressure
memory_pressure
storage_pressure
timing_pressure
arithmetic_pressure
display_pressure
audio_pressure
Classification Labels
word_starved
memory_starved
no_fpu
fixed_point_bound
scanline_bound
tracker_bound
input_sparse
balanced_retro
Role

Makes the scarcity surface explicit before any invariant analysis begins.

v147.3 — Arithmetic Scarcity Survival Layer (Mandatory)
Purpose

Enforce a hard arithmetic pressure filter on all QEC invariants.

This is the module that answers the central research question:
Which invariants survive when arithmetic becomes small, ugly, and explicit?

Anything that fails here is not promoted as an invariant.

Model
full precision state / invariant data
→ fixed-point lane
→ reduced bit-width (8-bit / 4-bit)
→ invariant recomputation
→ ArithmeticScarcityReceipt
Primary Module
src/qec/analysis/arithmetic_scarcity_kernel.py
Inputs
invariant structures
phase / spectral / geometry summaries
transition and refinement receipts
action capsule descriptors
RetroConstraintProfileReceipt
quantization policy
fixed-point policy
optional float-emulation policy
Output
ArithmeticScarcityReceipt
Bounded Metrics [0,1]
quantization_drift
fixed_point_stability
approximation_pressure
invariant_retention
float_emulation_cost
replay_stability
Required Classifications
exact_survives
bounded_degradation
emulation_required
instability_exposed
invalid_under_budget
Required Assertions

Every run must assert:

canonical tie-breaking under reduced arithmetic
declared rounding policy before analysis begins
no floating-point tie-breaking in fixed-point mode
invariant retention measured against declared arithmetic regime, not a hidden full-precision baseline
Role

The arithmetic proving wall.
Anything classified invalid_under_budget or instability_exposed is removed from the promoted invariant set.

v147.4 — Float-Emulation Audit Module
Purpose

Audit how QEC analysis behavior changes across the arithmetic spectrum:

native double precision
explicitly emulated float
reduced precision
fixed-point lanes

This is a semantic audit, not a performance benchmark.

Model
QEC analysis result (full precision)
+ declared arithmetic regime sequence
→ regime-by-regime equivalence audit
→ FloatEmulationAuditReceipt
Primary Module
src/qec/analysis/float_emulation_audit.py
Inputs
baseline analysis artifacts
declared arithmetic regimes
RetroConstraintProfileReceipt
Output
FloatEmulationAuditReceipt
Bounded Metrics [0,1]
regime_equivalence
divergence_onset_threshold
structural_invariant_fraction
magnitude_dependent_fraction
emulation_overhead_ratio
Role

Separates structurally grounded invariants from precision-dependent claims.

v147.5 — Canonical Projection + Minimal Representation (Mandatory)
Purpose

Project QEC state, invariant summaries, and action-capsule structures into retro-native representational forms while enforcing minimal representation.

Projection loss is not failure — it is information.

If a QEC invariant cannot be expressed in a tile grid, sprite map, tracker pattern, register form, or BASIC-visible state without catastrophic loss, that loss is evidence about the structure itself.

Model
high-dimensional state / invariant receipt / action capsule
+ RetroTargetReceipt
+ projection policy
→ minimal representation
→ retro-native projection
→ canonical form
→ RetroProjectionReceipt
Primary Module
src/qec/analysis/retro_projection_kernel.py
Inputs
invariant receipts
transition / refinement / governance receipts
action capsules
RetroTargetReceipt
projection policy
Output
RetroProjectionReceipt
Projection Families (All Enforced)
Family	Form
tile / grid	Character-grid representation
scanline / raster	Scanline-indexed event sequence
sprite / object	Fixed-count sprite arrangement
tracker pattern	Note/effect matrix projection
register / byte / nibble	Width-constrained numeric encoding
BASIC-visible state	BASIC variable assignment surface
Bounded Metrics [0,1]
projection_loss
motif_preservation
palette_pressure
channel_pressure
timing_coherence
action_readability
Role

Forces abstract structure into hostile representational forms. Turns structural claims into measurable projection receipts.

v147.6 — Action Capsule Retro Audit
Purpose

Specifically test the v146 action capsule representation under retro constraint profiles.

This is the direct test of the execution-bridge claims under retro pressure.

Model
action capsule set
+ RetroConstraintProfileReceipt
+ ArithmeticScarcityReceipt
→ capsule projection under constraint
→ hash stability audit
→ ordering audit
→ ActionCapsuleRetroAuditReceipt
Primary Module
src/qec/analysis/action_capsule_retro_audit.py
Inputs
action capsule archives
RetroConstraintProfileReceipt
ArithmeticScarcityReceipt
tie-breaking policy declaration
Output
ActionCapsuleRetroAuditReceipt
Bounded Metrics [0,1]
capsule_reconstruction_fidelity
hash_stability_under_reduction
ordering_canonicality
provenance_chain_integrity
semantic_content_retention
Role

If action capsules are genuinely proof-carrying and semantics-preserving, they must survive this audit.

v147.7 — Redundancy Collapse Program (Mandatory)
Purpose

Replace the vague claim that modern systems contain redundant compute with a measured, receipt-carrying proof.

The Redundancy Collapse Program applies QEC invariant analysis to a modern system trace, extracts the invariant core, eliminates provably redundant structure, and projects the result onto constrained surfaces to measure what survives.

Model
modern system trace
→ invariant extraction
→ redundancy identification
→ redundancy elimination
→ constrained projection
→ measurable reduction
→ ComputeReductionReceipt
Primary Module
src/qec/analysis/redundancy_collapse.py
Inputs
modern system execution trace
QEC invariant analysis pipeline
RetroConstraintProfileReceipt for constrained comparison
Outputs
ComputeReductionReceipt
RedundancyMap
InvariantCoreSet
Bounded Metrics [0,1]
compute_waste_eliminated
invariant_retention
structural_loss
Hard Conditions
compute_waste_eliminated must be measured, not asserted
invariant_retention must remain above declared floor
structural_loss must remain strictly bounded
unbounded structural loss is failure
Role

Redundancy claims without measurement are claims without evidence. This module converts claims into receipts.

v147.8 — Cross-Emulator Determinism Proof (Mandatory)
Purpose

Enforce that the same workload, run through multiple emulator variants and host environments, produces identical canonical traces and identical analysis hashes.

Emulator implementation drift is the primary threat to cross-platform replay identity.

Model
same workload
→ multiple emulators / hardware profiles
→ canonical trace normalization
→ QEC analysis
→ hash comparison
→ CrossPlatformReplayReceipt
Primary Module
src/qec/analysis/cross_emulator_determinism_proof.py
Inputs
canonical workload specification
RetroTargetReceipt for each platform
normalized RetroTraceReceipt from each platform
hash comparison policy
Output
CrossPlatformReplayReceipt
Bounded Metrics [0,1]
hash_consensus_fraction
provenance_drift_magnitude
normalization_recovery_rate
residual_mismatch_fraction
cross_platform_homotopy
Required Assertion

All analysis output hashes must match across:

emulator variants
host environments
replay runs

If any mismatch occurs:

it must be recorded in the Failure Ledger
it must not be silently absorbed
the run is not successful
Role

Makes the platform-independence claim falsifiable.

v147.9 — Offline Atari Policy Audit
Purpose

Use Atari 2600 trajectories and offline-control corpora as a static, deterministic policy stress surface for QEC governance and action capsule logic.

The environment is fixed. The ROM is fixed. The trajectory corpus is fixed. The evaluation is replay-only.

Model
offline trajectory corpus
+ canonical ROM / target profile
+ governance / action logic
→ deterministic policy audit
→ OfflineAtariPolicyAuditReceipt
Primary Module
src/qec/analysis/offline_atari_policy_audit.py
Inputs
offline trajectory bundles
ROM hash metadata
target metadata
transition / refinement / governance receipts
action capsules
audit policy
Output
OfflineAtariPolicyAuditReceipt
Bounded Metrics [0,1]
corpus_coverage
action_consistency
counterfactual_stability
replay_alignment
control_entropy_proxy
governance_repeatability
Role

Turns offline control corpora into lawful benchmark surfaces without live stochastic contamination.

v147.10 — Operationalization Battery: Real-World Compute Systems (Mandatory)
Purpose

Apply QEC-style invariant reduction, redundancy elimination, and proof-carrying analysis to real-world compute substrates where the claims of the framework should produce measurable, benchmarkable results.

This branch makes the redundant compute elimination thesis real.

Common Model
system trace archive
+ QEC invariant analysis
→ redundancy map
→ canonical redundancy receipt
→ comparison against baseline
Primary Module
src/qec/analysis/operationalization_battery.py
v147.10.a — Transformer Training Loop Redundancy Audit
Model
transformer training checkpoint archive
+ QEC invariant analysis over gradient update sequences
→ redundancy map
→ TransformerTrainingRedundancyReceipt
Output
TransformerTrainingRedundancyReceipt
Bounded Metrics [0,1]
redundant_layer_execution
activation_similarity
compute_eliminated
qec_analysis_overhead
Role

Measures redundant gradient updates, near-degenerate attention structures, and plateau phases that consume compute without meaningful movement.

v147.10.b — Diffusion Schedule Redundancy Audit
Model
diffusion denoising trajectory archive
+ QEC step-equivalence analysis
→ redundant step map
→ DiffusionRedundancyReceipt
Output
DiffusionRedundancyReceipt
Bounded Metrics [0,1]
step_equivalence_fraction
noise_schedule_redundancy
compute_eliminated
trajectory_stability
Role

Identifies denoising steps that are structurally equivalent to bounded no-ops.

v147.10.c — QEC Decoding Strategy Comparison
Model
QEC decoding policy archive
+ canonical syndrome corpus
→ deterministic decoding comparison
→ DecodingStrategyComparisonReceipt
Output
DecodingStrategyComparisonReceipt
Bounded Metrics [0,1]
decoding_path_redundancy
correction_efficiency
invariant_stability
logical_error_suppression
Role

Benchmarks decoding strategies on the actual target metric, not hand-wavy proxies.

v147.10.d — Scheduling / Orchestration Redundancy Audit
Model
multi-agent task descriptions
→ static graph compilation
→ typed node registry validation
→ execution trace mapping
→ scheduling redundancy extraction
→ SchedulingRedundancyReceipt
Output
SchedulingRedundancyReceipt
Bounded Metrics [0,1]
deterministic_scheduling_fidelity
task_graph_compression
elimination_of_probabilistic_routing
Role

Eradicates probabilistic orchestration bloat in favor of graph-compiled, typed, deterministic workflows.

v147.11 — Adversarial / Pathological Stress Battery (Mandatory)
Purpose

Explicitly try to break the framework.

This battery does not exist to demonstrate that everything works. It exists to find the edge of the system and document it precisely.

Model
bounded deterministic action trajectory
→ adversarial perturbation and schema drift injection
→ invariant recomputation
→ boundary and stability audit
→ AdversarialStressBatteryReceipt
Primary Module
src/qec/analysis/adversarial_stress_battery.py
Threat Classes
Class I — Non-Deterministic Environment Contamination

Targets fail-fast rejection.

Examples:

wall-clock contamination
PRNG field injection
provenance drift injection
Class II — Adversarial Input Construction

Targets ordering canonicality and hash stability.

Examples:

tie-breaking ambiguity under fixed-point arithmetic
projection loss maximizers
semantically distinct inputs engineered to collide structurally
provenance-manipulated action capsule sequences
Class III — Pathological Convergence Cases

Targets bounded-output guarantees.

Examples:

slow convergence lattices
oscillating refinement sequences
near-degenerate invariant spaces
boundary cases between bounded_degradation and instability_exposed
Outputs
AdversarialStressBatteryReceipt
PathologicalConvergenceReceipt
Bounded Metrics [0,1]
AdversarialRobustnessScore
FalseEquilibriumDetection
ConvergenceBoundIntegrity
HashStabilityUnderAttack
Required Assertions

Every run must assert:

all Threat Class I inputs are rejected before analysis
no Threat Class II input can violate hash stability or canonical ordering
all Threat Class III inputs terminate within declared bounds
all failures are classified, not suppressed
Role

The pass criterion is not “all tests pass.” It is “all failures are correctly classified and bounded.”

v147.12 — Failure Ledger (Mandatory)
Purpose

Make failure a first-class output.

Silent error absorption, fallback logic, and unbounded retries are prohibited.

All modules in v147.x must emit:

success metrics
failure conditions
Model
module analysis result
→ success / failure bifurcation
→ canonical metric classification
→ FailureLedgerReceipt
Primary Module
src/qec/analysis/failure_ledger.py
Inputs
module result receipt
failure evidence artifact
failure classification policy
Output
FailureLedgerReceipt
Failure Classes
invalid_under_budget
provenance_mismatch
hash_instability_detected
schema_violation
bound_violation
convergence_failure
projection_overflow
invariant_collapse
invalid_input
Bounded Metrics [0,1]
failure_classification_coverage
suppression_rate (must always be 0)
evidence_completeness
bounded_failure_fraction
Role

A failure in the stress battery is not a defect of the test. It is scientific data extracted by the test.

v147.13 — RETRO Evaluation Pack
Purpose

Aggregate the full retro-constrained proving ground into a singular, reproducible, publication-grade evidence pack.

Nothing proceeds to v148 without this pack.

Model
all v147 receipts
→ canonical aggregation
→ evaluation tables
→ replay evidence bundle
→ RETROEvaluationPackReceipt
Primary Module
src/qec/analysis/retro_evaluation_pack.py
Inputs
all generated receipts from v147.0 through v147.12
Outputs
RETROEvaluationPackReceipt
markdown research summary
benchmark tables
deterministically generated figures / diagrams
replay evidence bundle
Bounded Metrics [0,1]
DocumentationCompleteness
ReproducibilityScore
ConsensusCoverage
FailureLedgerClosure
Deliverables
comparative design summary
arithmetic austerity report
float-emulation audit summary
projection fidelity report
replay certification tables
redundancy collapse report
cross-emulator determinism report
offline policy audit summary
adversarial battery ledger
operationalization benchmark summary
integration notes for v148 promotion
Role

Marks the end of the retro foundation arc. v148 does not begin without this pack.

Research Questions (Answered by Structure)
1. Which invariants survive 4-bit arithmetic, fixed-point lanes, no-FPU systems, and scanline / timing constraints?

Answered by:

v147.3 Arithmetic Scarcity Survival Layer
v147.4 Float-Emulation Audit
v147.5 Canonical Projection + Minimal Representation
2. Do action capsules remain interpretable, hash-stable, and replay-safe under extreme constraint?

Answered by:

v147.6 Action Capsule Retro Audit
v147.8 Cross-Emulator Determinism Proof
3. Can retro systems serve as canonical benchmark surfaces, invariant filters, and determinism proofs?

Answered by:

v147.0 Retro Target Registry
v147.1 Retro Trace Intake Bridge
v147.9 Offline Atari Policy Audit
4. Can redundancy elimination be proven and measured in transformers, diffusion systems, schedulers, and decoding systems?

Answered by:

v147.7 Redundancy Collapse Program
v147.10 Operationalization Battery
5. Where does the system break?

Answered by:

v147.11 Adversarial / Pathological Stress Battery
v147.12 Failure Ledger
v148.x — Optional Promotion to Certified Retro Experiment Layer and Publication

This arc is conditional.

It proceeds only if the v147.x evaluation pack demonstrates all of the following without exception:

retro-constrained testing produced useful, non-trivial evidence
no unclassified violation of determinism, replay identity, boundedness, or architecture cleanliness occurred
the adversarial battery found bounded, classified failures rather than unbounded ones
the operationalization battery produced measurable, reproducible benchmark receipts
at least one cross-target invariant consensus pattern has been documented

If these conditions are not met, v148 is deferred. There is no partial promotion.

v148.0 — Promotion Entry Gate
Purpose

Codify the promotion criteria from v147.x into a single admissibility gate.

Model
RETROEvaluationPackReceipt
+ FailureLedgerReceipt aggregate
→ promotion admissibility check
→ PromotionGateReceipt
Primary Module
src/qec/promotion/promotion_gate.py
Inputs
RETROEvaluationPackReceipt
aggregated failure ledger
promotion criteria policy
Output
PromotionGateReceipt
Bounded Metrics [0,1]
promotion_readiness
evidence_sufficiency
bounded_failure_acceptability
Role

Prevents premature promotion.

v148.1 — Proof-Carrying Retro Experiment Capsule
Purpose

Bind complete retro experiment descriptions to proof-carrying capsule representations using the action capsule architecture established in v146.

This is not live execution.

Model
retro target
+ experiment definition
+ invariant test
+ policy audit
→ proof-carrying experiment capsule
→ RetroExperimentCapsuleReceipt
Primary Module
src/qec/promotion/retro_experiment_capsule.py
Inputs
retro target profile
experiment specification
relevant v147 receipts
promotion policy
Output
RetroExperimentCapsuleReceipt
Bounded Metrics [0,1]
capsule_certification_fidelity
experiment_replay_closure
target_binding_integrity
Role

Packages proven experiment descriptions into certifiable, non-live artifacts.

v148.2 — Cross-Target Invariant Consensus Bridge
Purpose

Determine whether the same invariant pattern survives across divergent target families.

Model
same analysis
across multiple registered targets
→ consensus map
→ divergence map
→ CrossTargetConsensusReceipt
Primary Module
src/qec/promotion/cross_target_consensus.py
Inputs
per-target invariant receipts
cross-target comparison policy
promotion thresholds
Output
CrossTargetConsensusReceipt
Bounded Metrics [0,1]
cross_target_consensus_fraction
majority_retention_fraction
divergence_classification_coverage
Role

Produces the first rigorous cross-family stability map.

v148.3 — Adversarial Hardening Loop
Purpose

Close the loop between adversarial failure and deterministic hardening.

Hardening means tighter validation, tighter bound declarations, and better fail-fast coverage — not relaxed invariants.

Model
adversarial battery failure ledger
→ mitigation specification
→ validation layer update
→ re-run adversarial battery
→ HardeningReceipt
Primary Module
src/qec/promotion/adversarial_hardening_loop.py
Inputs
adversarial battery ledger
mitigation policy
validation update specification
Output
HardeningReceipt
Bounded Metrics [0,1]
mitigation_effectiveness
failure_recurrence_reduction
bound_tightening_quality
Role

Produces an auditable record of what was broken and how it was fixed.

v148.4 — Operationalization Evidence Publication
Purpose

Prepare the operationalization battery results for external publication.

Model
operationalization receipts
→ comparative benchmark synthesis
→ reproducibility bundle
→ OperationalizationPublicationReceipt
Primary Module
src/qec/promotion/operationalization_publication.py
Inputs
all operationalization receipts
baseline comparison tables
reproducibility artifacts
Outputs
OperationalizationPublicationReceipt
benchmark report
reproducibility bundle
receipt archive
Bounded Metrics [0,1]
publication_readiness
benchmark_reproducibility
claim_receipt_coverage
Role

Ensures every claim about compute reduction is backed by a receipt.

v148.5 — Retro Publication and Documentation Release
Purpose

Finalize the public-facing research and documentation material for the complete retro arc.

Model
v147 evaluation pack
+ v148 consensus results
→ paper draft
→ benchmark appendix
→ documentation release
→ RetroPublicationReceipt
Primary Module
src/qec/promotion/retro_publication_release.py
Inputs
RETROEvaluationPackReceipt
CrossTargetConsensusReceipt
operationalization publication receipts
Outputs
RetroPublicationReceipt
paper draft
README updates
technical documentation
benchmark appendix
replay certification summary
Bounded Metrics [0,1]
methods_completeness
results_coverage
limitations_honesty
documentation_integrity
Role

Turns the retro arc into a reproducible public research artifact.

v148.6 — Optional Demonstration Layer
Purpose

If v148.1–v148.5 produce a stable retro experiment capsule system, define a narrow, tightly bounded demonstration surface.

This layer is strictly analysis-output-only. It does not execute on retro hardware. It does not drive emulators. It does not produce stochastic outputs.

Model
certified receipt
→ deterministic projection rendering
→ human-readable demonstration artifact
→ DemonstrationLayerReceipt
Primary Module
src/qec/promotion/demonstration_layer.py
Inputs
certified receipts
projection receipts
rendering policy
Outputs
DemonstrationLayerReceipt
tile grids
scanline diagrams
tracker-pattern tables
fixed-register summaries
Bounded Metrics [0,1]
visualization_integrity
receipt_render_fidelity
projection_traceability
Role

A receipt renderer, not an executor.

Guardrails for the New Roadmap
Never Allowed
stochastic execution
live RL inside core QEC
PRNG-driven routing
wall-clock dependence
hidden emulator-state assumptions
decoder-core modification
architecture-specific silent fallback
dynamic heap growth as an untracked escape hatch
probabilistic orchestration dressed up as determinism
suppressing failures in any battery
using retro targets as an excuse for non-canonical shortcuts
Required Mitigations
quantization rules must be explicit and canonical
tie-breaking rules must be explicit and canonical
float-emulation policy must be explicit and auditable
emulator provenance must be declared and reproducible
all traces must be normalized before comparison
all receipts must serialize canonically and hash stably
all adversarial failures must be typed and logged
all promotion claims must be backed by receipts
no result may be published without corresponding canonical evidence
Final Strategic Direction
v143.5
→ completed deterministic runtime proof

v144.x – v146.x
→ orchestration, governance, benchmarking, and proof-carrying action representation

v147.x
→ retro-constrained proving ground for invariants, action capsules, and “quantumy” structure

v148.x
→ optional promotion to certified retro experiment layer and publication

The post-v146 phase is not an extension of what came before. It is a deliberate shift from construction to empirical pressure.

Every module in v147.x exists to test a claim made by v143–v146.

Every receipt in v147.x exists to certify that the testing was rigorous.

The retro arc is not nostalgia and not toy research.

The operationalization arc is not vague futurism.

The adversarial arc is not decorative.

The law holds throughout:

same input
→ same ordering
→ same canonical JSON
→ same stable hash
→ same bytes
