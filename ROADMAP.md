# QSOLKCB / QEC — ROADMAP.md

## Deterministic Reasoning • Canonical Identity • Replay-Safe Proof Artifacts • QEC Systems • Operator Surfaces • Reproducible Scientific Workflows

---

# 🧭 Stable Tip Metadata

```text
latest completed release → v162.2
current frontier → v163.0
next work                → Heavy Dependency Invariant Discovery
active arc               → v163.x — Heavy Dependency Invariant Discovery
completed arc            → v162.x — IRC Operator Control Surface
```

Published tags are authoritative.

If this roadmap disagrees with published release history:

- published release history wins
- roadmap/context/prompt must be corrected

Stable lineage remains anchored to v137.* compatibility contracts.

## 🧠 Core Identity

QEC is a deterministic, replay-safe proof system for:

- quantum error correction
- invariant-driven computation
- canonical identity
- proof receipts
- semantic resonance
- governance
- distributed convergence
- lattice/router/readout proof
- multi-scale invariance
- entropy and decay signatures
- GameWorld interaction boundaries
- perturbation and substrate contracts
- bounded recursive proof loops
- reality-loop composition
- global validation / truth / replay receipts
- operator control surfaces
- scientific backend receipts
- reproducible build receipts
- symbolic grammar boundaries

QEC is NOT:

- a probabilistic optimizer
- a loose AI-agent framework
- a symbolic metaphor engine without receipts
- a runtime that silently trusts external interpretation
- a game-playing bot
- an RL trainer
- a source-code execution sandbox
- a world simulator
- a gameplay automation framework
- a renderer
- a stochastic policy engine
- a substrate/hardware authority
- a philosophical truth engine
- a medical or biological claim engine

QEC does not need to make messy worlds deterministic.

QEC makes the boundary deterministic.

Core identity:

**QEC does not merely compute. QEC proves.**

## 🧠 Core Law — System Invariant

```text
same input
→ same ordering
→ same canonical JSON
→ same stable SHA-256 hash
→ same bytes
→ same proof artifact
→ same outcome
```

Violation:

```text
SYSTEM INVALID
```

Every roadmap item must produce:

- contract
- artifact
- hash
- receipt
- canonical JSON
- validation rule
- failure mode
- deterministic replay test

If an idea cannot produce those, it remains inspiration — not QEC.

## ✅ Completed Proof Spine — v151 → v161

- v151 → ingestion / semantic resonance / replay validation
- v152 → reversible layers / compression
- v153 → lattice / router / readout / masks / shifts / replay proofs
- v154 → multi-scale invariance / Sierpinski compression / scale receipts
- v155 → entropy drift / decay signatures / decay-resistance proofs
- v156 → GameWorld intake / adapters / observations / traces / probes / reports
- v157 → perturbation contracts / stress receipts
- v158 → substrate constraint contracts / material encoding / substrate drift
- v159 → bounded recursive proof loops / convergence receipts
- v160 → reality-loop composition / cross-arc identity links / proof receipts
- v161 → global validation index / threshold truth receipts / global replay proofs

Current terminal proof chain:

```text
reality_loop_proof_receipt_hash
→ global_validation_entry_hash
→ global_validation_index_hash
→ global_threshold_contract_hash
→ global_truth_receipt_hash
→ replay_record_hash
→ global_replay_proof_hash
→ irc_replay_audit_hash                             (v162.2)
→ heavy_dependency_discovery_manifest_hash          (v163.0)
→ dependency_hotpath_receipt_hash                   (v163.1)
→ backend_invariant_candidate_hash                  (v163.2)
→ cross_backend_equivalence_receipt_hash            (v163.3)
→ optimization_opportunity_index_hash               (v163.4)
→ optimization_contract_hash                        (v164.0)
→ lightweight_adapter_spec_hash                     (v164.1)
→ cached_canonical_kernel_receipt_hash              (v164.2)
→ fast_path_equivalence_receipt_hash                (v164.3)
→ optimization_implementation_receipt_hash          (v164.4)
→ dependency_reduction_receipt_hash                 (v164.5)
→ optimized_simulation_spec_hash                    (v165.0)
→ backend_equivalence_replay_receipt_hash           (v165.1)
→ optimized_qec_benchmark_receipt_hash              (v165.2)
→ optimized_telemetry_receipt_hash                  (v165.3)
→ optimized_simulation_report_hash                  (v165.4)
→ qldpc_construction_receipt_hash (v166.x)
→ syndrome_stream_receipt_hash
→ control_plane_manifest_hash
→ resource_overhead_receipt_hash
→ ieee754_precision_format_manifest_hash
→ safe_bit_reinterpretation_policy_hash
→ fast_approximation_receipts_hash
→ fast_inv_sqrt_receipt_hash
→ fast_exp_receipt_hash
→ sign_bit_operation_receipts_hash
→ ulp_epsilon_receipt_hash
→ float_ordering_receipt_hash
→ reduced_precision_adapter_receipt_hash
→ float_integer_test_receipt_hash
→ hardware_float_adapter_boundary_hash
→ contextuality_threshold_receipt_hash
→ qml_boundary_receipt_hash
→ tool_dispatch_replay_proof_hash
→ interpretability_boundary_receipt_hash
→ reproducible_publication_receipt_hash
→ benchmark_ladder_receipt_hash
→ materials_signal_boundary_receipt_v2_hash
→ symbolic_compiler_manifest_hash
→ audit_trail_receipt_hash
→ experiment_scheduler_replay_proof_hash
→ cross_environment_replay_receipt_hash
→ federated_operator_audit_receipt_hash
→ hermetic_environment_receipt_hash
→ global_proof_composition_v2_hash
```

## ✅ Stabilization Releases

- v161.2.1 → documentation / installer / dependency normalization
- v161.2.2 → pytest environment cleanup / warning cleanup / deterministic local tool stubs / qldpc upstream-source dependency policy clarification

## 🧰 Post-v161 Repository Hygiene Priorities

These are not new proof arcs, but high-leverage stabilization work.

- **License and SPDX clarity**: resolve root license and file-level SPDX ambiguity; document software license policy clearly.
- **Version-line policy**: clarify relationship between research tags, Python package versions, and Rust TUI versions.
- **Dependency governance**: distinguish default/dev/science/external extras; document upstream-source dependency policy; avoid unrestricted dependency resolution in deterministic environments.
- **Installer and release provenance hardening**: move beyond raw `curl | sh`; add checksum/signature/manifest receipts; fail closed on installer mismatch.
- **Declarative artifact-chain registry**: reduce hard-coded global validation tables; preserve byte-identical deterministic ordering; registry itself must be hash-bound.
- **Backend adapter receipts**: wrap Stim / PyMatching / Qiskit Aer / QuTiP / qldpc outputs as receipts; external tools are adapters, never authorities.
- **Contributor and test matrix**: distinguish required vs optional backends; document skip policy; document upstream-source dependency workflow.

## Phase: v162.x — IRC Operator Control Surface

**Status**
PLANNED

**Source-Grounded Motivation**
Operator interaction requires a bounded control path with deterministic command parsing and replay-safe logs. Local IRC command surfaces provide a narrow interface that can be audited without expanding proof authority.

The phase is constrained to local-first operator workflows and deterministic routing contracts. No network-driven authority or autonomous external decision-making is introduced.

**Arc Reinterpretation**
"Local operator commands become deterministic receipts, not trusted execution authority."

**Planned Releases**
- v162.0 → Local IRC Server Core
- v162.1 → QEC IRC Command Router
- v162.2 → IRC Replay Audit Receipt

**Expected Modules**
- operator command parser
- local IRC loop adapter
- replay log normalizer

**Expected Artifacts**
- `irc_command_manifest.json`
- `irc_replay_audit_receipt.json`
- `irc_replay_audit_hash`

**Core Rule**
```text
same local command stream
+ same canonical IRC normalization
→ same irc_replay_audit_hash
```

**Acceptance Gates**
- pytest: IRC command parsing determinism tests pass
- pytest: replay receipt hash-stability tests pass
- dependency boundary: local-only mode required by default

**Must Not Do**
- no remote command authority
- no hidden operator side-channel
- no nondeterministic command ordering

**Dependency Boundaries**
- depends on: v161 replay/receipt contracts
- does not modify: src/qec/decoder/

**Relationship to Existing Roadmap**
Retains v162 scope while tightening local-only and replay-audit constraints.

## Phase: v163.x — Heavy Dependency Invariant Discovery

**Status**
PLANNED

**Source-Grounded Motivation**
QEC now deliberately supports heavyweight scientific dependencies because they are useful for simulation, comparison, visualization, and quantum-toolchain interoperability. However, accepting heavy dependencies does not mean trusting them silently or letting them become permanent runtime burdens.

The purpose of v163.x is to inspect heavy dependencies and discover deterministic invariants:

- stable input/output equivalence classes
- import-time and runtime hot paths
- matrix-shape / dtype invariants
- sparse/dense conversion boundaries
- backend parity relationships
- repeated canonicalization opportunities
- render-free plotting data paths
- simulator configuration invariants
- deterministic fast-path candidates
- cacheable kernels
- dependency surface minimization opportunities

This is discovery only.

v163.x does not implement optimizations yet.

It produces receipt-bound evidence about where optimizations may safely exist.

**Arc Reinterpretation**
"Heavy dependency" means an external scientific backend whose useful behavior must be mapped into deterministic invariants before QEC can rely on it.

The dependency is not the authority.
The invariant is not the optimization.
The discovery receipt is the boundary.

**Planned Releases**
- v163.0 → HeavyDependencyDiscoveryManifest
- v163.1 → DependencyImportAndHotPathReceipt
- v163.2 → BackendInvariantCandidateReceipt
- v163.3 → CrossBackendEquivalenceReceipt
- v163.4 → OptimizationOpportunityIndex

**Expected Modules**
- src/qec/analysis/heavy_dependency_discovery.py
- src/qec/analysis/dependency_hotpath_receipts.py
- src/qec/analysis/backend_invariant_candidate_receipts.py
- src/qec/analysis/cross_backend_equivalence_receipts.py
- src/qec/analysis/optimization_opportunity_index.py

**Expected Artifacts**
- HeavyDependencyDiscoveryManifest
- DependencyImportAndHotPathReceipt
- BackendInvariantCandidateReceipt
- CrossBackendEquivalenceReceipt
- OptimizationOpportunityIndex
- HeavyDependencyDiscoveryReceipt

**Expected Hashes**
- heavy_dependency_discovery_manifest_hash
- dependency_hotpath_receipt_hash
- backend_invariant_candidate_hash
- cross_backend_equivalence_receipt_hash
- optimization_opportunity_index_hash
- heavy_dependency_discovery_receipt_hash

**Core Rule**
```text
same dependency manifest
+ same canonical input corpus
+ same backend versions
+ same precision policy
+ same discovery probes
→ same invariant candidates
→ same heavy_dependency_discovery_receipt_hash
```

**Acceptance Gates**
- pytest: discovery manifest canonical JSON is stable
- pytest: backend inventory ordering is deterministic
- pytest: optional missing backend is represented as unavailable, not failure
- pytest: no unrestricted PyPI resolution
- pytest: no live network calls
- pytest: import hotpath receipt is deterministic across PYTHONHASHSEED values
- pytest: invariant candidates include source backend, input corpus hash, and precision policy
- pytest: cross-backend equivalence uses declared ULP / exact / structural comparison mode
- pytest: optimization opportunity index is deterministic and sorted by declared ranking tuple
- dependency boundary: no import from src/qec/decoder/
- dependency boundary: external backends are adapters, never authorities

**Must Not Do**
- no optimization implementation in v163.x
- no replacing dependencies yet
- no performance claim without benchmark receipt
- no hidden randomness
- no live cloud backend calls
- no network dependency in tests
- no unbounded benchmark loops
- no backend output accepted without receipt wrapping
- no mutation of proof artifacts

**Dependency Boundaries**
- depends on: v162.x operator surface for local inspection commands later
- depends on: v161 global replay contracts for deterministic receipt discipline
- uses: QuTiP / Qiskit / Qiskit Aer / SciPy / NumPy / pandas / matplotlib / qldpc / Stim / PyMatching / mido as optional adapters
- does not modify: src/qec/decoder/
- does not implement: optimization fast paths

**Relationship to Existing Roadmap**
v163.x replaces and expands the prior Scientific Backend Normalization arc. It keeps backend normalization but adds invariant discovery, hot-path analysis, equivalence class detection, and optimization opportunity indexing.

This prepares v164.x to safely exploit the discovered invariants.

## Phase: v164.x — Invariant-Based Heavy Dependency Optimization

**Status**
PLANNED

**Source-Grounded Motivation**
v163.x discovers deterministic invariants in heavy dependencies. v164.x turns those invariants into explicit optimization contracts.

**Arc Reinterpretation**
"Optimization" means exploiting a discovered deterministic invariant under a declared equivalence contract.

A fast path is not accepted because it is fast.
A fast path is accepted only if replay proves equivalence.

**Planned Releases**
- v164.0 → OptimizationContract
- v164.1 → LightweightAdapterSpec
- v164.2 → CachedCanonicalKernelReceipt
- v164.3 → FastPathEquivalenceReceipt
- v164.4 → OptimizationImplementationReceipt
- v164.5 → DependencyReductionReceipt

**Expected Modules**
- src/qec/analysis/optimization_contracts.py
- src/qec/analysis/lightweight_adapter_specs.py
- src/qec/analysis/cached_canonical_kernel_receipts.py
- src/qec/analysis/fast_path_equivalence_receipts.py
- src/qec/analysis/optimization_implementation_receipts.py
- src/qec/analysis/dependency_reduction_receipts.py

**Expected Artifacts**
- OptimizationContract
- LightweightAdapterSpec
- CachedCanonicalKernelReceipt
- FastPathEquivalenceReceipt
- OptimizationImplementationReceipt
- DependencyReductionReceipt

**Expected Hashes**
- optimization_contract_hash
- lightweight_adapter_spec_hash
- cached_canonical_kernel_receipt_hash
- fast_path_equivalence_receipt_hash
- optimization_implementation_receipt_hash
- dependency_reduction_receipt_hash

**Dependency Boundaries**
- depends on: v163.x HeavyDependencyDiscoveryReceipt
- does not modify: src/qec/decoder/

## Phase: v165.x — Optimized QEC Simulation Backends

**Status**
PLANNED

**Source-Grounded Motivation**
After v163.x discovers invariants and v164.x implements optimization contracts, QEC needs an applied simulation layer that uses those contracts safely.

**Planned Releases**
- v165.0 → OptimizedSimulationSpec
- v165.1 → BackendEquivalenceReplayReceipt
- v165.2 → OptimizedQECBenchmarkReceipt
- v165.3 → OptimizedTelemetryReceipt
- v165.4 → OptimizedSimulationReport

**Expected Hashes**
- optimized_simulation_spec_hash
- backend_equivalence_replay_receipt_hash
- optimized_qec_benchmark_receipt_hash
- optimized_telemetry_receipt_hash
- optimized_simulation_report_hash

**Dependency Boundaries**
- depends on: v164.x OptimizationImplementationReceipt
- depends on: v163.x HeavyDependencyDiscoveryReceipt
- does not modify: src/qec/decoder/

## Phase: v166.x — QLDPC / Hashing-Bound Code Receipts

**Status**
PLANNED

## Phase: v167.x — Qudit / Ququart / High-Dimensional Stabilizer Receipts

**Status**: PLANNED

## Phase: v168.x — Proof Telemetry / MIDI / Sonification Receipts
**Status**: PLANNED

## Phase: v169.x — Symbolic Geometry Grammar / Cosmovirus Sandbox
**Status**: PLANNED

## Phase: v170.x — Reproducible Build / Supply-Chain Receipts
**Status**: PLANNED

## Phase: v171.x — Deterministic Knowledge Base / Agent Memory Receipts
**Status**: PLANNED

## Phase: v172.x — Graphene / Photonic / Diamond / Materials Signal Receipts
**Status**: PLANNED

## Phase: v173.x — Interactive Proof Worlds / Citizen-Science Game Receipts
**Status**: PLANNED

## Phase: v174.x — BP Dynamics / Fixed-Point Trap Receipts
**Status**: PLANNED

## Phase: v175.x — Operator Console Unification
**Status**: PLANNED

## Phase: v176.x — Real-Time Syndrome Streaming Receipts
**Status**: PLANNED

**Planned Releases**
- v176.0 → SyndromeStreamManifest
- v176.1 → StreamingWindowReceipt
- v176.2 → SyndromeReplayReceipt

## Phase: v177.x — Hardware Abstraction / Control-Plane Receipts
**Status**: PLANNED

## Phase: v178.x — Fault-Tolerant Resource Accounting Receipts
**Status**: PLANNED

**Planned Releases**
- v178.0 → DistillationOverheadReceipt
- v178.1 → LogicalCycleOverheadReceipt
- v178.2 → ResourceBudgetReplayReceipt

## Phase: v178.5.x — IEEE 754 Precision & Approximation Receipts

**Status**
PLANNED

**Source-Grounded Motivation**
IEEE 754 floating-point numbers encode sign, biased exponent, and mantissa as structured bit fields. Reinterpreting a positive normalized float as an unsigned integer exposes monotonic ordering and ULP-distance relationships. This enables deterministic precision contracts for ULP comparison, fast inverse square root, fast approximate powers, fast natural exponentiation, fast log2/exp2 approximations, exponent-floor extraction, geometric mean via integer representation, sign-bit operations, direct integer comparison of positive normalized floats, integer tests over float bit patterns, FP16/BF16 boundaries, and hardware exponent-construction paths.

QEC does not treat these as unchecked performance tricks. QEC treats them as bounded approximation contracts. Every approximation must declare precision format, safe bit reinterpretation policy, input domain, error bound, refinement/Newton iteration count, constants, overflow/subnormal/NaN/Inf behavior, signed-zero policy, NaN payload policy, and validation rule before proof-pipeline use.

Reduced-precision formats such as FP16 and BF16 are especially important because they change mantissa width and rounding behavior. BF16 preserves FP32 exponent range with reduced mantissa precision; FP16 reduces both range and precision. Hardware paths (including AMD Vitis-style exponent construction) are adapter signals only and must be wrapped as receipts.

**Arc Reinterpretation**
```text
"IEEE 754 bit manipulation" means deterministic precision contracts
over declared floating-point formats.

It does not mean unchecked performance hacks.
It does not mean silent float equality.
It does not mean hardware authority.
It does not mean approximation without an error bound.
It does not mean unsafe pointer-punning.

The receipt is the boundary.
The hack is not the proof.
```

**Core IEEE 754 Structural Facts**
```text
FP32 layout:
  sign      = 1 bit
  exponent  = 8 bits
  mantissa  = 23 bits
  bias      = 127

FP64 layout:
  sign      = 1 bit
  exponent  = 11 bits
  mantissa  = 52 bits
  bias      = 1023

FP16 layout:
  sign      = 1 bit
  exponent  = 5 bits
  mantissa  = 10 bits
  bias      = 15

BF16 layout:
  sign      = 1 bit
  exponent  = 8 bits
  mantissa  = 7 bits
  bias      = 127
```

```text
Special values:
exponent all 1s + mantissa zero      → infinity
exponent all 1s + mantissa nonzero   → NaN
exponent zero  + mantissa zero       → signed zero
exponent zero  + mantissa nonzero    → subnormal
```

**Historical IEEE 754 Hack Families (Receipt-Bound)**
```text
historical pseudocode
source signal
not implementation guidance
```

- `fast_inv_sqrt(x)` source signal (Quake-style), magic constant `0x5f3759df`; requires declared input domain, constants, iteration count, error bounds, special-value behavior → `fast_inv_sqrt_receipt_hash`.
- `fast_exp(x)` exponent-field construction; requires declared scale/bias constants, overflow boundary, domain, error bound → `fast_exp_receipt_hash`.
- `fast_pow(x, c)` affine transform of integer representation; requires declared bias, parameter, domain, refinement policy, error bound → `fast_pow_receipt_hash`.
- `fast_log2(x)` and `fast_log2_floor(x)` require explicit approximation mode (`exponent_floor` for floor path), mantissa-correction policy, and error semantics → `fast_log2_receipt_hash`, `fast_log2_floor_receipt_hash`.
- `fast_exp2(x)` and `fast_geometric_mean(array[n])` require bounded domains, deterministic ordering, and declared precision → `fast_exp2_receipt_hash`, `fast_gmean_receipt_hash`.
- sign-bit operations (`fast_abs`, `fast_negate`) require signed-zero and NaN payload policy → `sign_bit_abs_receipt_hash`, `sign_bit_negate_receipt_hash`.
- positive normalized floats integer ordering and float integer tests require sign-domain and special-value policy declarations.

Safe bit reinterpretation only:
```text
C/C++: std::memcpy or std::bit_cast (C++20)
Rust:  f32::to_bits / f32::from_bits, f64::to_bits / f64::from_bits
Python: struct.pack/unpack with explicit endian policy
NumPy: explicit dtype/view conversions with width/endian policy
```

Unsafe pointer-punning (e.g., `*(int*)&x`) is strict-aliasing unsafe and is a source signal only.

**Planned Releases**
- v176.5.0 → IEEE754PrecisionFormatManifest
- v176.5.1 → SafeBitReinterpretationPolicy
- v176.5.2 → FastApproximationReceipts
- v176.5.3 → SignBitOperationReceipts
- v176.5.4 → ULPEpsilonComparisonReceipt
- v176.5.5 → FloatOrderingReceipt
- v176.5.6 → ReducedPrecisionAdapterReceipt
- v176.5.7 → FloatIntegerTestReceipt
- v176.5.8 → HardwareFloatAdapterBoundary

**Expected Modules**
- src/qec/analysis/ieee754_precision_format_manifest.py
- src/qec/analysis/safe_bit_reinterpretation_policy.py
- src/qec/analysis/fast_approximation_receipts.py
- src/qec/analysis/sign_bit_operation_receipts.py
- src/qec/analysis/ulp_epsilon_receipts.py
- src/qec/analysis/float_ordering_receipts.py
- src/qec/analysis/reduced_precision_adapter_receipts.py
- src/qec/analysis/float_integer_test_receipts.py
- src/qec/analysis/hardware_float_adapter_boundary.py

**Expected Artifacts**
- IEEE754PrecisionFormatManifest
- SafeBitReinterpretationPolicy
- FastApproximationReceipt
- FastInvSqrtReceipt
- FastPowReceipt
- FastExpReceipt
- FastExp2Receipt
- FastLog2Receipt
- FastLog2FloorReceipt
- FastGeometricMeanReceipt
- SignBitAbsReceipt
- SignBitNegateReceipt
- ULPEpsilonComparisonReceipt
- FloatOrderingReceipt
- ReducedPrecisionAdapterReceipt
- FloatIntegerTestReceipt
- HardwareFloatAdapterBoundaryReceipt

**Expected Hashes**
- ieee754_precision_format_manifest_hash
- safe_bit_reinterpretation_policy_hash
- fast_approximation_receipts_hash
- fast_inv_sqrt_receipt_hash
- fast_pow_receipt_hash
- fast_exp_receipt_hash
- fast_exp2_receipt_hash
- fast_log2_receipt_hash
- fast_log2_floor_receipt_hash
- fast_gmean_receipt_hash
- sign_bit_abs_receipt_hash
- sign_bit_negate_receipt_hash
- ulp_epsilon_receipt_hash
- float_ordering_receipt_hash
- reduced_precision_adapter_receipt_hash
- float_integer_test_receipt_hash
- hardware_float_adapter_boundary_hash

**Core Rule**
```text
same float input
+ same declared precision format
+ same safe bit reinterpretation policy
+ same approximation primitive
+ same magic / bias constants
+ same refinement count
+ same input-domain policy
+ same special-value policy
→ same bounded approximation output
→ same approximation receipt hash
```

**Acceptance Gates**
- pytest: IEEE754PrecisionFormatManifest produces canonical JSON
- pytest: FP32 / FP64 / FP16 / BF16 manifests remain stable across PYTHONHASHSEED values
- pytest: SafeBitReinterpretationPolicy rejects pointer-punning / strict-aliasing unsafe methods
- pytest: fast_inv_sqrt receipt declares `0x5f3759df` or explicit alternative
- pytest: fast_pow receipt declares bias constant and refinement count
- pytest: fast_exp receipt declares scale/bias constants and overflow domain
- pytest: fast_log2_floor receipt is labelled exponent_floor, not exact logarithm
- pytest: sign_bit operations declare NaN payload and signed-zero policies
- pytest: ULP epsilon receipt rejects NaN unless NaN policy is explicitly declared
- pytest: FloatOrderingReceipt rejects negative inputs unless sign-corrected ordering is declared
- pytest: ReducedPrecisionAdapterReceipt declares mantissa width and rounding policy
- pytest: FloatIntegerTestReceipt declares normal/subnormal/zero/NaN/Inf behavior
- pytest: HardwareFloatAdapterBoundaryReceipt has `adapter_only=true`
- dependency boundary: no import from src/qec/decoder/
- dependency boundary: no hardware SDK dependency in tests
- dependency boundary: no C extension / SIMD intrinsic required

**Must Not Do**
- no Silent float equality in proof-sensitive validators
- no approximation without declared error bound
- no undeclared precision format
- no unsafe C/C++ pointer-punning as implementation guidance
- no strict-aliasing undefined behavior
- no fast_log2_floor labelled as exact log2
- no sign-bit operation without signed-zero and NaN policy
- no FP16 / BF16 output without reduced-precision receipt
- no hardware authority claim

**Dependency Boundaries**
- depends on: v178.x resource accounting receipts
- used by: v179.x contextuality / topology receipts
- used by: v180.x QML boundary receipts
- used by: v184.x benchmark ladder receipts
- used by: v185.x materials / photonic / device signal receipts
- used by: v189.x cross-environment replay receipts
- does not modify: src/qec/decoder/
- does not require: hardware SDKs, SIMD intrinsics, C extensions
- authoritative standard: IEEE 754-2019
- source signals: leegao/float-hacks, randomascii, AMD Vitis tutorial, FP16/BF16 guides, kaatinga IEEE754 notes

**Relationship to Existing Roadmap**
v178.x establishes resource accounting receipts with float-bearing metrics. v178.5.x adds the precision-contract layer before v179+ phases consume those values.

## Phase: v179.x — Quantum Geometry / Contextuality / Topological Toolkit Receipts
**Status**: PLANNED

**Planned Releases**
- v179.0 → ContextualityThresholdReceipt
- v179.1 → TopologicalBoundaryReceipt
- v179.2 → GeometryReplayValidationReceipt

## Phase: v180.x — Deterministic Quantum ML Boundary Receipts
**Status**: PLANNED

## Phase: v181.x — Local Agent / Tool Dispatch Receipts
**Status**: PLANNED

## Phase: v182.x — Interpretability / Sparse Feature Receipts
**Status**: PLANNED

## Phase: v183.x — Reproducible Research Publication Receipts
**Status**: PLANNED

## Phase: v184.x — Benchmark Ladder / External Comparator Receipts
**Status**: PLANNED

## Phase: v185.x — Photonic / Materials / Device Signal Receipts v2
**Status**: PLANNED

## Phase: v186.x — Symbolic Diagram Compiler v2 / Grammar Extension
**Status**: PLANNED

**Planned Releases**
- v186.0 → SymbolicTokenizerReceipt v2
- v186.1 → DiagramCompilerManifest v2
- v186.2 → SymbolicBoundaryReplayReceipt v2

## Phase: v187.x — Human Audit / Red-Team Receipts
**Status**: PLANNED

## Phase: v188.x — Deterministic Experiment Scheduler
**Status**: PLANNED

## Phase: v189.x — Cross-Environment Hardware/OS Replay Receipts
**Status**: PLANNED

## Phase: v190.x — Operator / IRC / TUI / CLI Federation Receipts v2
**Status**: PLANNED

## Phase: v191.x — Reproducible Build / Hermetic Environment Receipts v2
**Status**: PLANNED

## Phase: v192.x — Global Proof Composition v2

**Status**
PLANNED

Purpose:
- register all post-v173 receipts
- bind source-grounded claims
- bind benchmark receipts
- bind operator receipts
- bind reproducible build receipts
- bind symbolic claim boundaries
- produce `global_proof_composition_v2_hash`

**Planned Releases**
- v192.0 → PostV173ArtifactRegistry
- v192.1 → SourceClaimCompositionReceipt
- v192.2 → BenchmarkAndHardwareBoundaryCompositionReceipt
- v192.3 → GlobalProofCompositionV2

**Core rule**
```text
same post-v173 artifact registry
+ same source claim receipts
+ same benchmark boundary receipts
+ same operator/reproducibility receipts
→ same global_proof_composition_v2_hash
```

**Must Not Do**
- no global truth expansion beyond registered receipts
- no semantic truth claims
- no hardware authority claims
- no symbolic claim drift
- no benchmark marketing claims

## Research Basis

| Cluster | Representative Sources |
|---|---|
| QEC hardware progress | Google Willow, Riverlane roadmap, neutral-atom QEC, two-gross code |
| QLDPC and hashing bound | npj Quantum Information APM-LDPC, parity-unfolded distillation |
| Real-time decoding | LEGO decoder architecture |
| Quantum geometry and contextuality | contextuality/code-switching, quantum geometry toolkit |
| Materials and photonic devices | diamond SiV sensing, Nature Photonics CV optics, QKD, Nature Materials |
| Reproducible builds | Debian Forky mandatory reproducibility |
| Agent memory and tool dispatch | SkillOS, FAMA, hybrid memory, button-pushing explorers, LLM shebang |
| Interpretability | Qwen-Scope SAE |
| Recursive multi-agent | RecursiveMAS |
| Scientific backends | Qiskit v2.4, Stim, PyMatching, Qiskit Aer |
| Heavy dependency invariant discovery | QuTiP, Qiskit, Qiskit Aer, SciPy, NumPy, pandas, matplotlib, qldpc, Stim, PyMatching |
| Invariant-based dependency optimization | import hot-path analysis, sparse/dense boundary detection, backend equivalence testing, canonical kernel caching |
| Optimized QEC simulation backends | QuTiP/Qiskit parity, internal QLDPC construction, optimized telemetry, benchmark receipts |
| Symbolic grammar | qec_theory_diagram.txt |
| Benchmarking | Riverlane, Google Willow, Qiskit, Phoronix, quantization comparison |
| IEEE 754 precision / approximation | IEEE 754-2019, leegao/float-hacks, randomascii floating-point format tricks, AMD Vitis IEEE-754 Format Trick, FP16/BF16 precision guides, kaatinga IEEE754 bit-inspection notes |

Some sources are research signals rather than primary authority. Inaccessible sources must be marked as `source_inaccessible` and cannot be sole justification for a phase.

## Updated Global Artifact Hash Chain (v151 → v192)

```text
reality_loop_proof_receipt_hash
→ global_validation_entry_hash
→ global_validation_index_hash
→ global_threshold_contract_hash
→ global_truth_receipt_hash
→ replay_record_hash
→ global_replay_proof_hash
→ syndrome_stream_receipt_hash
→ control_plane_manifest_hash
→ resource_overhead_receipt_hash
→ ieee754_precision_format_manifest_hash
→ safe_bit_reinterpretation_policy_hash
→ fast_approximation_receipts_hash
→ fast_inv_sqrt_receipt_hash
→ fast_exp_receipt_hash
→ sign_bit_operation_receipts_hash
→ ulp_epsilon_receipt_hash
→ float_ordering_receipt_hash
→ reduced_precision_adapter_receipt_hash
→ float_integer_test_receipt_hash
→ hardware_float_adapter_boundary_hash
→ contextuality_threshold_receipt_hash
→ qml_boundary_receipt_hash
→ tool_dispatch_replay_proof_hash
→ interpretability_boundary_receipt_hash
→ reproducible_publication_receipt_hash
→ benchmark_ladder_receipt_hash
→ materials_signal_boundary_receipt_v2_hash
→ symbolic_compiler_manifest_hash
→ audit_trail_receipt_hash
→ experiment_scheduler_replay_proof_hash
→ cross_environment_replay_receipt_hash
→ federated_operator_audit_receipt_hash
→ hermetic_environment_receipt_hash
→ global_proof_composition_v2_hash
```

## Near-Term Implementation Priority

1. v163.0 → HeavyDependencyDiscoveryManifest
2. v163.1 → DependencyImportAndHotPathReceipt
3. v163.2 → BackendInvariantCandidateReceipt
4. v163.3 → CrossBackendEquivalenceReceipt
5. v163.4 → OptimizationOpportunityIndex
6. v164.0 → OptimizationContract
7. v164.1 → LightweightAdapterSpec
8. v164.2 → CachedCanonicalKernelReceipt
9. v164.3 → FastPathEquivalenceReceipt
10. v164.4 → OptimizationImplementationReceipt
11. v165.0 → OptimizedSimulationSpec
12. v165.1 → BackendEquivalenceReplayReceipt
13. v165.2 → OptimizedQECBenchmarkReceipt

Execution order rationale:
discover invariants
→ define optimization contracts
→ prove equivalence
→ implement safe fast paths
→ apply to optimized QEC simulations

## Publication / Zenodo Opportunities

| Phase | Publication Opportunity |
|---|---|
| v176.x | Syndrome streaming receipt schema |
| v178.x | Fault-tolerant resource accounting receipts |
| v178.5.x | IEEE 754 precision and bounded approximation receipts |
| v179.x | Contextuality classification receipts |
| v183.x | Reproducible publication receipts |
| v184.x | Benchmark ladder receipts |
| v186.x | Symbolic diagram compiler |
| v189.x | Cross-environment replay receipts |
| v191.x | Hermetic environment receipts |
| v192.x | Global proof composition v2 |

## Risk Register

| Risk | Preventing Receipt/Gate |
|---|---|
| dependency bloat | heavy_dependency_discovery_manifest + pinned dependency gate |
| heavy dependency becomes silent authority | BackendInvariantCandidateReceipt + CrossBackendEquivalenceReceipt |
| optimization changes numerical semantics | FastPathEquivalenceReceipt + IEEE754 precision receipts |
| speedup claim without benchmark | OptimizedQECBenchmarkReceipt |
| lightweight adapter diverges from reference backend | BackendEquivalenceReplayReceipt |
| cache key hides input mutation | CachedCanonicalKernelReceipt |
| dependency reduction removes required behavior | DependencyReductionReceipt + rollback condition |
| optimized simulation claims hardware relevance | OptimizedSimulationReport hardware-claim boundary |
| unsafe pointer-punning becomes implementation guidance | SafeBitReinterpretationPolicy |
| fast inverse square root uses undeclared magic constant | FastInvSqrtReceipt |
| fast exp overflows silently | FastExpReceipt input-domain policy |
| fast log2 floor is mistaken for exact logarithm | FastLog2FloorReceipt approximation-mode field |
| sign-bit abs/negate mishandles NaN payload or signed zero | SignBitAbsReceipt / SignBitNegateReceipt |
| positive-float ordering is applied to negative values | FloatOrderingReceipt sign-domain policy |
| silent float equality in proof-sensitive validators | ULPEpsilonComparisonReceipt |
| undeclared reduced-precision format | ReducedPrecisionAdapterReceipt |
| fast approximation used without declared error bound | FastApproximationReceipt |
| hardware float path treated as authority | HardwareFloatAdapterBoundaryReceipt |
| subnormal / NaN / Inf behavior silently differs across environments | IEEE754PrecisionFormatManifest |
| approximation benchmark marketed as quantum advantage | BenchmarkLadderReceipt + FastApproximationReceipt |
| license ambiguity | SPDX/license policy receipt gate |
| version drift | version-line compatibility gate |
| installer supply-chain risk | installer provenance manifest + checksum verification gate |
| symbolic claim drift | symbolic_compiler_manifest_hash |
| hardware authority drift | hardware boundary receipt gate |
| agent autonomy drift | tool_dispatch_replay_proof_hash + local-only execution gate |
| benchmark marketing drift | benchmark_ladder_receipt_hash + comparator disclosure gate |
| runtime nondeterminism | deterministic replay test gate |
| network dependence | offline replay gate |
| hidden randomness | seeded-or-rejected randomness gate |
| unbounded loops | bounded recursion scheduler gate |
| cross-environment drift | cross_environment_replay_receipt_hash |
| inaccessible source | source accessibility gate (`source_inaccessible` required) |
| LLM output as evidence | source citation and artifact verification gate |
| contextuality overclaim | contextuality_threshold_receipt_hash |
| declarative registry drift | hash-bound registry consistency gate |

## Safety / Interpretation Notes

**Heavy Dependency Reminder**
A heavy dependency is an adapter, not an authority.

QEC may use QuTiP, Qiskit, SciPy, matplotlib, pandas, NumPy, Stim,
PyMatching, qldpc, and related tools as reference backends, discovery
surfaces, or simulation helpers.

QEC must not trust their output silently.

Every useful backend behavior must be wrapped as:
  - invariant candidate
  - equivalence receipt
  - optimization contract
  - replay test
before becoming a QEC fast path.

**Optimization Reminder**
Optimization is not proof.

A faster path is valid only when:
  - its invariant source is recorded
  - its equivalence policy is declared
  - its benchmark is bounded
  - its rollback condition is explicit
  - its replay hash is stable

No speedup claim without a benchmark receipt.
No optimization without equivalence proof.


- GameWorld Reminder: observation contracts do not grant world authority.
- Perturbation Reminder: stress inputs must stay contract-bounded.
- Substrate Reminder: substrate descriptors are constraints, not truth sources.
- Recursion Reminder: loops must remain finite and receipt-bounded.
- Reality Loop Reminder: composition links receipts; it does not infer semantics.
- Global Truth Reminder: truth receipts are scoped to registered artifacts.
- Cosmovirus Reminder: symbolic sandboxing is claim-bounded and non-metaphysical.
- Syndrome Streaming Reminder: stream ingestion must be replay-safe and canonical.
- Hardware Profile Reminder: hardware profiles are adapter contexts, not authorities.
- Resource Accounting Reminder: overhead claims require accounting receipts.
- Contextuality Reminder: contextuality classifications are thresholded, not ontological.
- Agent Boundary Reminder: local dispatch only; no hidden autonomy.
- Symbolic Compiler Reminder: `claim_mode = SYMBOLIC_ONLY` for symbolic surfaces.
- Benchmark Reminder: no benchmark claims without comparator receipts.
- Reproducible Build Reminder: build claims require hermetic receipts.
- Dependency Policy Reminder: unpinned dependencies are non-compliant.
- Installer Provenance Reminder: installer mismatch must fail closed.

### IEEE 754 Precision Reminder

```text
Fast floating-point approximations are deterministic only when the input,
precision format, safe bit reinterpretation policy, constants, input domain,
special-value policy, and refinement count are all fixed.

Historical C pointer-punning examples such as *(int*)&x are source signals,
not QEC implementation guidance. Modern implementations must use safe bit
reinterpretation: memcpy, std::bit_cast, Rust to_bits/from_bits, Python struct,
or explicit dtype views with width/endian policy.

Fast inverse square root, fast exp, fast log2, fast pow, fast geometric mean,
sign-bit abs/negate, positive-float ordering, and float integer tests are all
bounded operations. They are not exact unless the receipt says so and proves it.

A QEC receipt for a float hack MUST declare:
  - precision format
  - safe bit reinterpretation policy
  - constants / magic numbers
  - input domain
  - error bound
  - refinement / Newton iteration count
  - NaN / Inf / zero / subnormal policy
  - signed-zero policy
  - NaN payload policy
  - ULP comparison rule, where applicable

Reduced precision formats such as FP16 and BF16 introduce mantissa truncation.
Any backend that uses reduced precision MUST declare that format in a receipt.

Hardware backends that use IEEE 754 exponent-bit construction are adapters.
QEC wraps their output as receipts.
QEC does not claim hardware authority over silicon implementations.

Silent float equality is not acceptable in proof-sensitive validators.
Use declared ULP-distance comparison receipts instead.

IEEE 754-2019 is the authoritative standard.
Blog posts, code examples, and hardware tutorials are research signals only.
```

## References

1. [IEEE754-2019] IEEE 754-2019 — IEEE Standard for Floating-Point Arithmetic — https://standards.ieee.org/ieee/754/6210/ (authoritative standard).
2. [IEEE754-hacks] leegao/float-hacks — Floating Point Hacks — https://github.com/leegao/float-hacks (research signal).
3. [randomascii-fp] Bruce Dawson — Tricks With the Floating-Point Format — https://randomascii.wordpress.com/2012/01/11/tricks-with-the-floating-point-format/ (research signal).
4. [amd-vitis-ieee754] AMD Vitis Tutorials — IEEE-754 Format Trick — https://docs.amd.com/r/en-US/Vitis-Tutorials-AI-Engine-Development/IEEE-754-Format-Trick (hardware adapter signal).
5. [fp16-bf16-guide] Complete Guide to Floating Point Representation: IEEE 754 & Half Precision Formats — https://medium.com/@adnaan525/complete-guide-to-floating-point-representation-ieee-754-half-precision-formats-ff4c4aa49227 (research signal).
6. [kaatinga-ieee754] Deep understanding of IEEE754 floating point numbers — https://gist.github.com/kaatinga/cecd8c26f544e270dd2008290818a20c (research signal).
7. Google Quantum AI updates (research signal).
8. Riverlane QEC roadmap and public benchmarking notes.
9. Stim documentation and source repository.
10. PyMatching documentation and source repository.
11. Qiskit v2.4 and Qiskit Aer documentation.
12. Debian reproducible builds documentation.
13. Nature/npj publications referenced in roadmap extension materials.
14. qec_theory_diagram.txt (repository symbolic grammar reference).

Do not modify src/qec/decoder/.
