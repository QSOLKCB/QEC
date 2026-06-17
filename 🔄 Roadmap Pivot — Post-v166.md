> **Superseded planning note:** The active roadmap now reserves v167.x for
> Symbolic Sonification Runtime & Event Mapping. QEC OS Runtime & Benchmark Reset
> work described here has been deferred to v180.0–v180.9; v167.x QEC OS labels in
> this historical pivot note should be read as old aliases, not current targets.

---

## 🔄 Roadmap Pivot — Post-v166.8

### Direction Statement

v166.8 completes the decoder-governance arc. The receipt-chain governance discipline
established in v163.x–v166.x is preserved as infrastructure. It is not discarded.

However, the project now pivots from receipt-only governance toward executable
Quantum Error Correction work. Every phase after v166.8 must produce:

- runnable code in `src/qec/`
- passing pytest tests against real syndrome inputs or real decoder outputs
- golden JSON fixtures for cross-language replay
- canonical receipts (as always)
- benchmark measurements (where applicable)

**Governance approval is not runtime activation.** v166.8 declares governance
approval for decoder promotion. Runtime activation occurs in v168.x
(DecoderRuntimeActivation), subject to golden corpus verification. The canonical
baseline (v166.0) remains callable and immutable. The promoted decoder is activated
as an additional runtime path via an explicit router. No silent source replacement
occurs.

The following arcs from the previous roadmap are **deleted** from the near-term plan
and moved to the research backlog:
- v168.x (MIDI/Sonification Receipts)
- v169.x (Symbolic Geometry Grammar / Cosmovirus Sandbox)
- v171.x (Deterministic Knowledge Base / Agent Memory Receipts)
- v172.x (Graphene / Photonic / Diamond / Materials Signal Receipts)
- v173.x (Interactive Proof Worlds / Citizen-Science Game Receipts)
- v178.5.x (IEEE 754 Precision & Approximation Receipts)
- v179.x (Quantum Geometry / Contextuality / Topological Toolkit Receipts)
- v180.x (Deterministic Quantum ML Boundary Receipts)
- v181.x (Local Agent / Tool Dispatch Receipts)
- v182.x (Interpretability / Sparse Feature Receipts)
- v185.x (Photonic / Materials / Device Signal Receipts v2)
- v186.x (Symbolic Diagram Compiler v2 / Grammar Extension)
- v190.x (Operator / IRC / TUI / CLI Federation Receipts v2)

The following arcs are **deferred** to post-v175:
- v170.x (Reproducible Build / Supply-Chain Receipts)
- v175.x (Operator Console Unification)
- v178.x (Fault-Tolerant Resource Accounting Receipts)
- v183.x (Reproducible Research Publication Receipts)
- v187.x (Human Audit / Red-Team Receipts)
- v188.x (Deterministic Experiment Scheduler)
- v189.x (Cross-Environment Hardware/OS Replay Receipts)
- v191.x (Reproducible Build / Hermetic Environment Receipts v2)
- v192.x (Global Proof Composition v2)

The following arcs are **pivoted** from receipt-only to executable milestones:

---

## Phase: v167.x — QEC OS Runtime & Benchmark Reset

**Status:** PLANNED

**Purpose:** Declare the QEC OS architecture. Activate the existing canonical decoder
baseline as a real runtime. Establish the module layout, API contracts, and golden
fixture format that all subsequent phases depend on.

**Real Modules to Build:**
- `src/qec/os/__init__.py` — QEC OS entry point
- `src/qec/os/registry.py` — Code family registry
- `src/qec/os/api.py` — Decoder runtime API surface
- `src/qec/os/fixture.py` — Golden fixture serializer/deserializer
- `tests/fixtures/` — Directory for golden JSON fixtures
- `tests/os/` — QEC OS unit tests

**Golden Fixture Format:**
```json
{
  "fixture_type": "syndrome_batch | parity_check | correction | benchmark",
  "code_family": "repetition | surface | toric | steane | hgp",
  "code_params": {"n": 7, "k": 1, "d": 3},
  "noise_model": "code_capacity | phenomenological | circuit_level",
  "noise_rate": 0.01,
  "seed": 42,
  "data": {},
  "sha256": "...",
  "created_at": "v167.0"
}
```

**Planned Releases:**
- v167.0 → QECOSArchitectureManifest
- v167.1 → CodeFamilyRegistryReceipt
- v167.2 → DecoderRuntimeAPIReceipt
- v167.3 → GoldenFixtureFormatReceipt
- v167.4 → BaselineActivationReceipt
- v167.5 → QECOSReplayEquivalenceReceipt

**Expected Hashes:**
```text
qec_os_architecture_manifest_hash           (v167.0)
code_family_registry_receipt_hash           (v167.1)
decoder_runtime_api_receipt_hash            (v167.2)
golden_fixture_format_receipt_hash          (v167.3)
baseline_activation_receipt_hash            (v167.4)
qec_os_replay_equivalence_receipt_hash      (v167.5)
```

**Core Rule:**
```text
same code family + same parameters + same seed
→ same golden fixture
→ same sha256
→ same qec_os_replay_equivalence_receipt_hash
```

**Acceptance Gates:**
- pytest: QECOSArchitectureManifest declares all layer names and module paths
- pytest: CodeFamilyRegistryReceipt lists all supported code families with [[n,k,d]] parameters
- pytest: DecoderRuntimeAPIReceipt declares decoder interface (input: syndrome batch, output: correction)
- pytest: BaselineActivationReceipt confirms canonical decoder baseline is callable via runtime API
- pytest: All fixture round-trips are hash-stable across PYTHONHASHSEED values
- pytest: No live network calls in tests
- pytest: No hardware authority claims

**Must Not Do:**
- no new decoders in this phase
- no benchmark claims without measurement
- no mutation of `src/qec/decoder/` without replay equivalence proof
- no receipt-only artifacts with no executable code

**Definition of Done:** The canonical decoder baseline from v166.0 is callable via
the QEC OS runtime API. A repetition code syndrome batch passes through the decoder
and returns a correction. Input and output are serialized as golden JSON fixtures
with stable SHA-256 hashes. `pytest -q` passes.

**Dependency Boundaries:**
- depends on: v166.8 DecoderPromotionReceipt (canonical decoder baseline)
- feeds into: v168.x Decoder Runtime Activation
- does not modify: `src/qec/decoder/`

---

## Phase: v168.x — Decoder Runtime Activation

**Status:** PLANNED

**Purpose:** Activate the governance-promoted decoder as a real runtime. Converts
the v166.8 governance decision into an actual code change. The promoted decoder is
activated as an additional runtime path, not a silent replacement. The canonical
baseline remains callable and immutable.

**Real Modules to Build:**
- `src/qec/decoder/runtime.py` — Decoder runtime activation layer
- `src/qec/decoder/baseline.py` — Canonical baseline (preserved, immutable)
- `src/qec/decoder/promoted.py` — Promoted decoder (activated here)
- `src/qec/decoder/router.py` — Routes syndrome batches to baseline or promoted decoder
- `tests/decoder/test_activation.py` — Activation tests
- `tests/decoder/test_golden_corpus.py` — Golden corpus replay tests
- `tests/fixtures/repetition_code/` — Repetition code golden fixtures
- `tests/fixtures/surface_code_d3/` — Surface code d=3 golden fixtures

**Planned Releases:**
- v168.0 → DecoderRuntimeActivationManifest
- v168.1 → PromotedDecoderActivationReceipt
- v168.2 → BaselinePreservationReceipt
- v168.3 → GoldenCorpusVerificationReceipt
- v168.4 → DecoderRouterReceipt
- v168.5 → ActivationRollbackReceipt

**Expected Hashes:**
```text
decoder_runtime_activation_manifest_hash    (v168.0)
promoted_decoder_activation_receipt_hash    (v168.1)
baseline_preservation_receipt_hash          (v168.2)
golden_corpus_verification_receipt_hash     (v168.3)
decoder_router_receipt_hash                 (v168.4)
activation_rollback_receipt_hash            (v168.5)
```

**Golden Corpus Requirements:**
- Repetition code d=3,5,7: 100 syndrome batches each, seed=42
- Surface code d=3: 100 syndrome batches, code capacity noise p=0.01, seed=42
- Surface code d=3: 100 syndrome batches, phenomenological noise p=0.01, seed=42
- All fixtures include: syndrome bits, expected correction, decoder used, noise model, seed

**Acceptance Gates:**
- pytest: Promoted decoder produces same output as canonical baseline on golden corpus
- pytest: Canonical baseline remains callable after activation
- pytest: Rollback to canonical baseline works
- pytest: All golden corpus fixtures are hash-stable
- pytest: No silent decoder replacement (router is explicit and logged)
- pytest: No live hardware calls in tests

**Must Not Do:**
- no silent replacement of `src/qec/decoder/`
- no removal of canonical baseline replay path
- no activation without rollback receipt
- no golden corpus bypass

**Definition of Done:** Promoted decoder is callable via `src/qec/decoder/runtime.py`.
Passes all golden corpus tests for repetition code and surface code d=3. Canonical
baseline remains callable. Rollback receipt exists. `pytest -q` passes.

**Dependency Boundaries:**
- depends on: v167.x QEC OS Runtime & Benchmark Reset
- feeds into: v169.x Stabilizer & QLDPC Construction Engine
- does not silently modify: `src/qec/decoder/`

---

## Phase: v169.x — Stabilizer & QLDPC Construction Engine

**Status:** PLANNED

**Purpose:** Build a real code construction layer. Implement parity-check matrix
construction for repetition codes, surface codes, Steane code, and hypergraph
product (HGP) QLDPC codes. Expose all codes as canonical parity-check matrices
in the golden fixture format. Integrate with qLDPC as an adapter, not an authority.

**Real Modules to Build:**
- `src/qec/codes/__init__.py` — Code construction entry point
- `src/qec/codes/repetition.py` — Repetition code construction
- `src/qec/codes/surface.py` — Surface code (rotated, unrotated) construction
- `src/qec/codes/steane.py` — Steane [7,1,3] code construction
- `src/qec/codes/hgp.py` — Hypergraph product code construction (adapter to qLDPC)
- `src/qec/codes/gf2.py` — GF(2) arithmetic (row reduction, rank, nullspace)
- `src/qec/codes/canonical.py` — Canonical form for parity-check matrices
- `tests/codes/` — Code construction tests
- `tests/fixtures/codes/` — Golden parity-check matrix fixtures

**Planned Releases:**
- v169.0 → CodeConstructionManifest
- v169.1 → RepetitionCodeReceipt
- v169.2 → SurfaceCodeReceipt
- v169.3 → SteaneCodeReceipt
- v169.4 → HGPCodeReceipt
- v169.5 → GF2OperationsReceipt
- v169.6 → CodeConstructionReplayEquivalenceReceipt

**Expected Hashes:**
```text
code_construction_manifest_hash             (v169.0)
repetition_code_receipt_hash                (v169.1)
surface_code_receipt_hash                   (v169.2)
steane_code_receipt_hash                    (v169.3)
hgp_code_receipt_hash                       (v169.4)
gf2_operations_receipt_hash                 (v169.5)
code_construction_replay_equivalence_receipt_hash (v169.6)
```

**Acceptance Gates:**
- pytest: RepetitionCodeReceipt produces correct H matrix for d=3,5,7
- pytest: SurfaceCodeReceipt produces correct H_X, H_Z for d=3 (hand-verified)
- pytest: SteaneCodeReceipt produces correct [[7,1,3]] parameters
- pytest: HGPCodeReceipt produces valid CSS code (H_X * H_Z^T = 0 mod 2)
- pytest: GF2OperationsReceipt: row reduction, rank, nullspace correct on hand-verified examples
- pytest: qLDPC is adapter-only (output verified against golden fixtures, not trusted directly)
- pytest: All fixtures are hash-stable across PYTHONHASHSEED values

**Must Not Do:**
- no code distance claim without verification receipt
- no qLDPC output accepted as ground truth without golden fixture comparison
- no decoder implementation in this phase
- no QLDPC performance claims without benchmarks

**Definition of Done:** Repetition, surface (d=3,5,7), Steane, and HGP codes are
constructible. All produce canonical parity-check matrices that match golden fixtures.
GF(2) row reduction, rank, and nullspace pass hand-verified tests. `pytest -q` passes.

**Dependency Boundaries:**
- depends on: v167.x QEC OS Runtime & Benchmark Reset
- feeds into: v170.x Syndrome Stream & Noise Model Runtime

---

## Phase: v170.x — Syndrome Stream & Noise Model Runtime

**Status:** PLANNED

**Purpose:** Build a real syndrome ingestion layer and noise model runtime. Integrate
Stim as the primary syndrome simulator. Implement code capacity, phenomenological,
and circuit-level depolarizing noise models. Stim is an adapter, not an authority.

**Real Modules to Build:**
- `src/qec/syndrome/__init__.py` — Syndrome stream entry point
- `src/qec/syndrome/batch.py` — Syndrome batch representation
- `src/qec/syndrome/stream.py` — Syndrome stream API (windowing, ordering)
- `src/qec/syndrome/stim_adapter.py` — Stim adapter (adapter-only)
- `src/qec/noise/__init__.py` — Noise model entry point
- `src/qec/noise/code_capacity.py` — Code capacity noise model
- `src/qec/noise/phenomenological.py` — Phenomenological noise model
- `src/qec/noise/circuit_level.py` — Circuit-level depolarizing noise model (via Stim)
- `tests/fixtures/syndromes/` — Golden syndrome fixtures

**Planned Releases:**
- v170.0 → SyndromeStreamManifest
- v170.1 → NoiseModelRegistryReceipt
- v170.2 → CodeCapacityNoiseReceipt
- v170.3 → PhenomenologicalNoiseReceipt
- v170.4 → CircuitLevelNoiseReceipt
- v170.5 → StimAdapterBoundaryReceipt
- v170.6 → SyndromeStreamReplayEquivalenceReceipt

**Acceptance Gates:**
- pytest: Same seed + same noise model + same code → same syndrome batch (deterministic)
- pytest: StimAdapterBoundaryReceipt has adapter_only=true
- pytest: No live hardware connections in tests
- pytest: All syndrome fixtures are hash-stable

**Must Not Do:**
- no live hardware syndrome ingestion
- no real-time performance claims without measurement
- no mixing noise model outputs without declared noise model type
- no Stim output treated as ground truth

**Definition of Done:** Syndrome batches for repetition code (code capacity) and
surface code (phenomenological, circuit-level) are generatable. All syndrome batches
are deterministic given seed and noise model. `pytest -q` passes.

**Dependency Boundaries:**
- depends on: v169.x Stabilizer & QLDPC Construction Engine
- feeds into: v171.x Benchmark Corpus & Logical Error Rate Harness

---

## Phase: v171.x — Benchmark Corpus & Logical Error Rate Harness

**Status:** PLANNED

**Purpose:** Build a real benchmark harness. Run actual decoder benchmarks. Measure
logical error rates, decoder latency, and syndrome throughput. Use sinter as the
primary Monte Carlo sampling harness. All results include Wilson score confidence
intervals. Benchmark results are bounded observations, not correctness proofs.

**Real Modules to Build:**
- `src/qec/benchmark/__init__.py` — Benchmark harness entry point
- `src/qec/benchmark/runner.py` — Benchmark runner (wraps sinter)
- `src/qec/benchmark/metrics.py` — Metrics: latency, throughput, LER, Wilson CI
- `src/qec/benchmark/comparator.py` — Comparator registry (Stim, PyMatching, qldpc)
- `src/qec/benchmark/report.py` — Benchmark report serializer
- `tests/fixtures/benchmarks/` — Golden benchmark results

**Planned Releases:**
- v171.0 → BenchmarkHarnessManifest
- v171.1 → LogicalErrorRateReceipt
- v171.2 → DecoderLatencyReceipt
- v171.3 → SyndromeThroughputReceipt
- v171.4 → ThresholdSweepReceipt
- v171.5 → BenchmarkComparatorReceipt
- v171.6 → BenchmarkReplayEquivalenceReceipt

**Acceptance Gates:**
- pytest: LogicalErrorRateReceipt declares hardware, corpus, decoder, noise model, shot count
- pytest: LogicalErrorRateReceipt includes Wilson score 68% confidence intervals
- pytest: DecoderLatencyReceipt declares hardware type and measurement method
- pytest: ThresholdSweepReceipt includes at least 3 code distances and 10 noise rate points
- pytest: BenchmarkComparatorReceipt has adapter_only=true for all external comparators
- pytest: No benchmark claim without declared hardware

**Must Not Do:**
- no benchmark results without declared hardware
- no logical error rate without Wilson score confidence intervals
- no threshold claim without sweep data
- no external comparator treated as authority

**Definition of Done:** Logical error rate curves exist for repetition code (d=3,5,7)
and surface code (d=3,5) under circuit-level noise. All results include Wilson score
confidence intervals and declared hardware. `pytest -q` passes.

**Dependency Boundaries:**
- depends on: v170.x Syndrome Stream & Noise Model Runtime
- feeds into: v172.x Cross-Backend Differential Testing

---

## Phase: v172.x — Cross-Backend Differential Testing

**Status:** PLANNED

**Purpose:** Build a differential testing harness comparing QEC OS decoder outputs
against PyMatching, qldpc BP+OSD, and union-find on the same syndrome inputs.
Identify correctness discrepancies. Log disagreement cases as fixtures.

**Real Modules to Build:**
- `src/qec/differential/__init__.py` — Differential testing entry point
- `src/qec/differential/harness.py` — Differential test harness
- `src/qec/differential/comparator.py` — External decoder adapter
- `tests/differential/` — Differential test suite

**Planned Releases:**
- v172.0 → DifferentialTestManifest
- v172.1 → PyMatchingComparatorReceipt
- v172.2 → BPOSDComparatorReceipt
- v172.3 → UnionFindComparatorReceipt
- v172.4 → DifferentialTestResultReceipt
- v172.5 → DifferentialReplayEquivalenceReceipt

**Acceptance Gates:**
- pytest: All comparators have declared version, source, and adapter_only=true
- pytest: Differential test results declare agreement rate and disagreement cases
- pytest: Disagreement cases logged as fixtures for investigation
- pytest: No comparator treated as ground truth

**Must Not Do:**
- no QEC OS decoder superiority claims without benchmark evidence
- no PyMatching output treated as ground truth
- no differential tests without declared comparator version

**Definition of Done:** QEC OS decoder outputs compared against PyMatching and
BP+OSD on surface code d=3 and at least one QLDPC code. Agreement rates measured
and logged. `pytest -q` passes.

**Dependency Boundaries:**
- depends on: v171.x Benchmark Corpus & Logical Error Rate Harness
- feeds into: v173.x Odin Port Readiness Layer

---

## Phase: v173.x — Odin Port Readiness Layer

**Status:** PLANNED

**Purpose:** Establish all artifacts required before an Odin rewrite can begin.
This phase does not write any Odin code. It produces: complete golden fixture corpus,
cross-language test vector specification, serialization format specification,
performance baseline document, and Odin fork readiness checklist.

**Real Modules to Build:**
- `docs/odin_migration/fixture_format.md` — Fixture format specification
- `docs/odin_migration/api_contracts.md` — API contracts for Odin implementation
- `docs/odin_migration/performance_baselines.md` — Python performance baselines
- `docs/odin_migration/readiness_checklist.md` — Odin fork readiness checklist
- `scripts/validate_fixtures.py` — Fixture validation script

**Planned Releases:**
- v173.0 → OdinMigrationManifest
- v173.1 → GoldenCorpusCompletenessReceipt
- v173.2 → FixtureFormatSpecificationReceipt
- v173.3 → PerformanceBaselineReceipt
- v173.4 → CrossLanguageTestVectorReceipt
- v173.5 → OdinReadinessChecklistReceipt

**Acceptance Gates:**
- pytest: Golden corpus covers all code families implemented in v169.x
- pytest: Fixture format specification is machine-readable (JSON schema)
- pytest: Performance baseline document includes wall-clock times for all benchmarks
- pytest: Odin readiness checklist is complete (all items checked)

**Must Not Do:**
- no Odin code in this phase
- no Odin performance claims without Odin benchmarks
- no Odin rewrite started before this phase is complete

**Definition of Done:** Golden fixture corpus is complete. Fixture format is specified.
Performance baselines are documented. Odin fork readiness checklist is complete.
`pytest -q` passes.

**Dependency Boundaries:**
- depends on: v172.x Cross-Backend Differential Testing
- feeds into: v174.x Odin Runtime Prototype

---

## Phase: v174.x — Odin Runtime Prototype

**Status:** PLANNED

**Purpose:** Build the first Odin implementation of QEC OS core modules. Start with
GF(2) arithmetic and repetition code construction. Compare Odin output against Python
golden fixtures. This is a prototype; the Python implementation remains primary.

**Real Odin Modules to Build (in `qec-odin/` subdirectory):**
- `qec-odin/src/gf2/gf2.odin` — GF(2) arithmetic
- `qec-odin/src/codes/repetition.odin` — Repetition code construction
- `qec-odin/src/codes/surface.odin` — Surface code construction (d=3)
- `qec-odin/src/fixture/loader.odin` — Fixture loader (reads Python golden fixtures)
- `qec-odin/tests/` — Odin test suite (`odin test`)

**Planned Releases:**
- v174.0 → OdinPrototypeManifest
- v174.1 → OdinGF2ImplementationReceipt
- v174.2 → OdinRepetitionCodeReceipt
- v174.3 → OdinSurfaceCodeReceipt
- v174.4 → OdinFixtureParityReceipt
- v174.5 → OdinPrototypeReplayEquivalenceReceipt

**Acceptance Gates:**
- odin test: GF(2) row reduction matches Python golden fixtures
- odin test: Repetition code H matrix matches Python golden fixtures
- odin test: Surface code d=3 H_X, H_Z matrices match Python golden fixtures
- pytest: Python fixture validation script confirms all Odin outputs match

**Must Not Do:**
- no Python implementation replacement in this phase
- no Odin performance claims without benchmarks
- no module porting without golden fixture coverage

**Definition of Done:** Odin GF(2) arithmetic, repetition code, and surface code d=3
produce output matching Python golden fixtures byte-for-byte (after JSON normalization).
`odin test qec-odin/tests/` passes.

**Dependency Boundaries:**
- depends on: v173.x Odin Port Readiness Layer
- feeds into: v175.x QEC OS Integration Milestone

---

## Phase: v175.x — QEC OS Integration Milestone

**Status:** PLANNED

**Purpose:** Integration milestone. Verify all QEC OS layers work together end-to-end.
Run a complete experiment: construct surface code → generate syndrome batches under
circuit-level noise → decode with activated decoder → measure logical error rate →
compare against PyMatching → produce canonical benchmark report.

**Real Modules to Build:**
- `src/qec/experiment/__init__.py` — Experiment runner entry point
- `src/qec/experiment/runner.py` — End-to-end experiment runner
- `scripts/run_experiment.py` — CLI experiment runner
- `tests/integration/` — Integration test suite

**Planned Releases:**
- v175.0 → QECOSIntegrationManifest
- v175.1 → EndToEndExperimentReceipt
- v175.2 → IntegrationBenchmarkReceipt
- v175.3 → QECOSCompletenessReceipt

**Acceptance Gates:**
- pytest: End-to-end experiment runs without errors
- pytest: Logical error rate measured for surface code d=3,5 under circuit-level noise
- pytest: All QEC OS layers covered by integration tests
- pytest: Odin prototype passes parity tests for all implemented modules

**Must Not Do:**
- no production readiness claims without full test suite passing
- no hardware relevance claims
- no quantum advantage claims

**Definition of Done:** Complete QEC OS experiment runnable from CLI. All results are
canonical JSON with SHA-256 hashes. Odin prototype passes parity tests. `pytest -q`
passes including integration tests.

**Dependency Boundaries:**
- depends on: v174.x Odin Runtime Prototype
- completes: QEC OS pivot arc (v167.x–v175.x)

---

## 🚫 Must Not Do Rules (Post-v166.8)

1. Do not produce a receipt-only phase with no executable code.
2. Do not activate the promoted decoder without golden corpus verification.
3. Do not claim benchmark results without declared hardware and comparator version.
4. Do not claim logical error rate without Wilson score confidence intervals.
5. Do not claim threshold without sweep data across at least 3 code distances.
6. Do not treat Stim, PyMatching, or qLDPC output as ground truth.
7. Do not start the Odin rewrite before the golden corpus is complete (v173.x).
8. Do not replace `src/qec/decoder/` silently.
9. Do not claim hardware authority.
10. Do not claim quantum advantage.
11. Do not add cosmological, MIDI, agent memory, or materials signal phases to the near-term roadmap.
12. Do not produce a phase that only registers receipts from other phases.

---

## 📋 Risk Register (Post-v166.8)

| Risk | Severity | Mitigation |
|------|----------|------------|
| Promoted decoder fails golden corpus tests | High | Investigate before activation; keep canonical baseline callable |
| Stim API changes break syndrome generation | Medium | Pin Stim version; test against pinned version |
| PyMatching API changes break differential tests | Medium | Pin PyMatching version; use adapter pattern |
| qLDPC output differs from hand-verified fixtures | Medium | Verify all qLDPC output against hand-computed small cases |
| GF(2) implementation has edge cases | Medium | Test against hand-verified examples; use property tests |
| Odin JSON parsing differs from Python | Low | Use canonical JSON normalization; test round-trips |
| Benchmark results are hardware-dependent | Medium | Declare hardware; do not claim universal performance |
| Odin ecosystem lacks sparse matrix support | High | Implement sparse GF(2) from scratch; do not rely on external libs |
| CI benchmark regression detection | Medium | Set regression thresholds; fail CI on >20% regression |

---

## 🔗 Terminal Proof Chain Extension (Post-v166.8)

```text
decoder_promotion_receipt_hash                (v166.8)
→ qec_os_architecture_manifest_hash           (v167.0)
→ code_family_registry_receipt_hash           (v167.1)
→ decoder_runtime_api_receipt_hash            (v167.2)
→ golden_fixture_format_receipt_hash          (v167.3)
→ baseline_activation_receipt_hash            (v167.4)
→ qec_os_replay_equivalence_receipt_hash      (v167.5)
→ decoder_runtime_activation_manifest_hash    (v168.0)
→ promoted_decoder_activation_receipt_hash    (v168.1)
→ baseline_preservation_receipt_hash          (v168.2)
→ golden_corpus_verification_receipt_hash     (v168.3)
→ decoder_router_receipt_hash                 (v168.4)
→ activation_rollback_receipt_hash            (v168.5)
→ code_construction_manifest_hash             (v169.0)
→ repetition_code_receipt_hash                (v169.1)
→ surface_code_receipt_hash                   (v169.2)
→ steane_code_receipt_hash                    (v169.3)
→ hgp_code_receipt_hash                       (v169.4)
→ gf2_operations_receipt_hash                 (v169.5)
→ code_construction_replay_equivalence_receipt_hash (v169.6)
→ syndrome_stream_manifest_hash               (v170.0)
→ noise_model_registry_receipt_hash           (v170.1)
→ code_capacity_noise_receipt_hash            (v170.2)
→ phenomenological_noise_receipt_hash         (v170.3)
→ circuit_level_noise_receipt_hash            (v170.4)
→ stim_adapter_boundary_receipt_hash          (v170.5)
→ syndrome_stream_replay_equivalence_receipt_hash (v170.6)
→ benchmark_harness_manifest_hash             (v171.0)
→ logical_error_rate_receipt_hash             (v171.1)
→ decoder_latency_receipt_hash                (v171.2)
→ syndrome_throughput_receipt_hash            (v171.3)
→ threshold_sweep_receipt_hash                (v171.4)
→ benchmark_comparator_receipt_hash           (v171.5)
→ benchmark_replay_equivalence_receipt_hash   (v171.6)
→ differential_test_manifest_hash             (v172.0)
→ pymatching_comparator_receipt_hash          (v172.1)
→ bposd_comparator_receipt_hash               (v172.2)
→ union_find_comparator_receipt_hash          (v172.3)
→ differential_test_result_receipt_hash       (v172.4)
→ differential_replay_equivalence_receipt_hash (v172.5)
→ odin_migration_manifest_hash                (v173.0)
→ golden_corpus_completeness_receipt_hash     (v173.1)
→ fixture_format_specification_receipt_hash   (v173.2)
→ performance_baseline_receipt_hash           (v173.3)
→ cross_language_test_vector_receipt_hash     (v173.4)
→ odin_readiness_checklist_receipt_hash       (v173.5)
→ odin_prototype_manifest_hash                (v174.0)
→ odin_gf2_implementation_receipt_hash        (v174.1)
→ odin_repetition_code_receipt_hash           (v174.2)
→ odin_surface_code_receipt_hash              (v174.3)
→ odin_fixture_parity_receipt_hash            (v174.4)
→ odin_prototype_replay_equivalence_receipt_hash (v174.5)
→ qec_os_integration_manifest_hash            (v175.0)
→ end_to_end_experiment_receipt_hash          (v175.1)
→ integration_benchmark_receipt_hash          (v175.2)
→ qec_os_completeness_receipt_hash            (v175.3)
```

