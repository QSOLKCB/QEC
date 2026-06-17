# QEC Second-Pass Roadmap Tightening — Post-v166.8

> **Superseded planning note:** This document records the earlier second-pass QEC OS
> tightening plan. The active roadmap now reserves v167.x for Symbolic
> Sonification Runtime & Event Mapping and defers this QEC OS Runtime & Benchmark
> Reset sequence to v180.0–v180.9. Treat any v167.x QEC OS references below as
> historical aliases for the deferred v180.x sequence.

**Repository:** QSOLKCB/QEC  
**Completed Release:** v166.8 — DecoderPromotionReceipt  
**Document Type:** Roadmap Tightening Pass 2 — Ready-to-Paste ROADMAP.md Replacement Tail

---

## A. Executive Summary

v166.8 completed the decoder-governance arc with `DecoderPromotionReceipt`. That receipt records governance eligibility for decoder promotion. It does not activate the promoted decoder at runtime, does not replace `src/qec/decoder/`, and does not constitute executable QEC OS work.

The first-pass roadmap pivot correctly identified the direction: pivot toward executable Quantum Error Correction work. However, it spread that work across nine phases (v167.x–v175.x) with approximately 50 sub-releases, many of which remained receipt-only manifests. The first pass did not compress the pivot into the required v167.0–v167.9 window.

This second-pass tightening compresses all executable QEC OS work into exactly 10 releases: **v167.0 through v167.9**. Every release ships at least one of: executable runtime code, pytest tests, golden JSON fixtures, benchmark harness code, decoder/code-construction runtime code, cross-backend differential test machinery, or Odin migration artifacts that are directly testable. The old receipt-only arcs from the original roadmap (Qudit, MIDI, Symbolic Geometry, Agent Memory, Materials, Citizen-Science, Operator Federation, Global Proof Composition v2) are removed from the near-term roadmap and classified below.

---

## B. Diagnosis of First-Pass Roadmap Problems

The first-pass pivot was directionally correct but structurally insufficient. The following problems required a second pass.

| Problem | Evidence | Impact |
|---|---|---|
| **Spread across too many version families** | Work distributed v167.x–v175.x (9 phase families) | Near-term pivot not visible; looks like another 50-release receipt arc |
| **Sub-releases still receipt-only** | v167.0 = "QECOSArchitectureManifest", v167.1 = "CodeFamilyRegistryReceipt" | Violates the hard requirement that every release ships executable code/tests/fixtures |
| **Decoder activation deferred to v168.x** | First pass says "Runtime activation occurs in v168.x" | Activation is still 6+ sub-releases away from v167.0; not compressed |
| **Odin work deferred to v173.x–v174.x** | Odin readiness at v173.x, prototype at v174.x | Odin work is 30+ sub-releases from current position |
| **Phase names still receipt-flavored** | "QECOSArchitectureManifest", "GoldenFixtureFormatReceipt" as release names | Signals receipt-first thinking rather than execution-first thinking |
| **README contradiction unresolved** | README still declares "v167.0 — QuditDimensionPolicyManifest" as frontier | Qudit arc is a pure receipt arc; must be explicitly superseded |
| **v166.8 contradiction not stated clearly enough** | First pass mentions it but does not make it a standalone mandatory section | Ambiguity remains about what v166.8 actually delivered |
| **Old receipt arcs not fully classified** | First pass lists deletions but does not produce a classification table | Deferred vs. deleted vs. collapsed is unclear |

---

## C. Ready-to-Paste ROADMAP.md Replacement Tail

The following section is ready to paste into ROADMAP.md immediately after the v166.8 entry.

---

```
## QEC OS Pivot — Post-v166.8

v166.8 completed decoder governance.

Governance approval is not runtime activation.

The next roadmap segment turns QEC back toward executable Quantum Error Correction
work: runtime APIs, real code construction, syndrome fixtures, decoder routing,
benchmark harnesses, cross-backend differential tests, and Odin migration readiness.

Every release from v167.0 through v167.9 must ship executable code, tests, fixtures,
benchmarks, or migration artifacts. Receipt-only releases are no longer accepted in
the near-term roadmap.

### v166.8 Clarification

v166.8 declares receipt-chain promotion eligibility. It does not replace decoder
source, activate runtime execution, or mutate `src/qec/decoder/`.

Runtime activation must occur explicitly through a tested router and golden corpus
verification. This happens in v167.3 (PromotedDecoderRuntimeRouter), after the
runtime API skeleton (v167.2) and golden corpus seed (v167.1) are in place.

The README declaration "Current frontier: v167.0 — QuditDimensionPolicyManifest"
is superseded by this pivot. The Qudit arc is deferred to the research backlog.
The v167.x version family is now reserved for QEC OS executable work.

---

## v167.0 — QECOSRuntimeSkeleton

**Purpose:** Create the QEC OS module skeleton, define the runtime API shape,
define the golden fixture schema, and prove fixture hash determinism. No decoder
activation yet.

**Files / Modules to Add or Change:**
- `src/qec/os/__init__.py` — QEC OS package entry point
- `src/qec/os/api.py` — Runtime API shape: `decode(syndrome_batch) -> correction`
- `src/qec/os/fixture.py` — Golden fixture serializer / deserializer
- `src/qec/os/registry.py` — Code family registry (repetition, surface, steane, hgp)
- `tests/os/test_fixture_determinism.py` — Fixture hash determinism tests
- `tests/fixtures/` — Empty directory with `.gitkeep` and schema file

**Golden Fixture Schema (canonical JSON):**
```json
{
  "fixture_type": "syndrome_batch | parity_check | correction | benchmark",
  "code_family": "repetition | surface | steane | hgp",
  "code_params": {"n": 7, "k": 1, "d": 3},
  "noise_model": "code_capacity | phenomenological | circuit_level",
  "noise_rate": 0.01,
  "seed": 42,
  "data": {},
  "sha256": "...",
  "created_at": "v167.0"
}
```

**Tests to Add:**
- `test_fixture_round_trip`: serialize → deserialize → hash must be identical
- `test_fixture_hash_stable_across_pythonhashseed`: run with PYTHONHASHSEED=0,1,42
- `test_registry_lists_all_code_families`: registry must enumerate all supported families
- `test_no_live_network_calls`: monkeypatch socket to reject network calls in tests

**Fixtures to Add:**
- `tests/fixtures/schema_v167.json` — canonical fixture schema definition

**Benchmarks:** None in this release.

**Acceptance Gates:**
- `pytest tests/os/` passes with PYTHONHASHSEED varied
- Fixture round-trip produces identical SHA-256 across three PYTHONHASHSEED values
- Registry enumerates: repetition, surface, steane, hgp
- No live network calls in tests
- No decoder activation

**Definition of Done:** `src/qec/os/` exists. Fixture schema is defined. Hash
determinism is proven by tests. `pytest -q` passes.

**Must Not Do:**
- no decoder activation
- no benchmark claims
- no mutation of `src/qec/decoder/`
- no receipt-only artifacts with no executable code

**Dependency Boundaries:**
- depends on: v166.8 DecoderPromotionReceipt (proof chain anchor)
- feeds into: v167.1 GoldenCorpusSeed
- does not modify: `src/qec/decoder/`

---

## v167.1 — GoldenCorpusSeed

**Purpose:** Add the first golden fixture corpus: repetition code and tiny syndrome
fixtures. Add a fixture validator script. All fixtures must be canonical JSON with
stable SHA-256.

**Files / Modules to Add or Change:**
- `tests/fixtures/repetition_d3_syndromes.json` — 10 syndrome batches, seed=42
- `tests/fixtures/repetition_d5_syndromes.json` — 10 syndrome batches, seed=42
- `tests/fixtures/repetition_d3_parity_check.json` — H matrix for d=3
- `tests/fixtures/repetition_d5_parity_check.json` — H matrix for d=5
- `scripts/validate_fixtures.py` — CLI fixture validator (reads fixture, recomputes SHA-256)
- `tests/os/test_golden_corpus.py` — Golden corpus round-trip tests

**Tests to Add:**
- `test_repetition_d3_fixture_hash_stable`: SHA-256 must match across runs
- `test_repetition_d5_fixture_hash_stable`: SHA-256 must match across runs
- `test_parity_check_fixture_valid`: H matrix dimensions correct for d=3, d=5
- `test_validate_fixtures_script_exits_zero`: CLI validator exits 0 on valid corpus
- `test_validate_fixtures_script_exits_nonzero_on_tamper`: exits nonzero if SHA-256 tampered

**Fixtures to Add:**
- Repetition code d=3: 10 syndrome batches (code-capacity noise, p=0.01, seed=42)
- Repetition code d=5: 10 syndrome batches (code-capacity noise, p=0.01, seed=42)
- Repetition code d=3: parity-check matrix H (hand-verified)
- Repetition code d=5: parity-check matrix H (hand-verified)

**Benchmarks:** None in this release.

**Acceptance Gates:**
- All fixture SHA-256 values are stable across three separate Python interpreter runs
- `scripts/validate_fixtures.py` exits 0 on valid corpus, nonzero on tampered corpus
- Parity-check matrices match hand-verified values
- `pytest tests/os/` passes

**Definition of Done:** Repetition code golden fixtures exist and are SHA-256 stable.
Fixture validator script is runnable. `pytest -q` passes.

**Must Not Do:**
- no decoder invocation yet
- no surface code or Steane fixtures yet (those come in v167.4)
- no benchmark claims

**Dependency Boundaries:**
- depends on: v167.0 QECOSRuntimeSkeleton
- feeds into: v167.2 BaselineDecoderRuntimeAPI

---

## v167.2 — BaselineDecoderRuntimeAPI

**Purpose:** Expose the existing canonical baseline decoder (from v166.0) through an
explicit runtime API. No source replacement. No promoted decoder activation yet.
Regression tests around the baseline call path.

**Files / Modules to Add or Change:**
- `src/qec/decoder/baseline_api.py` — Explicit runtime API wrapping the existing baseline
- `src/qec/decoder/__init__.py` — Export `baseline_decode(syndrome_batch) -> correction`
- `tests/decoder/test_baseline_api.py` — Baseline API regression tests
- `tests/decoder/test_baseline_golden_corpus.py` — Baseline against repetition-code fixtures

**Tests to Add:**
- `test_baseline_decode_repetition_d3`: baseline decode on repetition d=3 golden fixtures
- `test_baseline_decode_repetition_d5`: baseline decode on repetition d=5 golden fixtures
- `test_baseline_api_returns_canonical_json`: output is serializable to canonical JSON
- `test_baseline_api_is_deterministic`: same input → same output across 3 calls
- `test_baseline_api_no_side_effects`: baseline call does not mutate input syndrome batch
- `test_src_qec_decoder_not_mutated`: assert no files in `src/qec/decoder/` were modified

**Fixtures to Add:**
- `tests/fixtures/repetition_d3_corrections.json` — expected corrections from baseline
- `tests/fixtures/repetition_d5_corrections.json` — expected corrections from baseline

**Benchmarks:** None in this release (latency measurement deferred to v167.7).

**Acceptance Gates:**
- `baseline_decode()` callable and returns correct corrections for repetition d=3, d=5
- Output is deterministic (same input → same output, 3 runs)
- Output serializes to canonical JSON with stable SHA-256
- `src/qec/decoder/` source files are unmodified (verified by git diff)
- `pytest tests/decoder/` passes

**Definition of Done:** Canonical baseline decoder is callable via `baseline_decode()`.
Passes golden corpus regression tests. `src/qec/decoder/` is unmodified. `pytest -q` passes.

**Must Not Do:**
- no promoted decoder activation
- no mutation of `src/qec/decoder/`
- no new decoder implementation
- no benchmark claims without measurement

**Dependency Boundaries:**
- depends on: v167.1 GoldenCorpusSeed (repetition-code fixtures)
- feeds into: v167.3 PromotedDecoderRuntimeRouter

---

## v167.3 — PromotedDecoderRuntimeRouter

**Purpose:** Add an explicit baseline/promoted decoder router. The promoted decoder
path is opt-in. The baseline remains the default and is always callable. The rollback-
to-baseline path is tested.

**Files / Modules to Add or Change:**
- `src/qec/decoder/router.py` — Decoder router: routes to baseline or promoted path
- `src/qec/decoder/promoted_stub.py` — Promoted decoder stub (calls baseline until
  real promoted decoder is wired; stub is explicit, not silent)
- `tests/decoder/test_router.py` — Router tests
- `tests/decoder/test_rollback.py` — Rollback-to-baseline tests

**Tests to Add:**
- `test_router_default_is_baseline`: router with no config routes to baseline
- `test_router_opt_in_promoted`: router with `decoder="promoted"` routes to promoted stub
- `test_router_promoted_stub_matches_baseline`: promoted stub output == baseline output
- `test_router_rollback_to_baseline`: set `decoder="baseline"` after promoted → baseline
- `test_router_logs_decoder_selection`: router logs which decoder was selected (no silent routing)
- `test_router_no_silent_replacement`: assert promoted stub is explicit, not a monkey-patch

**Fixtures to Add:**
- `tests/fixtures/router_config_baseline.json` — router config declaring baseline
- `tests/fixtures/router_config_promoted.json` — router config declaring promoted

**Benchmarks:** None in this release.

**Acceptance Gates:**
- Default router path is baseline
- Promoted path is opt-in and explicit
- Rollback from promoted to baseline works and is tested
- Router logs decoder selection (no silent routing)
- `pytest tests/decoder/` passes

**Definition of Done:** Decoder router exists. Promoted path is opt-in. Baseline
remains callable. Rollback is tested. `pytest -q` passes.

**Must Not Do:**
- no silent decoder replacement
- no promoted decoder source activation (stub only in this release)
- no removal of baseline call path
- no mutation of `src/qec/decoder/` source files

**Dependency Boundaries:**
- depends on: v167.2 BaselineDecoderRuntimeAPI
- feeds into: v167.4 GF2AndStabilizerCore

---

## v167.4 — GF2AndStabilizerCore

**Purpose:** Implement or consolidate GF(2) rank, row-reduction, and nullspace.
Add repetition, Steane [7,1,3], and small surface-code parity-check fixtures.
All tests are hand-verified.

**Files / Modules to Add or Change:**
- `src/qec/codes/__init__.py` — Code construction package entry point
- `src/qec/codes/gf2.py` — GF(2) arithmetic: rank, row_reduce, nullspace, css_commute
- `src/qec/codes/repetition.py` — Repetition code H matrix construction
- `src/qec/codes/steane.py` — Steane [7,1,3] H_X, H_Z construction
- `src/qec/codes/surface.py` — Rotated surface code d=3 H_X, H_Z construction
- `tests/codes/test_gf2.py` — GF(2) algebra correctness tests
- `tests/codes/test_stabilizer_codes.py` — Stabilizer code construction tests

**Tests to Add:**
- `test_gf2_rank_hand_verified`: rank of known matrices matches hand-computed values
- `test_gf2_row_reduce_idempotent`: row_reduce(row_reduce(M)) == row_reduce(M)
- `test_gf2_nullspace_orthogonal`: M @ nullspace(M).T == 0 mod 2
- `test_repetition_h_matrix_d3`: H for d=3 matches [[1,1,0],[0,1,1]]
- `test_repetition_h_matrix_d5`: H for d=5 matches hand-verified 4×5 matrix
- `test_steane_css_commutation`: H_X @ H_Z.T == 0 mod 2
- `test_steane_parameters`: [[7,1,3]] — n=7, k=1, d=3
- `test_surface_d3_css_commutation`: H_X @ H_Z.T == 0 mod 2
- `test_surface_d3_parameters`: n=13, k=1, d=3 (rotated surface code)

**Fixtures to Add:**
- `tests/fixtures/codes/steane_h_x.json` — Steane H_X (hand-verified)
- `tests/fixtures/codes/steane_h_z.json` — Steane H_Z (hand-verified)
- `tests/fixtures/codes/surface_d3_h_x.json` — Surface d=3 H_X (hand-verified)
- `tests/fixtures/codes/surface_d3_h_z.json` — Surface d=3 H_Z (hand-verified)

**Benchmarks:** None in this release.

**Acceptance Gates:**
- GF(2) rank, row-reduction, nullspace pass hand-verified tests
- CSS commutation condition H_X @ H_Z.T == 0 mod 2 holds for Steane and surface d=3
- Steane parameters are [[7,1,3]]
- All fixture SHA-256 values are stable
- `pytest tests/codes/` passes

**Definition of Done:** GF(2) algebra and stabilizer code construction are correct
and tested. Golden parity-check fixtures exist for Steane and surface d=3. `pytest -q` passes.

**Must Not Do:**
- no code distance claim without verification
- no decoder implementation in this release
- no QLDPC construction yet (that is v167.5)
- no performance claims

**Dependency Boundaries:**
- depends on: v167.0 QECOSRuntimeSkeleton (fixture schema)
- feeds into: v167.5 QLDPCConstructionHarness

---

## v167.5 — QLDPCConstructionHarness

**Purpose:** Add a hypergraph product (HGP) / QLDPC construction harness. If qldpc
is used, it is adapter-only. Verify the CSS condition H_X * H_Z.T == 0 mod 2. No
performance claims.

**Files / Modules to Add or Change:**
- `src/qec/codes/hgp.py` — Hypergraph product code construction (native or qldpc adapter)
- `src/qec/codes/qldpc_adapter.py` — qldpc library adapter (adapter_only=true)
- `src/qec/codes/canonical.py` — Canonical form for parity-check matrices (sorted rows)
- `tests/codes/test_hgp.py` — HGP construction tests
- `tests/codes/test_qldpc_adapter.py` — qldpc adapter boundary tests

**Tests to Add:**
- `test_hgp_css_commutation`: H_X @ H_Z.T == 0 mod 2 for small HGP code
- `test_hgp_parameters_declared`: [[n,k,d]] parameters are declared (not claimed as proven)
- `test_qldpc_adapter_is_adapter_only`: adapter_only=true in adapter manifest
- `test_qldpc_output_matches_golden_fixture`: qldpc output verified against hand-computed small case
- `test_canonical_form_stable`: canonical form of same matrix is identical across runs
- `test_canonical_form_row_sorted`: canonical form has rows in lexicographic order

**Fixtures to Add:**
- `tests/fixtures/codes/hgp_small_h_x.json` — Small HGP H_X (hand-verified, [[n,k,d]] declared)
- `tests/fixtures/codes/hgp_small_h_z.json` — Small HGP H_Z (hand-verified)

**Benchmarks:** None in this release (construction time measured in v167.7).

**Acceptance Gates:**
- CSS commutation holds for HGP code
- qldpc adapter has adapter_only=true
- qldpc output verified against golden fixture (not trusted directly)
- Canonical form is stable and deterministic
- `pytest tests/codes/` passes

**Definition of Done:** HGP construction harness exists. CSS commutation is verified.
qldpc is adapter-only. `pytest -q` passes.

**Must Not Do:**
- no QLDPC performance claims without benchmarks
- no qldpc output accepted as ground truth
- no code distance proof without verification receipt
- no decoder implementation

**Dependency Boundaries:**
- depends on: v167.4 GF2AndStabilizerCore
- feeds into: v167.6 SyndromeNoiseRuntime

---

## v167.6 — SyndromeNoiseRuntime

**Purpose:** Add syndrome batch/stream representation, code-capacity / phenomenological
/ circuit-level noise fixtures. Stim is adapter-only. Deterministic seeds are required.

**Files / Modules to Add or Change:**
- `src/qec/syndrome/__init__.py` — Syndrome package entry point
- `src/qec/syndrome/batch.py` — Syndrome batch representation (bits, metadata, seed)
- `src/qec/syndrome/stream.py` — Syndrome stream API (windowed iteration)
- `src/qec/noise/__init__.py` — Noise model package entry point
- `src/qec/noise/code_capacity.py` — Code-capacity depolarizing noise (native, no Stim)
- `src/qec/noise/phenomenological.py` — Phenomenological noise (native)
- `src/qec/noise/stim_adapter.py` — Stim circuit-level noise adapter (adapter_only=true)
- `tests/syndrome/test_syndrome_batch.py` — Syndrome batch tests
- `tests/syndrome/test_noise_models.py` — Noise model determinism tests

**Tests to Add:**
- `test_code_capacity_noise_deterministic`: same seed → same syndrome batch
- `test_phenomenological_noise_deterministic`: same seed → same syndrome batch
- `test_stim_adapter_is_adapter_only`: adapter_only=true in stim adapter manifest
- `test_syndrome_batch_serializes_to_canonical_json`: batch → JSON → SHA-256 stable
- `test_syndrome_stream_windowed_iteration`: stream yields correct window sizes
- `test_noise_model_seed_required`: noise model raises if seed not provided

**Fixtures to Add:**
- `tests/fixtures/syndromes/repetition_d3_code_capacity_p001_seed42.json`
- `tests/fixtures/syndromes/surface_d3_phenomenological_p001_seed42.json`
- `tests/fixtures/syndromes/surface_d3_circuit_level_p001_seed42.json` (via Stim adapter)

**Benchmarks:** None in this release (throughput measured in v167.7).

**Acceptance Gates:**
- Code-capacity and phenomenological noise are deterministic given seed
- Stim adapter has adapter_only=true
- All syndrome fixtures are SHA-256 stable
- Noise model raises on missing seed
- `pytest tests/syndrome/` passes

**Definition of Done:** Syndrome batches for repetition code (code-capacity) and
surface code (phenomenological, circuit-level) are generatable and deterministic.
`pytest -q` passes.

**Must Not Do:**
- no live hardware syndrome ingestion
- no Stim output treated as ground truth
- no real-time performance claims
- no mixing noise model outputs without declared noise model type

**Dependency Boundaries:**
- depends on: v167.5 QLDPCConstructionHarness
- feeds into: v167.7 BenchmarkHarnessAndLER

---

## v167.7 — BenchmarkHarnessAndLER

**Purpose:** Add benchmark runner, logical error rate (LER) fixtures, Wilson score
confidence intervals, decoder latency measurements, and syndrome throughput measurements.
No benchmark marketing.

**Files / Modules to Add or Change:**
- `src/qec/benchmark/__init__.py` — Benchmark package entry point
- `src/qec/benchmark/runner.py` — Benchmark runner (wraps sinter or native sampler)
- `src/qec/benchmark/metrics.py` — LER, Wilson score CI, latency, throughput
- `src/qec/benchmark/report.py` — Benchmark report serializer (canonical JSON)
- `tests/benchmark/test_ler.py` — LER calculation and CI tests
- `tests/benchmark/test_latency.py` — Decoder latency measurement tests
- `tests/benchmark/test_throughput.py` — Syndrome throughput measurement tests
- `scripts/run_benchmark.py` — CLI benchmark runner

**Tests to Add:**
- `test_wilson_score_ci_calculation`: Wilson score 68% CI for known p, n values
- `test_ler_requires_shot_count`: LER report raises if shot_count not declared
- `test_ler_requires_hardware_declaration`: LER report raises if hardware not declared
- `test_decoder_latency_measurement`: latency measured in wall-clock ms, declared hardware
- `test_syndrome_throughput_measurement`: throughput in syndromes/sec, declared hardware
- `test_benchmark_report_canonical_json`: report serializes to canonical JSON with SHA-256

**Fixtures to Add:**
- `tests/fixtures/benchmarks/baseline_ler_repetition_d3.json` — LER with CI
- `tests/fixtures/benchmarks/baseline_ler_repetition_d5.json` — LER with CI
- `tests/fixtures/benchmarks/baseline_latency_repetition_d3.json` — latency measurement
- `tests/fixtures/benchmarks/baseline_throughput.json` — syndrome throughput

**Benchmarks to Add (CI cadence: weekly):**

| Benchmark | Metric | Fixture/Corpus | CI Cadence |
|---|---|---|---|
| Baseline decoder latency | ms per syndrome batch | repetition d=3,5 | weekly |
| Promoted decoder latency | ms per syndrome batch | repetition d=3,5 | weekly |
| Syndrome batch throughput | syndromes/sec | repetition d=3,5 | weekly |
| GF(2) rank / row-reduction time | ms per matrix | hand-verified matrices | weekly |
| Parity-check construction time | ms per code | repetition, Steane, surface d=3 | weekly |
| HGP construction time | ms per code | small HGP | weekly |
| Syndrome generation throughput | syndromes/sec | code-capacity, phenomenological | weekly |
| Logical error rate sampling | LER ± Wilson CI | repetition d=3,5,7 | weekly |
| Threshold sweep smoke benchmark | LER vs p, 3 distances | surface d=3,5,7 | weekly |

**Acceptance Gates:**
- Wilson score 68% CI is computed for all LER measurements
- All benchmark reports declare hardware type
- All benchmark reports declare shot count
- Benchmark runner exits nonzero if hardware not declared
- `pytest tests/benchmark/` passes

**Definition of Done:** LER curves exist for repetition code d=3,5. All results
include Wilson score CIs and declared hardware. Benchmark runner is CLI-callable.
`pytest -q` passes.

**Must Not Do:**
- no benchmark results without declared hardware
- no LER without Wilson score CI
- no threshold claim without sweep data
- no benchmark marketing (no "QEC advantage" claims)

**Dependency Boundaries:**
- depends on: v167.6 SyndromeNoiseRuntime
- feeds into: v167.8 CrossBackendDifferentialHarness

---

## v167.8 — CrossBackendDifferentialHarness

**Purpose:** Compare QEC OS decoder outputs against PyMatching, BP+OSD, and union-find
where available. External backends are adapters, not truth. Disagreement cases become
fixtures.

**Files / Modules to Add or Change:**
- `src/qec/differential/__init__.py` — Differential testing package entry point
- `src/qec/differential/harness.py` — Differential test harness
- `src/qec/differential/pymatching_adapter.py` — PyMatching adapter (adapter_only=true)
- `src/qec/differential/bposd_adapter.py` — BP+OSD adapter (adapter_only=true)
- `src/qec/differential/union_find_adapter.py` — Union-find adapter (adapter_only=true)
- `tests/differential/test_pymatching_comparison.py` — PyMatching differential tests
- `tests/differential/test_bposd_comparison.py` — BP+OSD differential tests
- `tests/differential/test_disagreement_logging.py` — Disagreement fixture logging

**Tests to Add:**
- `test_pymatching_adapter_is_adapter_only`: adapter_only=true, version declared
- `test_bposd_adapter_is_adapter_only`: adapter_only=true, version declared
- `test_differential_agreement_rate_logged`: agreement rate is computed and logged
- `test_disagreement_cases_become_fixtures`: disagreement cases serialized to JSON fixtures
- `test_no_external_backend_as_ground_truth`: harness raises if any adapter has authority=true
- `test_differential_harness_on_repetition_d3`: run differential test on repetition d=3

**Fixtures to Add:**
- `tests/fixtures/differential/repetition_d3_pymatching_comparison.json`
- `tests/fixtures/differential/repetition_d3_disagreements.json` (may be empty)

**Benchmarks to Add:**

| Benchmark | Metric | Fixture/Corpus | CI Cadence |
|---|---|---|---|
| PyMatching differential comparison time | ms per batch | repetition d=3 | weekly |

**Acceptance Gates:**
- All external adapters have adapter_only=true and declared version
- Agreement rates are logged
- Disagreement cases are serialized as fixtures
- No external backend treated as ground truth
- `pytest tests/differential/` passes

**Definition of Done:** QEC OS decoder outputs compared against PyMatching on
repetition d=3. Agreement rates measured and logged. Disagreement cases are fixtures.
`pytest -q` passes.

**Must Not Do:**
- no QEC OS decoder superiority claims
- no PyMatching output as ground truth
- no differential tests without declared comparator version

**Dependency Boundaries:**
- depends on: v167.7 BenchmarkHarnessAndLER
- feeds into: v167.9 OdinPortReadinessAndParitySpec

---

## v167.9 — OdinPortReadinessAndParitySpec

**Purpose:** Add Odin migration docs and specs, cross-language fixture format,
performance baseline snapshot, and Odin fork readiness checklist. No Odin rewrite
unless fixtures and parity tests exist (they now do).

**Files / Modules to Add or Change:**
- `docs/odin_migration/fixture_format.md` — Canonical JSON fixture format spec
- `docs/odin_migration/api_contracts.md` — API contracts for Odin implementation
- `docs/odin_migration/performance_baselines.md` — Python performance baselines (from v167.7)
- `docs/odin_migration/readiness_checklist.md` — Odin fork readiness checklist
- `scripts/validate_fixtures.py` — CLI fixture validator (updated for full corpus)
- `scripts/odin_parity_check.py` — Python-side parity check script for Odin output
- `tests/odin/test_fixture_format_spec.py` — Fixture format spec is machine-readable
- `tests/odin/test_odin_parity_contract.py` — Python/Odin parity contract tests

**Tests to Add:**
- `test_fixture_format_spec_is_json_schema`: fixture_format.md embeds valid JSON schema
- `test_performance_baselines_complete`: baselines cover all benchmarks from v167.7
- `test_readiness_checklist_complete`: all checklist items are checked
- `test_odin_parity_script_runs`: `odin_parity_check.py` runs without error on golden corpus
- `test_fixture_validator_full_corpus`: validator passes on complete golden corpus

**Fixtures to Add:**
- `tests/fixtures/odin/parity_contract_v167.json` — Python/Odin parity contract declaration

**Benchmarks to Add:**

| Benchmark | Metric | Fixture/Corpus | CI Cadence |
|---|---|---|---|
| Odin parity fixture validation time | ms per fixture | full golden corpus | weekly |

**Acceptance Gates:**
- Fixture format spec is machine-readable (JSON schema embedded)
- Performance baselines cover all benchmarks from v167.7
- Odin fork readiness checklist is complete (all items checked)
- `scripts/validate_fixtures.py` passes on full golden corpus
- `pytest tests/odin/` passes

**Definition of Done:** Golden fixture corpus is complete. Fixture format is specified.
Performance baselines are documented. Odin fork readiness checklist is complete.
Parity contract exists. `pytest -q` passes.

**Must Not Do:**
- no Odin code in this release (readiness only)
- no Odin performance claims without Odin benchmarks
- no Odin rewrite started before this release is complete

**Dependency Boundaries:**
- depends on: v167.8 CrossBackendDifferentialHarness
- enables: Odin port (post-v167.9 backlog)
- completes: v167.x QEC OS Pivot arc
```

---

## D. 10-Release Table: v167.0–v167.9

| Release | Name | Primary Deliverable | Key Tests | Key Fixtures | Benchmarks |
|---|---|---|---|---|---|
| v167.0 | QECOSRuntimeSkeleton | `src/qec/os/` API shape + fixture schema | fixture hash determinism | schema_v167.json | none |
| v167.1 | GoldenCorpusSeed | repetition-code golden fixtures + validator script | fixture round-trip, SHA-256 stability | repetition d=3,d=5 syndromes + parity checks | none |
| v167.2 | BaselineDecoderRuntimeAPI | `baseline_decode()` runtime API | baseline regression, determinism, no-side-effects | repetition corrections | none |
| v167.3 | PromotedDecoderRuntimeRouter | explicit decoder router + promoted stub | router default=baseline, opt-in promoted, rollback | router config fixtures | none |
| v167.4 | GF2AndStabilizerCore | GF(2) algebra + Steane + surface d=3 construction | GF(2) correctness, CSS commutation, hand-verified | Steane H_X/H_Z, surface d=3 H_X/H_Z | none |
| v167.5 | QLDPCConstructionHarness | HGP construction + qldpc adapter | CSS commutation, adapter_only, canonical form | small HGP H_X/H_Z | none |
| v167.6 | SyndromeNoiseRuntime | syndrome batch/stream + code-capacity/phenomenological/circuit-level noise | noise determinism, Stim adapter_only, seed required | syndrome fixtures (3 noise models) | none |
| v167.7 | BenchmarkHarnessAndLER | benchmark runner + LER + Wilson CI + latency + throughput | Wilson CI, hardware declaration, shot count | LER + latency + throughput fixtures | 9 benchmarks (weekly CI) |
| v167.8 | CrossBackendDifferentialHarness | differential harness + PyMatching/BP+OSD/union-find adapters | adapter_only, agreement rate, disagreement fixtures | differential comparison fixtures | PyMatching comparison time |
| v167.9 | OdinPortReadinessAndParitySpec | Odin migration docs + fixture format spec + readiness checklist | fixture format JSON schema, checklist complete, parity contract | parity contract fixture | Odin fixture validation time |

---

## E. Benchmark Matrix

| Benchmark Name | Metric | Fixture / Corpus | CI Cadence | Introduced |
|---|---|---|---|---|
| Baseline decoder latency | ms per syndrome batch | repetition d=3, d=5 | weekly | v167.7 |
| Promoted decoder latency | ms per syndrome batch | repetition d=3, d=5 | weekly | v167.7 |
| Syndrome batch throughput | syndromes/sec | repetition d=3, d=5 | weekly | v167.7 |
| GF(2) rank / row-reduction time | ms per matrix | hand-verified test matrices | weekly | v167.7 |
| Parity-check construction time | ms per code | repetition, Steane, surface d=3 | weekly | v167.7 |
| HGP construction time | ms per code | small HGP code | weekly | v167.7 |
| Syndrome generation throughput | syndromes/sec | code-capacity, phenomenological | weekly | v167.7 |
| Logical error rate sampling | LER ± Wilson 68% CI | repetition d=3, d=5, d=7 | weekly | v167.7 |
| Threshold sweep smoke benchmark | LER vs p, 3 distances | surface d=3, d=5, d=7 | weekly | v167.7 |
| PyMatching differential comparison time | ms per batch | repetition d=3 | weekly | v167.8 |
| Odin parity fixture validation time | ms per fixture | full golden corpus | weekly | v167.9 |

**Benchmark Rules:**
- All benchmark reports must declare hardware type (CPU model, RAM, OS)
- All LER measurements must include Wilson score 68% confidence intervals
- All latency measurements must declare measurement method (wall-clock, perf_counter)
- No benchmark claim without declared shot count
- No threshold claim without sweep data across at least 3 code distances
- Benchmark results are bounded observations, not correctness proofs
- No benchmark marketing; no "QEC advantage" claims

---

## F. Test Matrix

| Test Category | Purpose | Required Fixtures |
|---|---|---|
| Golden fixture round-trip | Serialize → deserialize → SHA-256 must be identical | schema_v167.json |
| Baseline decoder runtime API | `baseline_decode()` returns correct corrections | repetition d=3,d=5 corrections |
| Promoted router equivalence | Promoted stub output == baseline output | router config fixtures |
| Rollback-to-baseline | Router reverts to baseline on explicit config | router config fixtures |
| GF(2) algebra correctness | Rank, row-reduce, nullspace match hand-verified values | hand-verified matrices (inline) |
| Stabilizer construction correctness | H matrix dimensions and values match hand-verified | Steane H_X/H_Z, surface d=3 H_X/H_Z |
| CSS commutation | H_X @ H_Z.T == 0 mod 2 for all CSS codes | Steane, surface d=3, HGP fixtures |
| Syndrome noise determinism | Same seed → same syndrome batch, 3 runs | repetition d=3 code-capacity fixture |
| LER confidence interval calculation | Wilson score 68% CI for known p, n | LER fixtures |
| Cross-backend differential disagreement logging | Disagreement cases serialized as fixtures | differential comparison fixtures |
| Odin fixture compatibility | Parity contract fixture validates against JSON schema | parity_contract_v167.json |

---

## G. Odin Readiness Checklist

The following artifacts must exist before an Odin fork can begin. All are produced by v167.0–v167.9.

- [ ] **Stable canonical JSON fixture schema** — `tests/fixtures/schema_v167.json` (v167.0)
- [ ] **Golden syndrome fixtures** — repetition d=3,d=5; surface d=3 phenomenological and circuit-level (v167.1, v167.6)
- [ ] **Golden correction fixtures** — baseline corrections for repetition d=3,d=5 (v167.2)
- [ ] **Golden parity-check fixtures** — repetition, Steane, surface d=3, HGP (v167.1, v167.4, v167.5)
- [ ] **GF(2) algebra fixtures** — hand-verified rank, row-reduce, nullspace test cases (v167.4)
- [ ] **Benchmark baselines** — wall-clock times for all 11 benchmarks, declared hardware (v167.7)
- [ ] **CLI fixture validator** — `scripts/validate_fixtures.py` exits 0 on valid corpus (v167.1, updated v167.9)
- [ ] **Python/Odin parity contract** — `tests/fixtures/odin/parity_contract_v167.json` (v167.9)
- [ ] **Deterministic ordering rules** — canonical form spec in `docs/odin_migration/fixture_format.md` (v167.9)
- [ ] **Error/failure fixture corpus** — disagreement cases from cross-backend differential (v167.8)

---

## H. Deferred / Removed From Near-Term Roadmap

| Old Arc | Old Version | Classification | Reason |
|---|---|---|---|
| Qudit / Ququart / High-Dimensional Stabilizer Receipts | v167.x (old) | **Deferred to research backlog** | Pure receipt arc; no executable QEC OS work |
| MIDI / Sonification Receipts | v168.x (old) | **Deleted from near-term roadmap** | No executable QEC work; no tests; no fixtures |
| Symbolic Geometry Grammar / Cosmovirus Sandbox | v169.x (old) | **Deleted from near-term roadmap** | Mystical/cosmological detour; no executable QEC work |
| Reproducible Build / Supply-Chain Receipts | v170.x (old) | **Deferred to post-v167.9 backlog** | Infrastructure; not blocking QEC OS pivot |
| Deterministic Knowledge Base / Agent Memory Receipts | v171.x (old) | **Deleted from near-term roadmap** | Agent memory; no executable QEC work |
| Materials / Photonic / Diamond / Device Signal Receipts | v172.x (old) | **Deleted from near-term roadmap** | Hardware signal receipts; no executable QEC work |
| Interactive Proof Worlds / Citizen-Science Game Receipts | v173.x (old) | **Deleted from near-term roadmap** | Gamification; no executable QEC work |
| Fault-Tolerant Resource Accounting Receipts | v178.x (old) | **Deferred to post-v167.9 backlog** | Useful but not blocking QEC OS pivot |
| IEEE 754 Precision & Approximation Receipts | v178.5.x (old) | **Deferred to post-v167.9 backlog** | Useful but not blocking QEC OS pivot |
| Quantum Geometry / Contextuality / Topological Toolkit | v179.x (old) | **Deferred to research backlog** | Research signal; not blocking QEC OS pivot |
| Deterministic Quantum ML Boundary Receipts | v180.x (old) | **Deleted from near-term roadmap** | No executable QEC work |
| Local Agent / Tool Dispatch Receipts v2 | v181.x (old) | **Deleted from near-term roadmap** | Agent dispatch; no executable QEC work |
| Interpretability / Sparse Feature Receipts | v182.x (old) | **Deleted from near-term roadmap** | Interpretability; no executable QEC work |
| Reproducible Research Publication Receipts | v183.x (old) | **Deferred to post-v167.9 backlog** | Publication infrastructure; not blocking |
| Photonic / Materials / Device Signal Receipts v2 | v185.x (old) | **Deleted from near-term roadmap** | Duplicate of v172.x deletion |
| Symbolic Diagram Compiler v2 / Grammar Extension | v186.x (old) | **Deleted from near-term roadmap** | Symbolic grammar; no executable QEC work |
| Operator / IRC / TUI / CLI Federation Receipts v2 | v190.x (old) | **Deleted from near-term roadmap** | Operator federation; no executable QEC work |
| Hermetic Environment Receipts v2 | v191.x (old) | **Deferred to post-v167.9 backlog** | Infrastructure; not blocking QEC OS pivot |
| Global Proof Composition v2 | v192.x (old) | **Deleted from near-term roadmap** | Terminal composition receipt; no executable QEC work |

**Collapsed into QEC OS Support (not deleted, absorbed into v167.x):**

| Old Arc | Absorbed Into |
|---|---|
| v167.x QEC OS Runtime & Benchmark Reset (first-pass) | v167.0 QECOSRuntimeSkeleton |
| v168.x Decoder Runtime Activation (first-pass) | v167.2 + v167.3 |
| v169.x Stabilizer & QLDPC Construction Engine (first-pass) | v167.4 + v167.5 |
| v170.x Syndrome Stream & Noise Model Runtime (first-pass) | v167.6 |
| v171.x Benchmark Corpus & Logical Error Rate Harness (first-pass) | v167.7 |
| v172.x Cross-Backend Differential Testing (first-pass) | v167.8 |
| v173.x Odin Port Readiness Layer (first-pass) | v167.9 |
| v174.x Odin Runtime Prototype (first-pass) | Post-v167.9 backlog (readiness now complete) |
| v175.x QEC OS Integration Milestone (first-pass) | Post-v167.9 backlog |

---

## Deferred Research Backlog

The following material is not near-term executable QEC OS work. It is preserved for
future consideration but must not appear in the v167.x roadmap.

- **Qudit / Ququart / High-Dimensional Stabilizer work** — Requires GF(d) arithmetic beyond GF(2); defer until v167.4 GF(2) core is stable
- **Odin Runtime Prototype** — Enabled by v167.9 readiness checklist; can begin as a separate fork after v167.9 is complete
- **QEC OS Integration Milestone** — End-to-end experiment runner; natural follow-on after v167.9
- **Fault-Tolerant Resource Accounting** — Overhead estimation for fault-tolerant circuits; useful but not blocking
- **Reproducible Build / Supply-Chain** — Hermetic build receipts; useful but not blocking QEC OS pivot
- **Quantum Geometry / Contextuality** — Research signal; requires primary source verification
- **Threshold sweep at scale** — Full threshold sweep beyond smoke benchmark; requires more compute
- **Odin performance benchmarks** — Cannot exist until Odin implementation exists
- **Publication / Zenodo artifacts** — Natural output after v167.9 corpus is complete
- **Operator Console Unification** — IRC/TUI/CLI federation; not blocking QEC OS work

---

## I. Final Next 3 Codex Tasks After v166.8

### Task 1: v167.0 — QECOSRuntimeSkeleton

**Implementation Summary:** Create the `src/qec/os/` package with four modules: `__init__.py` (package entry point and version declaration), `api.py` (defines the `decode(syndrome_batch: dict) -> dict` runtime API shape as an abstract interface with type annotations), `fixture.py` (implements `serialize_fixture(data: dict) -> dict` that adds `sha256` field using `hashlib.sha256` over `json.dumps(data, sort_keys=True, separators=(',', ':'))`, and `deserialize_fixture(path: str) -> dict` that recomputes and validates the hash), and `registry.py` (a simple dict-based registry mapping code family names to parameter schemas). Tests in `tests/os/test_fixture_determinism.py` run the round-trip three times with different `PYTHONHASHSEED` values using `subprocess` and assert identical SHA-256 output each time.

**Expected Files:**
- `src/qec/os/__init__.py`
- `src/qec/os/api.py`
- `src/qec/os/fixture.py`
- `src/qec/os/registry.py`
- `tests/os/__init__.py`
- `tests/os/test_fixture_determinism.py`
- `tests/fixtures/schema_v167.json`

---

### Task 2: v167.1 — GoldenCorpusSeed

**Implementation Summary:** Generate the first golden fixture corpus using the fixture serializer from v167.0. Write a small `scripts/generate_fixtures.py` that constructs the repetition code parity-check matrix for d=3 and d=5 by hand (no external libraries needed: H for d=3 is `[[1,1,0],[0,1,1]]`), generates 10 syndrome batches per distance using a simple XOR-based code-capacity noise model with `numpy.random.default_rng(seed=42)`, serializes each batch to canonical JSON with SHA-256, and writes to `tests/fixtures/`. The fixture validator `scripts/validate_fixtures.py` reads every JSON file in `tests/fixtures/`, recomputes SHA-256, and exits nonzero if any hash mismatches. Tests assert that the validator exits 0 on the valid corpus and nonzero when a fixture is intentionally tampered.

**Expected Files:**
- `scripts/generate_fixtures.py`
- `scripts/validate_fixtures.py`
- `tests/fixtures/repetition_d3_syndromes.json`
- `tests/fixtures/repetition_d5_syndromes.json`
- `tests/fixtures/repetition_d3_parity_check.json`
- `tests/fixtures/repetition_d5_parity_check.json`
- `tests/os/test_golden_corpus.py`

---

### Task 3: v167.2 — BaselineDecoderRuntimeAPI

**Implementation Summary:** Add `src/qec/decoder/baseline_api.py` that imports the existing canonical decoder from `src/qec/decoder/` (without modifying any existing file) and exposes a single public function `baseline_decode(syndrome_batch: dict) -> dict` that validates the input against the fixture schema, calls the existing decoder, wraps the output in a canonical JSON structure with `decoder="baseline"`, `version="v166.0"`, and a SHA-256 hash, and returns it. The function must be deterministic: same input dict → same output dict on three consecutive calls. Tests in `tests/decoder/test_baseline_api.py` run the baseline against the repetition d=3 and d=5 golden fixtures from v167.1, assert determinism, assert no mutation of the input dict, and use `git diff --exit-code src/qec/decoder/` to assert that no source files in `src/qec/decoder/` were modified.

**Expected Files:**
- `src/qec/decoder/baseline_api.py`
- `src/qec/decoder/__init__.py` (updated to export `baseline_decode`)
- `tests/decoder/__init__.py`
- `tests/decoder/test_baseline_api.py`
- `tests/decoder/test_baseline_golden_corpus.py`
- `tests/fixtures/repetition_d3_corrections.json`
- `tests/fixtures/repetition_d5_corrections.json`

---

## Hard Rejection Rules (Enforcement Checklist)

The following rules apply to every release in v167.0–v167.9. Any release that violates one of these rules must be rejected and revised before merging.

| Rule | Applies To |
|---|---|
| No release that only defines a manifest with no executable test or fixture | All releases |
| No release that only adds prose | All releases |
| No release that only registers receipts | All releases |
| No release that claims performance without benchmark measurements | v167.7+ |
| No release that claims correctness from a benchmark | All releases |
| No release that treats Stim, PyMatching, qldpc, or any backend as authority | v167.5, v167.6, v167.8 |
| No release that starts Odin rewrite before fixture and parity specs exist | Pre-v167.9 |
| No release that silently replaces `src/qec/decoder/` | All releases |
| No release that claims hardware authority or QEC advantage | All releases |

---

*End of QEC Second-Pass Roadmap Tightening Document*
