# QSOL QEC Architectural Constitution

## Canonical Engineering Constitution — v133.x Hardened

This document governs **all AI-assisted activity** inside the `QSOLKCB/QEC` repository.

This file is treated as **always-active repository law**.

It is assumed to be read on **every prompt / every turn** by AI coding systems.

This is not guidance.

This is the **constitutional layer of the codebase**.

All code generation, testing, refactoring, release preparation, commits, and architectural decisions must obey this file.

---

# Core Values

The QEC framework is governed by **seven non-negotiable principles**:

1. Determinism
2. Safety
3. Decoder Stability
4. Architectural Layering
5. Minimal Complexity
6. Scientific Transparency
7. Reproducibility Guarantees

These principles override convenience.

---

# 0. Operating Mode — Direct Commit Constitutional Workflow

QEC is operated as a **direct-commit deterministic repository**.

## Rules

* DO NOT create PRs
* DO NOT create feature branches
* DO NOT fork internal workflow
* work directly on `main`
* commits must be minimal and single-purpose
* tags are the release boundary

PR workflow is considered **architecturally invalid** for this repository.

## Safety Conditions

Direct commits are allowed only if:

* tests pass
* determinism preserved
* decoder untouched
* schemas stable
* no invariant regression

---

# 0A. Dependency Acquisition Law

Do not install from live package indexes unless explicitly approved.

Preferred order:

1. stdlib
2. existing repository dependencies
3. canonical upstream release tarball / source archive
4. package index only as last resort

All third-party additions must be:

* version pinned
* source traceable
* checksum verifiable
* minimally scoped

Live package indexes are **disallowed by default**.

---

# 1. Architectural Layer Model (NON-NEGOTIABLE)

Dependency flow is strictly downward.

| Layer | Path                   | Role                      |
| ----- | ---------------------- | ------------------------- |
| 1     | `src/qec/decoder/`     | Protected decoder core    |
| 2     | `src/qec/channel/`     | Channel / LLR models      |
| 3     | `src/qec/diagnostics/` | Observational signals     |
| 4     | `src/qec/analysis/`    | Supervisory intelligence  |
| 5     | `src/qec/experiments/` | Controlled experiments    |
| 6     | `src/bench/`           | Deterministic harness     |
| 7     | `src/qec/sims/`        | Simulation & universe lab |

## Rules

* lower layers must never import higher layers
* decoder must never import analysis
* decoder must never import experiments
* decoder must never import sims
* analysis must not mutate decoder internals
* no circular imports
* no upward leakage

Layer boundaries are **hard invariants**.

Violation is forbidden.

---

# 1A. Simulation Layer Law (v133.x)

Simulation systems are now a **first-class architectural layer**.

Protected simulation paths include:

* `src/qec/sims/`
* `src/qec/simulation/`

## Rules

* simulation layers must remain additive
* simulation code must never mutate decoder semantics
* simulation adapters must remain optional
* external backend integrations must not become hard dependencies
* experiment layers must consume simulation layers, not redefine them
* simulation outputs must remain replay-safe
* tuple-only collections preferred
* frozen dataclasses strongly preferred

Simulation kernels must remain:

* pure
* immutable
* deterministic
* backend-agnostic

---

# 2. Determinism is Architecture

Determinism is not a feature.

It is a hard structural requirement.

All code must preserve **byte-identical replay**.

## Required invariants

* no hidden randomness
* no implicit RNG state
* no time-based behavior
* no async nondeterminism
* deterministic ordering
* stable floating reductions
* canonical serialization
* stable dict / tuple ordering
* frozen dataclasses preferred

## Required techniques

* explicit seed injection
* SHA-256 deterministic sub-seeds
* sorted iteration everywhere
* TypedDict / stable schemas
* immutable state objects

## Guarantee

If configuration is fixed:

```text
same input
→ same output
→ same bytes
```

Anything else is invalid.

---

# 3. Decoder Core Protection (SACRED)

Protected path:

```text
src/qec/decoder/
```

## Default rule

Do not modify the decoder core.

## Forbidden without explicit instruction

* BP update equations
* scheduling semantics
* iteration ordering
* decoder refactors
* adaptive logic inside decoder
* supervisory logic leakage
* temporal verification logic inside decoder

The decoder is sacred infrastructure.

Minor releases must not alter decoder semantics.

---

# 4. Supervisory Control Law

System identity:

```text
diagnostics
→ control
→ supervision
→ verification
→ explainability
```

## Allowed

* deterministic control systems
* supervisory state machines
* hysteresis controllers
* temporal verifiers
* policy memory
* theorem verification
* explainability systems

## Forbidden

* stochastic control
* ML black-box control
* hidden learning state
* decoder-side supervision

---

# 5. Safety Law

Safety always dominates performance.

Fail-safe precedence must be preserved.

## Required

* absorbing safe states
* escalation locks
* fail-safe latching
* bounded-time recovery
* deterministic temporal legality

## Forbidden

* unsafe recovery bypass
* fail-safe override without explicit approval
* unsafe temporal transitions

---

# 6. Minimal Diff Discipline

Every commit must be minimal and single-purpose.

## Forbidden

* broad refactors
* rename-only edits
* style-only commits
* formatting churn
* unrelated cleanup

## Required

* surgical changes
* smallest viable diff
* deterministic impact

Parallelism is allowed only for independent modules and tests.

Do not create merge ambiguity.

---

# 7. Dependency Policy

## Rules

* prefer stdlib
* prefer NumPy / SciPy
* NetworkX allowed when justified
* no heavy frameworks
* no new dependencies without explicit approval

Architectural bloat is forbidden.

---

# 8. Sparse Linear Algebra Rules

Spectral and graph work must scale.

## Forbidden

* dense graph matrices
* full eigendecomposition on large systems
* O(N²) memory blowups

## Required

* sparse operators
* `scipy.sparse`
* iterative solvers
* memory scaling with graph edges

---

# 9. Formal Verification Direction

Allowed future integrations:

* Coq
* Lean
* TLA+
* UPPAAL
* SMT / SAT systems

All integrations must remain deterministic and optional.

---

# 10. Test Discipline

Untested code is unshipped code.

## Required

* unit tests
* deterministic replay tests
* regression tests
* schema stability tests
* boundary-condition tests

## Rules

* do not widen tolerances to hide defects
* fix code, not tests
* every new module must ship with tests

---

# 10A. Experimental Reproducibility

All simulation experiments must be replayable.

## Required

* deterministic initial state
* explicit law configuration
* fixed step count
* stable sweep ordering
* tuple-only result ordering
* canonical summary serialization

Comparative experiments must yield identical results under repeated execution.

---

# 11. Commit Discipline

Commit only if:

* tests pass
* schemas stable
* determinism preserved
* decoder untouched
* safety invariants preserved

Passing tests alone is insufficient.

---

# 12. Escalation Rule

If a change affects:

* decoder semantics
* schema contracts
* determinism
* fail-safe logic
* supervisory transitions
* serialization
* hashing
* simulation law semantics

STOP.

Explain risk.

Request instruction.

---

# 13A. Theory Ingestion Priority

When ingesting theory from the `/papers` corpus:

1. Use `ROADMAP.md` first
2. Use existing Layer 4 modules second
3. Use `papers/*.md` when available
4. Use `papers/*.pdf` only when explicitly required

## Rules

* text-first ingestion is mandatory
* PDF rendering toolchains (`pdftoppm`) are not guaranteed in all environments
* missing `pdftoppm` is an environment-only limitation, not a repository failure
* do not retry failed PDF render paths
* do not treat PDF toolchain absence as a defect
* always check for `.md` equivalent before attempting `.pdf`

This rule is **durable workflow law** from `v137.0.19` onward.

---

# 13B. Execution Parallelism Law

Parallelism is **preferred by default**.

Claude Code must use parallel execution whenever tasks are logically independent and can be executed concurrently. Sequential execution is permitted only when outputs are dependency-coupled, ordering affects correctness, or later steps require earlier results.

## Mandatory parallel candidates

The following operations must be parallelized when multiple instances occur in the same pass:

* reading multiple markdown files
* reading papers / theory documents
* scanning roadmap + implementation docs simultaneously
* running independent test groups
* reading multiple source modules
* analyzing multiple release artifacts
* file discovery / glob operations
* import hygiene checks across independent modules

## Sequential execution conditions

Sequential execution is allowed only when:

* outputs are dependency-coupled (one result feeds into the next)
* ordering affects correctness
* later steps require earlier results

## Rules

* independent file reads must be batched into parallel tool calls
* independent searches must be batched into parallel tool calls
* do not serialize work that has no data dependency
* do not wait for one independent result before requesting another
* parallel execution must not compromise determinism guarantees

This rule is **durable engineering law** from `v137.0.19` onward.

---

# 13. Governing Principle

When uncertain:

* preserve determinism
* preserve safety
* preserve stability
* prefer minimal change
* read before writing
* verify before committing

Capability may grow.

Stability must never regress.

If it cannot be replayed byte-for-byte, it is not a valid result.
