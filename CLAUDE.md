# QSOL QEC Architectural Constitution (v99 Hardened)

This document governs all AI-assisted activity inside the QSOL QEC repository.

It applies to Claude when performing:

- code generation
- refactoring
- testing
- experiments
- commits
- release preparation

This document is not guidance.

It is the architectural constitution of the repository.

Claude must obey these rules when operating inside this codebase.

---

# Core Values

The QEC framework is governed by six non-negotiable principles:

1. Determinism  
2. Decoder Stability  
3. Architectural Layering  
4. Minimal Complexity  
5. Scientific Transparency  
6. Reproducibility Guarantees  

---

# 0. Direct Commit Mode (Non-PR Workflow)

⚠️ FIRST

If you create a pull request (PR), the task is considered FAILED.

Claude must operate in **Direct Commit Mode**.

## Rules

- DO NOT create a PR  
- DO NOT create a new branch  
- Work directly on `main`  
- You MAY commit and push  
- Commit directly to `main` only  

## Rationale

The repository is:

- single-operator controlled  
- deterministic  
- release-tag driven  

Branching/PR workflows introduce:

- merge ambiguity  
- sync confusion  
- non-deterministic repo state  

## Safety Conditions

Direct commits are allowed only if:

- all tests pass  
- determinism is preserved  
- decoder core is untouched  
- architectural invariants hold  

All escalation rules remain in effect.

---

# 1. Architectural Layer Model (Non-Negotiable)

Dependencies may only flow downward.

| Layer | Path | Role |
|------|------|------|
| 1 | src/qec/decoder/ | Decoder core (protected) |
| 2 | src/qec/channel/ | Channel models |
| 3 | src/qec/diagnostics/ | Observational diagnostics |
| 4 | src/qec/predictors/ | Instability predictors |
| 5 | src/qec/analysis/ | Metrics, attractors, strategy logic |
| 6 | src/qec/experiments/ | Experimental orchestration |
| 7 | src/bench/ | Benchmark harness |

## Rules

- Lower layers must never import higher layers  
- decoder must never import experiments  
- src/qec must never import src/bench  
- predictors must not depend on experiments  

Layer boundaries are architectural invariants.

Violation is forbidden.

---

# 2. Determinism is Architecture

Determinism is a structural requirement.

All code must preserve deterministic execution.

## Required invariants

- no hidden randomness  
- no implicit RNG state  
- explicit seed injection  
- deterministic ordering  
- no use of Python hash()  
- no unordered floating reductions  

## Required techniques

- SHA-256 deterministic sub-seeds  
- canonical serialization  
- stable iteration ordering  

## Guarantee

If:

runtime_mode = "off"

then outputs must be byte-identical across runs.

---

# 3. Artifact & Identity Stability

Artifacts are immutable experiment records.

## Rules

- no mutation after hashing  
- canonical serialization only  
- identity independent of memory layout  
- identical configs → identical outputs  

Identity drift without version bump is forbidden.

---

# 4. Decoder Core Protection

Protected path:

src/qec/decoder/

## Default rule

Do not modify the decoder core.

## Forbidden without explicit instruction

- BP message updates  
- scheduling semantics  
- iteration ordering  
- decoder refactors  
- adaptive logic inside decoder  

The decoder must remain bit-stable across minor releases.

---

# 5. Adaptive System Constraint (v99+)

The system includes:

metrics → strategy → evaluation → adaptation → memory

## Rules

- adaptation must remain deterministic  
- memory must be bounded  
- no stochastic learning  
- no hidden state outside defined structures  

## Allowed

- score adjustments  
- biasing strategy selection  
- evaluation-based feedback  

## Forbidden

- randomness  
- model training  
- implicit learning  
- mutation of past history  

---

# 6. Diagnostics Layer Protection

Diagnostics are observational only.

## Must be

- deterministic  
- side-effect free  
- opt-in  

## Must NOT

- modify decoder inputs  
- alter BP messages  
- mutate arrays in-place  

Diagnostics may operate on copies only.

---

# 7. Predictor Layer Protection

Predictors estimate instability pre-decode.

## Must

- use diagnostics outputs only  
- be deterministic  
- produce signals only  

## Must NOT

- modify decoder  
- modify inputs  

---

# 8. Controller Layer Protection

Controllers run controlled experiments.

## May

- modify inputs  
- run multiple passes  

## Must NOT

- modify decoder implementation  
- introduce randomness  

---

# 9. Sparse Linear Algebra Rules

Spectral methods must scale.

## Forbidden

- dense Hashimoto matrices  
- full eigendecomposition  

## Required

- sparse operators  
- scipy.sparse.linalg.eigs  
- linear operator interfaces  

Memory must scale with |E|, not |E|².

---

# 10. Tanner Graph Constraints

QLDPC constraint:

H_X H_Z^T = 0

Graph modifications must preserve commutativity.

Invalid transformations must be rejected.

---

# 11. Minimal Diff Discipline

Changes must be minimal and targeted.

## Forbidden

- large refactors  
- renaming identifiers  
- style-only edits  
- reformatting  
- import shuffling  

Each commit must be single-purpose.

---

# 12. Dependency Policy

## Rules

- prefer stdlib  
- prefer NumPy / SciPy  
- no new dependencies without approval  
- no frameworks  

Architectural bloat is forbidden.

---

# 13. Test Discipline

Untested code is unshipped code.

## Required

- unit tests  
- determinism tests  
- regression tests  

## Rules

- do not widen tolerances to pass tests  
- fix code, not tests  

---

# 14. Commit & Push Discipline

Claude may commit only if:

- tests pass  
- determinism preserved  
- decoder untouched  
- schema stable  
- identity stable  

## Critical rule

Passing tests alone are insufficient.

---

# 15. Escalation Rule

If a change affects:

- decoder semantics  
- scheduling  
- schema  
- serialization  
- hashing  
- determinism  

Claude must:

1. STOP  
2. Explain the risk  
3. Request instruction  

Silence is not consent.

---

# 16. Governing Principle

When uncertain:

- preserve stability  
- avoid refactoring  
- prefer doing nothing  
- read before writing  
- maintain invariants  

Capability grows.  
Stability does not regress.

If it cannot be reproduced byte-for-byte,
it is not a valid result.
