# QSOL-IMC QEC Architectural Constitution

## Canonical Engineering Constitution — v150.x

This document governs **all AI-assisted activity** inside the `QSOLKCB/QEC` repository.

This is the **constitutional layer of the system**.

All code generation, testing, refactoring, and architectural decisions must obey this file.

This is not guidance.

This is **law**.

---

# Core System Principle

```text
same input
→ same ordering
→ same canonical form
→ same hash
→ same bytes
→ same proof
```

Any violation invalidates the system.

---

# 0. Operating Model — Deterministic Direct-Commit System

QEC operates as a **deterministic, proof-driven repository**.

## Rules

* work directly on `main`
* no feature branches
* no PR-first workflow
* commits must be minimal and single-purpose
* tags define release boundaries

## Commit Preconditions

* full test suite passes
* determinism preserved
* identity + hashing invariants preserved
* decoder untouched
* schemas stable

---

# 1. Architectural Layer Model (HARD INVARIANT)

| Layer | Path                   | Role                       |
| ----- | ---------------------- | -------------------------- |
| 1     | `src/qec/decoder/`     | Protected core             |
| 2     | `src/qec/channel/`     | Signal models              |
| 3     | `src/qec/diagnostics/` | Observability              |
| 4     | `src/qec/analysis/`    | Deterministic intelligence |
| 5     | `src/qec/experiments/` | Controlled experiments     |
| 6     | `src/bench/`           | Harness                    |
| 7     | `src/qec/sims/`        | Simulation                 |

## Rules

* lower layers NEVER import higher layers
* decoder NEVER imports analysis
* no circular imports
* no upward leakage

Violation is forbidden.

---

# 2. Determinism is Architecture

Determinism is not optional.

## Forbidden

* randomness
* wall-clock dependence
* async nondeterminism
* implicit ordering

## Required

* canonical ordering everywhere
* explicit identity
* stable serialization
* deterministic reduction

---

# 3. Identity Law (v150.x)

Identity is the root of all system truth.

## Rules

* ALL identity-bearing tuples MUST pass `canonical_hash_identity`
* identity must be:

  * sorted
  * unique
  * canonical
* identity MUST NOT be inferred or reconstructed heuristically
* duplicate identity elements are forbidden

Invalid identity → invalid system state

---

# 4. Hashing & Proof Law

All proofs are hash-based.

## Rules

* hashes MUST be computed over canonical JSON
* keys sorted
* compact separators
* UTF-8 encoding

## Critical Rule

* self-referential hash fields MUST be excluded from hash bodies

## Guarantee

* all hashes MUST recompute exactly
* mismatch = `ValueError("INVALID_INPUT")`

---

# 5. Decoder Core Protection (SACRED)

Protected path:

```
src/qec/decoder/
```

## Forbidden

* modifying BP logic
* scheduling changes
* adaptive logic
* supervisory leakage

Decoder is immutable infrastructure.

---

# 6. Supervisory & Analysis Law

System structure:

```text
diagnostics
→ analysis
→ decision
→ verification
→ proof
```

## Allowed

* deterministic control
* governance systems
* verification logic
* explainability

## Forbidden

* stochastic control
* hidden learning state
* decoder-side supervision

---

# 7. Message & Protocol Law (v150.4)

All inter-agent communication MUST be deterministic.

## Rules

* message identity = `message_hash`
* payloads MUST be canonical
* duplicate messages are forbidden
* ordering MUST be canonical:

```text
(message_hash, sender_id, receiver_id)
```

## Forbidden

* timestamps
* async ordering
* external sequencing

---

# 8. Convergence Law (v150.5)

Systems MUST stabilize or fail.

## Rules

* convergence MUST be finite
* convergence MUST be deterministic
* convergence MUST be provable
* convergence flags MUST be derived, not set

## Enforcement

* inconsistent convergence state = INVALID_INPUT
* non-convergent system = INVALID_INPUT

---

# 9. Receipt Integrity Law

Receipts are proof artifacts.

## Rules

* receipts MUST be immutable
* receipts MUST be canonical
* receipts MUST be hash-verifiable
* receipt hash MUST exclude itself

## Guarantee

* receipt MUST recompute exactly under replay

Invalid receipt = invalid system

---

# 10. Safety Law

Safety overrides all.

## Required

* fail-safe states
* bounded recovery
* deterministic transitions

## Forbidden

* unsafe bypass
* hidden override

---

# 11. Minimal Diff Discipline

Every change must be surgical.

## Forbidden

* broad refactors
* style-only edits
* unrelated cleanup

## Required

* smallest viable change
* single-purpose commits

---

# 12. Test Discipline

Untested code is invalid.

## Required

* deterministic replay tests
* invariant tests
* boundary tests
* hash stability tests

## Rule

* fix code, not tests

---

# 13. Full-System Validation Rule

Before merge:

```bash
pytest -q
```

Must pass fully.

No exceptions.

---

# 14. Parallelism Law

Parallelism is allowed only when:

* tasks are independent
* determinism is preserved

Sequential required when:

* ordering affects outcome

---

# 15. Dependency Law

## Order

1. stdlib
2. existing deps
3. pinned source
4. package index (last resort)

## Rules

* version pinned
* checksum verifiable
* minimal scope

---

# 16. Escalation Rule

If change affects:

* identity
* hashing
* convergence
* protocol
* decoder
* safety

STOP.

Explain risk.

---

# 17. Governing Principle

When uncertain:

* preserve determinism
* preserve identity
* preserve proof
* prefer minimal change
* verify before commit

---

# Final Law

If it cannot be reproduced byte-for-byte,

it is not a valid result.
