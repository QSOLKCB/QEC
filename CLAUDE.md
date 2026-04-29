# QSOL QEC Architectural Constitution

## Canonical Engineering Constitution — v150.x

This document governs **all AI-assisted activity** inside the `QSOLKCB/QEC` repository.

This is the **constitutional layer of the system**.

All code generation, testing, refactoring, release preparation, commits, and architectural decisions must obey this file.

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

Violation invalidates the system.

---

# Core Values

1. Determinism
2. Safety
3. Decoder Stability
4. Architectural Layering
5. Minimal Complexity
6. Scientific Transparency
7. Reproducibility

---

# 0. Operating Model — Deterministic Direct-Commit System

## Rules

* work directly on `main`
* no feature branches
* no PR-first workflow
* commits must be minimal and single-purpose
* tags define release boundaries

## Commit Preconditions

* tests pass (see Validation Law)
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

---

# 2. Determinism is Architecture

## Forbidden

* randomness
* wall-clock dependence
* async nondeterminism
* implicit ordering

## Required

* canonical ordering everywhere
* stable serialization
* deterministic reduction
* immutable state

---

# 3. Identity Law (v150.x)

Identity is the root of all system truth.

## Rules

* ALL identity-bearing tuples MUST pass `canonical_hash_identity`
* identity must be:

  * sorted
  * unique
  * canonical
* identity MUST NOT be inferred or reconstructed

Invalid identity = invalid system

---

# 4. Hashing & Proof Law

## Rules

* hashes computed over canonical JSON
* keys sorted
* compact separators
* UTF-8 encoding

## Critical Rule

* self-referential hash fields MUST be excluded

## Guarantee

* hashes MUST recompute exactly
* mismatch → `ValueError("INVALID_INPUT")`

---

# 5. Decoder Core Protection (SACRED)

Path:

```text
src/qec/decoder/
```

## Forbidden

* modifying BP logic
* scheduling changes
* adaptive logic
* supervisory leakage

---

# 6. Supervisory & Analysis Law

System flow:

```text
diagnostics
→ analysis
→ decision
→ verification
→ proof
```

---

# 7. Message & Protocol Law (v150.4)

## Rules

* message identity = `message_hash`
* payloads MUST be canonical
* duplicates forbidden
* ordering MUST be:

```text
(message_hash, sender_id, receiver_id)
```

---

# 8. Convergence Law (v150.5)

## Rules

* convergence MUST be finite
* convergence MUST be deterministic
* convergence MUST be provable
* flags MUST be derived

Non-convergence → `INVALID_INPUT`

---

# 9. Conflict Classification Law (v150.6)

## Rules

* classification MUST be deterministic
* classification MUST be symmetric
* payload comparison MUST be canonical

Allowed classes:

```text
IDENTICAL
EQUIVALENT
DOMINATED
INCONSISTENT
```

---

# 10. Receipt Integrity Law

Receipts are proof artifacts.

## Rules

* immutable
* canonical
* hash-verifiable
* self-hash excluded

Invalid receipt = invalid system

---

# 11. Safety Law

Safety overrides performance.

---

# 12. Minimal Diff Discipline

* smallest viable change
* single-purpose commits
* no refactor noise

---

# 13. Test Discipline

* deterministic replay tests
* invariant tests
* boundary tests
* hash stability tests

---

# 🔴 14. Validation Law (v150.7 — NEW)

Validation is **not manual**.

It is **conditionally mandatory**.

---

## 🔴 Validation Escalation Rule

You MUST run:

```bash
pytest -q
```

IF ANY of the following are touched:

### Identity Layer

* `canonical_hash_identity`
* identity-bearing tuples
* `input_memory_hashes`

### Hashing Layer

* canonical JSON
* hashing logic
* receipt hash computation

### Receipt Layer

* ANY receipt dataclass
* ANY proof artifact

### Multi-Agent System

* decisions
* protocol
* convergence
* conflict classification
* governance

### Ordering / Canonicalization

* sorting logic
* deduplication
* canonical transformations

---

## 🔴 Enforcement

If escalation is triggered:

* full suite MUST pass
* failures must be fixed if caused by change
* otherwise STOP

If full suite is not executed:

→ PATCH IS INVALID

---

## 🟡 Local Test Rule

If escalation NOT triggered:

* module-level tests allowed

BUT:

* escalation must be re-evaluated before commit

---

## 🧠 Principle

```text
validation is triggered by invariant impact
```

NOT by developer memory.

---

# 15. Dependency Law

* stdlib first
* pinned dependencies only
* minimal scope

---

# 16. Escalation Rule

If touching:

* identity
* hashing
* convergence
* protocol
* governance
* decoder

STOP and explain risk.

---

# Final Law

If it cannot be reproduced byte-for-byte,

it is not a valid result.
