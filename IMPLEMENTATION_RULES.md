# IMPLEMENTATION_RULES.md  
QSOL QEC Engineering Rules for AI-Assisted Development (v99–v100)

Author: Trent Slade — QSOL-IMC  
ORCID: 0009-0002-4515-9237  

---

## ⚠️ Purpose

This document defines **engineering discipline** for AI agents working in the QEC repository.

- `CLAUDE.md` → architectural law  
- `IMPLEMENTATION_RULES.md` → implementation behavior  

Both must be obeyed.

---

# 🧠 Core Principle

QEC is a **deterministic scientific instrument**.

Not a simulator.  
Not an optimizer.  
Not an ML system.

Code must prioritize:

- reproducibility  
- transparency  
- structural correctness  
- deterministic behavior  

---

# 📖 1. Read Before Writing

Before writing code, Claude must:

1. Read relevant modules  
2. Identify architectural layer  
3. Confirm constraints from `CLAUDE.md`  
4. Propose minimal change  

Claude must **never write code blindly**.

---

# ✂️ 2. Minimal Change Principle

All changes must be:

- local  
- surgical  
- minimal  

Rules:

- modify smallest number of lines possible  
- do not refactor unrelated code  
- do not rename identifiers unnecessarily  
- do not reformat files  
- do not restructure modules without instruction  

Large diffs = high risk.

---

# 🔁 3. Determinism is Mandatory

All algorithms must be deterministic.

Required:

- explicit seed control  
- stable iteration ordering  
- deterministic sorting  
- canonical serialization  

Forbidden:

- `random.random()`  
- unseeded `np.random`  
- unordered `set` iteration  
- implicit dict iteration ordering  

No hidden randomness. Ever.

---

# 🧠 4. Structured Memory Discipline (v99+)

QEC now includes **system memory**.

Memory must follow:

```python
Dict[(regime_key, strategy), List[event]]

Where:

event = {
    "step": int,
    "score": float,
    "metrics": Dict[str, float],
}

Rules:

memory must be deterministic
ordering must be stable
structure must not drift
keys must be hashable and reproducible

Forbidden:

untyped dict nesting
mixing schemas across modules
implicit structure changes

Memory is a first-class system component.

🧮 5. Sparse Linear Algebra First

All spectral operations must be sparse.

Forbidden:

dense NB matrices
O(|E|²) constructions

Required:

scipy.sparse
LinearOperator
scipy.sparse.linalg.eigs

Pattern:

eigs(LinearOperator(...), k=1, which="LR")

Memory must scale with |E|.

🧪 6. Analysis vs Control Separation

QEC has distinct roles:

Analysis (observational)
metrics
attractors
topology
spectral diagnostics
Control (external)
strategy selection
adaptation
scheduling

Rules:

analysis must not modify behavior
control must not modify decoder internals
🧱 7. Decoder Core Protection

Protected layer:

src/qec/decoder/

Forbidden:

modifying BP updates
changing message passing
injecting hooks
altering convergence logic

The decoder is:

a fixed experimental object

🔬 8. Experimental Modules Must Be External

All experiments must live in:

src/qec/analysis/
src/qec/experiments/

They must:

wrap the decoder
never modify it
remain opt-in
⚙️ 9. Input Modification Rules

Allowed:

LLR modification
schedule selection
graph structure (if valid)

Forbidden:

modifying BP internals
altering message updates
injecting state into decoder
🧠 10. Strategy & Adaptation Rules (v99+)

Strategy logic must be:

deterministic
memory-driven
regime-aware

Required:

explicit scoring functions
stable ranking
no stochastic exploration

Forbidden:

reinforcement learning
policy gradients
exploration noise

This is deterministic adaptation, not ML.

📊 11. Artifact Discipline

All outputs must be:

JSON-safe
canonicalized
deterministic

Artifacts must include:

config
parameters
results
memory snapshot (if applicable)

No non-deterministic metadata.

🧪 12. Testing Requirements

All new features must include:

determinism tests
regression tests
structural validation
edge case handling

Tests must verify:

correctness
reproducibility
stability
🧱 13. Graph Constraints

QLDPC constraints must always hold:

H_X H_Z^T = 0

Any graph modification must:

preserve row degree
preserve column degree
preserve commutativity

If uncertain → STOP.

🚫 14. Forbidden Techniques

Not allowed in QEC:

stochastic optimization
simulated annealing
reinforcement learning
neural networks
heuristic search without justification

Allowed:

spectral methods
linear algebra
deterministic graph algorithms
⚠️ 15. Error Handling

Errors must be explicit:

raise ValueError("Invalid configuration")

Forbidden:

silent correction
implicit fallback
hidden state repair
📚 16. Documentation Discipline

Every module must include:

purpose
inputs
outputs
algorithm explanation

Focus on why, not just what.

🔁 17. Commit Discipline

Each commit must:

implement one feature
include tests
preserve determinism
avoid unrelated edits
❓ 18. When Uncertain

Claude must:

stop
explain uncertainty
ask for clarification

Never guess.

## Theory-Coupled Release Rule (mandatory)

Every tagged release must integrate at least one explicit executable construct from the `/papers` theory corpus.

This requirement is mandatory for all releases from `v137.0.13` onward.

Acceptable integrations include:

- mathematical invariant
- operator rule
- field equation term
- symmetry prior
- qutrit / ternary state logic
- E8 structural mapping
- resonance constraint
- cosmological scaling law
- symbolic ontology mapping
- reproducible simulation path

The integration must be:

- executable
- testable
- deterministic
- documented in release notes

Each release note must include:

Theory Upgrade Source:
- file:
- concept:
- implementation:
- invariant tested:

Pure naming or thematic alignment does NOT satisfy this rule.

The release must absorb real theory semantics from the `/papers` corpus.

---

# 19. Release Hardening Scope Fence

## SCOPE FENCE (mandatory)

AI-assisted hardening passes are **strictly scoped** to the current release boundary.

### AI-assisted hardening passes may ONLY modify:

1. Files explicitly created or changed in the current release
2. Direct dependency files required for the current release to function
3. Tests directly targeting modules introduced or modified in the current release

### AI-assisted hardening passes MUST NOT modify:

* Historical tests unrelated to the current release
* Prior release modules
* Legacy scripts
* Unrelated CI or pytest failures
* Older bench, simulation, or experiment files
* Any file outside the current release's dependency cone

### Unrelated failure handling

If repository-wide pytest surfaces failures outside the current release scope:

* **Report** the failures explicitly
* **Do NOT repair** them during the hardening pass
* Repairs are permitted only if the failure is **directly caused** by the current release

### Dedicated follow-up requirement

Repository-wide pytest failures outside current release scope must be:

* Logged as a separate concern
* Resolved in a **dedicated follow-up pass**
* Never conflated with release hardening work

This fence is **durable engineering law**, not a temporary guideline.

Violation of scope fence constitutes an invalid hardening pass.

---

# 20. PDF / Papers Ingestion Rule

## Priority Order

When ingesting theory from the `/papers` corpus:

1. Prefer `ROADMAP.md` first for theory-coupled release context
2. Prefer existing Layer 4 modules
3. Prefer `papers/*.md` when available
4. Use `papers/*.pdf` only when explicitly required by instruction

## Rationale

* `pdftoppm` and other PDF rendering toolchains are **not guaranteed** in all environments
* Missing `pdftoppm` is an **environment-only limitation**, not a repository failure
* Repeated PDF render attempts against unavailable tooling waste ingestion passes
* Text-first ingestion is faster, deterministic, and sufficient for theory coupling

## Rules

* Do not attempt PDF rendering unless explicitly instructed
* Do not treat PDF render failure as a repository defect
* Do not retry failed PDF toolchain paths
* Always check for a `.md` equivalent before attempting `.pdf`

---

🧠 Final Principle

QEC evolves by:

structure → stability → capability

Not:

feature → complexity → chaos
🧠 Closing

If a result cannot be reproduced byte-for-byte:

it is not part of the system

If a change weakens structure:

it must not be merged

Small is beautiful.
Determinism is holy.
Stability is engineered.
