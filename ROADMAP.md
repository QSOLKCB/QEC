# QSOLKCB / QEC — ROADMAP.md
## Post-v146.1

---

## Core Law (Invariant)

```text
same input
→ same ordering
→ same canonical JSON
→ same stable hash
→ same bytes

All modules MUST:

be deterministic
be replay-safe
produce bounded outputs
fail fast on invalid input
use canonical JSON (sorted, compact)
use stable SHA-256 hashing
exclude randomness, wall-clock, async
preserve decoder immutability
remain analysis-layer only

Violation → module invalid.

State
v143.5 → proof
v144.x → structure
v145.x → governance
v146.0 → action representation
v146.1 → integrity + replay closure

System is:

deterministic proof system
+ invariant analysis engine
+ governance substrate
+ proof-carrying action layer

Architecture is complete.

Direction

NOT:

build more architecture

YES:

prove under constraint
→ quantify under scarcity
→ test under adversary
→ validate across platforms
→ extract measurable reductions
ROADMAP SPINE
v146.x → complete (representation + integrity)

v147.x → constraint proving ground

v148.x → promotion + publication
v147.x — CONSTRAINT PROVING GROUND

Goal:

invalidate weak invariants
measure redundancy
test action capsules under constraint
produce canonical receipts

All claims MUST produce receipts.

v147.0 — Retro Target Registry
target spec → canonical receipt

Defines constraint space.

Module
retro_target_registry.py

Output
RetroTargetReceipt

v147.1 — Trace Intake
raw trace + target → canonical trace

Unifies input surface.

Module
retro_trace_intake_bridge.py

Output
RetroTraceReceipt

v147.2 — Constraint Profile
target + trace → pressure profile

Defines scarcity.

Module
retro_constraint_profile_kernel.py

Output
RetroConstraintProfileReceipt

v147.3 — Arithmetic Scarcity (MANDATORY)
full precision → reduced arithmetic → invariant test

Filters valid invariants.

Output
ArithmeticScarcityReceipt

Reject:

invalid_under_budget
instability_exposed
v147.4 — Float Audit
multi-regime comparison → equivalence map

Separates structural vs precision-bound invariants.

v147.5 — Projection (MANDATORY)
state → minimal retro representation

Forces structure into constraint.

Output
RetroProjectionReceipt

v147.6 — Action Capsule Audit
capsules + constraint → stability check

Tests v146 under pressure.

v147.7 — Redundancy Collapse (MANDATORY)
system trace → invariant core → compute reduction

Produces measurable:

compute_waste_eliminated
invariant_retention
v147.8 — Cross-Emulator Determinism (MANDATORY)
same workload → multiple environments → hash equality

Failure → recorded, not absorbed.

v147.9 — Offline Policy Audit
fixed corpus → deterministic policy evaluation

No stochastic environments.

v147.10 — Operationalization (MANDATORY)

Apply QEC to:

transformers
diffusion
decoding
scheduling

Output:

RedundancyReceipt

Measure:

compute_eliminated
v147.11 — Adversarial Battery (MANDATORY)
valid input → adversarial perturbation → invariant test

Must:

reject nondeterministic inputs
preserve hash stability
terminate within bounds
v147.12 — Failure Ledger (MANDATORY)
result → classified failure or success

Rules:

suppression_rate = 0
all failures typed
v147.13 — Evaluation Pack
all receipts → canonical bundle

Output:

reproducibility artifacts
benchmark tables
evidence bundle
Research Questions (Encoded)
Which invariants survive constraint?
Are capsules stable under reduction?
Are results platform-invariant?
Is redundancy measurable?
Where does it break?
v148.x — PROMOTION (CONDITIONAL)

Requires:

deterministic integrity preserved
failures bounded + classified
measurable results produced
cross-target consensus observed

Else: STOP.

v148.0 — Promotion Gate
evaluation pack → admissibility check
v148.1 — Experiment Capsules
experiment → proof-carrying artifact
v148.2 — Cross-Target Consensus
multi-target invariants → consensus map
v148.3 — Hardening Loop
failure → mitigation → re-test
v148.4 — Operational Publication
receipts → benchmark report
v148.5 — Final Release
evaluation + consensus → publication
v148.6 — Demonstration (Optional)
receipt → deterministic visualization

No execution.

Guardrails
Forbidden
randomness
wall-clock dependence
async
decoder modification
probabilistic routing
silent failure
non-canonical serialization
Required
explicit quantization
explicit tie-breaking
canonical JSON
stable hashing
reproducible traces
typed failure outputs
Final Direction
v143–146 → build + validate

v147 → stress + prove

v148 → certify + publish
Law (unchanged)
same input
→ same ordering
→ same canonical JSON
→ same stable hash
→ same bytes

---
