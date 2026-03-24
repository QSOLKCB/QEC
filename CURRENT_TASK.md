# CURRENT_TASK.md  
Active Development Target — v99.1.x  
Deterministic Strategy Memory Integration & Hardening

---

## ⚠️ Scope

This document defines the **current implementation task**.

Only implement features required for:

> **v99.1.x — Strategy Memory Stabilization**

Do NOT implement roadmap features beyond this scope.

---

# 🧠 Goal

Stabilize and harden the **deterministic adaptive control loop** by introducing:

> **bounded, per-strategy memory with deterministic biasing**

This completes the transition from:


stateless strategy selection
→
experience-aware deterministic system


---

# 🔬 System Context

The system now operates as:


metrics → attractor → strategy → evaluation → adaptation → memory


v99.1 introduces:

- local specialization per strategy  
- performance-informed decision bias  
- deterministic “learning” without randomness  

---

# 🎯 Core Objective

Enable **memory-aware strategy selection** while preserving:

- determinism  
- architectural separation  
- zero regression in baseline behavior  

---

# 🧩 Features To Implement / Harden

---

## 1. Strategy Memory Structure

Canonical structure:

```python
Dict[str, List[Dict[str, Any]]]

Where:

key = strategy_id
value = bounded list of recent performance records
Requirements
deterministic append-only updates
enforce fixed capacity (default: 10)
no in-place mutation
stable ordering (oldest → newest)
2. Performance Scoring

Compute per-strategy performance:

performance =
  0.6 * avg_score
+ 0.3 * improvement_rate
- 0.1 * instability_penalty

Constraints:

clamp to [-1, 1]
deterministic aggregation
no floating drift (stable ordering + rounding if needed)
3. Bias Computation

Convert performance → selection bias:

bias = 0.2 * performance

Clamp:

[-0.2, +0.2]

Two components:

local bias (per strategy)
global bias (optional aggregate signal)
4. Memory-Aware Strategy Scoring

Final scoring:

final_score =
  base_score
+ global_bias
+ local_bias

Constraints:

clamp to [0, 1]
preserve deterministic ranking

Tie-breaking (strict):

score
confidence
simplicity
lexicographic id
5. Deterministic Selection Integration

Update:

select_next_strategy(...)

Behavior:

if memory is provided → use memory-aware scoring
otherwise → fallback to baseline logic
6. Metrics Integration

Update evaluation loop:

maintain strategy_memory across iterations
update after each evaluation
expose bias in logs:
MEM bias=+X.XX (local) +X.XX (global)
7. Type Safety (Hard Requirement)

Enforce consistent typing across system:

Dict[str, List[Dict[str, Any]]]

Applies to:

strategy_memory
function signatures
selection interfaces

No Any-typed containers for memory.

8. Determinism Guarantees

Must preserve:

identical outputs across runs
stable ordering
no hidden randomness
no mutation side effects
🧪 Tests

All functionality must be covered by deterministic tests:

Required coverage
memory append + cap enforcement
no-mutation guarantees
performance scoring correctness
bias clamping
selection determinism
integration with selector
📦 Output Guarantees
no schema changes
no artifact drift
baseline behavior unchanged when memory disabled
🚫 Explicit Non-Goals

Do NOT implement:

multi-step planning
predictive strategy selection
long-horizon memory
decay functions
reinforcement learning
stochastic adaptation

These belong to v100+

🧱 Architectural Constraints

Must obey:

CLAUDE.md
ROADMAP.md
layering rules
decoder protection

Forbidden:

modifying decoder core
introducing randomness
cross-layer leakage
✅ Success Condition

v99.1.x is complete when:

strategy memory is deterministic and bounded
performance scoring is correct and stable
bias influences selection deterministically
selector integrates memory without regressions
all tests pass
outputs are byte-identical across runs
🔜 Next Step (Do Not Implement Yet)

v100 — Strategy Ecology

strategy specialization patterns
cross-regime dominance
behavioral taxonomy
🧠 Final Principle

This system must become:

adaptive without randomness

If it learns, it must do so deterministically.

If it changes behavior, it must be explainable.

If it cannot be reproduced, it is not valid.
