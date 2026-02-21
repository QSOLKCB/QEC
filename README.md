# QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![v2.5.0](https://img.shields.io/badge/version-v2.5.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v2.5.0)
&nbsp;&nbsp;
[![v2.4.0](https://img.shields.io/badge/version-v2.4.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v2.4.0)
&nbsp;&nbsp;
![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)

Deterministic quantum error correction framework for QLDPC CSS codes with algebraic construction guarantees, numerically stable belief propagation, statistically rigorous FER simulation, and modular decoder utilities.

---

# Release Lineage

---

## v2.5.0 — Deterministic Statistical Rigor + Layered Decoding

### Highlights

Wilson score confidence intervals for FER simulation  
- Continuity-corrected Wilson interval (`ci_method="wilson"`)  
- Configurable `alpha`  
- `gamma >= 0.0` (set `gamma=0` to disable correction)  
- Integer-grounded computation (no float reconstruction)  

Deterministic early termination  
- `early_stop_epsilon` stops trials once CI width is sufficiently tight  
- Fully reproducible termination point  

Layered (serial) BP scheduling  
- `schedule="layered"`  
- Incremental LLR update invariant  
- O(nnz(H)) per iteration  
- Deterministic check-node traversal  

OSD-1 post-processing  
- `postprocess="osd1"`  
- Single least-reliable pivot flip  
- Deterministic tie-breaking  
- Never-degrade guarantee preserved  

All new features are opt-in.  
Default parameters remain bit-identical to v2.4.0.

247 / 247 core tests passing.

---

## v2.4.0 — Performance Hardening + Deterministic FER Harness

### Highlights

Multi-mode BP decoder  
- `sum_product`
- `min_sum`
- `norm_min_sum`
- `offset_min_sum`

Message damping and magnitude clipping

OSD-0 post-processing for BP failure recovery

Dict-based asymmetric channel bias in `channel_llr`

Deterministic Monte Carlo FER simulation harness

Optional FER plotting utility

Backward-compatible API (`max_iter` alias preserved)

168 / 168 tests passing

Construction layer remains algebraically guaranteed (v2.1.0).

---

## v2.3.0 — Decoder Utility Formalization and Stability Refinement

This release formalized the decoder layer into standalone utilities while preserving full backward compatibility with JointSPDecoder.

### Highlights

Explicit detection → inference → correction separation

Standalone `bp_decode` with per-variable LLR input

Pauli-frame update abstraction (`update_pauli_frame`)

Channel LLR modeling with optional scalar or vector bias

Enforced input validation (`p ∈ (0, 1)`)

Reduced per-iteration overhead in BP early-stop logic

101 / 101 tests passing

Construction layer remains algebraically guaranteed (v2.1.0).

---

## v2.2.0 — Belief Propagation Stability Hardening

Numerical stability refinement of the sum-product decoder.

### Highlights

Correct handling of degree-1 check nodes

Eliminated artificial LLR amplification from `atanh(≈1)`

Removed false confidence injection in sparse Tanner graphs

Stabilized BP behavior under irregular parity structures

No architectural changes. Decoder logic stabilization only.

---

## v2.1.0 — Additive Lift Invariant Hardening

Transition from empirically stable lifting to algebraically guaranteed construction.

Additive lift structure:


s(i, j) = (r_i + c_j) mod L


### Highlights

Algebraic guarantee of lifted CSS orthogonality

Sparse-safe GF(2) rank computation

Deterministic seeded construction

89 / 89 invariant tests passing

Construction layer transitioned from probabilistic behavior → structural invariance.

---

## v2.0.0 — Architectural Expansion

Initial multidimensional QLDPC CSS stack:

- Protograph-based construction  
- GF(2^e) lifting  
- Ternary Golay [[11,1,5]]₃  
- Ququart stabilizer + D4 lattice prior  
- Deterministic construction framework  

---

# Current System State (v2.5.0)

Construction layer is algebraically enforced.

Decoder layer supports:
- Multi-mode BP
- Layered scheduling
- OSD-0 and OSD-1 post-processing

Detection, inference, and correction are modular and independently test-covered.

Deterministic FER simulation harness with:
- Wilson confidence intervals
- Early termination
- Seeded RNG workflow

Fully deterministic execution path.

247 total core tests passing.

---

# Architecture Overview

Channel Model      → `channel_llr`  
Detection          → `syndrome` / `detect`  
Inference          → `bp_decode` / `infer`  
Correction         → `update_pauli_frame`  
Construction Layer → Additive invariant QLDPC CSS lift  

---

The framework cleanly separates algebraic construction guarantees from numerically stable belief-propagation decoding, enabling deterministic, test-covered workflows from channel modeling through Pauli-frame correction and statistically rigorous simulation.
