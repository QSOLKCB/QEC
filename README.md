# QEC v2.0 — Ququart + Qutrit Golay + Quantum LDPC + High-Density Geometry Layer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17742258.svg)](https://doi.org/10.5281/zenodo.17742258)
![GitHub release (latest by tag)](https://img.shields.io/github/v/release/QSOLKCB/QEC?label=release)

---

# QEC v2.0 — Multidimensional Stabilizer Stack + Golay-Class Logic + Quantum LDPC

This release extends QEC beyond ququart stabilizers into **ternary Golay-class quantum logic** and **protograph-based quantum LDPC codes near the hashing bound**, enabling direct experimentation with **qutrit-perfect codes**, **high-rate CSS codes**, and the existing **ℤ₄ ququart + lattice geometry** framework.

---

## 🧬 What's New in v2.0

---

### 🔷 Protograph-Based Quantum LDPC Codes (Komoto–Kasai 2025)

**New Module:**

* `src/qec_qldpc_codes.py`

Implementation of **CSS quantum LDPC codes over GF(2ᵉ)** from:

> D. Komoto & K. Kasai, "Quantum Error Correction near the Coding Theoretical Bound,"
> *npj Quantum Information* **11**, 154 (2025). [arXiv:2412.21171](https://arxiv.org/abs/2412.21171)

Key features:

* **Protograph-based construction**: J×L template base graph lifted with P×P circulant permutations
* **Finite field extension over GF(2ᵉ)**: field elements replaced by e×e companion matrices for binary expansion
* **CSS orthogonality by construction**: self-orthogonal paired-column design in characteristic 2 — no iterative patching
* **Shared circulant lifts**: same permutation π_j for both H_X and H_Z at each column, so (C(a)⊗π)(C(b)⊗π)ᵀ = C(abᵀ)⊗I
* **Joint X/Z sum-product decoder** (belief propagation) for depolarizing channel
* **Hard invariant**: `ConstructionInvariantError` raised if H_X · H_Zᵀ ≠ 0 mod 2 — construction never silently fails

---

### 📈 Code Rates & Hashing Bound

Code rates follow **R = 1 − 2J/L** with predefined configurations:

| Rate | J | L  | Description              |
|------|---|----|--------------------------|
| 0.50 | 1 | 4  | Half-rate baseline       |
| 0.60 | 2 | 10 | Mid-rate code            |
| 0.75 | 2 | 16 | High-rate near capacity  |

**Hashing bound** for the depolarizing channel:

```
R_hash(p) = 1 + (1−p)·log₂(1−p) + p·log₂(p/3)
```

The paper achieves FER 10⁻⁴ at p_phys = 9.45% with 104K logical / 312K physical qubits using e=8, P=8192.

---

### 🧮 GF(2ᵉ) Arithmetic Engine

```python
from src.qec_qldpc_codes import GF2e

gf = GF2e(e=3)           # GF(8), primitive poly x³+x+1
a, b = 5, 3
print(gf.mul(a, b))      # Field multiplication
print(gf.companion_matrix(a))  # 3×3 binary companion matrix
```

* Full arithmetic: add (XOR), multiply (via log/exp tables), inverse
* Companion matrix homomorphism: C(a+b) = C(a)+C(b), C(a·b) = C(a)@C(b) mod 2
* Supports any extension degree e ≥ 2

---

### 🛠️ Quick Start — QLDPC Codes

```python
from src.qec_qldpc_codes import create_code, simulate_frame_error_rate, hashing_bound

# Create a rate-0.50 code with lifting parameter P=32
code = create_code(rate=0.50, P=32, e=3)
print(f"Physical qubits: {code.n}")
print(f"Logical qubits:  {code.k}")
print(f"Code rate:        {code.rate:.3f}")

# Simulate frame error rate
fer = simulate_frame_error_rate(code, p_phys=0.01, num_trials=1000)
print(f"FER at p=0.01:    {fer:.4f}")

# Hashing bound at this noise level
print(f"Hashing bound:    {hashing_bound(0.01):.4f}")
```

---

### 🟣 Ternary Golay Qutrit Code ([[11,1,5]]₃)

**New Module:**

* `src/qec_golay.py`

This release adds a full implementation of the **ternary Golay code**, the unique perfect linear code over **GF(3)**:

* Classical parameters: **[11, 6, 5]₃**
* Quantum CSS lift: **[[11,1,5]]₃**
* Corrects **any single-qutrit error**
* Protects **one logical qutrit inside eleven physical qutrits**

---

### 📐 Parity-Check Matrix (GF(3))

Used for both X- and Z-type stabilizers:

```
H = [
 [1 0 0 0 0 1 1 1 2 2 0]
 [0 1 0 0 0 1 1 2 1 0 2]
 [0 0 1 0 0 1 2 1 0 1 2]
 [0 0 0 1 0 1 2 0 1 2 1]
 [0 0 0 0 1 1 0 2 2 1 1]
]
```

* Self-orthogonal over **GF(3)**
* Nullspace generates **729 exact codewords**
* CSS-compatible for qutrit stabilizers

---

### 🧮 Generator Matrix

Automatically computed from the nullspace of **H**, producing:

* **6 independent generators**
* Full **dimension-729 logical subspace**
* Deterministic encoding via:

```python
encode_message(m)
```

---

### 🧠 Quantum Role

This Golay layer enables:

* **Perfect qutrit error correction**
* **Magic-state distillation pipelines**
* **Ternary stabilizer benchmarking**
* **Direct comparison: binary (d=2), ququart (d=4), and qutrit (d=3)**

---

## 🟦 Ququart Stabilizer Code (d = 4)

Unchanged from prior releases:

**File:**
`src/qec_ququart.py`

**Codewords:**

```
|jₗ⟩ = |j, j, j⟩   for j ∈ {0,1,2,3}
```

**Stabilizers:**

```
S₁ = Z₁ · Z₂⁻¹
S₂ = Z₂ · Z₃⁻¹
```

**Logical Operators:**

```
Xₗ = X₁ · X₂ · X₃
Zₗ = Z₁
```

---

## 🧊 High-Density Geometry Layer (D₄)

**File:**
`src/ququart_lattice_prior.py`

Projects logical amplitudes into:

* **ℤ⁴** → baseline cubic
* **D₄** → dense E8-surrogate lattice

Acts as a **geometric pre-decoder** that:

* Compresses noise
* Sharpens amplitudes
* Raises effective threshold
* Produces lattice-stabilized logical states

---

## 📊 Threshold Benchmarks

* `ququart_threshold.png`
* `ququart_lattice_prior_threshold.png`

**Result:**
D₄ prior strictly reduces logical error rates across all tested pₚₕᵧₛ.

---

## 🎧 Sonic / QEC Cross-Mapping

| Regime       | Physical Error  | Sonic State        |
| ------------ | --------------- | ------------------ |
| Stable       | < 1×10⁻⁵        | Clean, narrow-band |
| Transitional | 1×10⁻⁵ → 1×10⁻³ | Spectral pressure  |
| Critical     | > 1×10⁻³        | Saturated collapse |

Ternary Golay introduces **triplet-locked harmonic fields** distinct from ququart D₄ geometry.

---

## ⚙️ Simulation Engine

### Core Stack

* `src/steane_numpy_fast.py`
* `src/qec_ququart.py`
* `src/qudit_stabilizer.py`
* `src/ququart_lattice_prior.py`
* `src/qec_golay.py`
* `src/qec_qldpc_codes.py` **(NEW — Quantum LDPC)**

### Example Scripts

* `examples/ququart_threshold_demo.py`
* `examples/ququart_threshold_with_prior.py`
* `examples/qldpc_hashing_bound_demo.py` **(NEW — QLDPC hashing bound & FER simulation)**

---

## 🧪 Test Suite

```bash
pytest tests/ -v
```

**97 tests** covering:

* Steane [[7,1,3]] code (32 tests)
* Quantum LDPC codes (65 tests):
  * GF(2ᵉ) arithmetic & companion matrices
  * Protograph pair orthogonality (parametrized across J, L)
  * CSS invariants (determinism, no-zero rows/cols, H_X·H_Zᵀ=0)
  * Joint X/Z sum-product decoder
  * Depolarizing channel statistics
  * Hashing bound invariants & monotonicity
  * Frame error rate simulation trends
  * All predefined code configurations

---

## 📚 References

* Komoto, D. & Kasai, K. "Quantum Error Correction near the Coding Theoretical Bound." *npj Quantum Information* **11**, 154 (2025). [doi:10.1038/s41534-025-01090-1](https://doi.org/10.1038/s41534-025-01090-1)
* Calderbank, A. R. & Shor, P. W. "Good quantum error-correcting codes exist." *Phys. Rev. A* **54**, 1098 (1996).
* Steane, A. M. "Error Correcting Codes in Quantum Theory." *Phys. Rev. Lett.* **77**, 793 (1996).

---

## 🧾 License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

---

## 🔖 Citation (Updated)

```bibtex
@software{slade_2025_qsolkcb,
  author       = {Slade, T.},
  title        = {QSOLKCB/QEC: QEC v2.0 — Ququart + Qutrit Golay + Quantum LDPC + Geometry Layer},
  year         = {2025},
  version      = {v2.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17742258},
  url          = {https://doi.org/10.5281/zenodo.17742258}
}
```

---

## 🏷️ Keywords (Expanded)

quantum error correction · qutrit · ququart · Golay code · ternary stabilizer · qudit stabilizer · D4 lattice · quantum LDPC · protograph codes · CSS codes · GF(2^e) · hashing bound · sum-product decoder · belief propagation · spectral algebraics · sonification · QSOL-IMC · E8-inspired · threshold physics

---
