Beautiful — here’s a **drop-in replacement README section** that cleanly extends your current v1.5 document to include the new **qec_golay.py ternary Golay / qutrit layer**, without breaking your existing ququart + geometry narrative.

You can paste this directly over your current README, or splice just the new Golay blocks if you prefer.

---

# QEC v1.6 — Ququart + Qutrit Golay + High-Density Geometry Layer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17742258.svg)](https://doi.org/10.5281/zenodo.17742258)
![GitHub release (latest by tag)](https://img.shields.io/github/v/release/QSOLKCB/QEC?label=release)

---

# QEC v1.6 — Multidimensional Stabilizer Stack + Golay-Class Logic

This release extends QEC beyond ququart stabilizers into **ternary Golay-class quantum logic**, enabling direct experimentation with **qutrit-perfect codes** alongside the existing **ℤ₄ ququart + lattice geometry** framework.

---

## 🧬 What’s New in v1.6

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

Unchanged from v1.5:

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
* ✅ `src/qec_golay.py`  **(NEW)**

### Example Scripts

* `examples/ququart_threshold_demo.py`
* `examples/ququart_threshold_with_prior.py`

---

## 🧾 License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

---

## 🔖 Citation (Updated)

```bibtex
@software{slade_2025_qsolkcb,
  author       = {Slade, T.},
  title        = {QSOLKCB/QEC: QEC v1.6 — Ququart + Qutrit Golay + Geometry Layer},
  year         = {2025},
  version      = {v1.6-golay-qutrit},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17742258},
  url          = {https://doi.org/10.5281/zenodo.17742258}
}
```

---

## 🏷️ Keywords (Expanded)

quantum error correction · qutrit · ququart · Golay code · ternary stabilizer · qudit stabilizer · D4 lattice · spectral algebraics · sonification · QSOL-IMC · E8-inspired · threshold physics

---
