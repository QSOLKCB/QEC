#!/usr/bin/env python3
"""
Quantum LDPC Codes Near the Hashing Bound — Demo
=================================================

Demonstrates protograph-based CSS quantum LDPC codes following:
    Komoto & Kasai, "Quantum Error Correction near the Coding
    Theoretical Bound", npj Quantum Information 11, 154 (2025).

This script:
  1. Shows the hashing bound curve for the depolarizing channel
  2. Constructs codes at rates R = 0.50, 0.60, 0.75
  3. Verifies CSS orthogonality
  4. Runs a small Monte Carlo FER simulation
  5. Compares measured FER to hashing bound predictions
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.qec_qldpc_codes import (
    GF2e,
    QuantumLDPCCode,
    JointSPDecoder,
    create_code,
    hashing_bound,
    hashing_bound_threshold,
    simulate_frame_error_rate,
    PREDEFINED_CODES,
)


def main():
    print("=" * 70)
    print("  Quantum LDPC Codes Near the Hashing Bound")
    print("  Komoto & Kasai, npj Quantum Information 11, 154 (2025)")
    print("=" * 70)

    # ── 1. Hashing Bound Table ──────────────────────────────────────
    print("\n[1] Hashing bound for the depolarizing channel\n")
    print(f"  {'p_phys':>8s}  {'R_hash':>8s}")
    print(f"  {'─'*8}  {'─'*8}")
    for p in [0.001, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.189]:
        print(f"  {p:8.3f}  {hashing_bound(p):8.4f}")

    print("\n  Hashing-bound thresholds for target rates:")
    for name in sorted(PREDEFINED_CODES):
        rate = float(name.split('_')[1])
        p_th = hashing_bound_threshold(rate)
        print(f"    R = {rate:.2f}  →  p_max = {p_th:.4f}  ({p_th*100:.2f}%)")

    # ── 2. Finite Field Companion Matrices ──────────────────────────
    print("\n[2] GF(2^3) = GF(8) companion matrices\n")
    gf = GF2e(3)
    for elem in [1, 2, 3]:
        C = gf.companion_matrix(elem)
        print(f"  C({elem}) =")
        for row in C:
            print(f"    {list(row)}")
        print()

    # ── 3. Construct Codes ──────────────────────────────────────────
    print("[3] Constructing predefined codes (P=16)\n")
    codes = {}
    for name in sorted(PREDEFINED_CODES):
        code = create_code(name, lifting_size=16, seed=42)
        codes[name] = code
        print(f"  {name}: {code}")
        print(f"    CSS orthogonal: {code.verify_css_orthogonality()}")

    # ── 4. Monte Carlo Simulation ───────────────────────────────────
    print("\n[4] Monte Carlo FER simulation (rate-1/2 code, P=16)\n")
    code = codes['rate_0.50']
    decoder = JointSPDecoder(code, max_iter=50)

    print(f"  Code: n={code.n}, k={code.k}, rate={code.rate:.4f}")
    p_th = hashing_bound_threshold(0.50)
    print(f"  Hashing bound threshold for R=0.50: p = {p_th:.4f}\n")

    print(f"  {'p_phys':>8s}  {'FER':>8s}  {'FER_X':>8s}  {'FER_Z':>8s}  {'DecFail':>8s}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for p in [0.005, 0.01, 0.02, 0.03, 0.05]:
        res = simulate_frame_error_rate(
            code, decoder, p, n_frames=50, seed=42
        )
        print(
            f"  {p:8.3f}  {res['fer']:8.4f}  "
            f"{res['fer_x']:8.4f}  {res['fer_z']:8.4f}  "
            f"{res['decode_failure_rate']:8.4f}"
        )

    # ── 5. Scaling with lifting size ────────────────────────────────
    print("\n[5] Code size scaling with lifting parameter P\n")
    print(f"  {'P':>6s}  {'n (phys)':>10s}  {'k (log)':>10s}  {'rate':>8s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}")
    for P in [8, 16, 32, 64]:
        c = create_code('rate_0.50', lifting_size=P, seed=42)
        print(f"  {P:6d}  {c.n:10d}  {c.k:10d}  {c.rate:8.4f}")

    print("\n" + "=" * 70)
    print("  At large P, these codes approach the hashing bound with")
    print("  FER ~ 10^{-4} near p_phys = 9.45% (Komoto-Kasai 2025).")
    print("=" * 70)


if __name__ == '__main__':
    main()
