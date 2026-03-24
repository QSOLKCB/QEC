#!/usr/bin/env python
"""
v3.9.0 Release Verification & DPS Experiment Runner.

Performs all 8 verification steps required before tagging v3.9.0.
"""

from __future__ import annotations

import json
import sys
import hashlib
from typing import Any

import numpy as np

from bench_reports.dps_v381_eval import (
    run_mode, run_evaluation, run_determinism_check,
    compute_dps_slope, MODES, MODE_ORDER,
    DEFAULT_SEED, DEFAULT_DISTANCES, DEFAULT_P_VALUES,
    DEFAULT_TRIALS, DEFAULT_MAX_ITERS, DEFAULT_BP_MODE,
    _pre_generate_instances,
)
from qec_qldpc_codes import (
    bp_decode, syndrome, channel_llr, create_code,
)
from qec.decoder.rpc import StructuralConfig


# ──────────────────────────────────────────────────────────────
# Step 1 — Baseline Invariant Verification
# ──────────────────────────────────────────────────────────────

def step1_baseline_invariant():
    print("=" * 72)
    print("STEP 1 — BASELINE INVARIANT VERIFICATION")
    print("=" * 72)

    seed = DEFAULT_SEED
    distances = DEFAULT_DISTANCES
    p_values = DEFAULT_P_VALUES
    trials = DEFAULT_TRIALS

    all_match = True

    for run_id in range(2):
        rng = np.random.default_rng(seed)
        results_run = {}

        for distance in distances:
            H = create_code(name="rate_0.50", lifting_size=distance, seed=seed).H_X
            for p in p_values:
                instances = _pre_generate_instances(H, p, trials, rng)
                r = run_mode("baseline", H, instances,
                             max_iters=DEFAULT_MAX_ITERS, bp_mode=DEFAULT_BP_MODE)
                results_run[(distance, p)] = r

        if run_id == 0:
            run1 = results_run
        else:
            run2 = results_run

    for key in run1:
        r1 = run1[key]
        r2 = run2[key]
        fer_match = r1["fer"] == r2["fer"]
        frame_match = r1["frame_errors"] == r2["frame_errors"]
        audit_match = r1["audit_summary"] == r2["audit_summary"]
        if not (fer_match and frame_match and audit_match):
            print(f"  MISMATCH at d={key[0]}, p={key[1]}")
            print(f"    FER: {r1['fer']} vs {r2['fer']}")
            print(f"    frame_errors: {r1['frame_errors']} vs {r2['frame_errors']}")
            all_match = False
        else:
            print(f"  d={key[0]}, p={key[1]}: FER={r1['fer']:.4f}, "
                  f"frame_errors={r1['frame_errors']}, "
                  f"mean_iters={r1['audit_summary']['mean_iters']:.2f} — MATCH")

    # Compute DPS slopes for both runs
    for p in p_values:
        fer1 = {d: run1[(d, p)]["fer"] for d in distances}
        fer2 = {d: run2[(d, p)]["fer"] for d in distances}
        slope1 = compute_dps_slope(fer1)
        slope2 = compute_dps_slope(fer2)
        if slope1 != slope2:
            print(f"  DPS slope MISMATCH at p={p}: {slope1} vs {slope2}")
            all_match = False
        else:
            print(f"  DPS slope at p={p}: {slope1:.6f} — MATCH")

    status = "PASS" if all_match else "FAIL"
    print(f"\nBASELINE INVARIANT: {status}")
    print()
    return all_match


# ──────────────────────────────────────────────────────────────
# Step 2 — Determinism Verification
# ──────────────────────────────────────────────────────────────

def step2_determinism():
    print("=" * 72)
    print("STEP 2 — DETERMINISM VERIFICATION")
    print("=" * 72)

    seed = DEFAULT_SEED

    # Run full evaluation twice
    eval1 = run_evaluation(seed=seed, enable_energy_trace=True)
    eval2 = run_evaluation(seed=seed, enable_energy_trace=True)

    all_match = True

    for mode_name in MODE_ORDER:
        for p in DEFAULT_P_VALUES:
            for d in DEFAULT_DISTANCES:
                r1 = eval1["results"][mode_name][p][d]
                r2 = eval2["results"][mode_name][p][d]

                fer_ok = r1["fer"] == r2["fer"]
                frame_ok = r1["frame_errors"] == r2["frame_errors"]
                audit_ok = r1["audit_summary"] == r2["audit_summary"]

                # Energy trace comparison
                et1 = r1.get("energy_traces", [])
                et2 = r2.get("energy_traces", [])
                energy_ok = len(et1) == len(et2)
                if energy_ok and et1:
                    for t1, t2 in zip(et1, et2):
                        if t1 != t2:
                            energy_ok = False
                            break

                if not (fer_ok and frame_ok and audit_ok and energy_ok):
                    print(f"  MISMATCH: {mode_name} | p={p} | d={d}")
                    if not fer_ok:
                        print(f"    FER: {r1['fer']} vs {r2['fer']}")
                    if not energy_ok:
                        print(f"    Energy trace mismatch")
                    all_match = False

    # Also check slopes
    for mode_name in MODE_ORDER:
        for p in DEFAULT_P_VALUES:
            s1 = eval1["slopes"][mode_name][p]
            s2 = eval2["slopes"][mode_name][p]
            if s1 != s2:
                print(f"  SLOPE MISMATCH: {mode_name} | p={p}: {s1} vs {s2}")
                all_match = False

    status = "PASS" if all_match else "FAIL"
    print(f"\nDETERMINISM CHECK: {status}")
    print()
    return all_match, eval1


# ──────────────────────────────────────────────────────────────
# Step 3 — DPS Evaluation Sweep
# ──────────────────────────────────────────────────────────────

def step3_dps_sweep(eval_result):
    print("=" * 72)
    print("STEP 3 — DPS EVALUATION SWEEP")
    print("=" * 72)

    results = eval_result["results"]
    slopes = eval_result["slopes"]
    config = eval_result["config"]

    print(f"\nConfig: seed={config['seed']}, distances={config['distances']}, "
          f"p_values={config['p_values']}, trials={config['trials']}")
    print()

    # FER table
    print(f"{'Mode':<22} {'p':>6} {'d':>4} {'FER':>8} {'Errors':>7} {'AvgIter':>8}")
    print("-" * 63)
    for mode_name in MODE_ORDER:
        for p in config["p_values"]:
            for d in config["distances"]:
                r = results[mode_name][p][d]
                print(f"{mode_name:<22} {p:>6.3f} {d:>4d} {r['fer']:>8.4f} "
                      f"{r['frame_errors']:>7d} "
                      f"{r['audit_summary']['mean_iters']:>8.2f}")

    # DPS slopes
    print()
    p_headers = [f"p={p}" for p in config["p_values"]]
    header = f"{'Mode':<22}" + "".join(f"{h:>12}" for h in p_headers)
    print(header)
    print("-" * len(header))
    for mode_name in MODE_ORDER:
        row = f"{mode_name:<22}"
        for p in config["p_values"]:
            row += f"{slopes[mode_name][p]:>12.6f}"
        print(row)

    print()
    return results, slopes


# ──────────────────────────────────────────────────────────────
# Step 4 — Energy Trace Diagnostics
# ──────────────────────────────────────────────────────────────

def step4_energy_diagnostics(eval_result):
    print("=" * 72)
    print("STEP 4 — ENERGY TRACE DIAGNOSTICS")
    print("=" * 72)

    results = eval_result["results"]
    config = eval_result["config"]

    # Collect per-mode energy stats (aggregate over all (d,p) trials)
    energy_stats = {}

    for mode_name in MODE_ORDER:
        starts = []
        ends = []
        drops = []
        monotonic_count = 0
        oscillation_counts = []
        total_traces = 0

        for p in config["p_values"]:
            for d in config["distances"]:
                r = results[mode_name][p][d]
                traces = r.get("energy_traces", [])
                for trace in traces:
                    if not trace:
                        continue
                    total_traces += 1
                    starts.append(trace[0])
                    ends.append(trace[-1])
                    drops.append(trace[-1] - trace[0])

                    # Monotonicity: check if energy is non-increasing
                    is_mono = all(trace[i+1] <= trace[i] + 1e-12
                                  for i in range(len(trace) - 1))
                    if is_mono:
                        monotonic_count += 1

                    # Oscillation count: number of direction changes
                    osc = 0
                    for i in range(1, len(trace) - 1):
                        if (trace[i] - trace[i-1]) * (trace[i+1] - trace[i]) < 0:
                            osc += 1
                    oscillation_counts.append(osc)

        if total_traces > 0:
            energy_stats[mode_name] = {
                "energy_start": np.mean(starts),
                "energy_end": np.mean(ends),
                "delta_energy": np.mean(drops),
                "monotonic_frac": monotonic_count / total_traces,
                "avg_oscillations": np.mean(oscillation_counts),
                "total_traces": total_traces,
            }
        else:
            energy_stats[mode_name] = None

    # Print table
    print()
    header = (f"{'Mode':<22} {'E_start':>10} {'E_end':>10} {'ΔEnergy':>10} "
              f"{'Monotonic':>10} {'Oscill':>8}")
    print(header)
    print("-" * len(header))

    for mode_name in MODE_ORDER:
        s = energy_stats[mode_name]
        if s is None:
            print(f"{mode_name:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10} "
                  f"{'N/A':>10} {'N/A':>8}")
        else:
            print(f"{mode_name:<22} {s['energy_start']:>10.2f} "
                  f"{s['energy_end']:>10.2f} {s['delta_energy']:>10.2f} "
                  f"{s['monotonic_frac']:>10.2%} "
                  f"{s['avg_oscillations']:>8.2f}")

    print()
    return energy_stats


# ──────────────────────────────────────────────────────────────
# Step 5 — DPS Sign Detection
# ──────────────────────────────────────────────────────────────

_DPS_EPS = 1e-9


def _dps_sign(x: float, eps: float = _DPS_EPS) -> str:
    """Classify DPS value with epsilon tolerance for floating-point noise."""
    if x > eps:
        return "positive"
    if x < -eps:
        return "negative"
    return "zero"


def step5_dps_sign(slopes):
    print("=" * 72)
    print("STEP 5 — DPS SIGN DETECTION")
    print("=" * 72)

    print()
    print(f"{'Mode':<22} {'p':>8} {'DPS':>12} {'Sign':>8}")
    print("-" * 52)

    negative_found = False
    negative_modes = []

    for mode_name in MODE_ORDER:
        for p in sorted(slopes[mode_name].keys()):
            dps = slopes[mode_name][p]
            sign_label = _dps_sign(dps)
            sign_display = {"negative": "NEG (<0)", "positive": "POS (>0)", "zero": "ZERO"}[sign_label]
            marker = " ***" if sign_label == "negative" else ""
            print(f"{mode_name:<22} {p:>8.3f} {dps:>12.6f} {sign_display:>8}{marker}")
            if sign_label == "negative":
                negative_found = True
                negative_modes.append((mode_name, p, dps))

    print()
    if negative_found:
        print("DPS < 0 DETECTED in the following configurations:")
        for mode, p, dps in negative_modes:
            print(f"  {mode} at p={p}: DPS = {dps:.6f}")
    else:
        print("No DPS < 0 detected in any mode/p configuration.")

    print()
    return negative_found, negative_modes


# ──────────────────────────────────────────────────────────────
# Step 6 — Comparative Analysis
# ──────────────────────────────────────────────────────────────

def step6_analysis(eval_result, energy_stats, negative_modes):
    print("=" * 72)
    print("STEP 6 — COMPARATIVE ANALYSIS")
    print("=" * 72)

    results = eval_result["results"]
    slopes = eval_result["slopes"]
    config = eval_result["config"]

    # Compare baseline FER vs intervention FER
    print("\n--- FER Comparison vs Baseline ---")
    for p in config["p_values"]:
        print(f"\n  p={p}:")
        baseline_fers = {d: results["baseline"][p][d]["fer"]
                         for d in config["distances"]}
        for mode_name in MODE_ORDER:
            if mode_name == "baseline":
                continue
            mode_fers = {d: results[mode_name][p][d]["fer"]
                         for d in config["distances"]}
            diffs = {d: mode_fers[d] - baseline_fers[d]
                     for d in config["distances"]}
            diff_str = ", ".join(f"d={d}: {diffs[d]:+.4f}" for d in config["distances"])
            print(f"    {mode_name:<22} {diff_str}")

    # Centered field effect
    print("\n--- Centered Field Effect ---")
    for p in config["p_values"]:
        slope_base = slopes["baseline"][p]
        slope_cen = slopes["centered"][p]
        print(f"  p={p}: baseline DPS={slope_base:.6f}, "
              f"centered DPS={slope_cen:.6f}, "
              f"delta={slope_cen - slope_base:.6f}")

    # Pseudo-prior symmetry breaking
    print("\n--- Pseudo-Prior Effect ---")
    for p in config["p_values"]:
        slope_base = slopes["baseline"][p]
        slope_prior = slopes["prior"][p]
        print(f"  p={p}: baseline DPS={slope_base:.6f}, "
              f"prior DPS={slope_prior:.6f}, "
              f"delta={slope_prior - slope_base:.6f}")

    # Combined intervention
    print("\n--- Combined Intervention (centered + prior) ---")
    for p in config["p_values"]:
        slope_base = slopes["baseline"][p]
        slope_cp = slopes["centered_prior"][p]
        print(f"  p={p}: baseline DPS={slope_base:.6f}, "
              f"centered_prior DPS={slope_cp:.6f}, "
              f"delta={slope_cp - slope_base:.6f}")

    # Energy trace analysis
    print("\n--- Energy Trace Analysis ---")
    for mode_name in MODE_ORDER:
        s = energy_stats.get(mode_name)
        if s is None:
            continue
        descent_type = "N/A"
        if s["delta_energy"] < -1.0:
            if s["monotonic_frac"] > 0.8:
                descent_type = "steep monotonic descent"
            elif s["avg_oscillations"] > 2.0:
                descent_type = "oscillatory convergence"
            else:
                descent_type = "moderate descent"
        elif abs(s["delta_energy"]) < 1.0:
            descent_type = "flatline"
        else:
            descent_type = "energy increase (divergent)"

        print(f"  {mode_name:<22} → {descent_type} "
              f"(ΔE={s['delta_energy']:.2f}, "
              f"mono={s['monotonic_frac']:.0%}, "
              f"osc={s['avg_oscillations']:.1f})")

    print()


# ──────────────────────────────────────────────────────────────
# Step 7 — Generate Report
# ──────────────────────────────────────────────────────────────

def step7_report(eval_result, energy_stats, negative_modes,
                 baseline_passed, determinism_passed):
    print("=" * 72)
    print("STEP 7 — GENERATING RELEASE EVALUATION REPORT")
    print("=" * 72)

    results = eval_result["results"]
    slopes = eval_result["slopes"]
    config = eval_result["config"]

    lines = []
    lines.append("# v3.9.0 — Geometry Intervention Results\n")

    lines.append("## Overview\n")
    lines.append("v3.9.0 introduces two deterministic channel-geometry interventions")
    lines.append("and a per-iteration BP energy trace diagnostic:\n")
    lines.append("1. **Centered syndrome-field projection** — removes uniform syndrome bias")
    lines.append("2. **Parity-derived pseudo-prior injection** — weak deterministic variable prior")
    lines.append("3. **BP energy trace** — measures LLR-belief correlation per iteration\n")

    lines.append("## Experimental Setup\n")
    lines.append(f"- Seed: {config['seed']}")
    lines.append(f"- Distances: {config['distances']}")
    lines.append(f"- P values: {config['p_values']}")
    lines.append(f"- Trials per (distance, p): {config['trials']}")
    lines.append(f"- Max iterations: {config['max_iters']}")
    lines.append(f"- BP mode: {config['bp_mode']}")
    lines.append(f"- Modes evaluated: {len(MODE_ORDER)}\n")

    lines.append("## Determinism Verification\n")
    lines.append(f"- Baseline invariant: **{'PASS' if baseline_passed else 'FAIL'}**")
    lines.append(f"- Full determinism check: **{'PASS' if determinism_passed else 'FAIL'}**")
    lines.append(f"- All modes produce identical results across repeated runs.\n")

    lines.append("## DPS Results\n")
    lines.append("| Mode | " + " | ".join(f"p={p}" for p in config["p_values"]) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in config["p_values"]) + " |")
    for mode_name in MODE_ORDER:
        row = f"| {mode_name} |"
        for p in config["p_values"]:
            s = slopes[mode_name][p]
            marker = " **" if _dps_sign(s) == "negative" else ""
            row += f" {s:.6f}{marker} |"
        lines.append(row)
    lines.append("")

    lines.append("## FER Results\n")
    lines.append("| Mode | p | d | FER | Frame Errors | Avg Iters |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for mode_name in MODE_ORDER:
        for p in config["p_values"]:
            for d in config["distances"]:
                r = results[mode_name][p][d]
                lines.append(
                    f"| {mode_name} | {p} | {d} | {r['fer']:.4f} | "
                    f"{r['frame_errors']} | "
                    f"{r['audit_summary']['mean_iters']:.2f} |"
                )
    lines.append("")

    lines.append("## Energy Trace Analysis\n")
    lines.append("| Mode | Energy Start | Energy End | Delta Energy | Monotonic | Oscillations |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for mode_name in MODE_ORDER:
        s = energy_stats.get(mode_name)
        if s is None:
            lines.append(f"| {mode_name} | N/A | N/A | N/A | N/A | N/A |")
        else:
            lines.append(
                f"| {mode_name} | {s['energy_start']:.2f} | "
                f"{s['energy_end']:.2f} | {s['delta_energy']:.2f} | "
                f"{s['monotonic_frac']:.0%} | {s['avg_oscillations']:.2f} |"
            )
    lines.append("")

    lines.append("## Intervention Comparison\n")

    # Centered field alone
    lines.append("### Centered Field Projection\n")
    for p in config["p_values"]:
        delta = slopes["centered"][p] - slopes["baseline"][p]
        lines.append(f"- p={p}: DPS delta = {delta:+.6f}")
    lines.append("")

    # Pseudo-prior alone
    lines.append("### Pseudo-Prior Injection\n")
    for p in config["p_values"]:
        delta = slopes["prior"][p] - slopes["baseline"][p]
        lines.append(f"- p={p}: DPS delta = {delta:+.6f}")
    lines.append("")

    # Combined
    lines.append("### Combined (Centered + Prior)\n")
    for p in config["p_values"]:
        delta = slopes["centered_prior"][p] - slopes["baseline"][p]
        lines.append(f"- p={p}: DPS delta = {delta:+.6f}")
    lines.append("")

    lines.append("## Conclusion\n")
    if negative_modes:
        lines.append("**DPS sign flip detected** in the following configurations:\n")
        for mode, p, dps in negative_modes:
            lines.append(f"- {mode} at p={p}: DPS = {dps:.6f}")
        lines.append("")
    else:
        lines.append("**No DPS sign flip detected.** All modes produce DPS >= 0")
        lines.append("under the evaluated parameter range.\n")

    safe = baseline_passed and determinism_passed
    lines.append(f"Release readiness: **{'SAFE TO TAG v3.9.0' if safe else 'DO NOT TAG'}**\n")

    report_text = "\n".join(lines)

    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/v3_9_0_geometry_intervention_results.md", "w") as f:
        f.write(report_text)

    print("Report written to: reports/v3_9_0_geometry_intervention_results.md")
    print()
    return report_text


# ──────────────────────────────────────────────────────────────
# Step 8 — Tag Recommendation
# ──────────────────────────────────────────────────────────────

def step8_tag(baseline_passed, determinism_passed):
    print("=" * 72)
    print("STEP 8 — TAG RECOMMENDATION")
    print("=" * 72)
    print()

    if not baseline_passed:
        print("DO NOT TAG — baseline invariant violated")
        return False
    if not determinism_passed:
        print("DO NOT TAG — determinism failure")
        return False

    print("SAFE TO TAG v3.9.0")
    return True


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("v3.9.0 Release Verification & DPS Experiment")
    print("=" * 72)
    print()

    # Step 1
    baseline_passed = step1_baseline_invariant()
    if not baseline_passed:
        print("ABORT: Baseline invariant violated. Halting.")
        step8_tag(False, False)
        sys.exit(1)

    # Step 2
    determinism_passed, eval_result = step2_determinism()
    if not determinism_passed:
        print("ABORT: Determinism failure. Halting.")
        step8_tag(True, False)
        sys.exit(1)

    # Step 3
    results, slopes = step3_dps_sweep(eval_result)

    # Step 4
    energy_stats = step4_energy_diagnostics(eval_result)

    # Step 5
    negative_found, negative_modes = step5_dps_sign(slopes)

    # Step 6
    step6_analysis(eval_result, energy_stats, negative_modes)

    # Step 7
    step7_report(eval_result, energy_stats, negative_modes,
                 baseline_passed, determinism_passed)

    # Step 8
    step8_tag(baseline_passed, determinism_passed)


if __name__ == "__main__":
    main()
