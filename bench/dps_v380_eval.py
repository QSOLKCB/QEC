#!/usr/bin/env python3
"""
v3.8.0 DPS Evaluation Harness — Activation Audit Edition.

Measures Distance Penalty Slope (DPS) under bsc_syndrome across four modes,
with full activation diagnostics to verify that structural interventions
(RPC augmentation, geom_v1 schedule) are actually engaged.

Modes:
  A_baseline   — flooding, no structural
  B_rpc_only   — flooding + RPC augmentation
  C_degree_only — geom_v1, no structural
  D_rpc_degree — geom_v1 + RPC augmentation

Diagnostics emitted per (mode, distance, p):
  - Schedule actually passed to bp_decode
  - RPC enabled flag and rows added
  - H checksum (deterministic: int(np.sum(H_used)))
  - Syndrome checksum (deterministic: sum over first trial)
  - Mean syndrome weight, fraction zero-syndrome
  - Mean error weight
  - Mean iterations, max iterations observed, fraction iters==1

All modes share identical error realizations via pre-generated instances.
No caching. No decoder modifications. Measurement only.

Usage:
    python bench/dps_v380_eval.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import os

import numpy as np

# Ensure project root is on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qec_qldpc_codes import bp_decode, syndrome, create_code, channel_llr
from src.qec.channel import get_channel_model
from src.qec.decoder.rpc import (
    RPCConfig,
    StructuralConfig,
    build_rpc_augmented_system,
)
from src.bench.geometry_diagnostics import compute_dps


# ═══════════════════════════════════════════════════════════════════
# Configuration — fixed, not swept
# ═══════════════════════════════════════════════════════════════════

SEED = 42
DISTANCES = [5, 7, 9, 12]
P_VALUES = [0.01, 0.015, 0.02]
TRIALS = 200
MAX_ITERS = 50
CHANNEL_MODEL = "bsc_syndrome"
BP_MODE = "min_sum"

RPC_CONFIG = StructuralConfig(
    rpc=RPCConfig(enabled=True, max_rows=10, w_min=2, w_max=50),
)

MODES = {
    "A_baseline":    {"schedule": "flooding",  "structural": None},
    "B_rpc_only":    {"schedule": "flooding",  "structural": RPC_CONFIG},
    "C_degree_only": {"schedule": "geom_v1",   "structural": None},
    "D_rpc_degree":  {"schedule": "geom_v1",   "structural": RPC_CONFIG},
}


# ═══════════════════════════════════════════════════════════════════
# Wilson CI
# ═══════════════════════════════════════════════════════════════════

def _wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    z2 = z * z
    p_hat = k / n
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)
    )
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return (lo, hi)


# ═══════════════════════════════════════════════════════════════════
# Pre-generate error instances (shared across all modes)
# ═══════════════════════════════════════════════════════════════════

def _derive_subseed(distance: int, p: float) -> int:
    """Deterministic sub-seed from (SEED, distance, p)."""
    payload = json.dumps({
        "base_seed": SEED, "distance": distance, "p": p,
    }, sort_keys=True)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def pregenerate_instances() -> dict:
    """Pre-generate all (distance, p) -> list of (error_vec, syndrome, llr).

    Shared across all modes so error realizations are identical.
    """
    channel = get_channel_model(CHANNEL_MODEL)
    instances = {}

    for d in DISTANCES:
        code = create_code("rate_0.50", lifting_size=d, seed=SEED)
        H = code.H_X
        n = H.shape[1]

        for p in P_VALUES:
            sub_seed = _derive_subseed(d, p)
            rng = np.random.default_rng(sub_seed)

            trials_data = []
            for _ in range(TRIALS):
                e = (rng.random(n) < p).astype(np.uint8)
                s = syndrome(H, e)
                llr = channel.compute_llr(p=p, n=n, error_vector=e)
                trials_data.append((e, s, llr))

            instances[(d, p)] = {"H": H, "n": n, "trials": trials_data}

    return instances


# ═══════════════════════════════════════════════════════════════════
# Core evaluation with activation diagnostics
# ═══════════════════════════════════════════════════════════════════

def evaluate_mode(
    mode_name: str,
    schedule: str,
    structural_config: StructuralConfig | None,
    instances: dict,
) -> tuple[list[dict], list[dict]]:
    """Run DPS evaluation for one mode.

    Returns (result_records, activation_records).
    """
    results = []
    activations = []

    for d in DISTANCES:
        for p in P_VALUES:
            inst = instances[(d, p)]
            H = inst["H"]
            n = inst["n"]
            trials_data = inst["trials"]
            m_orig = H.shape[0]

            frame_errors = 0
            total_iters = 0
            iter_counts = []
            syndrome_weights = []
            error_weights = []
            zero_syndrome_count = 0

            # Diagnostic: RPC row counts (per trial).
            rpc_added_rows_list = []

            # Diagnostic: checksums from first trial.
            first_H_checksum = None
            first_syn_checksum = None

            for t_idx, (e, s, llr) in enumerate(trials_data):
                # Track workload statistics.
                sw = int(np.sum(s))
                syndrome_weights.append(sw)
                if sw == 0:
                    zero_syndrome_count += 1
                error_weights.append(int(np.sum(e)))

                # Apply RPC augmentation if enabled.
                H_used = H
                s_used = s
                added_rows = 0
                if (structural_config is not None
                        and structural_config.rpc.enabled):
                    H_used, s_used = build_rpc_augmented_system(
                        H, s, structural_config,
                    )
                    added_rows = H_used.shape[0] - m_orig
                rpc_added_rows_list.append(added_rows)

                # Capture checksums from first trial only.
                if t_idx == 0:
                    first_H_checksum = int(np.sum(H_used))
                    first_syn_checksum = int(np.sum(s_used))

                # Decode.
                corr, iters = bp_decode(
                    H_used, llr,
                    max_iters=MAX_ITERS,
                    mode=BP_MODE,
                    schedule=schedule,
                    syndrome_vec=s_used,
                )

                total_iters += iters
                iter_counts.append(iters)

                # Check success against original error vector.
                residual = np.asarray(e) ^ np.asarray(corr)
                if np.any(residual):
                    frame_errors += 1

            fer = frame_errors / TRIALS
            ci_lo, ci_hi = _wilson_ci_95(frame_errors, TRIALS)
            mean_iters = total_iters / TRIALS
            max_iters_obs = max(iter_counts)
            frac_iters_eq_1 = sum(1 for i in iter_counts if i == 1) / TRIALS

            results.append({
                "decoder": mode_name,
                "distance": d,
                "p": p,
                "fer": fer,
                "fidelity": 1.0 - fer,
                "mean_iters": round(mean_iters, 4),
                "trials": TRIALS,
                "ci_95_lo": round(ci_lo, 6),
                "ci_95_hi": round(ci_hi, 6),
            })

            # Activation record.
            mean_added = sum(rpc_added_rows_list) / len(rpc_added_rows_list)
            min_added = min(rpc_added_rows_list)
            max_added = max(rpc_added_rows_list)

            activations.append({
                "mode": mode_name,
                "distance": d,
                "p": p,
                "schedule": schedule,
                "rpc_enabled": (structural_config is not None
                                and structural_config.rpc.enabled),
                "added_rows_mean": round(mean_added, 2),
                "added_rows_min": min_added,
                "added_rows_max": max_added,
                "H_rows_orig": m_orig,
                "H_rows_used": m_orig + max_added,
                "mean_syndrome_weight": round(
                    sum(syndrome_weights) / len(syndrome_weights), 4),
                "fraction_zero_syndrome": round(
                    zero_syndrome_count / TRIALS, 4),
                "mean_error_weight": round(
                    sum(error_weights) / len(error_weights), 4),
                "mean_iters": round(mean_iters, 4),
                "max_iters_observed": max_iters_obs,
                "fraction_iters_eq_1": round(frac_iters_eq_1, 4),
                "H_checksum": first_H_checksum,
                "syndrome_checksum": first_syn_checksum,
            })

    return results, activations


# ═══════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════

def print_activation_report(all_activations: dict[str, list[dict]]) -> None:
    """Print per-mode activation diagnostics."""
    print("\n" + "=" * 100)
    print("MODE ACTIVATION REPORT")
    print("=" * 100)

    for mode_name in sorted(all_activations.keys()):
        records = all_activations[mode_name]
        print(f"\n  [{mode_name}]")

        for rec in records:
            print(f"\n    d={rec['distance']}, p={rec['p']}")
            print(f"      schedule:              {rec['schedule']}")
            print(f"      rpc_enabled:           {rec['rpc_enabled']}")
            print(f"      added_rows:            "
                  f"mean={rec['added_rows_mean']}, "
                  f"min={rec['added_rows_min']}, "
                  f"max={rec['added_rows_max']}")
            print(f"      H_rows (orig/used):    "
                  f"{rec['H_rows_orig']} / {rec['H_rows_used']}")
            print(f"      mean_syndrome_weight:  {rec['mean_syndrome_weight']}")
            print(f"      fraction_zero_syndrome:{rec['fraction_zero_syndrome']}")
            print(f"      mean_error_weight:     {rec['mean_error_weight']}")
            print(f"      mean_iters:            {rec['mean_iters']}")
            print(f"      max_iters_observed:    {rec['max_iters_observed']}")
            print(f"      fraction_iters_eq_1:   {rec['fraction_iters_eq_1']}")
            print(f"      H_checksum:            {rec['H_checksum']}")
            print(f"      syndrome_checksum:     {rec['syndrome_checksum']}")

            # Warnings.
            if rec['rpc_enabled'] and rec['added_rows_max'] == 0:
                print(f"      WARNING: RPC produced zero additional rows!")
            if rec['fraction_iters_eq_1'] > 0.95:
                print(f"      WARNING: Decoder converges in 1 iteration "
                      f"for {rec['fraction_iters_eq_1']*100:.1f}% of trials.")
            if rec['fraction_zero_syndrome'] > 0.5:
                print(f"      WARNING: {rec['fraction_zero_syndrome']*100:.1f}%"
                      f" of trials have zero syndrome (degenerate workload).")


def print_schedule_verification(all_activations: dict[str, list[dict]]) -> None:
    """Verify schedule assignments match expectations."""
    print("\n" + "=" * 100)
    print("SCHEDULE VERIFICATION")
    print("=" * 100)

    expected = {
        "A_baseline":    "flooding",
        "B_rpc_only":    "flooding",
        "C_degree_only": "geom_v1",
        "D_rpc_degree":  "geom_v1",
    }

    all_ok = True
    for mode_name in sorted(all_activations.keys()):
        records = all_activations[mode_name]
        actual = records[0]["schedule"] if records else "N/A"
        exp = expected.get(mode_name, "???")
        match = actual == exp
        status = "OK" if match else "MISMATCH"
        if not match:
            all_ok = False
        print(f"  {mode_name:<16} expected={exp:<10} actual={actual:<10}  [{status}]")

    if all_ok:
        print("\n  All schedule assignments verified.")
    else:
        print("\n  WARNING: Schedule mismatch detected!")


def print_rpc_verification(all_activations: dict[str, list[dict]]) -> None:
    """Verify RPC augmentation actually adds rows where expected."""
    print("\n" + "=" * 100)
    print("RPC AUGMENTATION VERIFICATION")
    print("=" * 100)

    for mode_name in sorted(all_activations.keys()):
        records = all_activations[mode_name]
        rpc_on = records[0]["rpc_enabled"] if records else False
        print(f"\n  [{mode_name}]  rpc_enabled={rpc_on}")

        if rpc_on:
            all_zero = all(r["added_rows_max"] == 0 for r in records)
            if all_zero:
                print(f"    WARNING: RPC enabled but produced ZERO additional"
                      f" rows across all (d, p)!")
            else:
                for r in records:
                    print(f"    d={r['distance']}, p={r['p']}: "
                          f"added_rows mean={r['added_rows_mean']}, "
                          f"min={r['added_rows_min']}, "
                          f"max={r['added_rows_max']}")
        else:
            print(f"    (no RPC — skipped)")


def print_iteration_verification(all_activations: dict[str, list[dict]]) -> None:
    """Report iteration behavior per mode."""
    print("\n" + "=" * 100)
    print("ITERATION BEHAVIOR")
    print("=" * 100)

    for mode_name in sorted(all_activations.keys()):
        records = all_activations[mode_name]
        print(f"\n  [{mode_name}]")
        for r in records:
            frac1 = r["fraction_iters_eq_1"]
            flag = " <<<" if frac1 > 0.95 else ""
            print(f"    d={r['distance']}, p={r['p']:5.3f}: "
                  f"mean_iters={r['mean_iters']:6.2f}, "
                  f"max_iters={r['max_iters_observed']:3d}, "
                  f"frac_eq_1={frac1:.4f}{flag}")


def print_h_checksum_comparison(all_activations: dict[str, list[dict]]) -> None:
    """Compare H checksums across modes to detect actual matrix differences."""
    print("\n" + "=" * 100)
    print("H MATRIX CHECKSUM COMPARISON (first trial per cell)")
    print("=" * 100)

    # Group by (distance, p).
    for d in DISTANCES:
        for p in P_VALUES:
            print(f"\n  d={d}, p={p}:")
            for mode_name in sorted(all_activations.keys()):
                records = all_activations[mode_name]
                rec = next(r for r in records
                           if r["distance"] == d and r["p"] == p)
                print(f"    {mode_name:<16} H_cksum={rec['H_checksum']:8d}  "
                      f"syn_cksum={rec['syndrome_checksum']:4d}  "
                      f"H_rows={rec['H_rows_orig']}+{rec['added_rows_max']}")


def print_fer_table(all_results: dict[str, list[dict]]) -> None:
    """Print FER table across all modes."""
    print("\n" + "=" * 100)
    print("FER TABLE — bsc_syndrome, mode=min_sum, trials=%d, max_iters=%d"
          % (TRIALS, MAX_ITERS))
    print("=" * 100)

    for mode_name, records in sorted(all_results.items()):
        print(f"\n  [{mode_name}]")
        header = "    d\\p    "
        for p in P_VALUES:
            header += f"  p={p:<8}"
        print(header)

        for d in DISTANCES:
            row = f"    {d:<7}"
            for p in P_VALUES:
                rec = next(r for r in records
                           if r["distance"] == d and r["p"] == p)
                fer = rec["fer"]
                ci_lo = rec["ci_95_lo"]
                ci_hi = rec["ci_95_hi"]
                row += f"  {fer:.4f} [{ci_lo:.4f},{ci_hi:.4f}]"
            print(row)


def print_dps_table(all_dps: dict[str, list[dict]]) -> None:
    """Print DPS slope table across all modes."""
    print("\n" + "=" * 100)
    print("DPS SLOPE TABLE — log10(FER) vs distance")
    print("=" * 100)

    print(f"\n  {'Mode':<16}", end="")
    for p in P_VALUES:
        print(f"  {'p='+str(p):<14}", end="")
    print(f"  {'Inverted?':<12}")
    print("  " + "-" * (16 + len(P_VALUES) * 16 + 12))

    for mode_name in sorted(all_dps.keys()):
        dps_rows = all_dps[mode_name]
        dps_by_p = {r["p"]: r for r in dps_rows}

        print(f"  {mode_name:<16}", end="")
        for p in P_VALUES:
            row = dps_by_p.get(p)
            if row:
                slope = row["slope"]
                sign = "+" if slope > 0 else ("-" if slope < 0 else " ")
                print(f"  {sign}{abs(slope):<13.6f}", end="")
            else:
                print(f"  {'N/A':<14}", end="")

        any_inv = any(r.get("inverted", False) for r in dps_rows)
        inv_str = "YES" if any_inv else "no"
        print(f"  {inv_str:<12}")


def print_dps_delta_table(
    all_dps: dict[str, list[dict]],
    baseline_key: str = "A_baseline",
) -> None:
    """Print DPS delta vs baseline."""
    print("\n" + "=" * 100)
    print("DPS DELTA vs BASELINE")
    print("=" * 100)

    base_dps = {r["p"]: r["slope"] for r in all_dps[baseline_key]}

    print(f"\n  {'Mode':<16}", end="")
    for p in P_VALUES:
        print(f"  {'d@p='+str(p):<14}", end="")
    print(f"  {'Sign flip?':<12}")
    print("  " + "-" * (16 + len(P_VALUES) * 16 + 12))

    for mode_name in sorted(all_dps.keys()):
        if mode_name == baseline_key:
            continue
        dps_by_p = {r["p"]: r["slope"] for r in all_dps[mode_name]}

        print(f"  {mode_name:<16}", end="")
        flips = []
        for p in P_VALUES:
            s_mode = dps_by_p.get(p, 0.0)
            s_base = base_dps.get(p, 0.0)
            delta = s_mode - s_base
            print(f"  {delta:<+14.6f}", end="")
            if (s_base > 0 and s_mode <= 0) or (s_base <= 0 and s_mode > 0):
                flips.append(p)

        flip_str = ",".join(str(p) for p in flips) if flips else "no"
        print(f"  {flip_str:<12}")


def print_diagnostic_summary(
    all_activations: dict[str, list[dict]],
    all_dps: dict[str, list[dict]],
) -> None:
    """Print final diagnostic summary with root-cause assessment."""
    print("\n" + "=" * 100)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 100)

    # 1. Check if RPC adds rows.
    rpc_modes = ["B_rpc_only", "D_rpc_degree"]
    rpc_ever_adds = False
    for mode_name in rpc_modes:
        if mode_name in all_activations:
            for r in all_activations[mode_name]:
                if r["added_rows_max"] > 0:
                    rpc_ever_adds = True
                    break

    print(f"\n  1. RPC adds rows:            {'YES' if rpc_ever_adds else 'NO'}")
    if not rpc_ever_adds:
        print(f"     -> RPC augmentation is INERT (produces 0 additional rows)")

    # 2. Check if schedules differ.
    schedules_correct = True
    expected = {
        "A_baseline": "flooding", "B_rpc_only": "flooding",
        "C_degree_only": "geom_v1", "D_rpc_degree": "geom_v1",
    }
    for mode_name, exp in expected.items():
        if mode_name in all_activations:
            actual = all_activations[mode_name][0]["schedule"]
            if actual != exp:
                schedules_correct = False
    print(f"  2. Schedules correct:        {'YES' if schedules_correct else 'NO'}")

    # 3. Check iteration degeneracy.
    all_iters_eq_1 = True
    for mode_name in sorted(all_activations.keys()):
        for r in all_activations[mode_name]:
            if r["fraction_iters_eq_1"] < 0.95:
                all_iters_eq_1 = False
                break
    print(f"  3. Decoder converges iter=1: "
          f"{'YES (degenerate)' if all_iters_eq_1 else 'NO (multi-iter)'}")
    if all_iters_eq_1:
        print(f"     -> Degree scaling has NO EFFECT when convergence is "
              f"immediate")

    # 4. Check workload degeneracy (zero syndromes).
    max_zero_frac = 0.0
    for mode_name in sorted(all_activations.keys()):
        for r in all_activations[mode_name]:
            max_zero_frac = max(max_zero_frac, r["fraction_zero_syndrome"])
    print(f"  4. Max zero-syndrome frac:   {max_zero_frac:.4f}")
    if max_zero_frac > 0.5:
        print(f"     -> WARNING: Workload may be degenerate")

    # 5. Check H checksum differences.
    h_differs = False
    for d in DISTANCES:
        for p in P_VALUES:
            checksums = set()
            for mode_name in sorted(all_activations.keys()):
                rec = next(r for r in all_activations[mode_name]
                           if r["distance"] == d and r["p"] == p)
                checksums.add(rec["H_checksum"])
            if len(checksums) > 1:
                h_differs = True
                break
    print(f"  5. H matrix differs by mode: {'YES' if h_differs else 'NO'}")
    if not h_differs:
        print(f"     -> All modes decode with identical parity-check matrix")

    # 6. DPS identical across modes?
    dps_identical = True
    baseline_dps = {r["p"]: r["slope"] for r in all_dps.get("A_baseline", [])}
    for mode_name in sorted(all_dps.keys()):
        if mode_name == "A_baseline":
            continue
        for r in all_dps[mode_name]:
            if abs(r["slope"] - baseline_dps.get(r["p"], 0.0)) > 1e-10:
                dps_identical = False
    print(f"  6. DPS identical all modes:  {'YES' if dps_identical else 'NO'}")

    # Root cause assessment.
    print(f"\n  ROOT CAUSE ASSESSMENT:")
    causes = []
    if not rpc_ever_adds:
        causes.append("RPC produces zero additional rows (weight filter "
                       "rejects all XOR pairs)")
    if all_iters_eq_1:
        causes.append("Decoder converges in 1 iteration (degree scaling "
                       "cannot differentiate — only affects message "
                       "magnitudes, not hard decisions after 1 pass)")
    if not causes:
        causes.append("Structural interventions are engaged but produce "
                       "identical outcomes (genuine ineffectiveness at "
                       "this operating regime)")

    for i, cause in enumerate(causes, 1):
        print(f"    {i}. {cause}")

    print("\n" + "=" * 100)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 100)
    print("v3.8.0 DPS Evaluation — Activation Audit Edition")
    print("=" * 100)
    print(f"  seed={SEED}  trials={TRIALS}  max_iters={MAX_ITERS}")
    print(f"  distances={DISTANCES}")
    print(f"  p_values={P_VALUES}")
    print(f"  channel={CHANNEL_MODEL}  bp_mode={BP_MODE}")
    print(f"  RPC: max_rows={RPC_CONFIG.rpc.max_rows}, "
          f"w_min={RPC_CONFIG.rpc.w_min}, w_max={RPC_CONFIG.rpc.w_max}")

    # Pre-generate all error instances (shared across modes).
    print("\n>>> Pre-generating error instances ...")
    instances = pregenerate_instances()
    print(f"    done. ({len(instances)} (d, p) cells, {TRIALS} trials each)")

    all_results: dict[str, list[dict]] = {}
    all_dps: dict[str, list[dict]] = {}
    all_activations: dict[str, list[dict]] = {}

    for mode_name, mode_cfg in sorted(MODES.items()):
        print(f"\n>>> Evaluating {mode_name} "
              f"(schedule={mode_cfg['schedule']}, "
              f"structural={'RPC' if mode_cfg['structural'] else 'none'}) ...")
        records, activations = evaluate_mode(
            mode_name,
            mode_cfg["schedule"],
            mode_cfg["structural"],
            instances,
        )
        all_results[mode_name] = records
        all_activations[mode_name] = activations

        dps = compute_dps(records)
        all_dps[mode_name] = dps
        print(f"    done. ({len(records)} records)")

    # ── Activation diagnostics ──
    print_activation_report(all_activations)
    print_schedule_verification(all_activations)
    print_rpc_verification(all_activations)
    print_iteration_verification(all_activations)
    print_h_checksum_comparison(all_activations)

    # ── DPS tables ──
    print_fer_table(all_results)
    print_dps_table(all_dps)
    print_dps_delta_table(all_dps)

    # ── Diagnostic summary ──
    print_diagnostic_summary(all_activations, all_dps)


if __name__ == "__main__":
    main()
