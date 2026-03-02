"""
v3.8.1 — Structural Geometry DPS Evaluation Harness.

Controlled measurement of Distance Penalty Slope (DPS) across four modes:
  1. Baseline:   schedule="flooding", structural disabled
  2. RPC only:   schedule="flooding", structural.rpc.enabled=True
  3. geom_v1:    schedule="geom_v1", structural disabled
  4. RPC+geom:   schedule="geom_v1", structural.rpc.enabled=True

No decoder modifications.  No caching.  Fixed RNG.
Pre-generated error instances reused across all modes.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

# Ensure repo root is on sys.path for standalone execution.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from src.qec_qldpc_codes import bp_decode, syndrome, create_code
from src.qec.channel import get_channel_model
from src.qec.decoder.rpc import RPCConfig, StructuralConfig, build_rpc_augmented_system
from src.bench.geometry_diagnostics import compute_dps


# ── Configuration ──────────────────────────────────────────────────

SEED = 42
DISTANCES = [8, 12, 16]
P_VALUES = [0.01, 0.015, 0.02]
MAX_ITERS = 50
TRIALS = 200
CHANNEL = "bsc_syndrome"
MODE = "min_sum"

# RPC defaults
RPC_MAX_ROWS = 64
RPC_W_MIN = 2
RPC_W_MAX = 32


# ── Mode definitions ───────────────────────────────────────────────

MODES = {
    "baseline": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=RPCConfig(enabled=False)),
    },
    "rpc_only": {
        "schedule": "flooding",
        "structural": StructuralConfig(rpc=RPCConfig(
            enabled=True, max_rows=RPC_MAX_ROWS,
            w_min=RPC_W_MIN, w_max=RPC_W_MAX,
        )),
    },
    "geom_v1_only": {
        "schedule": "geom_v1",
        "structural": StructuralConfig(rpc=RPCConfig(enabled=False)),
    },
    "rpc_geom": {
        "schedule": "geom_v1",
        "structural": StructuralConfig(rpc=RPCConfig(
            enabled=True, max_rows=RPC_MAX_ROWS,
            w_min=RPC_W_MIN, w_max=RPC_W_MAX,
        )),
    },
}


# ── Error instance pre-generation ──────────────────────────────────

def pregenerate_instances(distances, p_values, trials, seed):
    """Pre-generate error instances per (distance, p).

    Returns dict[(distance, p)] -> list of (error_vector, syndrome, llr).
    All instances are generated once and reused across modes.
    """
    channel = get_channel_model(CHANNEL)
    instances = {}

    for d in distances:
        code = create_code("rate_0.50", lifting_size=d, seed=seed)
        H = code.H_X
        n = H.shape[1]

        for p in p_values:
            # Deterministic sub-seed per (distance, p)
            sub_seed = seed * 1000000 + d * 1000 + int(p * 10000)
            rng = np.random.default_rng(sub_seed)

            inst_list = []
            for _ in range(trials):
                e = (rng.random(n) < p).astype(np.uint8)
                s = syndrome(H, e)
                llr = channel.compute_llr(p=p, n=n, error_vector=e)
                inst_list.append((e, s, llr))

            instances[(d, p)] = inst_list

    return instances


# ── Single-mode evaluation ─────────────────────────────────────────

def evaluate_mode(mode_name, mode_cfg, distances, p_values, trials,
                  instances, seed):
    """Run DPS evaluation for one mode.

    Returns list of benchmark-compatible records and activation audit data.
    """
    schedule = mode_cfg["schedule"]
    structural = mode_cfg["structural"]
    rpc_enabled = structural.rpc.enabled

    records = []
    audit_data = []

    for d in distances:
        code = create_code("rate_0.50", lifting_size=d, seed=seed)
        H = code.H_X
        m_orig, n = H.shape

        for p in p_values:
            inst_list = instances[(d, p)]

            frame_errors = 0
            total_iters = 0
            iter_counts = []
            syndrome_weights = []
            error_weights = []

            for trial_idx in range(trials):
                e, s, llr = inst_list[trial_idx]

                # Apply RPC augmentation if enabled
                H_used = H
                s_used = s
                if rpc_enabled:
                    H_used, s_used = build_rpc_augmented_system(
                        H, s, structural.rpc,
                    )

                result = bp_decode(
                    H_used, llr, max_iters=MAX_ITERS,
                    mode=MODE, schedule=schedule,
                    syndrome_vec=s_used,
                )
                correction, iters = result[0], result[1]
                total_iters += iters
                iter_counts.append(iters)

                # Check success against original H
                residual = e ^ correction
                if np.any(residual):
                    frame_errors += 1

                syndrome_weights.append(int(np.sum(s)))
                error_weights.append(int(np.sum(e)))

            fer = float(frame_errors) / trials
            mean_iters = float(total_iters) / trials
            max_iters_obs = max(iter_counts) if iter_counts else 0
            fraction_iters_eq_1 = sum(
                1 for ic in iter_counts if ic == 1
            ) / trials

            # Augmented row info
            augmented_rows = H_used.shape[0]
            added_rows = augmented_rows - m_orig

            # Checksums
            H_checksum = int(np.sum(H_used.astype(np.int64)))
            syndrome_checksum = int(np.sum(s_used.astype(np.int64)))

            mean_syndrome_weight = float(np.mean(syndrome_weights))
            fraction_zero_syndrome = float(
                sum(1 for sw in syndrome_weights if sw == 0) / trials
            )
            mean_error_weight = float(np.mean(error_weights))

            # Record for DPS computation
            decoder_name = f"bp_{MODE}_{schedule}_none"
            records.append({
                "decoder": decoder_name,
                "distance": d,
                "p": p,
                "fer": fer,
                "fidelity": 1.0 - fer,
                "wer": fer,
                "mean_iters": round(mean_iters, 4),
                "trials": trials,
            })

            # Audit entry
            audit_entry = {
                "mode": mode_name,
                "distance": d,
                "p": p,
                "schedule": schedule,
                "rpc_enabled": rpc_enabled,
                "original_rows": m_orig,
                "augmented_rows": augmented_rows,
                "added_rows": added_rows,
                "mean_syndrome_weight": round(mean_syndrome_weight, 4),
                "fraction_zero_syndrome": round(fraction_zero_syndrome, 4),
                "mean_error_weight": round(mean_error_weight, 4),
                "mean_iters": round(mean_iters, 4),
                "max_iters_observed": max_iters_obs,
                "fraction_iters_eq_1": round(fraction_iters_eq_1, 4),
                "H_checksum": H_checksum,
                "syndrome_checksum": syndrome_checksum,
                "fer": fer,
            }
            audit_data.append(audit_entry)

    return records, audit_data


# ── Activation audit printing ──────────────────────────────────────

def print_activation_audit(audit_data):
    """Print activation audit with warnings."""
    print("\n" + "=" * 78)
    print("ACTIVATION AUDIT REPORT")
    print("=" * 78)

    for entry in audit_data:
        print(f"\n  Mode: {entry['mode']:<16}  "
              f"d={entry['distance']:<4}  p={entry['p']}")
        print(f"    schedule            = {entry['schedule']}")
        print(f"    rpc_enabled         = {entry['rpc_enabled']}")
        print(f"    original_rows       = {entry['original_rows']}")
        print(f"    augmented_rows      = {entry['augmented_rows']}")
        print(f"    added_rows          = {entry['added_rows']}")
        print(f"    mean_syndrome_weight = {entry['mean_syndrome_weight']}")
        print(f"    fraction_zero_syndrome = {entry['fraction_zero_syndrome']}")
        print(f"    mean_error_weight   = {entry['mean_error_weight']}")
        print(f"    mean_iters          = {entry['mean_iters']}")
        print(f"    max_iters_observed  = {entry['max_iters_observed']}")
        print(f"    fraction_iters_eq_1 = {entry['fraction_iters_eq_1']}")
        print(f"    H_checksum          = {entry['H_checksum']}")
        print(f"    syndrome_checksum   = {entry['syndrome_checksum']}")

        # Warnings
        if entry["added_rows"] == 0 and entry["rpc_enabled"]:
            print(f"    WARNING: added_rows == 0 despite RPC enabled")
        if entry["fraction_iters_eq_1"] > 0.95:
            print(f"    WARNING: fraction_iters_eq_1 > 0.95 "
                  f"({entry['fraction_iters_eq_1']})")
        expected_schedule = MODES[entry["mode"]]["schedule"]
        if entry["schedule"] != expected_schedule:
            print(f"    WARNING: schedule mismatch — "
                  f"expected {expected_schedule}, got {entry['schedule']}")


# ── DPS slope table ────────────────────────────────────────────────

def compute_and_print_dps_table(all_records):
    """Compute DPS slopes per mode and print summary table."""
    print("\n" + "=" * 78)
    print("DPS SLOPE TABLE")
    print("=" * 78)

    p_headers = "".join(f"  p={p:<10}" for p in P_VALUES)
    print(f"\n  {'Mode':<16}{p_headers}  {'Inverted?'}")
    print("  " + "-" * (16 + 14 * len(P_VALUES) + 12))

    mode_slopes = {}

    for mode_name in MODES:
        records = all_records[mode_name]
        dps = compute_dps(records)
        dps_by_p = {row["p"]: row for row in dps}
        mode_slopes[mode_name] = dps_by_p

        row_str = f"  {mode_name:<16}"
        any_inverted = False
        for p in P_VALUES:
            entry = dps_by_p.get(p)
            if entry:
                row_str += f"  {entry['slope']:<12.6f}"
                if entry["inverted"]:
                    any_inverted = True
            else:
                row_str += f"  {'N/A':<12}"
        inv_str = "INVERTED" if any_inverted else "normal"
        row_str += f"  {inv_str}"
        print(row_str)

    return mode_slopes


# ── Determinism check ──────────────────────────────────────────────

def run_determinism_check(instances, seed):
    """Re-run one config twice and confirm identical results."""
    print("\n" + "=" * 78)
    print("DETERMINISM CHECK")
    print("=" * 78)

    # Use baseline mode, smallest distance, first p value
    mode_name = "baseline"
    mode_cfg = MODES[mode_name]
    check_d = DISTANCES[0]
    check_p = P_VALUES[0]

    records_a, _ = evaluate_mode(
        mode_name, mode_cfg,
        distances=[check_d], p_values=[check_p],
        trials=TRIALS, instances=instances, seed=seed,
    )
    records_b, _ = evaluate_mode(
        mode_name, mode_cfg,
        distances=[check_d], p_values=[check_p],
        trials=TRIALS, instances=instances, seed=seed,
    )

    fer_a = records_a[0]["fer"]
    fer_b = records_b[0]["fer"]
    iters_a = records_a[0]["mean_iters"]
    iters_b = records_b[0]["mean_iters"]

    match = (fer_a == fer_b and iters_a == iters_b)

    print(f"\n  Config: mode={mode_name}, d={check_d}, p={check_p}")
    print(f"  Run A: FER={fer_a:.6f}, mean_iters={iters_a:.4f}")
    print(f"  Run B: FER={fer_b:.6f}, mean_iters={iters_b:.4f}")
    print(f"  Deterministic: {match}")

    if not match:
        print("  WARNING: Determinism violation detected!")

    return match


# ── Main ───────────────────────────────────────────────────────────

def main():
    """Run the full v3.8.1 DPS evaluation."""
    print("=" * 78)
    print("v3.8.1 Structural Geometry DPS Evaluation")
    print("=" * 78)
    print(f"  Seed:       {SEED}")
    print(f"  Distances:  {DISTANCES}")
    print(f"  P values:   {P_VALUES}")
    print(f"  Max iters:  {MAX_ITERS}")
    print(f"  Trials:     {TRIALS}")
    print(f"  Channel:    {CHANNEL}")
    print(f"  Mode:       {MODE}")

    # Phase 1: Pre-generate error instances
    print("\n  Pre-generating error instances...")
    instances = pregenerate_instances(DISTANCES, P_VALUES, TRIALS, SEED)
    print(f"  Generated {sum(len(v) for v in instances.values())} instances "
          f"across {len(instances)} (distance, p) pairs.")

    # Phase 2: Evaluate each mode
    all_records = {}
    all_audit = []

    for mode_name in MODES:
        print(f"\n  Evaluating mode: {mode_name}...")
        records, audit = evaluate_mode(
            mode_name, MODES[mode_name],
            DISTANCES, P_VALUES, TRIALS, instances, SEED,
        )
        all_records[mode_name] = records
        all_audit.extend(audit)

    # Phase 3: Activation audit
    print_activation_audit(all_audit)

    # Phase 4: DPS slope table
    compute_and_print_dps_table(all_records)

    # Phase 5: Determinism check
    det_ok = run_determinism_check(instances, SEED)

    print("\n" + "=" * 78)
    print("EVALUATION COMPLETE")
    print(f"  Determinism: {'PASS' if det_ok else 'FAIL'}")
    print("=" * 78)

    return 0 if det_ok else 1


if __name__ == "__main__":
    sys.exit(main())
