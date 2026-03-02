#!/usr/bin/env python3
"""
v3.8.0 DPS Evaluation Harness — Structural Geometry Interventions.

Measures Distance Penalty Slope (DPS) under bsc_syndrome across four modes:

  Mode A — Baseline         (flooding, no structural)
  Mode B — RPC Only         (flooding + RPC augmentation)
  Mode C — Degree Only      (geom_v1, no structural)
  Mode D — RPC + Degree     (geom_v1 + RPC augmentation)

Primary metric:   DPS slope sign (positive = inversion)
Secondary metric: DPS magnitude
Tertiary metric:  FER stability

All modes use identical:
  - RNG seed
  - Error realizations (same sub-seed derivation)
  - Trial count
  - Iteration budget
  - Code construction

No parameter sweeps.  No adaptive tuning.  No scheduling changes.
No decoder modifications.  Measurement only.

Usage:
    python bench/dps_v380_eval.py
"""
from __future__ import annotations

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
# Wilson CI (inline, matches src/simulation/fer.py)
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
# Core evaluation loop
# ═══════════════════════════════════════════════════════════════════

def evaluate_mode(
    mode_name: str,
    schedule: str,
    structural_config: StructuralConfig | None,
) -> list[dict]:
    """Run DPS evaluation for one mode across the full distance/p grid.

    Returns a list of result records compatible with compute_dps().
    """
    channel = get_channel_model(CHANNEL_MODEL)
    results = []

    for d in DISTANCES:
        code = create_code("rate_0.50", lifting_size=d, seed=SEED)
        H = code.H_X
        n = H.shape[1]

        # Pre-compute augmented H if RPC is enabled (once per distance).
        # Syndrome augmentation happens per-trial since s changes.
        H_aug_base = None
        if structural_config is not None and structural_config.rpc.enabled:
            # We need s to augment, so we build per-trial below.
            pass

        for p in P_VALUES:
            # Deterministic sub-seed: same across all modes for same (d, p).
            import hashlib, json
            payload = json.dumps({
                "base_seed": SEED, "distance": d, "p": p,
            }, sort_keys=True)
            sub_seed = int(hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()[:8], 16)
            rng = np.random.default_rng(sub_seed)

            frame_errors = 0
            total_iters = 0

            for _ in range(TRIALS):
                e = (rng.random(n) < p).astype(np.uint8)
                s = syndrome(H, e)
                llr = channel.compute_llr(p=p, n=n, error_vector=e)

                # Apply RPC augmentation if enabled.
                H_used = H
                s_used = s
                if (structural_config is not None
                        and structural_config.rpc.enabled):
                    H_used, s_used = build_rpc_augmented_system(
                        H, s, structural_config,
                    )

                corr, iters = bp_decode(
                    H_used, llr,
                    max_iters=MAX_ITERS,
                    mode=BP_MODE,
                    schedule=schedule,
                    syndrome_vec=s_used,
                )

                total_iters += iters

                # Check success against original error vector.
                residual = np.asarray(e) ^ np.asarray(corr)
                if np.any(residual):
                    frame_errors += 1

            fer = frame_errors / TRIALS
            ci_lo, ci_hi = _wilson_ci_95(frame_errors, TRIALS)

            results.append({
                "decoder": mode_name,
                "distance": d,
                "p": p,
                "fer": fer,
                "fidelity": 1.0 - fer,
                "mean_iters": round(total_iters / TRIALS, 4),
                "trials": TRIALS,
                "ci_95_lo": round(ci_lo, 6),
                "ci_95_hi": round(ci_hi, 6),
            })

    return results


# ═══════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════

def print_fer_table(all_results: dict[str, list[dict]]) -> None:
    """Print FER table across all modes."""
    print("\n" + "=" * 78)
    print("FER TABLE — bsc_syndrome, mode=min_sum, trials=%d, max_iters=%d"
          % (TRIALS, MAX_ITERS))
    print("=" * 78)

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
    print("\n" + "=" * 78)
    print("DPS SLOPE TABLE — log10(FER) vs distance")
    print("=" * 78)

    # Header.
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
    print("\n" + "=" * 78)
    print("DPS DELTA vs BASELINE")
    print("=" * 78)

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


def print_summary(all_dps: dict[str, list[dict]]) -> None:
    """Print final summary."""
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for mode_name in sorted(all_dps.keys()):
        dps_rows = all_dps[mode_name]
        any_inv = any(r.get("inverted", False) for r in dps_rows)
        slopes = [r["slope"] for r in dps_rows]
        mean_slope = sum(slopes) / len(slopes) if slopes else 0.0
        print(f"  {mode_name:<16}  mean_slope={mean_slope:+.6f}"
              f"  inverted={'YES' if any_inv else 'no'}")

    print("\n  DPS > 0 → distance inversion (FER increases with distance)")
    print("  DPS < 0 → correct scaling (FER decreases with distance)")
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 78)
    print("v3.8.0 DPS Evaluation — Structural Geometry Interventions")
    print("=" * 78)
    print(f"  seed={SEED}  trials={TRIALS}  max_iters={MAX_ITERS}")
    print(f"  distances={DISTANCES}")
    print(f"  p_values={P_VALUES}")
    print(f"  channel={CHANNEL_MODEL}  bp_mode={BP_MODE}")
    print(f"  RPC: max_rows={RPC_CONFIG.rpc.max_rows}, "
          f"w_min={RPC_CONFIG.rpc.w_min}, w_max={RPC_CONFIG.rpc.w_max}")

    all_results: dict[str, list[dict]] = {}
    all_dps: dict[str, list[dict]] = {}

    for mode_name, mode_cfg in sorted(MODES.items()):
        print(f"\n>>> Evaluating {mode_name} "
              f"(schedule={mode_cfg['schedule']}, "
              f"structural={'RPC' if mode_cfg['structural'] else 'none'}) ...")
        records = evaluate_mode(
            mode_name,
            mode_cfg["schedule"],
            mode_cfg["structural"],
        )
        all_results[mode_name] = records

        dps = compute_dps(records)
        all_dps[mode_name] = dps
        print(f"    done. ({len(records)} records)")

    # ── Output tables ──
    print_fer_table(all_results)
    print_dps_table(all_dps)
    print_dps_delta_table(all_dps)
    print_summary(all_dps)


if __name__ == "__main__":
    main()
