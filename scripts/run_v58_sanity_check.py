#!/usr/bin/env python
"""
v5.8 Sanity Check — Ternary Topology Diagnostics.

Runs a small deterministic decoding experiment on a minimal code instance
and prints ternary traces, transition metrics, metastability scores, and
basin probe fractions for visual inspection before a release tag.

This is a diagnostic helper only — not part of the core benchmark harness.

Usage:
    python scripts/run_v58_sanity_check.py

All seeds are fixed.  Output is fully deterministic and reproducible.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Ensure imports resolve from repository root.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np

from src.qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code
from qec.diagnostics.bp_phase_space import (
    compute_bp_phase_space,
    compute_metastability_score,
)
from qec.diagnostics.ternary_decoder_topology import (
    compute_ternary_decoder_topology,
)
from qec.diagnostics.basin_probe import probe_local_ternary_basin

# ── Configuration ─────────────────────────────────────────────────────

CODE_NAME = "rate_0.50"
LIFTING_SIZE = 8          # small but enough structure for multi-iteration decoding
CODE_SEED = 42
P_VALUES = [0.01, 0.03, 0.05]
TRIALS_PER_P = 5
MAX_ITERS = 30
BP_MODE = "min_sum"
SCHEDULE = "flooding"
BASIN_PROBE_SCALE = 1e-3
BASIN_PROBE_DIRECTIONS = 5
RNG_BASE_SEED = 12345


def run_sanity_check() -> list[dict[str, Any]]:
    """Run the sanity check experiment.  Returns all trial results."""
    code = create_code(name=CODE_NAME, lifting_size=LIFTING_SIZE, seed=CODE_SEED)
    H = code.H_X

    all_results: list[dict[str, Any]] = []

    for p in P_VALUES:
        # Deterministic RNG per p-value (derived from base seed).
        rng = np.random.default_rng(RNG_BASE_SEED + int(p * 10000))

        for trial_idx in range(TRIALS_PER_P):
            # Generate error instance.
            n = H.shape[1]
            e = (rng.random(n) < p).astype(np.uint8)
            s = syndrome(H, e)
            llr = channel_llr(e, p)

            # Decode with LLR history.
            correction, iters, llr_hist = bp_decode(
                H, llr,
                max_iters=MAX_ITERS,
                mode=BP_MODE,
                schedule=SCHEDULE,
                syndrome_vec=s,
                llr_history=MAX_ITERS,
            )

            # Phase-space diagnostic.
            trajectory_states = [
                llr_hist[i].copy() for i in range(llr_hist.shape[0])
            ]
            ps_result = compute_bp_phase_space(trajectory_states)

            # Syndrome residual for ternary classification.
            s_correction = syndrome(H, correction)
            final_sw = int(np.sum(s_correction != s))
            syndrome_residuals = [final_sw] * len(trajectory_states)

            # Ternary topology classification.
            tt_result = compute_ternary_decoder_topology(
                phase_space_result=ps_result,
                syndrome_residuals=syndrome_residuals,
            )

            # Metastability score.
            ms = compute_metastability_score(ps_result["residual_norms"])

            # Basin probe.
            def _decode_fn(llr_in: np.ndarray) -> np.ndarray:
                c, _, _ = bp_decode(
                    H, llr_in,
                    max_iters=MAX_ITERS,
                    mode=BP_MODE,
                    schedule=SCHEDULE,
                    syndrome_vec=s,
                    llr_history=1,
                )
                return c

            final_llr = llr_hist[-1].copy()
            basin_result = probe_local_ternary_basin(
                llr_vector=final_llr,
                decode_function=_decode_fn,
                perturbation_scale=BASIN_PROBE_SCALE,
                directions=min(BASIN_PROBE_DIRECTIONS, n),
                syndrome_function=lambda c: syndrome(H, c),
                syndrome_target=s,
            )

            trial_result = {
                "p": p,
                "trial": trial_idx,
                "ternary_trace": tt_result["ternary_trace"],
                "final_ternary_state": tt_result["final_ternary_state"],
                "boundary_crossings": tt_result["boundary_crossings"],
                "regime_switch_count": tt_result["regime_switch_count"],
                "first_success_iteration": tt_result["first_success_iteration"],
                "first_failure_iteration": tt_result["first_failure_iteration"],
                "metastability_score": ms,
                "basin_probe": {
                    "success_fraction": basin_result["success_fraction"],
                    "boundary_fraction": basin_result["boundary_fraction"],
                    "failure_fraction": basin_result["failure_fraction"],
                },
            }
            all_results.append(trial_result)

    return all_results


def print_results(results: list[dict[str, Any]]) -> None:
    """Print results in compact human-readable format."""
    print("=" * 60)
    print("v5.8 Ternary Topology Sanity Check")
    print("=" * 60)
    print(f"Code: {CODE_NAME}, lifting_size={LIFTING_SIZE}")
    print(f"BP: mode={BP_MODE}, schedule={SCHEDULE}, max_iters={MAX_ITERS}")
    print(f"p values: {P_VALUES}")
    print(f"Trials per p: {TRIALS_PER_P}")
    print("=" * 60)
    print()

    for r in results:
        print(f"p = {r['p']}")
        print(f"  trial = {r['trial']}")
        print(f"  ternary_trace: {r['ternary_trace']}")
        print(f"  final_state: {r['final_ternary_state']}")
        print(f"  boundary_crossings: {r['boundary_crossings']}")
        print(f"  regime_switch_count: {r['regime_switch_count']}")
        print(f"  first_success_iteration: {r['first_success_iteration']}")
        print(f"  first_failure_iteration: {r['first_failure_iteration']}")
        print(f"  metastability_score: {r['metastability_score']:.6f}")
        print(f"  basin_probe:")
        bp = r["basin_probe"]
        print(f"    success_fraction: {bp['success_fraction']:.2f}")
        print(f"    boundary_fraction: {bp['boundary_fraction']:.2f}")
        print(f"    failure_fraction: {bp['failure_fraction']:.2f}")
        print()

    # JSON serialization check.
    json_str = json.dumps(results, indent=2)
    print("-" * 60)
    print("JSON serialization: OK")
    print(f"Total trials: {len(results)}")
    print(f"JSON size: {len(json_str)} bytes")
    print("-" * 60)

    # Interpretation guide.
    print()
    print("=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print()
    print("Expected example patterns:")
    print()
    print("  Stable decode:")
    print("    [0, 0, 1, 1, 1]")
    print("    final_state: 1, boundary_crossings: 1")
    print()
    print("  Failure basin:")
    print("    [0, 0, -1, -1, -1]")
    print("    final_state: -1, boundary_crossings: 1")
    print()
    print("  Oscillatory / boundary:")
    print("    [0, -1, 0, -1, 0]")
    print("    final_state: 0, boundary_crossings >= 2")
    print()
    print("  Metastability score:")
    print("    < 0.01  → strong convergence")
    print("    0.01-0.1 → plateau behavior")
    print("    > 0.1   → oscillation")
    print()
    print("  Basin probe:")
    print("    success_fraction near 1.0 → deep in success basin")
    print("    mixed fractions → near decision boundary")
    print()


def main() -> int:
    """Entry point."""
    results = run_sanity_check()
    print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
