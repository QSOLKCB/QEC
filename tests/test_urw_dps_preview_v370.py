"""
v3.7.0 URW-BP DPS Preview — Lightweight Structural Probe.

Uses existing run_benchmark() and compute_dps() infrastructure.
No new DPS logic.  No new slope calculation.  Configuration-driven only.

Reduced grid for fast execution:
  - Distances: [8, 12]
  - Noise: p = [0.01, 0.02]
  - Trials: 25
  - Rho: [1.0, 0.85, 0.7]

Run with:  pytest tests/test_urw_dps_preview_v370.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bench.config import BenchmarkConfig, DecoderSpec
from src.bench.runner import run_benchmark
from src.bench.geometry_diagnostics import compute_dps


# ── Preview configuration ────────────────────────────────────────────

DISTANCES = [8, 12]
P_VALUES = [0.01, 0.02]
TRIALS = 25
MAX_ITERS = 30
SEED = 42
CHANNEL = "bsc_syndrome"

RHO_VALUES = [1.0, 0.85, 0.7]

# Number of trials used for gate-check validation (separate from main TRIALS sweep)
_RHO_GATE_CHECK_TRIALS = 15


# ── Helpers ──────────────────────────────────────────────────────────

def _make_config(mode: str, schedule: str = "flooding",
                 postprocess=None, urw_rho: float | None = None,
                 distances=None, p_values=None,
                 trials=None) -> BenchmarkConfig:
    """Build a BenchmarkConfig for one decoder arm."""
    params: dict = {"mode": mode, "schedule": schedule}
    if postprocess is not None:
        params["postprocess"] = postprocess
    if urw_rho is not None:
        params["urw_rho"] = urw_rho
    return BenchmarkConfig(
        seed=SEED,
        distances=distances or DISTANCES,
        p_values=p_values or P_VALUES,
        trials=trials or TRIALS,
        max_iters=MAX_ITERS,
        decoders=[DecoderSpec(adapter="bp", params=params)],
        runtime_mode="off",
        deterministic_metadata=True,
        channel_model=CHANNEL,
    )


def _extract_wer_table(results: list[dict]) -> dict[tuple[int, float], float]:
    """Extract (distance, p) -> WER mapping from benchmark results."""
    return {(rec["distance"], rec["p"]): rec["fer"] for rec in results}


# ══════════════════════════════════════════════════════════════════════
#  GATE CHECK
# ══════════════════════════════════════════════════════════════════════

class TestGateCheck:
    """Verify min_sum == min_sum_urw(rho=1.0) on identical inputs."""

    def test_rho1_matches_min_sum_under_bsc_syndrome(self):
        """Direct bp_decode comparison on same H, llr, syndrome."""
        from src.qec_qldpc_codes import bp_decode, syndrome, create_code
        from src.qec.channel import get_channel_model

        channel = get_channel_model("bsc_syndrome")
        code = create_code("rate_0.50", lifting_size=8, seed=SEED)
        H = code.H_X
        n = H.shape[1]

        rng = np.random.default_rng(SEED)
        p_test = 0.02

        for t in range(_RHO_GATE_CHECK_TRIALS):
            e = (rng.random(n) < p_test).astype(np.uint8)
            s = syndrome(H, e)
            llr = channel.compute_llr(p=p_test, n=n, error_vector=e)

            r_ms = bp_decode(H, llr, max_iters=MAX_ITERS, mode="min_sum",
                             schedule="flooding", syndrome_vec=s)
            r_urw = bp_decode(H, llr, max_iters=MAX_ITERS, mode="min_sum_urw",
                              urw_rho=1.0, schedule="flooding", syndrome_vec=s)

            np.testing.assert_array_equal(
                r_ms[0], r_urw[0],
                err_msg=f"Correction mismatch at trial {t}",
            )
            assert r_ms[1] == r_urw[1], \
                f"Iteration count mismatch at trial {t}: {r_ms[1]} vs {r_urw[1]}"


# ══════════════════════════════════════════════════════════════════════
#  DETERMINISM CHECK
# ══════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Two identical benchmark runs produce identical results."""

    def test_urw_benchmark_deterministic(self):
        """run_benchmark with URW produces identical results twice."""
        cfg = _make_config("min_sum_urw", urw_rho=0.8)
        r1 = run_benchmark(cfg)
        r2 = run_benchmark(cfg)

        for rec1, rec2 in zip(r1["results"], r2["results"]):
            assert rec1["fer"] == rec2["fer"], \
                f"FER mismatch at d={rec1['distance']}, p={rec1['p']}"
            assert rec1["mean_iters"] == rec2["mean_iters"], \
                f"mean_iters mismatch at d={rec1['distance']}, p={rec1['p']}"


# ══════════════════════════════════════════════════════════════════════
#  PREVIEW DPS SWEEP
# ══════════════════════════════════════════════════════════════════════

class TestURWDPSPreview:
    """Lightweight URW DPS preview under bsc_syndrome channel."""

    def test_preview_sweep(self):
        """Run reduced URW DPS sweep and print summary."""

        # ── Baseline: min_sum ──
        cfg_base = _make_config("min_sum")
        res_base = run_benchmark(cfg_base)
        wer_base = _extract_wer_table(res_base["results"])
        dps_base = compute_dps(res_base["results"])
        dps_base_by_p = {row["p"]: row["slope"] for row in dps_base}

        print("\n" + "=" * 70)
        print("BASELINE: mode=min_sum, postprocess=None")
        print("=" * 70)
        self._print_wer(wer_base, "min_sum baseline")
        self._print_dps(dps_base, "min_sum baseline")

        # ── URW sweep ──
        all_dps: dict[float, list[dict]] = {}
        all_wer: dict[float, dict] = {}

        for rho in RHO_VALUES:
            cfg = _make_config("min_sum_urw", urw_rho=rho)
            res = run_benchmark(cfg)
            wer = _extract_wer_table(res["results"])
            dps = compute_dps(res["results"])
            all_dps[rho] = dps
            all_wer[rho] = wer

            print(f"\n--- urw_rho={rho} ---")
            self._print_wer(wer, f"urw_rho={rho}")
            self._print_dps(dps, f"urw_rho={rho}")

        # ── Summary table ──
        print("\n" + "=" * 70)
        print("URW PREVIEW SUMMARY (postprocess=None)")
        print("=" * 70)

        print(f"\n  {'rho':<6}", end="")
        for p in P_VALUES:
            print(f"  {'DPS@'+str(p):<12}", end="")
        print(f"  {'Inverted?':<12}")

        # Baseline row.
        print(f"  {'base':<6}", end="")
        for p in P_VALUES:
            s = dps_base_by_p.get(p, float("nan"))
            print(f"  {s:<12.6f}", end="")
        any_inv = any(r["inverted"] for r in dps_base)
        print(f"  {str(any_inv):<12}")

        # URW rows.
        for rho in RHO_VALUES:
            dps_by_p = {row["p"]: row for row in all_dps[rho]}
            print(f"  {rho:<6.2f}", end="")
            for p in P_VALUES:
                row = dps_by_p.get(p)
                s = row["slope"] if row else float("nan")
                print(f"  {s:<12.6f}", end="")
            any_inv = any(r["inverted"] for r in all_dps[rho])
            print(f"  {str(any_inv):<12}")

        # Delta table.
        print(f"\n  Delta DPS vs baseline:")
        print(f"  {'rho':<6}", end="")
        for p in P_VALUES:
            print(f"  {'d@'+str(p):<12}", end="")
        print()

        for rho in RHO_VALUES:
            dps_by_p = {row["p"]: row["slope"] for row in all_dps[rho]}
            print(f"  {rho:<6.2f}", end="")
            for p in P_VALUES:
                s_urw = dps_by_p.get(p, float("nan"))
                s_base = dps_base_by_p.get(p, float("nan"))
                delta = s_urw - s_base
                print(f"  {delta:<+12.6f}", end="")
            print()

        # ── Analysis ──
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)

        # Sign flip check.
        for p in P_VALUES:
            base_slope = dps_base_by_p.get(p, 0.0)
            base_sign = 1 if base_slope > 0 else -1
            first_flip = None
            for rho in RHO_VALUES:
                dps_map = {row["p"]: row["slope"] for row in all_dps[rho]}
                s = dps_map.get(p, 0.0)
                s_sign = 1 if s > 0 else -1
                if s_sign != base_sign:
                    first_flip = rho
                    break
            if first_flip is not None:
                print(f"  p={p:.2f}: DPS sign flips at rho={first_flip}")
            else:
                print(f"  p={p:.2f}: no DPS sign flip across rho range")

        # Monotonic trend.
        print("\n  Monotonic trend check:")
        for p in P_VALUES:
            slopes = []
            for rho in RHO_VALUES:
                dps_map = {row["p"]: row["slope"] for row in all_dps[rho]}
                slopes.append(dps_map.get(p, float("nan")))
            diffs = [slopes[i+1] - slopes[i] for i in range(len(slopes)-1)]
            all_inc = all(d >= 0 for d in diffs)
            all_dec = all(d <= 0 for d in diffs)
            if all_inc:
                trend = "monotonically increasing (with decreasing rho)"
            elif all_dec:
                trend = "monotonically decreasing (with decreasing rho)"
            else:
                trend = "non-monotonic"
            print(f"  p={p:.2f}: DPS trend is {trend}")

        print("\n" + "=" * 70)
        print("SWEEP COMPLETE")
        print("=" * 70)

    # ── Printing helpers ──

    @staticmethod
    def _print_wer(table, label):
        print(f"\n  WER table — {label}")
        header = "    d\\p   " + "".join(f"  {p:<8}" for p in P_VALUES)
        print(header)
        for d in DISTANCES:
            row = f"    {d:<6}"
            for p in P_VALUES:
                fer = table.get((d, p), float("nan"))
                row += f"  {fer:<8.4f}"
            print(row)

    @staticmethod
    def _print_dps(dps_rows, label):
        print(f"\n  DPS slopes — {label}")
        for row in sorted(dps_rows, key=lambda r: r["p"]):
            inv = "INVERTED" if row["inverted"] else "normal"
            print(f"    p={row['p']:.2f}: slope={row['slope']:.6f}  ({inv})")
