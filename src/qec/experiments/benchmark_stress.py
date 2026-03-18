"""Deterministic benchmark + stress-test baseline for BP dynamics (v68.7).

Measures ``compute_bp_dynamics_metrics`` performance and failure modes
across deterministic stress scenarios.  Produces reproducible JSON artifacts.

Layer 5 (experiments) — imports only from Layer 3 (diagnostics).
No decoder modifications.  No hidden randomness.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from typing import Any, Optional

import numpy as np

from src.qec.diagnostics.bp_dynamics import compute_bp_dynamics_metrics


# ── Version / identity ───────────────────────────────────────────────

_BENCHMARK_VERSION = "68.7.0"


def _git_short_hash() -> Optional[str]:
    """Return short git hash or None (best-effort, no subprocess errors)."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ── Deterministic RNG helpers ────────────────────────────────────────

def _derive_seed(master_seed: int, label: str) -> int:
    """Derive a sub-seed deterministically using SHA-256."""
    data = struct.pack(">Q", master_seed) + label.encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _make_rng(master_seed: int, label: str) -> np.random.Generator:
    """Create a deterministic RNG from master seed + label."""
    return np.random.Generator(
        np.random.PCG64(_derive_seed(master_seed, label))
    )


# ── Trace generators (deterministic) ────────────────────────────────

def generate_converging_trace(
    n_vars: int, n_iters: int, seed: int,
) -> dict:
    """Smoothly converging BP trace — energy monotonically decreasing."""
    rng = _make_rng(seed, "converging")
    base = rng.standard_normal(n_vars).astype(np.float64)
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        scale = 1.0 + 5.0 * np.exp(-0.3 * t)
        noise = rng.standard_normal(n_vars).astype(np.float64) * 0.01 * np.exp(-0.3 * t)
        llr = base * scale + noise
        llr_trace.append(llr)
        energy_trace.append(float(np.sum(np.abs(llr - base)) + 0.1 * np.exp(-0.3 * t)))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def generate_high_noise_trace(
    n_vars: int, n_iters: int, seed: int,
) -> dict:
    """High-noise trace — large magnitude perturbations at every step."""
    rng = _make_rng(seed, "high_noise")
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        llr = rng.standard_normal(n_vars).astype(np.float64) * 10.0
        llr_trace.append(llr)
        energy_trace.append(float(np.sum(np.abs(llr))))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def generate_oscillating_trace(
    n_vars: int, n_iters: int, seed: int, period: int = 3,
) -> dict:
    """Periodic oscillating trace — signs flip with fixed period."""
    rng = _make_rng(seed, "oscillating")
    base = rng.standard_normal(n_vars).astype(np.float64)
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        sign = 1.0 if (t % period) < (period // 2 + 1) else -1.0
        llr = base * sign + rng.standard_normal(n_vars).astype(np.float64) * 0.01
        llr_trace.append(llr)
        energy_trace.append(float(5.0 + 0.5 * np.sin(2.0 * np.pi * t / period)))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def generate_long_iteration_trace(
    n_vars: int, n_iters: int, seed: int,
) -> dict:
    """Long iteration trace — slow convergence over many iterations."""
    rng = _make_rng(seed, "long_iter")
    base = rng.standard_normal(n_vars).astype(np.float64)
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        decay = 1.0 / (1.0 + 0.01 * t)
        noise = rng.standard_normal(n_vars).astype(np.float64) * decay * 0.1
        llr = base * (1.0 + decay) + noise
        llr_trace.append(llr)
        energy_trace.append(float(10.0 * decay + 0.01 * rng.standard_normal()))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def generate_small_window_trace(
    n_vars: int, seed: int,
) -> dict:
    """Minimal trace — only 3 iterations (below default tail_window)."""
    rng = _make_rng(seed, "small_window")
    llr_trace = []
    energy_trace = []
    for t in range(3):
        llr = rng.standard_normal(n_vars).astype(np.float64)
        llr_trace.append(llr)
        energy_trace.append(float(10.0 - t * 3.0))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def generate_large_window_trace(
    n_vars: int, n_iters: int, seed: int,
) -> dict:
    """Large trace with many iterations and high dimensionality."""
    rng = _make_rng(seed, "large_window")
    base = rng.standard_normal(n_vars).astype(np.float64)
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        decay = np.exp(-0.05 * t)
        llr = base * (1.0 + decay) + rng.standard_normal(n_vars).astype(np.float64) * 0.01
        llr_trace.append(llr)
        energy_trace.append(float(50.0 * decay))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def generate_pathological_trace(
    n_vars: int, n_iters: int, seed: int,
) -> dict:
    """Pathological trace — extreme values, near-zero, sign changes."""
    rng = _make_rng(seed, "pathological")
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        llr = np.zeros(n_vars, dtype=np.float64)
        # Mix of extreme, near-zero, and normal values
        quarter = max(1, n_vars // 4)
        llr[:quarter] = 1e10 * ((-1.0) ** t)
        llr[quarter : 2 * quarter] = 1e-15
        llr[2 * quarter : 3 * quarter] = rng.standard_normal(
            min(quarter, n_vars - 2 * quarter)
        ).astype(np.float64)
        remaining = n_vars - 3 * quarter
        if remaining > 0:
            llr[3 * quarter :] = rng.standard_normal(remaining).astype(np.float64) * 100.0
        llr_trace.append(llr)
        energy_trace.append(float(np.sum(np.abs(llr))))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def generate_diverging_trace(
    n_vars: int, n_iters: int, seed: int,
) -> dict:
    """Diverging trace — energy grows exponentially."""
    rng = _make_rng(seed, "diverging")
    base = rng.standard_normal(n_vars).astype(np.float64)
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        scale = np.exp(0.2 * t)
        llr = base * scale + rng.standard_normal(n_vars).astype(np.float64) * 0.1 * scale
        llr_trace.append(llr)
        energy_trace.append(float(np.sum(np.abs(llr))))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


# ── Stress scenario registry ────────────────────────────────────────

STRESS_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "converging_baseline",
        "description": "Smooth convergence (control)",
        "generator": generate_converging_trace,
        "kwargs": {"n_vars": 50, "n_iters": 30},
    },
    {
        "name": "high_noise",
        "description": "High-noise traces with large perturbations",
        "generator": generate_high_noise_trace,
        "kwargs": {"n_vars": 50, "n_iters": 30},
    },
    {
        "name": "oscillating_period3",
        "description": "Oscillating traces with period 3",
        "generator": generate_oscillating_trace,
        "kwargs": {"n_vars": 50, "n_iters": 30, "period": 3},
    },
    {
        "name": "oscillating_period2",
        "description": "Oscillating traces with period 2",
        "generator": generate_oscillating_trace,
        "kwargs": {"n_vars": 50, "n_iters": 30, "period": 2},
    },
    {
        "name": "long_iteration",
        "description": "Long iteration count (200 iters)",
        "generator": generate_long_iteration_trace,
        "kwargs": {"n_vars": 50, "n_iters": 200},
    },
    {
        "name": "small_window",
        "description": "Edge case: only 3 iterations",
        "generator": generate_small_window_trace,
        "kwargs": {"n_vars": 50},
    },
    {
        "name": "large_window",
        "description": "Large trace: 100 iters, 200 vars",
        "generator": generate_large_window_trace,
        "kwargs": {"n_vars": 200, "n_iters": 100},
    },
    {
        "name": "pathological_extreme",
        "description": "Pathological: extreme values, near-zero, sign flips",
        "generator": generate_pathological_trace,
        "kwargs": {"n_vars": 100, "n_iters": 30},
    },
    {
        "name": "diverging",
        "description": "Energy diverges exponentially",
        "generator": generate_diverging_trace,
        "kwargs": {"n_vars": 50, "n_iters": 30},
    },
]


# ── Outcome classification ──────────────────────────────────────────

def classify_outcome(result: dict) -> str:
    """Classify benchmark outcome using existing BP regime + energy trend.

    Returns one of: converged, oscillatory, unstable, diverged.
    Uses the regime from ``classify_bp_regime`` plus energy-based checks.
    """
    regime = result.get("regime", "")
    metrics = result.get("metrics", {})

    # Map BP regimes to outcome labels
    if regime == "stable_convergence":
        return "converged"
    if regime == "oscillatory_convergence":
        return "oscillatory"
    if regime in ("metastable_state", "correction_cycling"):
        return "unstable"
    if regime == "chaotic_behavior":
        return "diverged"
    if regime == "trapping_set_regime":
        # Check energy trend to distinguish unstable from diverged
        eds = metrics.get("eds_descent_fraction")
        if eds is not None and eds < 0.3:
            return "diverged"
        return "unstable"

    # Fallback: should not reach here with valid regimes
    return "unstable"


# ── Single-run benchmark ────────────────────────────────────────────

def run_single_benchmark(
    scenario_name: str,
    llr_trace: list,
    energy_trace: list,
    seed: int,
    params: Optional[dict] = None,
) -> dict:
    """Run compute_bp_dynamics_metrics and record benchmark data.

    Returns a deterministic dict with all required fields.
    """
    start = time.perf_counter()
    result = compute_bp_dynamics_metrics(
        llr_trace=llr_trace,
        energy_trace=energy_trace,
        params=params,
    )
    wall_time = time.perf_counter() - start

    outcome = classify_outcome(result)

    return {
        "scenario": scenario_name,
        "seed": seed,
        "n_iterations": len(energy_trace),
        "n_vars": len(llr_trace[0]) if llr_trace else 0,
        "wall_time_seconds": round(wall_time, 6),
        "regime": result["regime"],
        "outcome": outcome,
        "metrics": result["metrics"],
        "evidence": result["evidence"],
        "params": params if params is not None else {},
    }


# ── Full benchmark suite ────────────────────────────────────────────

def run_benchmark_suite(
    master_seed: int = 42,
    scenarios: Optional[list[dict]] = None,
    params: Optional[dict] = None,
) -> dict:
    """Run all stress scenarios and return deterministic JSON artifact.

    Parameters
    ----------
    master_seed : int
        Master seed for deterministic sub-seed derivation.
    scenarios : list[dict] or None
        Override scenario list (default: STRESS_SCENARIOS).
    params : dict or None
        Override params passed to compute_bp_dynamics_metrics.

    Returns
    -------
    dict
        Complete benchmark artifact (JSON-serializable, deterministic ordering).
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    runs: list[dict] = []
    outcome_counts: dict[str, int] = {
        "converged": 0,
        "oscillatory": 0,
        "unstable": 0,
        "diverged": 0,
    }

    for scenario in scenarios:
        name = scenario["name"]
        gen = scenario["generator"]
        kwargs = dict(scenario.get("kwargs", {}))
        sub_seed = _derive_seed(master_seed, name)
        kwargs["seed"] = sub_seed

        trace = gen(**kwargs)

        record = run_single_benchmark(
            scenario_name=name,
            llr_trace=trace["llr_trace"],
            energy_trace=trace["energy_trace"],
            seed=sub_seed,
            params=params,
        )
        record["description"] = scenario.get("description", "")
        runs.append(record)
        outcome_counts[record["outcome"]] = (
            outcome_counts.get(record["outcome"], 0) + 1
        )

    artifact = {
        "benchmark_version": _BENCHMARK_VERSION,
        "master_seed": master_seed,
        "git_hash": _git_short_hash(),
        "n_scenarios": len(runs),
        "outcome_summary": outcome_counts,
        "runs": runs,
    }
    return artifact


def serialize_artifact(artifact: dict) -> str:
    """Serialize benchmark artifact to deterministic JSON string."""
    return json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True)
