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

_BENCHMARK_VERSION = "68.7.2"


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


# ── BP trace fidelity proxies ────────────────────────────────────────
#
# These are *not* quantum state fidelity.  They measure iteration-to-iteration
# consistency of BP LLR traces as a proxy for convergence quality.

_NORM_EPS = 1e-300  # float64-safe guard against division by zero


def compute_mean_cosine_fidelity(llr_trace: list) -> Optional[float]:
    """Mean cosine similarity between consecutive normalized LLR vectors.

    Definition:
        For consecutive vectors v_t, v_{t+1}:
            cos(t) = dot(v_t, v_{t+1}) / (||v_t|| * ||v_{t+1}|| + eps)
        Result = mean over all consecutive pairs.

    Returns None if fewer than 2 iterations.
    This is a BP trace fidelity proxy, not quantum state fidelity.
    """
    if len(llr_trace) < 2:
        return None
    cosines = []
    for i in range(len(llr_trace) - 1):
        v_t = np.asarray(llr_trace[i], dtype=np.float64).ravel()
        v_next = np.asarray(llr_trace[i + 1], dtype=np.float64).ravel()
        norm_t = np.linalg.norm(v_t)
        norm_next = np.linalg.norm(v_next)
        norm_t = max(norm_t, _NORM_EPS)
        norm_next = max(norm_next, _NORM_EPS)
        denom = norm_t * norm_next
        cosines.append(float(np.dot(v_t, v_next) / denom))
    return float(np.mean(cosines))


def compute_mean_sign_fidelity(llr_trace: list) -> Optional[float]:
    """Mean sign agreement fraction between consecutive LLR vectors.

    Definition:
        For consecutive vectors v_t, v_{t+1}:
            sign_agree(t) = fraction of coordinates where sign(v_t) == sign(v_{t+1})
        sign(x) = +1 if x >= 0 else -1  (matches BP dynamics convention).
        Result = mean over all consecutive pairs.

    Returns None if fewer than 2 iterations.
    This is a BP trace fidelity proxy, not quantum state fidelity.
    """
    if len(llr_trace) < 2:
        return None
    fractions = []
    for i in range(len(llr_trace) - 1):
        v_t = np.asarray(llr_trace[i], dtype=np.float64).ravel()
        v_next = np.asarray(llr_trace[i + 1], dtype=np.float64).ravel()
        # sign: +1 for >= 0, -1 for < 0 (consistent with bp_dynamics._sign)
        s_t = np.where(v_t >= 0, 1, -1)
        s_next = np.where(v_next >= 0, 1, -1)
        fractions.append(float(np.mean(s_t == s_next)))
    return float(np.mean(fractions))


def compute_llr_stats(llr_trace: list) -> dict:
    """Compute max and mean absolute LLR across the entire trace.

    Returns dict with ``max_abs_llr`` and ``mean_abs_llr``.
    Returns zeros if trace is empty.
    """
    if not llr_trace:
        return {"max_abs_llr": 0.0, "mean_abs_llr": 0.0}
    abs_vals = [np.abs(np.asarray(v, dtype=np.float64).ravel()) for v in llr_trace]
    all_abs = np.concatenate(abs_vals)
    return {
        "max_abs_llr": float(np.max(all_abs)),
        "mean_abs_llr": float(np.mean(all_abs)),
    }


def compute_mean_quantum_fidelity(llr_trace: list) -> Optional[float]:
    """Mean quantum-style fidelity between consecutive normalized LLR vectors.

    Embeds each LLR vector into a Hilbert-style unit vector:
        psi_t = v_t / (||v_t|| + eps)
    Then computes:
        F(t) = (dot(psi_t, psi_{t+1}))^2
    Result = mean over all consecutive pairs.

    IMPORTANT: This is NOT true quantum state fidelity.  It is a
    Hilbert-space proxy derived from normalized LLR vectors.  It measures
    squared directional overlap, not convergence quality.

    Range: [0, 1].  F=1 means identical or anti-parallel directions;
    F=0 means orthogonal directions.

    Returns None if fewer than 2 iterations.
    """
    if len(llr_trace) < 2:
        return None
    fidelities = []
    for i in range(len(llr_trace) - 1):
        v_t = np.asarray(llr_trace[i], dtype=np.float64).ravel()
        v_next = np.asarray(llr_trace[i + 1], dtype=np.float64).ravel()
        norm_t = np.linalg.norm(v_t) + _NORM_EPS
        norm_next = np.linalg.norm(v_next) + _NORM_EPS
        overlap = float(np.dot(v_t, v_next) / (norm_t * norm_next))
        fidelities.append(overlap * overlap)
    return float(np.mean(fidelities))


def interpret_fidelity(
    cosine: Optional[float], sign: Optional[float],
) -> str:
    """Deterministic interpretation of fidelity metrics.

    Returns "high", "medium", or "low".
    - high:   cosine >= 0.95 AND sign >= 0.90
    - medium: cosine >= 0.70 AND sign >= 0.70
    - low:    otherwise
    """
    if cosine is None or sign is None:
        return "low"
    if cosine >= 0.95 and sign >= 0.90:
        return "high"
    if cosine >= 0.70 and sign >= 0.70:
        return "medium"
    return "low"


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

    cosine_fid = compute_mean_cosine_fidelity(llr_trace)
    sign_fid = compute_mean_sign_fidelity(llr_trace)
    quantum_fid = compute_mean_quantum_fidelity(llr_trace)
    llr_stats = compute_llr_stats(llr_trace)

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
        "mean_cosine_fidelity": cosine_fid,
        "mean_sign_fidelity": sign_fid,
        "mean_quantum_fidelity": quantum_fid,
        "max_abs_llr": llr_stats["max_abs_llr"],
        "mean_abs_llr": llr_stats["mean_abs_llr"],
        "fidelity_interpretation": interpret_fidelity(cosine_fid, sign_fid),
    }


# ── Gap analysis ────────────────────────────────────────────────────

def _compute_gap_analysis(runs: list[dict]) -> dict:
    """Compute aggregate gap analysis from benchmark runs.

    Deterministic: uses scenario name for tie-breaking (lexicographic).
    """
    if not runs:
        return {}

    # Slowest scenario (by wall time)
    slowest = max(runs, key=lambda r: (r["wall_time_seconds"], r["scenario"]))

    # Most unstable: prefer diverged > unstable > oscillatory > converged
    instability_rank = {"diverged": 3, "unstable": 2, "oscillatory": 1, "converged": 0}
    most_unstable = max(
        runs,
        key=lambda r: (instability_rank.get(r["outcome"], 0), r["scenario"]),
    )

    # Lowest / highest fidelity (by cosine, then sign, then name for ties)
    runs_with_fidelity = [r for r in runs if r.get("mean_cosine_fidelity") is not None]
    if runs_with_fidelity:
        lowest_fid = min(
            runs_with_fidelity,
            key=lambda r: (r["mean_cosine_fidelity"], r["mean_sign_fidelity"], r["scenario"]),
        )
        highest_fid = max(
            runs_with_fidelity,
            key=lambda r: (r["mean_cosine_fidelity"], r["mean_sign_fidelity"], r["scenario"]),
        )
        lowest_fid_entry = {
            "scenario": lowest_fid["scenario"],
            "mean_cosine_fidelity": lowest_fid["mean_cosine_fidelity"],
            "mean_sign_fidelity": lowest_fid["mean_sign_fidelity"],
        }
        highest_fid_entry = {
            "scenario": highest_fid["scenario"],
            "mean_cosine_fidelity": highest_fid["mean_cosine_fidelity"],
            "mean_sign_fidelity": highest_fid["mean_sign_fidelity"],
        }
    else:
        lowest_fid_entry = None
        highest_fid_entry = None

    return {
        "total_scenarios": len(runs),
        "slowest_scenario": {
            "scenario": slowest["scenario"],
            "wall_time_seconds": slowest["wall_time_seconds"],
        },
        "most_unstable_scenario": {
            "scenario": most_unstable["scenario"],
            "outcome": most_unstable["outcome"],
            "regime": most_unstable["regime"],
        },
        "lowest_fidelity_scenario": lowest_fid_entry,
        "highest_fidelity_scenario": highest_fid_entry,
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

    # ── Gap analysis aggregate ─────────────────────────────────────
    gap_analysis = _compute_gap_analysis(runs)

    artifact = {
        "benchmark_version": _BENCHMARK_VERSION,
        "master_seed": master_seed,
        "git_hash": _git_short_hash(),
        "n_scenarios": len(runs),
        "outcome_summary": outcome_counts,
        "gap_analysis": gap_analysis,
        "runs": runs,
    }
    return artifact


def serialize_artifact(artifact: dict) -> str:
    """Serialize benchmark artifact to deterministic JSON string."""
    return json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True)
