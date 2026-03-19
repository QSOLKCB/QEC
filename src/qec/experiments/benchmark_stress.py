"""Deterministic benchmark + stress framework for compute_bp_dynamics_metrics.

Generates 9 synthetic scenarios, runs them through the diagnostics pipeline,
and produces deterministic JSON-serializable results with fidelity metrics.

Version: v68.7.2
"""

import hashlib
import json
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.qec.diagnostics.bp_dynamics import compute_bp_dynamics_metrics


# ── Deterministic seed derivation ────────────────────────────────────────


def _derive_seed(label: str) -> int:
    """SHA-256 → first 8 bytes → int seed.  Fully deterministic."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return struct.unpack("<Q", digest[:8])[0]


# ── Scenario generators ─────────────────────────────────────────────────


def _make_converging_baseline(
    rng: np.random.Generator, n_vars: int, n_iters: int
) -> dict:
    """Smoothly converging LLR trace with monotonically decreasing energy."""
    base = rng.standard_normal(n_vars).astype(np.float64)
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        decay = 0.9 ** t
        noise = rng.standard_normal(n_vars).astype(np.float64) * 0.01 * decay
        llr = base + noise * decay
        llr_trace.append(llr)
        energy_trace.append(float(10.0 * decay))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def _make_high_noise(
    rng: np.random.Generator, n_vars: int, n_iters: int
) -> dict:
    """High-noise LLR trace — large random perturbations each step."""
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        llr = rng.standard_normal(n_vars).astype(np.float64) * 5.0
        llr_trace.append(llr)
        energy_trace.append(float(8.0 + rng.standard_normal() * 2.0))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def _make_oscillating_period3(
    rng: np.random.Generator, n_vars: int, n_iters: int
) -> dict:
    """Period-3 oscillation: cycles through 3 base vectors."""
    bases = [rng.standard_normal(n_vars).astype(np.float64) for _ in range(3)]
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        phase = t % 3
        noise = rng.standard_normal(n_vars).astype(np.float64) * 0.01
        llr_trace.append(bases[phase] + noise)
        energy_trace.append(float(5.0 + np.sin(2.0 * np.pi * t / 3.0)))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def _make_oscillating_period2(
    rng: np.random.Generator, n_vars: int, n_iters: int
) -> dict:
    """Period-2 oscillation: MUST flip sign each step.

    sign = 1.0 if (t % 2 == 0) else -1.0
    """
    base = rng.standard_normal(n_vars).astype(np.float64)
    # Ensure base has nonzero magnitude for meaningful sign flips
    base = base + 0.1 * np.sign(base)
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        sign = 1.0 if (t % 2 == 0) else -1.0
        noise = rng.standard_normal(n_vars).astype(np.float64) * 0.001
        llr_trace.append(sign * base + noise)
        energy_trace.append(float(5.0 + 0.5 * sign))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def _make_long_iteration(
    rng: np.random.Generator, n_vars: int, n_iters: int
) -> dict:
    """Long iteration run (3x normal iterations), slow convergence."""
    extended_iters = n_iters * 3
    base = rng.standard_normal(n_vars).astype(np.float64)
    llr_trace = []
    energy_trace = []
    for t in range(extended_iters):
        decay = 0.98 ** t
        noise = rng.standard_normal(n_vars).astype(np.float64) * 0.05 * decay
        llr_trace.append(base + noise)
        energy_trace.append(float(10.0 * decay))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def _make_small_window(
    rng: np.random.Generator, n_vars: int, _n_iters: int
) -> dict:
    """Very short trace — only 4 iterations."""
    llr_trace = []
    energy_trace = []
    for t in range(4):
        llr = rng.standard_normal(n_vars).astype(np.float64)
        llr_trace.append(llr)
        energy_trace.append(float(5.0 - t))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def _make_large_window(
    rng: np.random.Generator, n_vars: int, n_iters: int
) -> dict:
    """Large trace with many iterations, gradual convergence."""
    extended_iters = n_iters * 5
    base = rng.standard_normal(n_vars).astype(np.float64)
    llr_trace = []
    energy_trace = []
    for t in range(extended_iters):
        decay = 0.995 ** t
        noise = rng.standard_normal(n_vars).astype(np.float64) * 0.02 * decay
        llr_trace.append(base + noise)
        energy_trace.append(float(20.0 * decay))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def _make_pathological_extreme(
    rng: np.random.Generator, n_vars: int, n_iters: int
) -> dict:
    """Pathological scenario with extreme values and sparse structure.

    Critical fix:
        size = max(0, n_vars - 2 * quarter)
        size = min(quarter, size)
        if size > 0: fill slice with rng.standard_normal(size)
    """
    llr_trace = []
    energy_trace = []
    quarter = max(1, n_vars // 4)
    for t in range(n_iters):
        llr = np.zeros(n_vars, dtype=np.float64)
        # First quarter: extreme positive
        llr[:quarter] = 1e6
        # Second quarter: extreme negative
        llr[quarter:2 * quarter] = -1e6
        # Middle region: random fill with size constraints
        size = max(0, n_vars - 2 * quarter)
        size = min(quarter, size)
        if size > 0:
            llr[2 * quarter:2 * quarter + size] = rng.standard_normal(size).astype(np.float64)
        # Flip sign on odd iterations
        if t % 2 == 1:
            llr = -llr
        llr_trace.append(llr)
        energy_trace.append(float(1e6 * ((-1.0) ** t)))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


def _make_diverging(
    rng: np.random.Generator, n_vars: int, n_iters: int
) -> dict:
    """Diverging trace — energy grows exponentially."""
    base = rng.standard_normal(n_vars).astype(np.float64)
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        growth = 1.1 ** t
        noise = rng.standard_normal(n_vars).astype(np.float64) * 0.1
        llr_trace.append(base * growth + noise)
        energy_trace.append(float(1.0 * growth))
    return {"llr_trace": llr_trace, "energy_trace": energy_trace}


# ── Scenario registry ────────────────────────────────────────────────────

SCENARIOS = [
    ("converging_baseline", _make_converging_baseline),
    ("high_noise", _make_high_noise),
    ("oscillating_period3", _make_oscillating_period3),
    ("oscillating_period2", _make_oscillating_period2),
    ("long_iteration", _make_long_iteration),
    ("small_window", _make_small_window),
    ("large_window", _make_large_window),
    ("pathological_extreme", _make_pathological_extreme),
    ("diverging", _make_diverging),
]


# ── Fidelity metrics ────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with individual norm clamping."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    # Clamp norms individually
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    val = float(np.dot(a, b)) / (norm_a * norm_b)
    return float(np.clip(val, -1.0, 1.0))


def _sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of elements with matching signs."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) == 0:
        return 1.0
    return float(np.mean(np.sign(a) == np.sign(b)))


def _quantum_proxy(a: np.ndarray, b: np.ndarray) -> float:
    """Quantum fidelity proxy: (normalized dot product)^2."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    dot_normalized = float(np.dot(a, b)) / (norm_a * norm_b)
    return float(np.clip(dot_normalized, -1.0, 1.0) ** 2)


def compute_fidelity(llr_trace: list) -> dict:
    """Compute fidelity metrics between first and last LLR vectors."""
    if len(llr_trace) < 2:
        return {"cosine": 0.0, "sign_agreement": 1.0, "quantum_proxy": 0.0}
    first = np.asarray(llr_trace[0], dtype=np.float64).ravel()
    last = np.asarray(llr_trace[-1], dtype=np.float64).ravel()
    return {
        "cosine": _cosine_similarity(first, last),
        "sign_agreement": _sign_agreement(first, last),
        "quantum_proxy": _quantum_proxy(first, last),
    }


# ── Classification post-processing ──────────────────────────────────────


def classify_with_fallback(regime: str) -> str:
    """Map unknown regimes to 'unstable'."""
    known = {
        "stable_convergence",
        "oscillatory_convergence",
        "metastable_state",
        "trapping_set_regime",
        "correction_cycling",
        "chaotic_behavior",
    }
    if regime in known:
        return regime
    return "unstable"


# ── Main benchmark runner ────────────────────────────────────────────────


def run_benchmark_stress(
    n_vars: int = 50,
    n_iters: int = 30,
    base_seed_label: str = "benchmark_stress_v68.7.2",
) -> dict:
    """Run all 9 benchmark scenarios deterministically.

    Parameters
    ----------
    n_vars : int
        Number of LLR variables per iteration.
    n_iters : int
        Base number of iterations (some scenarios scale this).
    base_seed_label : str
        Label for SHA-256 seed derivation.

    Returns
    -------
    dict
        JSON-serializable results with scenario metrics, regimes,
        fidelity, and timing.
    """
    results = []

    for scenario_name, generator_fn in SCENARIOS:
        seed_label = f"{base_seed_label}:{scenario_name}"
        seed = _derive_seed(seed_label)
        rng = np.random.Generator(np.random.PCG64(seed))

        # Generate scenario
        t_start = time.monotonic()
        scenario_data = generator_fn(rng, n_vars, n_iters)
        t_gen = time.monotonic() - t_start

        # Run diagnostics
        t_start = time.monotonic()
        diagnostics_result = compute_bp_dynamics_metrics(
            llr_trace=scenario_data["llr_trace"],
            energy_trace=scenario_data["energy_trace"],
        )
        t_diag = time.monotonic() - t_start

        # Fidelity
        fidelity = compute_fidelity(scenario_data["llr_trace"])

        # Classification with fallback
        regime = classify_with_fallback(diagnostics_result["regime"])

        results.append({
            "scenario": scenario_name,
            "n_vars": n_vars,
            "n_iters": len(scenario_data["llr_trace"]),
            "seed": seed,
            "regime": regime,
            "metrics": diagnostics_result["metrics"],
            "evidence": diagnostics_result["evidence"],
            "fidelity": fidelity,
            "timing": {
                "generation_s": t_gen,
                "diagnostics_s": t_diag,
            },
        })

    return {
        "version": "v68.7.2",
        "base_seed_label": base_seed_label,
        "n_vars": n_vars,
        "n_iters_base": n_iters,
        "n_scenarios": len(results),
        "scenarios": results,
    }


def results_to_json(results: dict) -> str:
    """Serialize results to deterministic JSON (sorted keys)."""

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    return json.dumps(results, sort_keys=True, indent=2, default=_default)
