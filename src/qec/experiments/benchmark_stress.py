"""Deterministic benchmark + stress framework for compute_bp_dynamics_metrics.

Generates 9 synthetic scenarios, runs them through the diagnostics pipeline,
and produces deterministic JSON-serializable results with fidelity metrics.

Version: v68.9.0
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


# ── Dark-state detection ─────────────────────────────────────────────────


_DARK_EPS: float = 1e-6


def compute_dark_state_mask(
    llr_trace: List[np.ndarray],
    eps: float = _DARK_EPS,
) -> List[np.ndarray]:
    """Compute per-timestep boolean masks of dark-stable nodes.

    A node *i* at iteration *t* is dark-stable iff:
      - sign(v_i^t) == sign(v_i^{t-1})
      - abs(v_i^t - v_i^{t-1}) < eps

    Parameters
    ----------
    llr_trace : list[np.ndarray]
        LLR vectors per BP iteration (float64).
    eps : float
        Absolute tolerance for magnitude stability (default 1e-6).

    Returns
    -------
    list[np.ndarray]
        Boolean masks (same length / shapes as *llr_trace*).
        First timestep (t=0) is all-False — no previous state exists.
    """
    if len(llr_trace) == 0:
        return []

    # Fail fast on shape mismatch
    ref_shape = llr_trace[0].shape
    for idx, arr in enumerate(llr_trace):
        assert arr.shape == ref_shape, (
            f"llr_trace shape mismatch at index {idx}: "
            f"expected {ref_shape}, got {arr.shape}"
        )

    # t=0: no previous state → all False
    masks: List[np.ndarray] = [
        np.zeros(ref_shape, dtype=np.bool_)
    ]
    for t in range(1, len(llr_trace)):
        prev = np.asarray(llr_trace[t - 1], dtype=np.float64)
        curr = np.asarray(llr_trace[t], dtype=np.float64)
        same_sign = np.sign(prev) == np.sign(curr)
        small_delta = np.abs(curr - prev) < eps
        masks.append(same_sign & small_delta)
    return masks


def _dark_fractions(masks: List[np.ndarray]) -> List[float]:
    """Return dark-fraction per timestep."""
    fracs: List[float] = []
    for m in masks:
        n = m.size
        if n == 0:
            fracs.append(0.0)
        else:
            fracs.append(float(np.sum(m)) / float(n))
    return fracs


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


# ── Decoder genome ───────────────────────────────────────────────────────


def default_decoder_genome() -> dict:
    """Return the default decoder genome configuration.

    A decoder genome is a deterministic configuration of decoding behavior,
    applied as a lightweight transformation layer on LLR traces.
    """
    return {
        "alphabet": "binary",
        "clip_value": None,
        "damping": 0.0,
        "dark_skip": False,
    }


def apply_decoder_genome(
    llr_trace: List[np.ndarray],
    genome: dict,
) -> List[np.ndarray]:
    """Apply genome transformations to an LLR trace.

    Operations applied in order:
    1. Clipping (if clip_value is not None)
    2. Damping (if damping > 0)
    3. Dark skip (if dark_skip is True) — freeze dark-stable nodes
    4. Ternary projection (if alphabet == "ternary")

    Returns a new list (input is not mutated).
    """
    if len(llr_trace) == 0:
        return []

    clip_value = genome.get("clip_value", None)
    damping = float(genome.get("damping", 0.0))
    dark_skip = bool(genome.get("dark_skip", False))
    alphabet = genome.get("alphabet", "binary")

    # Deep copy to avoid mutation
    out = [np.array(v, dtype=np.float64) for v in llr_trace]

    # 1. Clipping
    if clip_value is not None:
        cv = float(clip_value)
        for t in range(len(out)):
            out[t] = np.clip(out[t], -cv, cv)

    # 2. Damping: v_t = (1 - damping) * v_t + damping * v_{t-1}
    if damping > 0.0:
        for t in range(1, len(out)):
            out[t] = (1.0 - damping) * out[t] + damping * out[t - 1]

    # 3. Dark skip: freeze dark-stable nodes to previous value
    if dark_skip:
        masks = compute_dark_state_mask(out)
        for t in range(1, len(out)):
            dark = masks[t]
            if np.any(dark):
                out[t][dark] = out[t - 1][dark]

    # 4. Ternary projection
    if alphabet == "ternary":
        for t in range(len(out)):
            v = out[t]
            result = np.zeros_like(v, dtype=np.float64)
            result[v > 1e-12] = 1.0
            result[v < -1e-12] = -1.0
            out[t] = result

    return out


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


# ── Single-scenario runner ───────────────────────────────────────────────


def run_single_benchmark(
    scenario_name: str,
    generator_fn,
    rng: np.random.Generator,
    n_vars: int,
    n_iters: int,
    genome: Optional[dict] = None,
) -> dict:
    """Run one benchmark scenario and return metrics including dark-state fractions.

    Parameters
    ----------
    genome : dict or None
        Decoder genome configuration.  If None, uses default_decoder_genome().

    Returns
    -------
    dict
        Contains scenario metrics, regime, fidelity, dark-state fractions,
        genome, and timing information.
    """
    if genome is None:
        genome = default_decoder_genome()

    t_start = time.monotonic()
    scenario_data = generator_fn(rng, n_vars, n_iters)
    t_gen = time.monotonic() - t_start

    # Apply genome transformation before diagnostics
    transformed_trace = apply_decoder_genome(
        scenario_data["llr_trace"], genome,
    )

    t_start = time.monotonic()
    diagnostics_result = compute_bp_dynamics_metrics(
        llr_trace=transformed_trace,
        energy_trace=scenario_data["energy_trace"],
    )
    t_diag = time.monotonic() - t_start

    fidelity = compute_fidelity(transformed_trace)
    regime = classify_with_fallback(diagnostics_result["regime"])

    # Dark-state invariants
    dark_masks = compute_dark_state_mask(transformed_trace)
    dark_fracs = _dark_fractions(dark_masks)
    if len(dark_fracs) > 0:
        mean_dark_fraction = float(np.mean(dark_fracs))
        final_dark_fraction = dark_fracs[-1]
    else:
        mean_dark_fraction = 0.0
        final_dark_fraction = 0.0

    return {
        "scenario": scenario_name,
        "n_vars": n_vars,
        "n_iters": len(transformed_trace),
        "regime": regime,
        "metrics": diagnostics_result["metrics"],
        "evidence": diagnostics_result["evidence"],
        "fidelity": fidelity,
        "genome": genome,
        "mean_dark_fraction": mean_dark_fraction,
        "final_dark_fraction": final_dark_fraction,
        "timing": {
            "generation_s": t_gen,
            "diagnostics_s": t_diag,
        },
    }


# ── Main benchmark runner ────────────────────────────────────────────────


def run_benchmark_stress(
    n_vars: int = 50,
    n_iters: int = 30,
    base_seed_label: str = "benchmark_stress_v68.7.2",
    genome: Optional[dict] = None,
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
    genome : dict or None
        Decoder genome configuration.  If None, uses default_decoder_genome().

    Returns
    -------
    dict
        JSON-serializable results with scenario metrics, regimes,
        fidelity, genome, and timing.
    """
    results = []

    for scenario_name, generator_fn in SCENARIOS:
        seed_label = f"{base_seed_label}:{scenario_name}"
        seed = _derive_seed(seed_label)
        rng = np.random.Generator(np.random.PCG64(seed))

        result = run_single_benchmark(
            scenario_name, generator_fn, rng, n_vars, n_iters,
            genome=genome,
        )
        result["seed"] = seed
        results.append(result)

    return {
        "version": "v68.9.0",
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
