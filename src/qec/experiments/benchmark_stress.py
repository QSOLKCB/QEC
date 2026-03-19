"""Deterministic benchmark + stress framework for compute_bp_dynamics_metrics.

Generates 9 synthetic scenarios, runs them through the diagnostics pipeline,
and produces deterministic JSON-serializable results with fidelity metrics.

Version: v70.0.0
"""

import hashlib
import json
import struct
import time
from collections import defaultdict
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


# Canonical key order for genome dicts — used by normalization and fingerprinting.
_GENOME_KEYS = ("alphabet", "clip_value", "damping", "dark_skip")


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


def normalize_decoder_genome(genome: Optional[dict]) -> dict:
    """Normalize a genome dict to canonical form.

    - Starts from defaults, overlays provided values.
    - Rejects unknown keys.
    - Validates types and ranges.
    - Returns a new dict with plain Python JSON-safe types in fixed key order.
    """
    base = default_decoder_genome()
    if genome is None:
        return base

    unknown = set(genome.keys()) - set(_GENOME_KEYS)
    if unknown:
        raise ValueError(f"Unknown genome keys: {sorted(unknown)}")

    for key in _GENOME_KEYS:
        if key in genome:
            base[key] = genome[key]

    # --- validate alphabet ---
    if base["alphabet"] not in ("binary", "ternary"):
        raise ValueError(
            f"Invalid alphabet: {base['alphabet']!r}, must be 'binary' or 'ternary'"
        )
    base["alphabet"] = str(base["alphabet"])

    # --- validate clip_value ---
    cv = base["clip_value"]
    if cv is not None:
        try:
            cv = float(cv)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid clip_value: {base['clip_value']!r}")
        if cv < 0:
            raise ValueError(f"clip_value must be >= 0, got {cv}")
        base["clip_value"] = cv

    # --- validate damping ---
    try:
        d = float(base["damping"])
    except (TypeError, ValueError):
        raise ValueError(f"Invalid damping: {base['damping']!r}")
    if not (0.0 <= d < 1.0):
        raise ValueError(f"damping must be in [0, 1), got {d}")
    base["damping"] = d

    # --- validate dark_skip ---
    base["dark_skip"] = bool(base["dark_skip"])

    return base


def fingerprint_decoder_genome(genome: dict) -> str:
    """Return a short deterministic fingerprint for a canonical genome.

    Uses compact JSON serialization (fixed key order) + SHA-256, first 12 hex chars.
    The genome should be normalized first for stable results.
    """
    # Serialize in fixed key order (not sort_keys — we control order explicitly)
    ordered = {k: genome[k] for k in _GENOME_KEYS}
    payload = json.dumps(ordered, separators=(",", ":"), sort_keys=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:12]


def _prepare_genome_list(
    genome: Optional[dict] = None,
    genomes: Optional[List[dict]] = None,
) -> List[dict]:
    """Validate and normalize genome inputs for single or sweep mode.

    Rules:
    - Exactly one of genome, genomes, or neither (default genome).
    - If both provided → raise ValueError.
    - Each genome is normalized via normalize_decoder_genome.
    - Duplicate genome_ids → raise ValueError.
    - Returns a list of canonical genome dicts.
    """
    if genome is not None and genomes is not None:
        raise ValueError(
            "Cannot provide both 'genome' and 'genomes'. Use one or neither."
        )

    if genomes is not None:
        if not isinstance(genomes, list) or len(genomes) == 0:
            raise ValueError("'genomes' must be a non-empty list of genome dicts.")
        canonical_list = [normalize_decoder_genome(g) for g in genomes]
    elif genome is not None:
        canonical_list = [normalize_decoder_genome(genome)]
    else:
        canonical_list = [normalize_decoder_genome(None)]

    # Enforce uniqueness by genome_id
    seen_ids: Dict[str, int] = {}
    for idx, g in enumerate(canonical_list):
        gid = fingerprint_decoder_genome(g)
        if gid in seen_ids:
            raise ValueError(
                f"Duplicate genome_id {gid!r} at indices {seen_ids[gid]} and {idx}."
            )
        seen_ids[gid] = idx

    return canonical_list


def apply_decoder_genome(
    llr_trace: List[np.ndarray],
    genome: dict,
) -> List[np.ndarray]:
    """Apply genome transformations to an LLR trace.

    Operator order is fixed and must not be reordered:
        clipping → damping → dark_skip → ternary

    The genome is normalized on entry — callers need not pre-normalize.
    Returns a new list (input is not mutated).
    """
    if len(llr_trace) == 0:
        return []

    # Enforce canonical form: no .get() fallbacks, no hidden defaults
    genome = normalize_decoder_genome(genome)
    clip_value = genome["clip_value"]
    damping = genome["damping"]
    dark_skip = genome["dark_skip"]
    alphabet = genome["alphabet"]

    # Deep copy to float64 — input is never mutated
    out = [np.array(v, dtype=np.float64) for v in llr_trace]

    # Step 1: Clipping (clip_value=0 is valid → all values become 0)
    if clip_value is not None:
        for t in range(len(out)):
            out[t] = np.clip(out[t], -clip_value, clip_value)

    # Step 2: Damping — uses previous timestep only (t-1)
    if damping > 0.0:
        for t in range(1, len(out)):
            out[t] = (1.0 - damping) * out[t] + damping * out[t - 1]

    # Step 3: Dark skip — freeze dark-stable nodes to previous value.
    # Mask source: compute_dark_state_mask on the *current* transformed trace
    # (i.e. after clipping+damping), so the mask reflects the state that
    # damping/clipping have already produced.
    if dark_skip:
        masks = compute_dark_state_mask(out)
        for t in range(1, len(out)):
            dark = masks[t]
            if np.any(dark):
                out[t][dark] = out[t - 1][dark]

    # Step 4: Ternary projection (threshold ±1e-12)
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
        canonical genome, genome_id, and timing information.
    """
    # Normalize and detach: caller mutations cannot affect stored result
    canonical = normalize_decoder_genome(genome)
    genome_id = fingerprint_decoder_genome(canonical)

    t_start = time.monotonic()
    scenario_data = generator_fn(rng, n_vars, n_iters)
    t_gen = time.monotonic() - t_start

    # Apply genome transformation before diagnostics
    transformed_trace = apply_decoder_genome(
        scenario_data["llr_trace"], canonical,
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
        "genome": canonical,
        "genome_id": genome_id,
        "mean_dark_fraction": mean_dark_fraction,
        "final_dark_fraction": final_dark_fraction,
        "timing": {
            "generation_s": t_gen,
            "diagnostics_s": t_diag,
        },
    }


# ── Main benchmark runner ────────────────────────────────────────────────


def _run_single_genome_suite(
    n_vars: int,
    n_iters: int,
    base_seed_label: str,
    genome: dict,
) -> dict:
    """Run all scenarios for one canonical genome. Returns a single result dict."""
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
        "version": "v70.0.0",
        "base_seed_label": base_seed_label,
        "n_vars": n_vars,
        "n_iters_base": n_iters,
        "n_scenarios": len(results),
        "scenarios": results,
    }


def run_benchmark_stress(
    n_vars: int = 50,
    n_iters: int = 30,
    base_seed_label: str = "benchmark_stress_v70.0.0",
    genome: Optional[dict] = None,
    genomes: Optional[List[dict]] = None,
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
    genomes : list[dict] or None
        Multiple genome configurations for sweep mode.
        Mutually exclusive with ``genome``.

    Returns
    -------
    dict
        JSON-serializable results.  When a single genome is used,
        returns the same structure as before (mode="single").
        When multiple genomes are provided, returns
        {"mode": "sweep", "results": [...]}.
    """
    genome_list = _prepare_genome_list(genome, genomes)

    if len(genome_list) == 1 and genomes is None:
        # Single-genome mode: preserve exact v68.9.2 output structure
        suite = _run_single_genome_suite(
            n_vars, n_iters, base_seed_label, genome_list[0],
        )
        suite["mode"] = "single"
        suite["table"] = build_experiment_table(suite)
        suite["comparisons"] = build_pairwise_comparison(suite)
        suite["pareto"] = build_pareto_frontier(suite)
        suite["scores"] = build_scores(suite)
        return suite

    # Sweep mode: deterministic sequential iteration
    sweep_results = []
    for g in genome_list:
        suite = _run_single_genome_suite(
            n_vars, n_iters, base_seed_label, g,
        )
        sweep_results.append(suite)

    sweep_result = {
        "mode": "sweep",
        "results": sweep_results,
    }
    sweep_result["table"] = build_experiment_table(sweep_result)
    sweep_result["comparisons"] = build_pairwise_comparison(sweep_result)
    sweep_result["pareto"] = build_pareto_frontier(sweep_result)
    sweep_result["scores"] = build_scores(sweep_result)
    return sweep_result


# ── Aggregation layer ────────────────────────────────────────────────────

# Keys excluded from pairwise delta computation — single source of truth
# shared by build_experiment_table (as reserved keys) and build_pairwise_comparison.
_EXCLUDED_KEYS = frozenset({
    "genome_id", "scenario", "version", "base_seed_label",
    "n_vars", "n_iters_base",
})


def build_experiment_table(result: dict) -> list:
    """Convert benchmark result (single or sweep) into a flat list of row dicts.

    Each row corresponds to one (genome, scenario) pair with flattened metrics.
    Input is not mutated.  Order follows genome order then scenario order.

    Parameters
    ----------
    result : dict
        Output of ``run_benchmark_stress`` (mode="single" or mode="sweep").

    Returns
    -------
    list[dict]
        Flat rows with genome_id, scenario, version, base_seed_label,
        n_vars, n_iters_base, and all flattened metric values.

    Raises
    ------
    ValueError
        If ``mode`` is not "single" or "sweep", if a suite is missing
        ``scenarios``, or if a metric key collides with a reserved row key.
    """
    mode = result.get("mode")
    if mode not in ("single", "sweep"):
        raise ValueError(f"Invalid result mode: {mode!r}")

    if mode == "sweep":
        suites = result["results"]
    else:
        suites = [result]

    rows: list = []
    for suite in suites:
        if "scenarios" not in suite or not isinstance(suite["scenarios"], list):
            raise ValueError("Malformed suite: missing or invalid 'scenarios'")

        version = suite.get("version", "")
        base_seed_label = suite.get("base_seed_label", "")
        n_vars = suite.get("n_vars")
        n_iters_base = suite.get("n_iters_base")

        for scenario in suite["scenarios"]:
            row: dict = {
                "genome_id": scenario["genome_id"],
                "scenario": scenario["scenario"],
                "version": version,
                "base_seed_label": base_seed_label,
                "n_vars": n_vars,
                "n_iters_base": n_iters_base,
            }
            # Flatten metrics dict with collision guard
            metrics = scenario.get("metrics", {})
            if metrics:
                overlap = _EXCLUDED_KEYS & metrics.keys()
                if overlap:
                    raise ValueError(
                        f"Metric key collision with reserved keys: {sorted(overlap)}"
                    )
                for k, v in metrics.items():
                    row[k] = v
            rows.append(row)

    return rows


def build_pairwise_comparison(result: dict) -> list:
    """Compute pairwise metric deltas between genomes for each scenario.

    For each scenario, iterates all ordered pairs (i, j) where i < j
    and computes (row_j[metric] - row_i[metric]) for all numeric fields.

    Parameters
    ----------
    result : dict
        Output of ``run_benchmark_stress``.  Must contain ``"table"`` key.

    Returns
    -------
    list[dict]
        Each row: genome_a, genome_b, scenario, and ``<metric>_delta`` fields.
        Deterministic ordering: scenario order preserved, genome order preserved.

    Raises
    ------
    ValueError
        If ``result`` has no ``"table"`` key.
    """
    if "table" not in result:
        raise ValueError("Result dict missing 'table' key")

    table = result["table"]

    # Group rows by scenario, preserving insertion order
    scenario_groups: dict = defaultdict(list)
    for row in table:
        scenario_groups[row["scenario"]].append(row)

    comparisons: list = []
    for scenario, rows in scenario_groups.items():
        if len(rows) < 2:
            continue
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                row_i = rows[i]
                row_j = rows[j]
                comp: dict = {
                    "genome_a": row_i["genome_id"],
                    "genome_b": row_j["genome_id"],
                    "scenario": scenario,
                }
                # Compute deltas for all numeric fields
                for key in row_i:
                    if key in _EXCLUDED_KEYS:
                        continue
                    val_i = row_i[key]
                    val_j = row_j.get(key)
                    if isinstance(val_i, (int, float)) and isinstance(val_j, (int, float)):
                        comp[f"{key}_delta"] = val_j - val_i
                comparisons.append(comp)

    return comparisons


def _is_finite_numeric(val) -> bool:
    """Return True if val is a finite int or float (not NaN, not inf)."""
    if not isinstance(val, (int, float)):
        return False
    if isinstance(val, bool):
        return False
    try:
        return not (np.isnan(val) or np.isinf(val))
    except (TypeError, ValueError):
        return False


def build_pareto_frontier(result: dict) -> list:
    """Identify Pareto-optimal genomes per scenario based on numeric metrics.

    For each scenario, a genome is Pareto-optimal (non-dominated) if no other
    genome is at least as good on all numeric metrics and strictly better on
    at least one.

    Parameters
    ----------
    result : dict
        Output of ``run_benchmark_stress``.  Must contain ``"table"`` key.

    Returns
    -------
    list[dict]
        Each entry: ``{"scenario": str, "pareto_genomes": list[str]}``.
        Ordered by scenario appearance in the table.

    Raises
    ------
    ValueError
        If ``result`` has no ``"table"`` key, if a comparison references an
        unknown scenario, or if a scenario has no valid numeric metrics.
    """
    if "table" not in result:
        raise ValueError("Result dict missing 'table' key")

    table = result["table"]

    # Build scenario → rows mapping (preserving insertion order)
    scenario_genomes: dict = {}
    for row in table:
        sc = row["scenario"]
        if sc not in scenario_genomes:
            scenario_genomes[sc] = []
        scenario_genomes[sc].append(row)

    # Validate that comparisons only reference known scenarios
    for comp in result.get("comparisons", []):
        if comp["scenario"] not in scenario_genomes:
            raise ValueError("unknown scenario")

    frontier: list = []
    for scenario, rows in scenario_genomes.items():
        # Identify numeric metric keys present and finite in all rows
        numeric_keys = sorted(
            k for k in rows[0]
            if k not in _EXCLUDED_KEYS
            and all(_is_finite_numeric(r.get(k)) for r in rows)
        )

        if not numeric_keys:
            raise ValueError("no valid numeric deltas")

        # Find non-dominated genomes
        non_dominated: list = []
        for i, row_i in enumerate(rows):
            dominated = False
            for j, row_j in enumerate(rows):
                if i == j:
                    continue
                # row_j dominates row_i iff row_j >= row_i on all and > on at least one
                all_geq = all(row_j[k] >= row_i[k] for k in numeric_keys)
                any_gt = any(row_j[k] > row_i[k] for k in numeric_keys)
                if all_geq and any_gt:
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(row_i["genome_id"])

        frontier.append({
            "scenario": scenario,
            "pareto_genomes": non_dominated,
        })

    return frontier


def build_scores(result: dict) -> list:
    """Compute normalized aggregate scores per genome and rank them.

    Scoring pipeline:
        1. Per genome, per metric: mean across scenarios (NaN excluded).
        2. Min-max normalization per metric across genomes.
           Constant columns → 0.5.
        3. Aggregate: mean of normalized metric values.
        4. Rank: score descending, genome_id ascending for tie-break.

    Parameters
    ----------
    result : dict
        Output of ``run_benchmark_stress``.  Must contain ``"table"`` key.

    Returns
    -------
    list[dict]
        Ranked list: ``{"genome_id": str, "score": float, "rank": int}``.

    Raises
    ------
    ValueError
        If ``result`` has no ``"table"`` key.
    """
    if "table" not in result:
        raise ValueError("Result dict missing 'table' key")

    table = result["table"]
    if not table:
        return []

    # Collect genome_ids in order of first appearance
    seen_genomes: dict = {}
    for row in table:
        gid = row["genome_id"]
        if gid not in seen_genomes:
            seen_genomes[gid] = []
        seen_genomes[gid].append(row)

    # Identify numeric metric keys across all rows (exclude NaN)
    all_keys: set = set()
    for row in table:
        for key in row:
            if key in _EXCLUDED_KEYS:
                continue
            if _is_finite_numeric(row[key]):
                all_keys.add(key)
    numeric_keys = sorted(all_keys)

    if not numeric_keys:
        return []

    # Per-genome, per-metric: compute mean across scenarios (exclude NaN)
    genome_agg: dict = {}
    for gid, rows in seen_genomes.items():
        agg: dict = {}
        for key in numeric_keys:
            values = [
                r[key] for r in rows
                if _is_finite_numeric(r.get(key))
            ]
            if values:
                agg[key] = sum(values) / len(values)
        genome_agg[gid] = agg

    # Min-max normalization per metric across genomes
    normalized: dict = {gid: {} for gid in genome_agg}
    for key in numeric_keys:
        values = [
            genome_agg[gid][key]
            for gid in genome_agg
            if key in genome_agg[gid]
        ]
        if not values:
            continue
        min_val = min(values)
        max_val = max(values)
        for gid in genome_agg:
            if key not in genome_agg[gid]:
                continue
            if max_val == min_val:
                normalized[gid][key] = 0.5
            else:
                normalized[gid][key] = (
                    (genome_agg[gid][key] - min_val) / (max_val - min_val)
                )

    # Aggregate: mean of normalized values
    scores: list = []
    for gid in seen_genomes:
        norm_vals = list(normalized[gid].values())
        if norm_vals:
            score = sum(norm_vals) / len(norm_vals)
        else:
            score = 0.0
        scores.append({
            "genome_id": gid,
            "score": score,
        })

    # Rank: score DESC, genome_id ASC for tie-break
    scores.sort(key=lambda x: (-x["score"], x["genome_id"]))
    for i, s in enumerate(scores):
        s["rank"] = i + 1

    return scores


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
