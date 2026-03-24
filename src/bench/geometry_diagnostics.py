"""
Geometry-aware syndrome-only diagnostics (v3.3.0).

Post-hoc analysis of benchmark results to explain distance scaling
behavior under syndrome-only inference.  Produces deterministic sidecar
artifacts separate from canonical benchmark output.

All aggregate functions are pure: they take benchmark result dicts and
return diagnostic dicts with deterministic field ordering via
:func:`canonicalize`.

No decoder defaults are altered.  No schema changes.  No new randomness.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ── Exceptions ────────────────────────────────────────────────────

class BSIConfigError(ValueError):
    """Raised when BSI base/2x benchmark records are misaligned."""
    pass


# ── Epsilon for log-domain safety ─────────────────────────────────
_LOG_EPS = 1e-30


# ═══════════════════════════════════════════════════════════════════
# 1. Distance Penalty Slope (DPS)
# ═══════════════════════════════════════════════════════════════════

def compute_dps(
    records: list[dict[str, Any]],
    eps: float = _LOG_EPS,
) -> list[dict[str, Any]]:
    """Slope of log10(FER + *eps*) vs distance for each group.

    Groups records by ``(decoder, p)``.  A positive slope indicates
    FER *increasing* with distance — the hallmark of distance scaling
    inversion under syndrome-only inference.

    Parameters
    ----------
    records:
        The ``results`` list from a benchmark output.
    eps:
        Small constant added before log transform.

    Returns
    -------
    list of dicts (sorted by key), each with:
        decoder, distances, intercept, inverted, log_fer, p, slope.
    """
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for rec in records:
        key = (rec["decoder"], rec["p"])
        groups.setdefault(key, []).append(rec)

    results: list[dict[str, Any]] = []

    for (decoder, p) in sorted(groups.keys()):
        group = sorted(groups[(decoder, p)], key=lambda r: r["distance"])
        distances = [r["distance"] for r in group]
        log_fer_raw = [math.log10(r["fer"] + eps) for r in group]

        if len(distances) < 2:
            slope = 0.0
            intercept = log_fer_raw[0] if log_fer_raw else 0.0
        else:
            d_arr = np.array(distances, dtype=np.float64)
            lf_arr = np.array(log_fer_raw, dtype=np.float64)
            A = np.vstack([d_arr, np.ones(len(d_arr))]).T
            coef, _, _, _ = np.linalg.lstsq(A, lf_arr, rcond=None)
            slope = float(coef[0])
            intercept = float(coef[1])

        log_fer = [round(v, 10) for v in log_fer_raw]
        slope = round(slope, 10)
        intercept = round(intercept, 10)

        results.append({
            "decoder": decoder,
            "distances": distances,
            "intercept": intercept,
            "inverted": slope > 0.0,
            "log_fer": log_fer,
            "p": p,
            "slope": slope,
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# 2. False-Convergence Rate (FCR)
# ═══════════════════════════════════════════════════════════════════

def compute_fcr(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """P(syndrome weight == 0 AND logical failure).

    Algebraically: ``FCR = SCR - (1 - FER)``.  Equivalent to the
    Inversion Index formalized in v3.2.1.  Clamped to ``>= 0``.

    Parameters
    ----------
    records:
        Benchmark result records with ``syndrome_success_rate``
        and ``fer`` (or ``fidelity``) fields.

    Returns
    -------
    list of dicts (deterministic order), each with:
        decoder, distance, fcr, fidelity, p, scr.
    """
    results: list[dict[str, Any]] = []
    for rec in sorted(records, key=lambda r: (r["decoder"], r["distance"], r["p"])):
        scr = rec.get("syndrome_success_rate", 0.0)
        fidelity = rec.get("fidelity", 1.0 - rec.get("fer", 0.0))
        fcr = max(0.0, round(scr - fidelity, 10))

        results.append({
            "decoder": rec["decoder"],
            "distance": rec["distance"],
            "fcr": fcr,
            "fidelity": round(fidelity, 10),
            "p": rec["p"],
            "scr": round(scr, 10),
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# 3. Budget Sensitivity Index (BSI)
# ═══════════════════════════════════════════════════════════════════

def compute_bsi(
    records_base: list[dict[str, Any]],
    records_2x: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """FER(base_max_iters) - FER(2x_base_max_iters).

    A large positive BSI indicates the decoder benefits significantly
    from additional iteration budget at that operating point.

    Parameters
    ----------
    records_base:
        Records from the base max_iters run.
    records_2x:
        Records from the 2x max_iters run.

    Returns
    -------
    list of matched dicts with: bsi, decoder, distance, fer_2x, fer_base, p.
    Raises BSIConfigError if any base record has no matching 2x record.
    Extra 2x records without base match are ignored.
    """
    lookup: dict[tuple[str, int, float], float] = {}
    for rec in records_2x:
        lookup[(rec["decoder"], rec["distance"], rec["p"])] = rec["fer"]

    results: list[dict[str, Any]] = []
    for rec in sorted(
        records_base, key=lambda r: (r["decoder"], r["distance"], r["p"])
    ):
        key = (rec["decoder"], rec["distance"], rec["p"])
        fer_2x = lookup.get(key)
        if fer_2x is None:
            raise BSIConfigError(f"BSI mismatch: no 2x record for {key}")

        results.append({
            "bsi": round(rec["fer"] - fer_2x, 10),
            "decoder": rec["decoder"],
            "distance": rec["distance"],
            "fer_2x": fer_2x,
            "fer_base": rec["fer"],
            "p": rec["p"],
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# 4. Schedule Sensitivity Index (SSI)
# ═══════════════════════════════════════════════════════════════════

def compute_ssi(
    records_by_schedule: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """max(FER) - min(FER) across schedules for each (decoder, distance, p).

    Parameters
    ----------
    records_by_schedule:
        Mapping from schedule name to benchmark records.

    Returns
    -------
    list of dicts with: decoder, distance, fer_by_schedule,
    fer_max, fer_min, p, ssi.  Coordinates with only one schedule
    are omitted.
    """
    by_coord: dict[tuple[str, int, float], dict[str, float]] = {}

    for sched in sorted(records_by_schedule.keys()):
        for rec in records_by_schedule[sched]:
            key = (rec["decoder"], rec["distance"], rec["p"])
            by_coord.setdefault(key, {})[sched] = rec["fer"]

    results: list[dict[str, Any]] = []
    for (decoder, distance, p) in sorted(by_coord.keys()):
        fer_map = by_coord[(decoder, distance, p)]
        if len(fer_map) < 2:
            continue
        fer_vals = list(fer_map.values())
        results.append({
            "decoder": decoder,
            "distance": distance,
            "fer_by_schedule": {k: fer_map[k] for k in sorted(fer_map)},
            "fer_max": max(fer_vals),
            "fer_min": min(fer_vals),
            "p": p,
            "ssi": round(max(fer_vals) - min(fer_vals), 10),
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# 5. Per-Iteration Summary (from LLR history)
# ═══════════════════════════════════════════════════════════════════

def compute_per_iteration_summary(
    H: np.ndarray,
    llr_history: np.ndarray,
    syndrome_target: np.ndarray,
) -> dict[str, Any]:
    """Derive syndrome_weight[t] and check_satisfaction_ratio[t].

    Purely post-hoc: takes the LLR history array already produced by
    ``bp_decode(llr_history=...)`` and computes diagnostic traces
    without re-running the decoder.

    Parameters
    ----------
    H:
        Parity-check matrix, shape (m, n).
    llr_history:
        Per-iteration L_total snapshots, shape (T, n).
    syndrome_target:
        Target syndrome vector, shape (m,).

    Returns
    -------
    dict with sorted keys:
        check_satisfaction_ratio, delta_syndrome,
        stall_count, stall_fraction, syndrome_weight.
    """
    if llr_history.ndim != 2 or llr_history.shape[0] == 0:
        return {
            "check_satisfaction_ratio": [],
            "delta_syndrome": [],
            "stall_count": 0,
            "stall_fraction": 0.0,
            "syndrome_weight": [],
        }

    T = llr_history.shape[0]
    m = H.shape[0]
    H32 = H.astype(np.int32, copy=False)
    target = np.asarray(syndrome_target, dtype=np.uint8)

    syndrome_weights: list[int] = []
    check_sat_ratios: list[float] = []

    for t in range(T):
        hard_t = (llr_history[t] < 0).astype(np.uint8)
        syn_t = ((H32 @ hard_t.astype(np.int32)) % 2).astype(np.uint8)
        diff = syn_t != target
        sw = int(np.sum(diff))
        csr = round(1.0 - float(np.mean(diff)), 10) if m > 0 else 1.0
        syndrome_weights.append(sw)
        check_sat_ratios.append(csr)

    delta_syndrome: list[int] = [0]
    for t in range(1, T):
        delta_syndrome.append(syndrome_weights[t] - syndrome_weights[t - 1])

    stall_count = sum(1 for d in delta_syndrome[1:] if d == 0) if T > 1 else 0
    stall_fraction = round(stall_count / max(T - 1, 1), 10) if T > 1 else 0.0

    return {
        "check_satisfaction_ratio": check_sat_ratios,
        "delta_syndrome": delta_syndrome,
        "stall_count": stall_count,
        "stall_fraction": stall_fraction,
        "syndrome_weight": syndrome_weights,
    }


# ═══════════════════════════════════════════════════════════════════
# 6. Residual Summary
# ═══════════════════════════════════════════════════════════════════

def compute_residual_summary(
    residual_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate per-iteration residual statistics.

    Parameters
    ----------
    residual_metrics:
        Dict from ``bp_decode(residual_metrics=True)``.

    Returns
    -------
    list of per-iteration dicts with energy, linf_*, l2_* statistics.
    Empty list when no residual data is available.
    """
    energies = residual_metrics.get("residual_energy", [])
    linf_list = residual_metrics.get("residual_linf", [])
    l2_list = residual_metrics.get("residual_l2", [])
    n_iters = max(len(energies), len(linf_list), len(l2_list))

    summaries: list[dict[str, Any]] = []
    for t in range(n_iters):
        entry: dict[str, Any] = {"iteration": t}

        if t < len(energies):
            entry["energy"] = round(float(energies[t]), 10)

        if t < len(linf_list):
            arr = np.asarray(linf_list[t], dtype=np.float64)
            if arr.size > 0:
                entry["linf_max"] = round(float(np.max(arr)), 10)
                entry["linf_mean"] = round(float(np.mean(arr)), 10)
                entry["linf_var"] = round(float(np.var(arr)), 10)

        if t < len(l2_list):
            arr = np.asarray(l2_list[t], dtype=np.float64)
            if arr.size > 0:
                entry["l2_max"] = round(float(np.max(arr)), 10)
                entry["l2_mean"] = round(float(np.mean(arr)), 10)
                entry["l2_var"] = round(float(np.var(arr)), 10)

        summaries.append(entry)

    return summaries


# ═══════════════════════════════════════════════════════════════════
# 7. Stall Metrics (aggregate)
# ═══════════════════════════════════════════════════════════════════

def compute_stall_metrics(
    per_iter_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate stall statistics across multiple trial summaries.

    Parameters
    ----------
    per_iter_summaries:
        List of dicts from :func:`compute_per_iteration_summary`.

    Returns
    -------
    dict with: max_stall_fraction, mean_stall_fraction,
    total_trials, trials_with_stall.
    """
    if not per_iter_summaries:
        return {
            "max_stall_fraction": 0.0,
            "mean_stall_fraction": 0.0,
            "total_trials": 0,
            "trials_with_stall": 0,
        }

    fracs = [s["stall_fraction"] for s in per_iter_summaries]
    return {
        "max_stall_fraction": round(float(max(fracs)), 10),
        "mean_stall_fraction": round(float(np.mean(fracs)), 10),
        "total_trials": len(per_iter_summaries),
        "trials_with_stall": sum(1 for f in fracs if f > 0.0),
    }


# ═══════════════════════════════════════════════════════════════════
# 8. Local Inconsistency Summary
# ═══════════════════════════════════════════════════════════════════

def compute_local_inconsistency(
    per_iter_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate local inconsistency (syndrome weight increases).

    A positive ``delta_syndrome`` at iteration *t* means the decoder
    moved *further* from convergence — a local inconsistency event.

    Parameters
    ----------
    per_iter_summaries:
        List of dicts from :func:`compute_per_iteration_summary`.

    Returns
    -------
    dict with: max_inconsistency_count, mean_inconsistency_fraction,
    total_trials, trials_with_inconsistency.
    """
    if not per_iter_summaries:
        return {
            "max_inconsistency_count": 0,
            "mean_inconsistency_fraction": 0.0,
            "total_trials": 0,
            "trials_with_inconsistency": 0,
        }

    inc_counts: list[int] = []
    inc_fracs: list[float] = []
    trials_with = 0

    for s in per_iter_summaries:
        deltas = s.get("delta_syndrome", [])
        positives = sum(1 for d in deltas[1:] if d > 0)
        total_steps = max(len(deltas) - 1, 1)
        inc_counts.append(positives)
        inc_fracs.append(positives / total_steps)
        if positives > 0:
            trials_with += 1

    return {
        "max_inconsistency_count": max(inc_counts),
        "mean_inconsistency_fraction": round(float(np.mean(inc_fracs)), 10),
        "total_trials": len(per_iter_summaries),
        "trials_with_inconsistency": trials_with,
    }


# ═══════════════════════════════════════════════════════════════════
# 9. Standalone Per-Iteration Collection
# ═══════════════════════════════════════════════════════════════════

def collect_per_iteration_data(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    decoder_params: dict[str, Any],
    max_iters: int = 50,
) -> dict[str, Any]:
    """Run ``bp_decode`` with instrumentation and return summaries.

    Enables ``llr_history`` and ``residual_metrics`` for a single
    diagnostic decode.  Does **not** alter any decoder defaults —
    instrumentation is purely read-only and opt-in.

    Parameters
    ----------
    H:
        Parity-check matrix, shape (m, n).
    llr:
        Per-variable LLR vector, shape (n,).
    syndrome_vec:
        Target syndrome vector, shape (m,).
    decoder_params:
        Decoder keyword arguments (mode, schedule, etc.).
    max_iters:
        Iteration budget.

    Returns
    -------
    dict with: converged, iterations, per_iteration, residual_summary.
    """
    from qec_qldpc_codes import bp_decode, syndrome

    params = dict(decoder_params)
    params.pop("H", None)  # H passed positionally
    params["max_iters"] = max_iters
    params["llr_history"] = max_iters
    params["residual_metrics"] = True

    result = bp_decode(H, llr, syndrome_vec=syndrome_vec, **params)
    # llr_history > 0 AND residual_metrics=True → 4-tuple.
    correction, iters, history, res_metrics = result

    syn_c = syndrome(H, correction)
    converged = bool(np.array_equal(syn_c, syndrome_vec))

    per_iter = compute_per_iteration_summary(H, history, syndrome_vec)
    res_summary = compute_residual_summary(res_metrics)

    return {
        "converged": converged,
        "iterations": int(iters),
        "per_iteration": per_iter,
        "residual_summary": res_summary,
    }


# ═══════════════════════════════════════════════════════════════════
# 10. Sidecar Artifact Builder
# ═══════════════════════════════════════════════════════════════════

def build_geometry_sidecar(
    records: list[dict[str, Any]],
    records_2x: list[dict[str, Any]] | None = None,
    records_by_schedule: dict[str, list[dict[str, Any]]] | None = None,
    per_iteration_data: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble a complete geometry diagnostics sidecar artifact.

    The returned dict is canonicalized (sorted keys, plain-Python
    types) and suitable for deterministic JSON serialization.

    Parameters
    ----------
    records:
        Primary benchmark ``results`` list.
    records_2x:
        Optional results from a 2x max_iters run (for BSI).
    records_by_schedule:
        Optional schedule→results mapping (for SSI).
    per_iteration_data:
        Optional per-trial instrumentation dicts (for stall /
        inconsistency / residual metrics).

    Returns
    -------
    Canonicalized sidecar dict.
    """
    from utils.canonicalize import canonicalize

    metrics: dict[str, Any] = {}

    metrics["dps"] = compute_dps(records)
    metrics["fcr"] = compute_fcr(records)

    if records_2x is not None:
        metrics["bsi"] = compute_bsi(records, records_2x)

    if records_by_schedule is not None and len(records_by_schedule) >= 2:
        metrics["ssi"] = compute_ssi(records_by_schedule)

    if per_iteration_data:
        pi_summaries = [
            d["per_iteration"] for d in per_iteration_data
            if "per_iteration" in d
        ]
        metrics["stall_metrics"] = compute_stall_metrics(pi_summaries)
        metrics["local_inconsistency"] = compute_local_inconsistency(
            pi_summaries,
        )
        all_res: list[list[dict[str, Any]]] = []
        for d in per_iteration_data:
            rs = d.get("residual_summary", [])
            if rs:
                all_res.append(rs)
        if all_res:
            metrics["residual_summaries"] = all_res

    sidecar: dict[str, Any] = {
        "diagnostic_version": "3.3.0",
        "metrics": metrics,
    }

    return canonicalize(sidecar)
