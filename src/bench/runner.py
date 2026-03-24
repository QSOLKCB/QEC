"""
Benchmark orchestration engine.

Produces a schema-compliant result object by sweeping over decoder
variants, code distances, and physical error probabilities in a
deterministic order.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import sys
from typing import Any

import numpy as np

from .config import BenchmarkConfig, DecoderSpec
from .schema import SCHEMA_VERSION, _SUPPORTED_SCHEMA_VERSIONS, canonicalize, validate_result
from .compare import (
    compute_threshold_table,
    compute_runtime_scaling,
    compute_iteration_histogram,
    aggregate_iteration_summaries,
)


# ── Adapter registry ────────────────────────────────────────────────

_ADAPTER_REGISTRY: dict[str, type] = {}


def _ensure_registry() -> None:
    """Lazily populate the adapter registry on first use."""
    if _ADAPTER_REGISTRY:
        return
    from .adapters.bp import BPAdapter  # noqa: local import
    _ADAPTER_REGISTRY["bp"] = BPAdapter


def _make_adapter(spec: DecoderSpec) -> Any:
    """Instantiate an adapter from a :class:`DecoderSpec`."""
    _ensure_registry()
    cls = _ADAPTER_REGISTRY.get(spec.adapter)
    if cls is None:
        raise ValueError(
            f"Unknown adapter {spec.adapter!r}. "
            f"Available: {sorted(_ADAPTER_REGISTRY)}"
        )
    adapter = cls()
    return adapter


# ── Code factory ────────────────────────────────────────────────────

def _make_code(distance: int, seed: int) -> Any:
    """Build a parity-check matrix for the given code distance.

    Uses ``create_code`` with lifting_size = distance to produce a
    code whose block length scales with distance.
    """
    from qec_qldpc_codes import create_code
    code = create_code(name="rate_0.50", lifting_size=distance, seed=seed)
    return code.H_X


# ── Deterministic sub-seed derivation ───────────────────────────────

def _derive_subseed(
    base_seed: int,
    decoder_identity: dict[str, Any],
    distance: int,
    p: float,
) -> int:
    """Derive a deterministic sub-seed from sweep coordinates.

    The seed depends only on the logical parameters — NOT on loop
    iteration order — so reordering decoders in config does not
    change per-record results.

    Uses SHA-256 over a JSON payload with sorted keys.
    """
    payload = json.dumps({
        "base_seed": base_seed,
        "decoder": decoder_identity,
        "distance": distance,
        "p": p,
    }, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ── Main entry point ────────────────────────────────────────────────

def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Execute a full benchmark sweep and return a schema-compliant result.

    Parameters
    ----------
    config:
        A :class:`BenchmarkConfig` describing the sweep grid.

    Returns
    -------
    dict conforming to schema version 3.0.0.
    """
    if config.schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Config schema_version {config.schema_version!r} not in "
            f"supported set {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )

    from qec_qldpc_codes import syndrome
    from qec.channel import get_channel_model

    channel = get_channel_model(config.channel_model)

    results: list[dict[str, Any]] = []

    # Deterministic sweep order: decoders → distances → p_values.
    for dec_spec in config.decoders:
        adapter = _make_adapter(dec_spec)

        for distance in config.distances:
            H = _make_code(distance, config.seed)
            _, n = H.shape

            # Initialize adapter with decoder params + H.
            init_params = dict(dec_spec.params)
            init_params["H"] = H
            init_params["max_iters"] = config.max_iters
            adapter.initialize(config=init_params)

            for p in config.p_values:
                # Per-(decoder, distance, p) sub-seed derived from
                # logical coordinates — independent of loop ordering.
                sub_seed = _derive_subseed(
                    config.seed,
                    adapter.serialize_identity(),
                    distance,
                    p,
                )
                sub_rng = np.random.default_rng(sub_seed)

                frame_errors = 0
                total_iters = 0
                iter_counts: list[int] = []
                syndrome_matches = 0

                for _ in range(config.trials):
                    e = (sub_rng.random(n) < p).astype(np.uint8)
                    s = syndrome(H, e)
                    llr = channel.compute_llr(p=p, n=n, error_vector=e)

                    dec_result = adapter.decode(
                        syndrome=s, llr=llr, error_vector=e,
                    )

                    iters = dec_result["iters"]
                    total_iters += iters
                    iter_counts.append(iters)
                    if not dec_result["success"]:
                        frame_errors += 1

                    # Syndrome consistency: does syndrome(H, c) == s?
                    correction = dec_result.get("correction")
                    if correction is not None:
                        s_c = syndrome(H, np.asarray(correction, dtype=np.uint8))
                        if np.array_equal(s_c, s):
                            syndrome_matches += 1

                fer = float(frame_errors) / config.trials
                fidelity = 1.0 - fer
                mean_iters = float(total_iters) / config.trials
                syndrome_success_rate = float(syndrome_matches) / config.trials

                record: dict[str, Any] = {
                    "decoder": adapter.name,
                    "decoder_identity": adapter.serialize_identity(),
                    "distance": distance,
                    "p": p,
                    "fer": fer,
                    "fidelity": fidelity,
                    "wer": fer,  # WER := FER in this project
                    "mean_iters": round(mean_iters, 4),
                    "syndrome_success_rate": syndrome_success_rate,
                    "trials": config.trials,
                }

                # Iteration histogram (opt-in).
                if config.collect_iter_hist:
                    hist = compute_iteration_histogram(iter_counts)
                    record["iter_histogram"] = hist["histogram"]

                # Runtime measurement (opt-in).
                if config.runtime_mode == "on":
                    # Build a deterministic workload for runtime measurement.
                    rt_rng = np.random.default_rng(sub_seed)
                    e_rt = (rt_rng.random(n) < p).astype(np.uint8)
                    s_rt = syndrome(H, e_rt)
                    llr_rt = channel.compute_llr(p=p, n=n, error_vector=e_rt)
                    workload = {
                        "llr": llr_rt,
                        "syndrome": s_rt,
                        "warmup": config.runtime.warmup,
                        "runs": config.runtime.runs,
                        "measure_memory": config.runtime.measure_memory,
                    }
                    record["runtime"] = adapter.measure_runtime(
                        workload=workload,
                    )
                else:
                    record["runtime"] = None

                results.append(record)

    # ── Build summaries ──
    # Group results by decoder name for threshold + scaling analysis.
    decoder_names = sorted(set(r["decoder"] for r in results))
    thresholds: list[dict[str, Any]] = []
    runtime_scaling: list[dict[str, Any]] = []

    for dname in decoder_names:
        dec_results = [r for r in results if r["decoder"] == dname]
        thresholds.append(compute_threshold_table(dec_results, dname))
        if config.runtime_mode == "on":
            runtime_scaling.append(
                compute_runtime_scaling(dec_results, dname)
            )

    iter_dists = aggregate_iteration_summaries(results)

    summaries: dict[str, Any] = {
        "thresholds": thresholds,
        "runtime_scaling": runtime_scaling,
        "iteration_distributions": iter_dists,
    }

    # ── Environment info ──
    env: dict[str, str] = {
        "platform": platform.platform(),
        "python_version": sys.version,
    }

    # ── Assemble top-level result ──
    try:
        from importlib.metadata import version as _pkg_version
        qec_version = _pkg_version("qec")
    except Exception:
        qec_version = "unknown"

    if config.deterministic_metadata:
        created_utc = "1970-01-01T00:00:00+00:00"
    else:
        created_utc = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )

    # ── Resource estimates (opt-in, v3.0.1) ──
    if (config.resource_model is not None
            and config.resource_model.enabled):
        from analysis.gate_cost import estimate_gate_costs, compare_costs
        dim = 2  # default
        if config.qudit is not None:
            dim = config.qudit.get("dimension", 2)
        qubit_est = estimate_gate_costs(
            dimension=dim,
            model=config.resource_model.model,
            assumptions=config.resource_model.assumptions,
        )
        native_est = estimate_gate_costs(
            dimension=dim,
            model=config.resource_model.native_model,
            assumptions=config.resource_model.assumptions,
        )
        comparison = compare_costs(qubit_est, native_est)
        summaries["resource_estimates"] = canonicalize({
            "qubit_decomposition": qubit_est,
            "native_qudit": native_est,
            "comparison": comparison,
        })

    result_obj: dict[str, Any] = {
        "schema_version": config.schema_version,
        "created_utc": created_utc,
        "qec_version": str(qec_version),
        "environment": env,
        "config": config.to_dict(),
        "results": results,
        "summaries": summaries,
    }

    # Canonicalize everything (numpy → Python, sorted keys).
    result_obj = canonicalize(result_obj)

    # Validate before returning.
    validate_result(result_obj)

    return result_obj
