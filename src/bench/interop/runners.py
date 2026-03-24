"""
Tool wrappers for interop benchmarks.

Each runner wraps a specific tool (QEC-native, Stim, PyMatching) and
produces schema-compliant records with proper ``benchmark_kind`` labels.

All third-party imports are gated via :mod:`imports`.  Runners for
unavailable tools return structured "skipped" records.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .imports import (
    HAS_STIM, HAS_PYMATCHING,
    stim, pymatching,
)
from .serialize import canonical_json, artifact_hash, config_hash
from .env import capture_environment


# ── Helpers ─────────────────────────────────────────────────────────

def _derive_subseed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed from a base seed and label."""
    payload = json.dumps(
        {"base_seed": base_seed, "label": label}, sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ── QEC-native runner ───────────────────────────────────────────────

def run_qec_native(
    *,
    distances: list[int],
    p_values: list[float],
    trials: int = 200,
    max_iters: int = 50,
    seed: int = 12345,
    runtime_mode: str = "off",
    deterministic_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Run QEC-native BP decoder benchmarks.

    Returns a list of schema-compliant interop records with
    ``benchmark_kind="direct_comparison"``.
    """
    from qec_qldpc_codes import create_code, bp_decode, syndrome
    from qec.channel import OracleChannel

    channel = OracleChannel()
    records: list[dict[str, Any]] = []

    for lift in sorted(distances):
        code = create_code(name="rate_0.50", lifting_size=lift, seed=seed)
        H = code.H_X
        _, n = H.shape

        for p in sorted(p_values):
            sub_seed = _derive_subseed(seed, f"qec_native_d{lift}_p{p}")
            rng = np.random.default_rng(sub_seed)

            frame_errors = 0
            total_iters = 0

            for _ in range(trials):
                e = (rng.random(n) < p).astype(np.uint8)
                s = syndrome(H, e)
                llr = channel.compute_llr(p=p, n=n, error_vector=e)

                result = bp_decode(
                    H, llr,
                    syndrome_vec=s,
                    mode="min_sum",
                    schedule="flooding",
                    max_iters=max_iters,
                )
                correction, iters = result[0], result[1]
                total_iters += int(iters)

                residual = np.asarray(e) ^ np.asarray(correction)
                if np.any(residual):
                    frame_errors += 1

            fer = float(frame_errors) / trials
            mean_iters = float(total_iters) / trials

            record_config = {
                "distance": lift,
                "max_iters": max_iters,
                "p": p,
                "seed": seed,
                "trials": trials,
            }

            record: dict[str, Any] = {
                "schema_version": "3.1.2",
                "tool": {
                    "name": "qec_bp",
                    "version": "native",
                    "category": "native",
                },
                "benchmark_kind": "direct_comparison",
                "code_family": "qldpc_css",
                "representation": "pcm",
                "seed": seed,
                "noise_model": "bsc_bitflip",
                "channel_model": "oracle",
                "trials": trials,
                "results": {
                    "logical_error_rate": round(fer, 8),
                    "mean_iters": round(mean_iters, 4),
                },
                "determinism": {
                    "canonical_json": {
                        "sort_keys": True,
                        "separators": [",", ":"],
                    },
                    "stable_sweep_hash": config_hash(record_config),
                    "artifact_hash": None,  # filled after serialization
                },
                "config": record_config,
            }

            if runtime_mode != "off":
                record["results"]["decoding_time_sec"] = None

            records.append(record)

    # Compute artifact hashes.
    for rec in records:
        rec["determinism"]["artifact_hash"] = artifact_hash(rec)

    return records


# ── Stim + PyMatching runner ────────────────────────────────────────

def _build_repetition_circuit(
    distance: int, rounds: int, p: float,
) -> Any:
    """Build a Stim repetition code circuit deterministically."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
    )
    return circuit


def _build_surface_code_circuit(
    distance: int, rounds: int, p: float,
) -> Any:
    """Build a Stim surface code circuit deterministically."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
    )
    return circuit


def run_stim_pymatching_baseline(
    *,
    code_family: str,
    distances: list[int],
    p_values: list[float],
    rounds_per_distance: int | None = None,
    trials: int = 1000,
    seed: int = 12345,
    runtime_mode: str = "off",
    deterministic_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Run a Stim -> DEM -> PyMatching reference baseline.

    Parameters
    ----------
    code_family:
        One of ``"repetition"`` or ``"surface_code"``.
    distances, p_values:
        Sweep parameters (sorted internally).
    rounds_per_distance:
        Number of syndrome rounds. Defaults to ``distance`` if None.
    trials:
        Number of samples per (distance, p) pair.
    seed:
        Base seed for deterministic sampling.
    runtime_mode:
        ``"off"`` omits timing; ``"on"`` includes it.

    Returns
    -------
    List of schema-compliant records with ``benchmark_kind="reference_baseline"``.
    """
    if not HAS_STIM:
        return [{
            "status": "skipped",
            "reason": "stim not installed",
            "tool": {"name": "stim", "version": None, "category": "third_party"},
            "benchmark_kind": "reference_baseline",
            "code_family": code_family,
        }]

    if not HAS_PYMATCHING:
        return [{
            "status": "skipped",
            "reason": "pymatching not installed",
            "tool": {"name": "pymatching", "version": None, "category": "third_party"},
            "benchmark_kind": "reference_baseline",
            "code_family": code_family,
        }]

    builders = {
        "repetition": _build_repetition_circuit,
        "surface_code": _build_surface_code_circuit,
    }
    builder = builders.get(code_family)
    if builder is None:
        raise ValueError(
            f"Unknown code_family {code_family!r}. "
            f"Available: {sorted(builders)}"
        )

    records: list[dict[str, Any]] = []

    for distance in sorted(distances):
        rounds = rounds_per_distance if rounds_per_distance is not None else distance

        for p in sorted(p_values):
            circuit = builder(distance, rounds, p)
            dem = circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)

            sub_seed = _derive_subseed(
                seed, f"stim_{code_family}_d{distance}_p{p}",
            )

            sampler = circuit.compile_detector_sampler(seed=sub_seed)
            detection_events, observable_flips = sampler.sample(
                trials, separate_observables=True,
            )

            predictions = matcher.decode_batch(detection_events)
            num_errors = int(np.sum(np.any(predictions != observable_flips, axis=1)))
            logical_error_rate = float(num_errors) / trials

            from .imports import STIM_VERSION, PYMATCHING_VERSION

            record_config = {
                "code_family": code_family,
                "distance": distance,
                "p": p,
                "rounds": rounds,
                "seed": seed,
                "trials": trials,
            }

            record: dict[str, Any] = {
                "schema_version": "3.1.2",
                "tool": {
                    "name": "stim_pymatching",
                    "version": f"stim={STIM_VERSION},pymatching={PYMATCHING_VERSION}",
                    "category": "third_party",
                },
                "benchmark_kind": "reference_baseline",
                "code_family": code_family,
                "representation": "stim_circuit",
                "seed": seed,
                "noise_model": "after_clifford_depolarization",
                "trials": trials,
                "results": {
                    "logical_error_rate": round(logical_error_rate, 8),
                },
                "determinism": {
                    "canonical_json": {
                        "sort_keys": True,
                        "separators": [",", ":"],
                    },
                    "stable_sweep_hash": config_hash(record_config),
                    "artifact_hash": None,
                },
                "config": record_config,
            }

            if runtime_mode != "off":
                record["results"]["decoding_time_sec"] = None

            records.append(record)

    # Compute artifact hashes.
    for rec in records:
        rec["determinism"]["artifact_hash"] = artifact_hash(rec)

    return records
