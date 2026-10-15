#!/usr/bin/env python3
"""Measure qutrit construction against the original scalar builder.

Run from any working directory. JSON goes to stdout; checks fail before timings.
This opt-in benchmark has no performance pass/fail thresholds.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import inspect
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
import tracemalloc
from types import MappingProxyType
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qec.decoder.qutrit import (  # noqa: E402
    ExactDecoder,
    Pauli,
    QutritStabilizerCode,
    cyclic_five_qutrit_code,
    shor_nine_qutrit_code,
    ternary_golay_qutrit_code,
)
from qec.decoder.qutrit.exact import _one_site_syndromes  # noqa: E402
from qec.decoder.qutrit.stabilizer import Syndrome, paulis_of_weight  # noqa: E402


# Body copied verbatim from the pinned pre-optimization exact.py, not from the
# live builder or its fast path. Inheriting retains the public decoding API.
PINNED_EXACT_SHA256 = "11b5cad38e88fad36dc37a9ab916496b17b75eb3d56df765ce90dd2e4153bfce"


@dataclass(frozen=True)
class ScalarReferenceDecoder(ExactDecoder):
    """Original scalar constructor, with the same frozen fields/table/API."""

    def __post_init__(self) -> None:
        if self.max_weight < 0:
            raise ValueError("max_weight must be non-negative")
        identity = Pauli((0,) * self.code.n, (0,) * self.code.n)
        table: dict[Syndrome, Pauli] = {self.code.syndrome(identity): identity}
        for weight in range(1, self.max_weight + 1):
            for candidate in paulis_of_weight(self.code.n, weight):
                syndrome = self.code.syndrome(candidate)
                previous = table.get(syndrome)
                if previous is None:
                    table[syndrome] = candidate
                    continue
                difference = candidate.compose(previous.inverse())
                if not self.code.is_stabilizer(difference):
                    raise ValueError(
                        "bounded error set violates exact correction conditions"
                    )
        object.__setattr__(self, "_table", MappingProxyType(table))


def k0_control() -> QutritStabilizerCode:
    """Two independent Z generators, n=2 and k=0; exercises degeneracy."""
    return QutritStabilizerCode(
        ((0, 0, 1, 0), (0, 0, 0, 1)), name="two-qutrit-k0-control"
    )


def summary(samples: list[int], unit: str) -> dict:
    return {"unit": unit, "raw_samples": samples, "median": statistics.median(samples)}


def elapsed(operation: Callable[[], object]) -> int:
    # Collection is outside the timer. Keep normal cyclic-GC behavior in the
    # measured operation and retain the result until after the timer stops.
    gc.collect()
    start = perf_counter_ns()
    result = operation()
    duration = perf_counter_ns() - start
    del result
    return duration


def peak_bytes(operation: Callable[[], object]) -> int:
    # Memory passes never contribute to latency samples; tracing is expensive.
    gc.collect()
    tracemalloc.start()
    try:
        result = operation()
        _, peak = tracemalloc.get_traced_memory()
        del result
        return peak
    finally:
        tracemalloc.stop()


def correct_batch(decoder: ExactDecoder, errors: tuple[Pauli, ...], repeats: int) -> None:
    for _ in range(repeats):
        for error in errors:
            decoder.correct(error)


def verify(code: QutritStabilizerCode, radius: int) -> dict:
    live, scalar = ExactDecoder(code, radius), ScalarReferenceDecoder(code, radius)
    if tuple(live._table.items()) != tuple(scalar._table.items()):
        raise AssertionError(f"ordered table mismatch: {code.name}, radius={radius}")
    if not isinstance(live._table, MappingProxyType):
        raise AssertionError("live table is no longer read-only")
    errors = tuple(
        error
        for weight in range(radius + 1)
        for error in paulis_of_weight(code.n, weight)
    )
    for error in errors:
        actual, expected = live.correct(error), scalar.correct(error)
        if actual != expected or not actual.success:
            raise AssertionError(f"correction mismatch: {code.name}, error={error}")
    # Fixed evenly spaced subset, including identity and the last error, used
    # identically by both reused-decoder workloads. No randomness or selection
    # based on measured results.
    count = min(16, len(errors))
    indexes = (0,) if count == 1 else tuple(
        index * (len(errors) - 1) // (count - 1) for index in range(count)
    )
    return {
        "ordered_table_entries": len(live._table),
        "verified_errors": len(errors),
        "errors": tuple(errors[index] for index in indexes),
        "live": live,
        "scalar": scalar,
    }


def benchmark_case(name: str, code: QutritStabilizerCode, radius: int,
                   verified: dict, samples: int, memory_samples: int,
                   correction_repeats: int) -> dict:
    errors = verified["errors"]
    constructors = {
        "live": lambda: ExactDecoder(code, radius),
        "scalar_reference": lambda: ScalarReferenceDecoder(code, radius),
    }
    batches = {
        "live": lambda: correct_batch(verified["live"], errors, correction_repeats),
        "scalar_reference": lambda: correct_batch(verified["scalar"], errors, correction_repeats),
    }
    construction = {key: [] for key in constructors}
    correction = {key: [] for key in constructors}
    construction_memory = {key: [] for key in constructors}
    correction_memory = {key: [] for key in constructors}
    keys = tuple(constructors)
    for sample in range(samples):
        # Alternate order to reduce systematic first/second measurement bias.
        for key in keys[::1 if sample % 2 == 0 else -1]:
            construction[key].append(elapsed(constructors[key]))
            correction[key].append(elapsed(batches[key]))
    for sample in range(memory_samples):
        for key in keys[::1 if sample % 2 == 0 else -1]:
            construction_memory[key].append(peak_bytes(constructors[key]))
            correction_memory[key].append(peak_bytes(batches[key]))
    preparation = None
    if radius > 0:
        prepare = lambda: _one_site_syndromes(code)
        preparation = {
            "latency": summary([elapsed(prepare) for _ in range(samples)], "ns"),
            "peak_traced_allocation": summary(
                [peak_bytes(prepare) for _ in range(memory_samples)], "bytes"
            ),
            "scope": "live private preparation helper only; also included in full constructor",
            "scalar_reference": "not applicable: scalar builder has no preparation plan",
        }
    measurements = {}
    for key in keys:
        measurements[key] = {
            "full_constructor": summary(construction[key], "ns"),
            "reused_correct_batch": summary(correction[key], "ns"),
            "constructor_peak_traced_allocation": summary(construction_memory[key], "bytes"),
            "reused_correct_batch_peak_traced_allocation": summary(correction_memory[key], "bytes"),
        }
    return {
        "case": name, "n": code.n, "k": code.k, "max_weight": radius,
        "verification": {
            "ordered_table_equal": True,
            "ordered_table_entries": verified["ordered_table_entries"],
            "all_bounded_corrections_equal_and_successful": True,
            "verified_errors": verified["verified_errors"],
        },
        "correct_batch": {
            "errors": [[list(error.x), list(error.z)] for error in errors],
            "repeats": correction_repeats,
            "calls_per_sample": len(errors) * correction_repeats,
        },
        "cold_plan_preparation": preparation,
        "measurements": measurements,
        "constructor_scalar_over_live_median_ratio": (
            statistics.median(construction["scalar_reference"])
            / statistics.median(construction["live"])
        ),
    }


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=positive_int, default=3)
    parser.add_argument("--memory-samples", type=positive_int, default=1)
    parser.add_argument("--correction-repeats", type=positive_int, default=2)
    args = parser.parse_args()
    # All code creation and exhaustive parity checks finish before ANY timing.
    cyclic, shor, golay, control = (
        cyclic_five_qutrit_code(), shor_nine_qutrit_code(),
        ternary_golay_qutrit_code(), k0_control(),
    )
    cases = (
        ("cyclic_five", cyclic, 1),
        ("shor_nine", shor, 1),
        ("golay_eleven", golay, 2),
        ("small_k0", control, 2),
        ("zero_weight", cyclic, 0),
    )
    checked = [verify(code, radius) for _, code, radius in cases]
    source_paths = [Path(__file__).resolve()] + [
        ROOT / "src/qec/decoder/qutrit" / name
        for name in ("exact.py", "stabilizer.py", "gf3.py", "codes.py")
    ]
    report = {
        "schema": "qutrit-construction-benchmark-v1",
        "environment": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "timer": "time.perf_counter_ns",
            "gc_enabled": gc.isenabled(),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "reference": {
            "origin": "pre-optimization qec/decoder/qutrit/exact.py",
            "pinned_original_file_sha256": PINNED_EXACT_SHA256,
            "embedded_builder_source_sha256": hashlib.sha256(
                inspect.getsource(ScalarReferenceDecoder.__post_init__).encode()
            ).hexdigest(),
            "independence": "original scalar constructor; shared unchanged Pauli, code algebra and public correct/decode",
        },
        "configuration": vars(args),
        "scope": {
            "claim": "constructor cost only, not FER, physical performance or general decoding throughput",
            "cold": "fresh decoder and fresh plan each call; imports and CPU caches are not cold",
            "excluded": "code factories, error generation, parity validation, GC collection and imports",
            "memory": "separate tracemalloc passes; Python traced allocations, not RSS or total native memory",
            "timing": "raw observations vary by host/load; no pass/fail thresholds",
        },
        "cases": [
            benchmark_case(name, code, radius, verified, args.samples,
                           args.memory_samples, args.correction_repeats)
            for (name, code, radius), verified in zip(cases, checked)
        ],
    }
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
