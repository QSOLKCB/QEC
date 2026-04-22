"""v142.4.2 — Cross-domain deterministic benchmark demo."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from qec.analysis.convergence_engine import evaluate_convergence_engine
from qec.analysis.cross_domain_convergence_benchmarks import evaluate_cross_domain_benchmark
from qec.analysis.deterministic_execution_wrapper import evaluate_deterministic_execution_wrapper
from qec.analysis.generalized_invariant_detector import evaluate_generalized_invariant_detector
from qec.analysis.iterative_system_abstraction_layer import (
    IterativeStateSnapshot,
    evaluate_iterative_system_abstraction,
)


def build_snapshots(
    state_ids: list[str],
    metrics: list[float],
) -> tuple[IterativeStateSnapshot, ...]:
    if len(state_ids) != len(metrics):
        raise ValueError("state_ids and metrics must have the same length")

    snapshots: list[IterativeStateSnapshot] = []
    for index, (state_id, metric) in enumerate(zip(state_ids, metrics)):
        snapshots.append(
            IterativeStateSnapshot(
                step_index=index,
                state_id=state_id,
                state_payload={"state": state_id},
                convergence_metric=metric,
                active=True,
            )
        )
    return tuple(snapshots)


def _run_domain(domain: str, state_ids: list[str], metrics: list[float]):
    snapshots = build_snapshots(state_ids=state_ids, metrics=metrics)
    execution_receipt = evaluate_iterative_system_abstraction(snapshots, version="v142.0")
    invariant_receipt = evaluate_generalized_invariant_detector(execution_receipt, version="v142.1")
    convergence_receipt = evaluate_convergence_engine(execution_receipt, invariant_receipt, version="v142.2")
    wrapper_receipt = evaluate_deterministic_execution_wrapper(
        execution_receipt,
        invariant_receipt,
        convergence_receipt,
        version="v142.3",
    )
    benchmark_receipt = evaluate_cross_domain_benchmark(
        domain,
        execution_receipt,
        invariant_receipt,
        convergence_receipt,
        wrapper_receipt,
        version="v142.4",
    )

    print(f"=== DOMAIN: {domain} ===")
    print("Canonical benchmark receipt")
    print(benchmark_receipt.to_canonical_json())
    print("Compact human-readable summary")
    print(
        "SUMMARY"
        f" | {domain}"
        f" | total={benchmark_receipt.signal.iterations_total}"
        f" | effective={benchmark_receipt.signal.iterations_effective}"
        f" | cutoff={benchmark_receipt.signal.cutoff_step}"
        f" | redundancy={benchmark_receipt.signal.structural_redundancy_ratio:.3f}"
        f" | efficiency={benchmark_receipt.signal.efficiency_gain:.3f}"
        f" | label={benchmark_receipt.decision.benchmark_label}"
    )

    return benchmark_receipt


def main() -> None:
    domains: tuple[tuple[str, list[str], list[float]], ...] = (
        (
            "transformers",
            ["T0", "T1", "T2", "T3", "T4", "T4", "T4", "T4"],
            [0.20, 0.50, 0.70, 0.85, 0.92, 0.95, 0.95, 0.95],
        ),
        (
            "diffusion",
            ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D6", "D6"],
            [0.10, 0.30, 0.50, 0.70, 0.85, 0.90, 0.91, 0.91, 0.91],
        ),
        (
            "gnn",
            ["A", "B", "A", "B", "C", "D", "D"],
            [0.30, 0.40, 0.35, 0.45, 0.50, 0.80, 0.95],
        ),
        (
            "physics",
            ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        ),
    )

    receipts = [
        _run_domain(domain=domain, state_ids=state_ids, metrics=metrics)
        for domain, state_ids, metrics in domains
    ]

    print("=== BENCHMARK TABLE ===")
    print(
        "| Domain | Total Steps | Effective Steps | Cutoff | Structural Redundancy"
        " | Invariant Density | Convergence Speedup | Efficiency Gain | Label |"
    )
    for receipt in receipts:
        print(
            f"| {receipt.domain}"
            f" | {receipt.signal.iterations_total}"
            f" | {receipt.signal.iterations_effective}"
            f" | {receipt.signal.cutoff_step}"
            f" | {receipt.signal.structural_redundancy_ratio:.3f}"
            f" | {receipt.signal.invariant_density:.3f}"
            f" | {receipt.signal.convergence_speedup:.3f}"
            f" | {receipt.signal.efficiency_gain:.3f}"
            f" | {receipt.decision.benchmark_label} |"
        )

    print("=== INTERPRETATION ===")
    for receipt in receipts:
        print(
            f"{receipt.domain}: IRIS detected {receipt.decision.benchmark_label} optimization potential"
            f" with redundancy={receipt.signal.structural_redundancy_ratio:.3f}"
            f" and efficiency_gain={receipt.signal.efficiency_gain:.3f}."
        )


if __name__ == "__main__":
    main()
