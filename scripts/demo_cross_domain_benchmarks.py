"""v142.4.2 — Cross-domain deterministic benchmark demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qec.analysis.convergence_engine import evaluate_convergence_engine
from qec.analysis.cross_domain_convergence_benchmarks import evaluate_cross_domain_benchmark
from qec.analysis.deterministic_execution_wrapper import evaluate_deterministic_execution_wrapper
from qec.analysis.demo_utils import build_snapshots
from qec.analysis.generalized_invariant_detector import evaluate_generalized_invariant_detector
from qec.analysis.iterative_system_abstraction_layer import evaluate_iterative_system_abstraction



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
    print("IRIS Cross-Domain Benchmark Demo (v142.4.2)")
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
        f"| {'Domain':<12} | {'Total Steps':>11} | {'Effective Steps':>15} | {'Cutoff':>6} | "
        f"{'Structural Redundancy':>22} | {'Invariant Density':>18} | {'Convergence Speedup':>20} | "
        f"{'Efficiency Gain':>16} | {'Label':<8} |"
    )
    for receipt in receipts:
        print(
            f"| {receipt.domain:<12} | "
            f"{receipt.signal.iterations_total:>11} | "
            f"{receipt.signal.iterations_effective:>15} | "
            f"{receipt.signal.cutoff_step:>6} | "
            f"{receipt.signal.structural_redundancy_ratio:>22.3f} | "
            f"{receipt.signal.invariant_density:>18.3f} | "
            f"{receipt.signal.convergence_speedup:>20.3f} | "
            f"{receipt.signal.efficiency_gain:>16.3f} | "
            f"{receipt.decision.benchmark_label:<8} |"
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
