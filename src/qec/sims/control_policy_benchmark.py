# SPDX-License-Identifier: MIT
"""Control policy sweep benchmarking — v134.6.0.

Deterministic benchmarking layer for comparing multiple supervisory
control policies on a qudit lattice.

Each policy is a frozen control table mapping stability labels to
(action, damping_factor, recovery_required) tuples.  The benchmark
runner executes a closed-loop simulation for each policy:

    for each step:
        1. coupled evolve
        2. analyze stability
        3. apply policy decision (damping)
        4. record trajectory

Results are scored deterministically:

    score = stability_ratio - 0.5 * recovery_ratio - 0.25 * oscillation_ratio

Higher score = better policy.

All operations are pure, immutable, tuple-only, and replay-safe.
No randomness. No plotting. No file IO. No heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.qudit_lattice_engine import (
    QuditFieldCell,
    QuditLatticeSnapshot,
)
from qec.sims.qudit_coupling_dynamics import coupled_evolve_step
from qec.sims.propagation_stability_analysis import _DIVERGENCE_THRESHOLD


# ── Policy tables ────────────────────────────────────────────────
# Each policy is a tuple of (label, action, damping_factor, recovery_required)
# rows sorted by label for canonical ordering.

_NOMINAL_POLICY: Tuple[Tuple[str, str, float, bool], ...] = (
    ("divergent", "recover", 0.1, True),
    ("fixed_point", "maintain", 1.0, False),
    ("oscillatory", "damp", 0.5, False),
    ("stable", "observe", 1.0, False),
)

_AGGRESSIVE_DAMPING_POLICY: Tuple[Tuple[str, str, float, bool], ...] = (
    ("divergent", "recover", 0.05, True),
    ("fixed_point", "maintain", 1.0, False),
    ("oscillatory", "damp", 0.2, False),
    ("stable", "observe", 0.9, False),
)

_RECOVERY_FIRST_POLICY: Tuple[Tuple[str, str, float, bool], ...] = (
    ("divergent", "recover", 0.01, True),
    ("fixed_point", "maintain", 1.0, False),
    ("oscillatory", "recover", 0.1, True),
    ("stable", "observe", 1.0, False),
)

BUILTIN_POLICIES: Tuple[Tuple[str, Tuple[Tuple[str, str, float, bool], ...]], ...] = (
    ("aggressive_damping", _AGGRESSIVE_DAMPING_POLICY),
    ("nominal", _NOMINAL_POLICY),
    ("recovery_first", _RECOVERY_FIRST_POLICY),
)


def _policy_to_map(
    policy: Tuple[Tuple[str, str, float, bool], ...],
) -> dict[str, Tuple[str, float, bool]]:
    """Convert a policy table to a label -> (action, damping, recovery) map."""
    return {
        label: (action, damping, recovery)
        for label, action, damping, recovery in policy
    }


# ── Result dataclass ─────────────────────────────────────────────


@dataclass(frozen=True)
class PolicyBenchmarkResult:
    """Frozen benchmark result for a single policy run.

    Fields
    ------
    policy_name : str
        Name of the policy that was benchmarked.
    mean_final_amplitude : float
        Mean field amplitude at the end of the simulation.
    steps_to_first_stability : int
        Step index at which fixed_point was first reached, or a
        repeated stable fingerprint was confirmed.
        Equal to total steps if stability was never reached.
    recovery_count : int
        Total number of steps where recovery was triggered.
    oscillation_count : int
        Total number of steps classified as oscillatory.
    score : float
        Deterministic composite score (higher is better).
    """

    policy_name: str
    mean_final_amplitude: float
    steps_to_first_stability: int
    recovery_count: int
    oscillation_count: int
    score: float


# ── Stability analysis (single-step) ────────────────────────────


def _single_step_stability_label(
    prev: QuditLatticeSnapshot,
    curr: QuditLatticeSnapshot,
) -> str:
    """Classify stability between two consecutive snapshots.

    Uses the same classification logic as PropagationStabilityReport
    but applied to a single transition.

    Returns one of: "fixed_point", "stable", "oscillatory", "divergent".
    """
    if curr.mean_field_amplitude > _DIVERGENCE_THRESHOLD:
        return "divergent"

    changed = 0
    for p, c in zip(prev.cells, curr.cells):
        if p.local_state != c.local_state:
            changed += 1

    if changed == 0:
        return "fixed_point"

    return "stable"


# ── Feedback application ─────────────────────────────────────────


def _apply_damping(
    snapshot: QuditLatticeSnapshot,
    damping_factor: float,
) -> QuditLatticeSnapshot:
    """Apply a damping factor to all cell amplitudes.

    Pure, deterministic. Returns snapshot unchanged when factor == 1.0.
    """
    if damping_factor == 1.0:
        return snapshot

    new_cells: list[QuditFieldCell] = []
    for cell in snapshot.cells:
        new_cells.append(QuditFieldCell(
            x_index=cell.x_index,
            y_index=cell.y_index,
            epoch_index=cell.epoch_index,
            qudit_dimension=cell.qudit_dimension,
            local_state=cell.local_state,
            field_amplitude=cell.field_amplitude * damping_factor,
        ))

    cells = tuple(new_cells)
    mean_amp = snapshot.mean_field_amplitude * damping_factor

    return QuditLatticeSnapshot(
        cells=cells,
        width=snapshot.width,
        height=snapshot.height,
        epoch_index=snapshot.epoch_index,
        mean_field_amplitude=mean_amp,
        active_state_count=snapshot.active_state_count,
    )


# ── Scoring ──────────────────────────────────────────────────────


def compute_policy_score(
    steps: int,
    first_stable_step: int,
    recovery_count: int,
    oscillation_count: int,
) -> float:
    """Compute a deterministic policy score.

    Formula:
        stability_ratio = 1.0 - (first_stable_step / steps)
        recovery_ratio  = recovery_count / steps
        oscillation_ratio = oscillation_count / steps
        score = stability_ratio - 0.5 * recovery_ratio - 0.25 * oscillation_ratio

    Higher stability + lower recovery count = better.

    Parameters
    ----------
    steps : int
        Total number of benchmark steps.
    first_stable_step : int
        Step index where stability was first reached (steps if never).
    recovery_count : int
        Number of steps triggering recovery.
    oscillation_count : int
        Number of steps classified as oscillatory.

    Returns
    -------
    float
        Deterministic composite score.
    """
    if steps == 0:
        return 0.0

    stability_ratio = 1.0 - (first_stable_step / steps)
    recovery_ratio = recovery_count / steps
    oscillation_ratio = oscillation_count / steps

    return stability_ratio - 0.5 * recovery_ratio - 0.25 * oscillation_ratio


# ── Single policy runner ─────────────────────────────────────────


def _run_single_policy(
    initial_snapshot: QuditLatticeSnapshot,
    policy_name: str,
    policy_table: Tuple[Tuple[str, str, float, bool], ...],
    steps: int,
) -> PolicyBenchmarkResult:
    """Run a closed-loop benchmark for a single policy.

    Closed-loop simulation:
        1. coupled evolve step
        2. classify stability (prev vs curr)
        3. look up policy action
        4. apply damping
        5. record metrics

    Returns a frozen PolicyBenchmarkResult.
    """
    policy_map = _policy_to_map(policy_table)
    current = initial_snapshot
    recovery_count = 0
    oscillation_count = 0
    first_stable_step = steps  # default: never reached

    # Fingerprint tracking for oscillation detection (mirrors
    # propagation_stability_analysis._state_only_fingerprint logic).
    seen_states: dict[Tuple[int, ...], int] = {}
    seen_states[tuple(c.local_state for c in current.cells)] = -1

    # Count consecutive fixed_point steps for stability confirmation.
    consecutive_fixed = 0

    for step_idx in range(steps):
        prev = current
        current = coupled_evolve_step(current)

        # Fingerprint-based label assignment
        fp = tuple(c.local_state for c in current.cells)
        base_label = _single_step_stability_label(prev, current)

        if base_label in ("stable", "fixed_point") and fp in seen_states:
            label = "oscillatory"
        else:
            label = base_label

        if label != "oscillatory":
            seen_states[fp] = step_idx

        # Look up policy; fall back to observe/no-change for unknown labels
        entry = policy_map.get(label, ("observe", 1.0, False))
        _action, damping, recovery = entry

        if recovery:
            recovery_count += 1
        if label == "oscillatory":
            oscillation_count += 1

        # Track first stable step: require fixed_point or two
        # consecutive stable observations to confirm stability.
        if label == "fixed_point":
            consecutive_fixed += 1
            if consecutive_fixed >= 1 and first_stable_step == steps:
                first_stable_step = step_idx
        else:
            consecutive_fixed = 0

        current = _apply_damping(current, damping)

    score = compute_policy_score(
        steps=steps,
        first_stable_step=first_stable_step,
        recovery_count=recovery_count,
        oscillation_count=oscillation_count,
    )

    return PolicyBenchmarkResult(
        policy_name=policy_name,
        mean_final_amplitude=current.mean_field_amplitude,
        steps_to_first_stability=first_stable_step,
        recovery_count=recovery_count,
        oscillation_count=oscillation_count,
        score=score,
    )


# ── Public benchmark runner ──────────────────────────────────────


def run_policy_benchmark(
    initial_snapshot: QuditLatticeSnapshot,
    policies: Tuple[Tuple[str, Tuple[Tuple[str, str, float, bool], ...]], ...] | None = None,
    steps: int = 20,
) -> Tuple[PolicyBenchmarkResult, ...]:
    """Benchmark multiple supervisory policies on a shared initial state.

    Each policy is executed independently from the same initial snapshot,
    producing comparable results.

    Parameters
    ----------
    initial_snapshot : QuditLatticeSnapshot
        Starting lattice state (shared across all policies).
    policies : tuple of (name, policy_table) pairs, optional
        Policies to benchmark. Defaults to BUILTIN_POLICIES
        (nominal, aggressive_damping, recovery_first).
    steps : int
        Number of closed-loop simulation steps per policy (default 20).

    Returns
    -------
    tuple of PolicyBenchmarkResult
        One result per policy, sorted by policy_name for deterministic ordering.

    Raises
    ------
    ValueError
        If steps < 1 or policies is empty.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    if policies is None:
        policies = BUILTIN_POLICIES

    if len(policies) == 0:
        raise ValueError("policies must not be empty")

    results: list[PolicyBenchmarkResult] = []
    for policy_name, policy_table in policies:
        result = _run_single_policy(
            initial_snapshot=initial_snapshot,
            policy_name=policy_name,
            policy_table=policy_table,
            steps=steps,
        )
        results.append(result)

    # Sort by policy_name for deterministic ordering
    results.sort(key=lambda r: r.policy_name)
    return tuple(results)


def render_benchmark_summary(
    results: Tuple[PolicyBenchmarkResult, ...],
) -> str:
    """Render a canonical ASCII summary of benchmark results.

    Parameters
    ----------
    results : tuple of PolicyBenchmarkResult
        Benchmark results to summarize.

    Returns
    -------
    str
        Deterministic ASCII table.
    """
    lines: list[str] = []
    lines.append("Policy Benchmark Summary")
    lines.append("=" * 60)
    header = (
        f"{'Policy':<24} {'Score':>8} {'Recov':>6} "
        f"{'Oscil':>6} {'FinalAmp':>10}"
    )
    lines.append(header)
    lines.append("-" * 60)

    for r in results:
        row = (
            f"{r.policy_name:<24} {r.score:>8.4f} "
            f"{r.recovery_count:>6d} {r.oscillation_count:>6d} "
            f"{r.mean_final_amplitude:>10.6f}"
        )
        lines.append(row)

    lines.append("-" * 60)
    if results:
        best = max(results, key=lambda r: r.score)
        lines.append(f"Best: {best.policy_name} (score={best.score:.4f})")

    return "\n".join(lines)
