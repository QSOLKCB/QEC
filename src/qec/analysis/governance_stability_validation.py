from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qec.analysis.canonical_hashing import sha256_hex


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _require_sha256(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise _invalid_input()
    return value


def _canonical_hashes(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise _invalid_input()
    validated = tuple(_require_sha256(value) for value in values)
    canonical = tuple(sorted(set(validated)))
    if not canonical:
        raise _invalid_input()
    return canonical


def _stability_payload(
    scenarios: tuple["GovernanceStabilityScenario", ...],
    result: "GovernanceStabilityResult",
) -> dict[str, object]:
    return {
        "scenarios": tuple(
            {
                "scenario_id": scenario.scenario_id,
                "input_memory_hashes": scenario.input_memory_hashes,
                "decision_hashes": scenario.decision_hashes,
                "selected_decision_hash": scenario.selected_decision_hash,
            }
            for scenario in scenarios
        ),
        "result": {
            "baseline_selected_hash": result.baseline_selected_hash,
            "stable": result.stable,
            "scenario_count": result.scenario_count,
            "perturbation_count": result.perturbation_count,
            "stability_score": result.stability_score,
        },
    }


@dataclass(frozen=True)
class GovernanceStabilityScenario:
    scenario_id: str
    input_memory_hashes: tuple[str, ...]
    decision_hashes: tuple[str, ...]
    selected_decision_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise _invalid_input()
        canonical_memory = _canonical_hashes(self.input_memory_hashes)
        canonical_decisions = _canonical_hashes(self.decision_hashes)
        selected = _require_sha256(self.selected_decision_hash)
        if selected not in canonical_decisions:
            raise _invalid_input()
        object.__setattr__(self, "input_memory_hashes", canonical_memory)
        object.__setattr__(self, "decision_hashes", canonical_decisions)
        object.__setattr__(self, "selected_decision_hash", selected)


@dataclass(frozen=True)
class GovernanceStabilityResult:
    baseline_selected_hash: str
    stable: bool
    scenario_count: int
    perturbation_count: int
    stability_score: float

    def __post_init__(self) -> None:
        _require_sha256(self.baseline_selected_hash)
        if self.stable is not True:
            raise _invalid_input()
        if not isinstance(self.scenario_count, int) or self.scenario_count <= 0:
            raise _invalid_input()
        if not isinstance(self.perturbation_count, int) or self.perturbation_count != self.scenario_count - 1:
            raise _invalid_input()
        if self.stability_score != 1.0:
            raise _invalid_input()


@dataclass(frozen=True)
class GovernanceStabilityReceipt:
    scenarios: tuple[GovernanceStabilityScenario, ...]
    result: GovernanceStabilityResult
    stability_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.scenarios, tuple) or not self.scenarios:
            raise _invalid_input()
        if not isinstance(self.result, GovernanceStabilityResult):
            raise _invalid_input()
        if self.result.scenario_count != len(self.scenarios):
            raise _invalid_input()
        if self.scenarios[0].selected_decision_hash != self.result.baseline_selected_hash:
            raise _invalid_input()
        stability_hash = _require_sha256(self.stability_hash)
        expected = sha256_hex(self._payload_without_hash())
        if stability_hash != expected:
            raise _invalid_input()
        object.__setattr__(self, "stability_hash", stability_hash)

    def _payload_without_hash(self) -> dict[str, object]:
        return _stability_payload(self.scenarios, self.result)


def validate_governance_stability(
    scenarios: Sequence[GovernanceStabilityScenario],
) -> GovernanceStabilityReceipt:
    if not isinstance(scenarios, Sequence):
        raise _invalid_input()
    if len(scenarios) == 0:
        raise _invalid_input()

    canonical_scenarios: list[GovernanceStabilityScenario] = []
    for scenario in scenarios:
        if not isinstance(scenario, GovernanceStabilityScenario):
            raise _invalid_input()
        canonical_scenarios.append(scenario)

    canonical_scenarios.sort(
        key=lambda s: (s.input_memory_hashes, s.decision_hashes, s.scenario_id)
    )

    seen_ids: set[str] = set()
    for scenario in canonical_scenarios:
        if scenario.scenario_id in seen_ids:
            raise _invalid_input()
        seen_ids.add(scenario.scenario_id)

    baseline = canonical_scenarios[0]
    for scenario in canonical_scenarios:
        if scenario.input_memory_hashes != baseline.input_memory_hashes:
            raise _invalid_input()
        if scenario.decision_hashes != baseline.decision_hashes:
            raise _invalid_input()
        if scenario.selected_decision_hash != baseline.selected_decision_hash:
            raise _invalid_input()

    result = GovernanceStabilityResult(
        baseline_selected_hash=baseline.selected_decision_hash,
        stable=True,
        scenario_count=len(canonical_scenarios),
        perturbation_count=len(canonical_scenarios) - 1,
        stability_score=1.0,
    )

    canonical_scenarios_tuple = tuple(canonical_scenarios)
    stability_hash = sha256_hex(_stability_payload(canonical_scenarios_tuple, result))
    return GovernanceStabilityReceipt(
        scenarios=canonical_scenarios_tuple,
        result=result,
        stability_hash=stability_hash,
    )
