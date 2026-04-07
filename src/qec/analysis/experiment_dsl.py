from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

EXPERIMENT_DSL_SCHEMA_VERSION = "v137.10.1"
_ALLOWED_COMPARATORS = frozenset({"lt", "le", "eq", "ge", "gt", "between"})


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | tuple["JsonValue", ...] | tuple[tuple[str, "JsonValue"], ...]


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_token(value: Any, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required")
    if callable(value):
        raise ValueError(f"{name} must not be callable")
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be a non-empty string")
    return token


def _normalize_optional_text(value: Any, *, name: str) -> str:
    if value is None:
        return ""
    if callable(value):
        raise ValueError(f"{name} must not be callable")
    return str(value).strip()


def _normalize_finite_float(value: Any, *, name: str) -> float:
    if callable(value):
        raise ValueError(f"{name} must not be callable")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _normalize_string_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    normalized = tuple(_normalize_token(item, name=name) for item in list(values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicates")
    return normalized


def _normalize_json_value(value: Any, *, path: str) -> JsonValue:
    if callable(value):
        raise ValueError(f"{path} must not be callable")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{path} floats must be finite")
        return numeric
    if isinstance(value, Mapping):
        items: list[tuple[str, JsonValue]] = []
        for key, sub in value.items():
            norm_key = _normalize_token(key, name=f"{path} key")
            items.append((norm_key, _normalize_json_value(sub, path=f"{path}.{norm_key}")))
        items.sort(key=lambda item: item[0])
        deduped: list[tuple[str, JsonValue]] = []
        seen: set[str] = set()
        for key, sub in items:
            if key in seen:
                raise ValueError(f"duplicate key after normalization at {path}: {key}")
            seen.add(key)
            deduped.append((key, sub))
        return tuple(deduped)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_normalize_json_value(item, path=path) for item in list(value))
    raise ValueError(f"{path} contains unsupported type: {type(value).__name__}")


def _json_value_to_python(value: JsonValue) -> Any:
    if isinstance(value, tuple):
        if value and isinstance(value[0], tuple) and len(value[0]) == 2 and isinstance(value[0][0], str):
            return {key: _json_value_to_python(sub) for key, sub in value}  # type: ignore[misc]
        return [_json_value_to_python(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperimentVariable:
    variable_id: str
    name: str
    role: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable_id": self.variable_id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
        }


@dataclass(frozen=True)
class ExperimentMeasurement:
    measurement_id: str
    observable_proxy: str
    units: str
    measurement_method: str
    variable_ids: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "observable_proxy": self.observable_proxy,
            "units": self.units,
            "measurement_method": self.measurement_method,
            "variable_ids": list(self.variable_ids),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ExperimentCriterion:
    criterion_id: str
    measurement_id: str
    comparator: str
    target_value: float | None
    lower_bound: float | None
    upper_bound: float | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "criterion_id": self.criterion_id,
            "measurement_id": self.measurement_id,
            "comparator": self.comparator,
        }
        if self.target_value is not None:
            payload["target_value"] = self.target_value
        if self.lower_bound is not None:
            payload["lower_bound"] = self.lower_bound
        if self.upper_bound is not None:
            payload["upper_bound"] = self.upper_bound
        return payload


@dataclass(frozen=True)
class ExperimentStep:
    step_order: int
    step_id: str
    action: str
    variable_ids: tuple[str, ...]
    measurement_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_order": self.step_order,
            "step_id": self.step_id,
            "action": self.action,
            "variable_ids": list(self.variable_ids),
            "measurement_ids": list(self.measurement_ids),
        }


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: str
    title: str
    objective: str
    linked_hypothesis_ids: tuple[str, ...]
    variables: tuple[ExperimentVariable, ...]
    controls: tuple[str, ...]
    procedure_steps: tuple[ExperimentStep, ...]
    measurements: tuple[ExperimentMeasurement, ...]
    acceptance_criteria: tuple[ExperimentCriterion, ...]
    tags: tuple[str, ...]
    notes: str
    provenance: tuple[tuple[str, JsonValue], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "objective": self.objective,
            "linked_hypothesis_ids": list(self.linked_hypothesis_ids),
            "variables": [item.to_dict() for item in self.variables],
            "controls": list(self.controls),
            "procedure_steps": [item.to_dict() for item in self.procedure_steps],
            "measurements": [item.to_dict() for item in self.measurements],
            "acceptance_criteria": [item.to_dict() for item in self.acceptance_criteria],
            "tags": list(self.tags),
            "notes": self.notes,
            "provenance": _json_value_to_python(self.provenance),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ExperimentReceipt:
    schema_version: str
    experiment_hash: str
    linked_hypothesis_ids: tuple[str, ...]
    variable_count: int
    measurement_count: int
    step_count: int
    criterion_count: int
    validation_passed: bool
    artifact_byte_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_hash": self.experiment_hash,
            "linked_hypothesis_ids": list(self.linked_hypothesis_ids),
            "variable_count": self.variable_count,
            "measurement_count": self.measurement_count,
            "step_count": self.step_count,
            "criterion_count": self.criterion_count,
            "validation_passed": self.validation_passed,
            "artifact_byte_length": self.artifact_byte_length,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_experiment_spec(raw_spec: Mapping[str, Any]) -> ExperimentSpec:
    if not isinstance(raw_spec, Mapping):
        raise ValueError("raw_spec must be a mapping")

    schema_version = _normalize_optional_text(raw_spec.get("schema_version"), name="schema_version") or EXPERIMENT_DSL_SCHEMA_VERSION
    title = _normalize_token(raw_spec.get("title"), name="title")
    objective = _normalize_token(raw_spec.get("objective"), name="objective")

    linked_hypothesis_ids = tuple(sorted(_normalize_string_tuple(raw_spec.get("linked_hypothesis_ids"), name="linked_hypothesis_ids")))
    controls = tuple(sorted(_normalize_string_tuple(raw_spec.get("controls"), name="controls")))
    tags = tuple(sorted(_normalize_string_tuple(raw_spec.get("tags"), name="tags")))
    notes = _normalize_optional_text(raw_spec.get("notes"), name="notes")

    raw_variables = raw_spec.get("variables") or []
    if isinstance(raw_variables, (str, bytes)) or not isinstance(raw_variables, Sequence):
        raise ValueError("variables must be a sequence")
    variables: list[ExperimentVariable] = []
    for idx, raw in enumerate(list(raw_variables)):
        if not isinstance(raw, Mapping):
            raise ValueError(f"variable at index {idx} must be a mapping")
        variables.append(
            ExperimentVariable(
                variable_id=_normalize_token(raw.get("variable_id"), name="variable_id"),
                name=_normalize_optional_text(raw.get("name"), name="variable.name"),
                role=_normalize_optional_text(raw.get("role"), name="variable.role"),
                description=_normalize_optional_text(raw.get("description"), name="variable.description"),
            )
        )
    variables = sorted(variables, key=lambda item: item.variable_id)

    raw_measurements = raw_spec.get("measurements") or []
    if isinstance(raw_measurements, (str, bytes)) or not isinstance(raw_measurements, Sequence):
        raise ValueError("measurements must be a sequence")
    measurements: list[ExperimentMeasurement] = []
    for idx, raw in enumerate(list(raw_measurements)):
        if not isinstance(raw, Mapping):
            raise ValueError(f"measurement at index {idx} must be a mapping")
        measurements.append(
            ExperimentMeasurement(
                measurement_id=_normalize_token(raw.get("measurement_id"), name="measurement_id"),
                observable_proxy=_normalize_token(raw.get("observable_proxy"), name="observable_proxy"),
                units=_normalize_token(raw.get("units"), name="units"),
                measurement_method=_normalize_token(raw.get("measurement_method"), name="measurement_method"),
                variable_ids=tuple(sorted(_normalize_string_tuple(raw.get("variable_ids"), name="measurement.variable_ids"))),
                notes=_normalize_optional_text(raw.get("notes"), name="measurement.notes"),
            )
        )
    measurements = sorted(measurements, key=lambda item: item.measurement_id)

    raw_criteria = raw_spec.get("acceptance_criteria") or []
    if isinstance(raw_criteria, (str, bytes)) or not isinstance(raw_criteria, Sequence):
        raise ValueError("acceptance_criteria must be a sequence")
    criteria: list[ExperimentCriterion] = []
    for idx, raw in enumerate(list(raw_criteria)):
        if not isinstance(raw, Mapping):
            raise ValueError(f"criterion at index {idx} must be a mapping")
        comparator = _normalize_token(raw.get("comparator"), name="criterion.comparator").lower()
        target = raw.get("target_value")
        lower = raw.get("lower_bound")
        upper = raw.get("upper_bound")
        criteria.append(
            ExperimentCriterion(
                criterion_id=_normalize_token(raw.get("criterion_id"), name="criterion_id"),
                measurement_id=_normalize_token(raw.get("measurement_id"), name="criterion.measurement_id"),
                comparator=comparator,
                target_value=None if target is None else _normalize_finite_float(target, name="target_value"),
                lower_bound=None if lower is None else _normalize_finite_float(lower, name="lower_bound"),
                upper_bound=None if upper is None else _normalize_finite_float(upper, name="upper_bound"),
            )
        )
    criteria = sorted(criteria, key=lambda item: item.criterion_id)

    raw_steps = raw_spec.get("procedure_steps") or []
    if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
        raise ValueError("procedure_steps must be a sequence")
    steps: list[ExperimentStep] = []
    for idx, raw in enumerate(list(raw_steps)):
        if not isinstance(raw, Mapping):
            raise ValueError(f"step at index {idx} must be a mapping")
        step_order = int(raw.get("step_order"))
        steps.append(
            ExperimentStep(
                step_order=step_order,
                step_id=_normalize_token(raw.get("step_id"), name="step_id"),
                action=_normalize_token(raw.get("action"), name="step.action"),
                variable_ids=tuple(sorted(_normalize_string_tuple(raw.get("variable_ids"), name="step.variable_ids"))),
                measurement_ids=tuple(sorted(_normalize_string_tuple(raw.get("measurement_ids"), name="step.measurement_ids"))),
            )
        )
    steps = sorted(steps, key=lambda item: (item.step_order, item.step_id))

    provenance_raw = raw_spec.get("provenance") or {}
    if not isinstance(provenance_raw, Mapping):
        raise ValueError("provenance must be a mapping")
    provenance = _normalize_json_value(dict(provenance_raw), path="provenance")
    if not isinstance(provenance, tuple):
        raise ValueError("provenance must normalize to an object")

    return ExperimentSpec(
        schema_version=schema_version,
        title=title,
        objective=objective,
        linked_hypothesis_ids=linked_hypothesis_ids,
        variables=tuple(variables),
        controls=controls,
        procedure_steps=tuple(steps),
        measurements=tuple(measurements),
        acceptance_criteria=tuple(criteria),
        tags=tags,
        notes=notes,
        provenance=provenance,
    )


def validate_experiment_spec(spec: ExperimentSpec) -> None:
    if not spec.title:
        raise ValueError("title must not be empty")
    if not spec.objective:
        raise ValueError("objective must not be empty")

    variable_ids = tuple(item.variable_id for item in spec.variables)
    if len(set(variable_ids)) != len(variable_ids):
        raise ValueError("duplicate variable IDs")

    measurement_ids = tuple(item.measurement_id for item in spec.measurements)
    if len(set(measurement_ids)) != len(measurement_ids):
        raise ValueError("duplicate measurement IDs")

    step_orders = tuple(step.step_order for step in spec.procedure_steps)
    if any(order <= 0 for order in step_orders):
        raise ValueError("step_order must be positive")
    if len(set(step_orders)) != len(step_orders):
        raise ValueError("duplicate step order values")

    variable_set = set(variable_ids)
    measurement_set = set(measurement_ids)

    for measurement in spec.measurements:
        if not measurement.observable_proxy:
            raise ValueError(f"measurement {measurement.measurement_id} missing observable_proxy")
        if not measurement.units:
            raise ValueError(f"measurement {measurement.measurement_id} missing units")
        if not measurement.measurement_method:
            raise ValueError(f"measurement {measurement.measurement_id} missing measurement_method")
        for variable_id in measurement.variable_ids:
            if variable_id not in variable_set:
                raise ValueError(f"measurement {measurement.measurement_id} references unknown variable_id: {variable_id}")

    for criterion in spec.acceptance_criteria:
        if criterion.measurement_id not in measurement_set:
            raise ValueError(f"criterion {criterion.criterion_id} references unknown measurement_id: {criterion.measurement_id}")
        if criterion.comparator not in _ALLOWED_COMPARATORS:
            raise ValueError(f"criterion {criterion.criterion_id} has invalid comparator: {criterion.comparator}")
        if criterion.comparator == "between":
            if criterion.lower_bound is None or criterion.upper_bound is None:
                raise ValueError(f"criterion {criterion.criterion_id} comparator between requires lower_bound and upper_bound")
            if criterion.lower_bound > criterion.upper_bound:
                raise ValueError(f"criterion {criterion.criterion_id} has malformed range bounds")
            if criterion.target_value is not None:
                raise ValueError(f"criterion {criterion.criterion_id} comparator between forbids target_value")
        else:
            if criterion.target_value is None:
                raise ValueError(f"criterion {criterion.criterion_id} comparator {criterion.comparator} requires target_value")
            if criterion.lower_bound is not None or criterion.upper_bound is not None:
                raise ValueError(f"criterion {criterion.criterion_id} comparator {criterion.comparator} forbids range bounds")

    for step in spec.procedure_steps:
        for variable_id in step.variable_ids:
            if variable_id not in variable_set:
                raise ValueError(f"step {step.step_id} references unknown variable_id: {variable_id}")
        for measurement_id in step.measurement_ids:
            if measurement_id not in measurement_set:
                raise ValueError(f"step {step.step_id} references unknown measurement_id: {measurement_id}")


def stable_experiment_hash(spec: ExperimentSpec) -> str:
    return hashlib.sha256(spec.to_canonical_bytes()).hexdigest()


def build_experiment_receipt(spec: ExperimentSpec) -> ExperimentReceipt:
    validate_experiment_spec(spec)
    return ExperimentReceipt(
        schema_version=spec.schema_version,
        experiment_hash=stable_experiment_hash(spec),
        linked_hypothesis_ids=spec.linked_hypothesis_ids,
        variable_count=len(spec.variables),
        measurement_count=len(spec.measurements),
        step_count=len(spec.procedure_steps),
        criterion_count=len(spec.acceptance_criteria),
        validation_passed=True,
        artifact_byte_length=len(spec.to_canonical_bytes()),
    )


def compile_experiment_spec(raw_spec: Mapping[str, Any]) -> ExperimentSpec:
    spec = normalize_experiment_spec(raw_spec)
    validate_experiment_spec(spec)
    return spec
