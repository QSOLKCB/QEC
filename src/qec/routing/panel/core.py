# SPDX-License-Identifier: MPL-2.0
"""Deterministic Panel separated-control exchange contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Final, Mapping

from qec.routing.strowger.receipts import validate_receipt as validate_strowger_receipt
from qec.sonify.canonical import (
    canonical_sha256,
    require_int,
    require_nonempty_text,
    validate_sha256,
)

QEC_VERSION = "171.5.0"
TOPOLOGY_SCHEMA = "qec.panel-topology.v1"
DIGIT_REGISTER_SCHEMA = "qec.panel-digit-register.v1"
SENDER_PROGRAM_SCHEMA = "qec.panel-sender-program.v1"
SENDER_REGISTER_RECEIPT_SCHEMA = "qec.panel-sender-register-receipt.v1"
ROUTE_RECEIPT_SCHEMA = "qec.panel-route-receipt.v1"
FAULT_BATTERY_SCHEMA = "qec.panel-fault-battery.v1"
CLAIM_VALIDATION_SCHEMA = "qec.panel-claim-validation.v1"
TRANSLATION_TABLE_SCHEMA = "qec.panel-translation-table.v1"
EQUIVALENCE_SCHEMA = "qec.panel-strowger-equivalence.v1"

_CLAIM_BOUNDARY_VALUES: Final = {
    "classical_routing_only": True,
    "decoder_replacement": False,
    "quantum_hardware_claim": False,
    "payload_mutation_permitted": False,
    "sender_may_force_accept": False,
    "browser_demo_is_canonical_evidence": False,
    "receipt_proves": "deterministic_separated_control_and_declared_route_verification",
    "receipt_does_not_prove": "physical_telephone_fidelity_or_quantum_advantage",
}
CLAIM_BOUNDARY: Final[Mapping[str, object]] = MappingProxyType(_CLAIM_BOUNDARY_VALUES)


def _exact_object(payload: object, *, label: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    if set(payload) != fields:
        raise ValueError(f"{label} must contain exactly the canonical fields")
    return payload


def _exact_list(payload: object, *, label: str) -> list[object]:
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a list")
    return payload


def _require_sorted_unique(values: tuple[str, ...], label: str) -> None:
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique and canonically sorted")


def _payload_sha256(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def _validate_hashed_artifact(payload: object, *, schema: str, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    if payload.get("schema") != schema:
        raise ValueError(f"unexpected {label} schema")
    observed = validate_sha256(payload.get("sha256"), f"{label}.sha256")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    if canonical_sha256(unsigned) != observed:
        raise ValueError(f"{label} hash mismatch")
    return payload


@dataclass(frozen=True)
class MotorGroup:
    name: str
    banks: tuple[str, ...]

    def __post_init__(self) -> None:
        require_nonempty_text(self.name, "motor_group.name")
        if not self.banks:
            raise ValueError("motor_group.banks must not be empty")
        for bank in self.banks:
            require_nonempty_text(bank, "motor_group.bank")
        _require_sorted_unique(self.banks, "motor_group.banks")

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "banks": list(self.banks)}

    @classmethod
    def from_dict(cls, payload: object) -> "MotorGroup":
        item = _exact_object(payload, label="motor group", fields={"name", "banks"})
        banks = _exact_list(item["banks"], label="motor group banks")
        return cls(item["name"], tuple(banks))


@dataclass(frozen=True)
class PanelBank:
    name: str
    motor_group: str
    capacity: int
    selector_min: int
    selector_max: int

    def __post_init__(self) -> None:
        require_nonempty_text(self.name, "bank.name")
        require_nonempty_text(self.motor_group, "bank.motor_group")
        require_int(self.capacity, "bank.capacity", minimum=1, maximum=4096)
        require_int(self.selector_min, "bank.selector_min", minimum=0, maximum=4096)
        require_int(self.selector_max, "bank.selector_max", minimum=0, maximum=4096)
        if self.selector_min > self.selector_max:
            raise ValueError("bank selector_min must be <= selector_max")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "motor_group": self.motor_group,
            "capacity": self.capacity,
            "selector_min": self.selector_min,
            "selector_max": self.selector_max,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PanelBank":
        item = _exact_object(
            payload,
            label="panel bank",
            fields={"name", "motor_group", "capacity", "selector_min", "selector_max"},
        )
        return cls(
            item["name"],
            item["motor_group"],
            item["capacity"],
            item["selector_min"],
            item["selector_max"],
        )


@dataclass(frozen=True)
class PanelPath:
    path_id: str
    bank: str
    selector_position: int
    destination: str

    def __post_init__(self) -> None:
        require_nonempty_text(self.path_id, "path.path_id")
        require_nonempty_text(self.bank, "path.bank")
        require_nonempty_text(self.destination, "path.destination")
        require_int(self.selector_position, "path.selector_position", minimum=0, maximum=4096)

    def as_dict(self) -> dict[str, object]:
        return {
            "path_id": self.path_id,
            "bank": self.bank,
            "selector_position": self.selector_position,
            "destination": self.destination,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PanelPath":
        item = _exact_object(
            payload,
            label="panel path",
            fields={"path_id", "bank", "selector_position", "destination"},
        )
        return cls(item["path_id"], item["bank"], item["selector_position"], item["destination"])


@dataclass(frozen=True)
class PanelTopology:
    topology_id: str
    motor_groups: tuple[MotorGroup, ...]
    banks: tuple[PanelBank, ...]
    paths: tuple[PanelPath, ...]

    def __post_init__(self) -> None:
        require_nonempty_text(self.topology_id, "topology_id")
        if not self.motor_groups or not self.banks or not self.paths:
            raise ValueError("panel topology requires motor groups, banks and paths")
        group_names = tuple(group.name for group in self.motor_groups)
        bank_names = tuple(bank.name for bank in self.banks)
        path_ids = tuple(path.path_id for path in self.paths)
        _require_sorted_unique(group_names, "motor group names")
        _require_sorted_unique(bank_names, "bank names")
        _require_sorted_unique(path_ids, "path ids")
        group_set = set(group_names)
        bank_set = set(bank_names)
        for bank in self.banks:
            if bank.motor_group not in group_set:
                raise ValueError(f"bank {bank.name} references unknown motor group")
        declared_membership = {
            bank_name
            for group in self.motor_groups
            for bank_name in group.banks
        }
        if declared_membership != bank_set:
            raise ValueError("motor-group bank membership must cover topology banks exactly")
        for group in self.motor_groups:
            expected = tuple(bank.name for bank in self.banks if bank.motor_group == group.name)
            if group.banks != expected:
                raise ValueError("motor-group bank membership disagrees with bank records")
        bank_map = {bank.name: bank for bank in self.banks}
        counts = {bank.name: 0 for bank in self.banks}
        coordinates: set[tuple[str, int]] = set()
        for path in self.paths:
            if path.bank not in bank_set:
                raise ValueError(f"path {path.path_id} references unknown bank")
            bank = bank_map[path.bank]
            if not bank.selector_min <= path.selector_position <= bank.selector_max:
                raise ValueError(f"path {path.path_id} exceeds bounded selector movement")
            coordinate = (path.bank, path.selector_position)
            if coordinate in coordinates:
                raise ValueError("panel paths must not share a bank/selector actuation coordinate")
            coordinates.add(coordinate)
            counts[path.bank] += 1
        for bank in self.banks:
            if counts[bank.name] > bank.capacity:
                raise ValueError(f"bank {bank.name} path inventory exceeds declared capacity")

    def as_dict(self) -> dict[str, object]:
        unsigned = {
            "schema": TOPOLOGY_SCHEMA,
            "contract_version": "171.0",
            "topology_id": self.topology_id,
            "motor_groups": [group.as_dict() for group in self.motor_groups],
            "banks": [bank.as_dict() for bank in self.banks],
            "paths": [path.as_dict() for path in self.paths],
        }
        return {**unsigned, "sha256": canonical_sha256(unsigned)}

    def sha256(self) -> str:
        return self.as_dict()["sha256"]  # type: ignore[return-value]

    def path(self, path_id: str) -> PanelPath:
        for path in self.paths:
            if path.path_id == path_id:
                return path
        raise ValueError(f"unknown panel path {path_id}")

    def bank(self, name: str) -> PanelBank:
        for bank in self.banks:
            if bank.name == name:
                return bank
        raise ValueError(f"unknown panel bank {name}")

    @classmethod
    def from_dict(cls, payload: object) -> "PanelTopology":
        item = _validate_hashed_artifact(payload, schema=TOPOLOGY_SCHEMA, label="panel topology")
        if set(item) != {"schema", "contract_version", "topology_id", "motor_groups", "banks", "paths", "sha256"}:
            raise ValueError("panel topology fields are not canonical")
        if item["contract_version"] != "171.0":
            raise ValueError("unexpected panel topology contract version")
        groups = _exact_list(item["motor_groups"], label="motor groups")
        banks = _exact_list(item["banks"], label="panel banks")
        paths = _exact_list(item["paths"], label="panel paths")
        return cls(
            item["topology_id"],
            tuple(MotorGroup.from_dict(value) for value in groups),
            tuple(PanelBank.from_dict(value) for value in banks),
            tuple(PanelPath.from_dict(value) for value in paths),
        )


@dataclass(frozen=True)
class TranslationEntry:
    digits: tuple[int, ...]
    destination: str
    path_candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.digits:
            raise ValueError("translation digits must not be empty")
        for index, digit in enumerate(self.digits):
            require_int(digit, f"translation.digits[{index}]", minimum=0, maximum=4096)
        require_nonempty_text(self.destination, "translation.destination")
        if not self.path_candidates:
            raise ValueError("translation path_candidates must not be empty")
        for value in self.path_candidates:
            require_nonempty_text(value, "translation.path_candidate")
        if len(set(self.path_candidates)) != len(self.path_candidates):
            raise ValueError("translation path_candidates must be unique")

    def as_dict(self) -> dict[str, object]:
        return {
            "digits": list(self.digits),
            "destination": self.destination,
            "path_candidates": list(self.path_candidates),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "TranslationEntry":
        item = _exact_object(
            payload,
            label="translation entry",
            fields={"digits", "destination", "path_candidates"},
        )
        digits = _exact_list(item["digits"], label="translation digits")
        candidates = _exact_list(item["path_candidates"], label="translation candidates")
        return cls(tuple(digits), item["destination"], tuple(candidates))


@dataclass(frozen=True)
class TranslationTable:
    table_id: str
    version: str
    fallback_rule: str
    entries: tuple[TranslationEntry, ...]

    def __post_init__(self) -> None:
        require_nonempty_text(self.table_id, "translation.table_id")
        require_nonempty_text(self.version, "translation.version")
        if self.fallback_rule not in {"none", "first_declared_free_path"}:
            raise ValueError("unsupported deterministic fallback rule")
        if not self.entries:
            raise ValueError("translation table must contain entries")
        keys = tuple((entry.digits, entry.destination) for entry in self.entries)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("translation entries must be unique and canonically sorted")
        for entry in self.entries:
            if len(entry.path_candidates) > 1 and self.fallback_rule == "none":
                raise ValueError("multiple path candidates require a declared deterministic fallback rule")

    def as_dict(self) -> dict[str, object]:
        unsigned = {
            "schema": TRANSLATION_TABLE_SCHEMA,
            "contract_version": "171.3",
            "table_id": self.table_id,
            "version": self.version,
            "fallback_rule": self.fallback_rule,
            "entries": [entry.as_dict() for entry in self.entries],
        }
        return {**unsigned, "sha256": canonical_sha256(unsigned)}

    def sha256(self) -> str:
        return self.as_dict()["sha256"]  # type: ignore[return-value]

    def lookup(self, digits: tuple[int, ...], destination: str) -> TranslationEntry:
        for entry in self.entries:
            if entry.digits == digits and entry.destination == destination:
                return entry
        raise ValueError("no exact panel translation entry")

    @classmethod
    def from_dict(cls, payload: object) -> "TranslationTable":
        item = _validate_hashed_artifact(payload, schema=TRANSLATION_TABLE_SCHEMA, label="translation table")
        if set(item) != {"schema", "contract_version", "table_id", "version", "fallback_rule", "entries", "sha256"}:
            raise ValueError("translation table fields are not canonical")
        if item["contract_version"] != "171.3":
            raise ValueError("unexpected translation table contract version")
        entries = _exact_list(item["entries"], label="translation entries")
        return cls(
            item["table_id"],
            item["version"],
            item["fallback_rule"],
            tuple(TranslationEntry.from_dict(value) for value in entries),
        )


@dataclass(frozen=True)
class PanelRequest:
    request_id: str
    digits: tuple[int, ...]
    epoch: int
    destination: str
    payload: bytes

    def __post_init__(self) -> None:
        require_nonempty_text(self.request_id, "request_id")
        if not self.digits:
            raise ValueError("digits must not be empty")
        for index, digit in enumerate(self.digits):
            require_int(digit, f"digits[{index}]", minimum=0, maximum=4096)
        require_int(self.epoch, "epoch", minimum=0)
        require_nonempty_text(self.destination, "destination")
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")

    @property
    def payload_sha256(self) -> str:
        return _payload_sha256(self.payload)


@dataclass(frozen=True)
class PanelFaultPlan:
    busy_banks: tuple[str, ...] = ()
    stalled_motor_groups: tuple[str, ...] = ()
    translation_corruption: bool = False
    sender_disagreement: bool = False
    unavailable_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, values in (
            ("busy_banks", self.busy_banks),
            ("stalled_motor_groups", self.stalled_motor_groups),
            ("unavailable_paths", self.unavailable_paths),
        ):
            for value in values:
                require_nonempty_text(value, name)
            _require_sorted_unique(values, name)
        if type(self.translation_corruption) is not bool or type(self.sender_disagreement) is not bool:
            raise TypeError("fault flags must be exact bools")

    def as_dict(self) -> dict[str, object]:
        return {
            "busy_banks": list(self.busy_banks),
            "stalled_motor_groups": list(self.stalled_motor_groups),
            "translation_corruption": self.translation_corruption,
            "sender_disagreement": self.sender_disagreement,
            "unavailable_paths": list(self.unavailable_paths),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PanelFaultPlan":
        item = _exact_object(
            payload,
            label="panel fault plan",
            fields={
                "busy_banks",
                "stalled_motor_groups",
                "translation_corruption",
                "sender_disagreement",
                "unavailable_paths",
            },
        )
        busy = _exact_list(item["busy_banks"], label="busy banks")
        stalled = _exact_list(item["stalled_motor_groups"], label="stalled motor groups")
        unavailable = _exact_list(item["unavailable_paths"], label="unavailable paths")
        return cls(
            tuple(busy),
            tuple(stalled),
            item["translation_corruption"],
            item["sender_disagreement"],
            tuple(unavailable),
        )


class PanelEventLog:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def append(self, phase: str, action: str, details: dict[str, object] | None = None) -> None:
        previous = self.events[-1]["event_sha256"] if self.events else None
        unsigned = {
            "sequence": len(self.events),
            "tick": len(self.events),
            "phase": phase,
            "action": action,
            "details": {} if details is None else details,
            "previous_event_sha256": previous,
        }
        self.events.append({**unsigned, "event_sha256": canonical_sha256(unsigned)})


def seal_digit_register(request: PanelRequest) -> dict[str, object]:
    unsigned = {
        "schema": DIGIT_REGISTER_SCHEMA,
        "contract_version": "171.1",
        "request_id": request.request_id,
        "digits": list(request.digits),
        "epoch": request.epoch,
        "destination": request.destination,
        "payload_hex": request.payload.hex(),
        "payload_sha256": request.payload_sha256,
        "payload_length": len(request.payload),
        "sealed": True,
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _request_from_register(register: object) -> PanelRequest:
    item = _validate_hashed_artifact(register, schema=DIGIT_REGISTER_SCHEMA, label="digit register")
    expected = {
        "schema", "contract_version", "request_id", "digits", "epoch", "destination",
        "payload_hex", "payload_sha256", "payload_length", "sealed", "sha256"
    }
    if set(item) != expected or item["contract_version"] != "171.1" or item["sealed"] is not True:
        raise ValueError("digit register is not canonically sealed")
    digits = _exact_list(item["digits"], label="digit register digits")
    payload_hex = item["payload_hex"]
    if not isinstance(payload_hex, str):
        raise ValueError("digit register payload_hex must be text")
    try:
        payload_bytes = bytes.fromhex(payload_hex)
    except ValueError as exc:
        raise ValueError("digit register payload_hex is invalid") from exc
    request = PanelRequest(item["request_id"], tuple(digits), item["epoch"], item["destination"], payload_bytes)
    if request.payload_sha256 != validate_sha256(item["payload_sha256"], "digit register payload_sha256"):
        raise ValueError("digit register payload identity mismatch")
    if len(payload_bytes) != item["payload_length"]:
        raise ValueError("digit register payload length mismatch")
    return request


def compile_sender_program(
    topology: PanelTopology,
    table: TranslationTable,
    register: dict[str, object],
) -> dict[str, object]:
    request = _request_from_register(register)
    entry = table.lookup(request.digits, request.destination)
    candidates: list[dict[str, object]] = []
    for ordinal, path_id in enumerate(entry.path_candidates):
        path = topology.path(path_id)
        if path.destination != request.destination:
            raise ValueError("translation candidate destination disagrees with sealed request")
        bank = topology.bank(path.bank)
        candidates.append({
            "ordinal": ordinal,
            "path_id": path.path_id,
            "bank": bank.name,
            "motor_group": bank.motor_group,
            "selector_position": path.selector_position,
        })
    unsigned = {
        "schema": SENDER_PROGRAM_SCHEMA,
        "contract_version": "171.1",
        "digit_register_sha256": register["sha256"],
        "topology_sha256": topology.sha256(),
        "translation_table_sha256": table.sha256(),
        "translation_table_id": table.table_id,
        "translation_table_version": table.version,
        "fallback_rule": table.fallback_rule,
        "payload_sha256": request.payload_sha256,
        "destination": request.destination,
        "candidate_paths": candidates,
        "sealed": True,
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _validate_sender_program(
    payload: object,
    *,
    register: object | None = None,
    topology: PanelTopology | None = None,
    table: TranslationTable | None = None,
) -> dict[str, object]:
    item = _validate_hashed_artifact(payload, schema=SENDER_PROGRAM_SCHEMA, label="sender program")
    expected_fields = {
        "schema",
        "contract_version",
        "digit_register_sha256",
        "topology_sha256",
        "translation_table_sha256",
        "translation_table_id",
        "translation_table_version",
        "fallback_rule",
        "payload_sha256",
        "destination",
        "candidate_paths",
        "sealed",
        "sha256",
    }
    if set(item) != expected_fields or item["contract_version"] != "171.1" or item["sealed"] is not True:
        raise ValueError("sender program fields are not canonical")
    validate_sha256(item["digit_register_sha256"], "sender program digit_register_sha256")
    validate_sha256(item["topology_sha256"], "sender program topology_sha256")
    validate_sha256(item["translation_table_sha256"], "sender program translation_table_sha256")
    validate_sha256(item["payload_sha256"], "sender program payload_sha256")
    require_nonempty_text(item["translation_table_id"], "sender program translation_table_id")
    require_nonempty_text(item["translation_table_version"], "sender program translation_table_version")
    require_nonempty_text(item["destination"], "sender program destination")
    if item["fallback_rule"] not in {"none", "first_declared_free_path"}:
        raise ValueError("sender program fallback rule is not canonical")
    candidates = _exact_list(item["candidate_paths"], label="sender candidate paths")
    if not candidates:
        raise ValueError("sender program must contain at least one candidate")
    for ordinal, candidate_payload in enumerate(candidates):
        candidate = _exact_object(
            candidate_payload,
            label=f"sender candidate[{ordinal}]",
            fields={"ordinal", "path_id", "bank", "motor_group", "selector_position"},
        )
        if candidate["ordinal"] != ordinal:
            raise ValueError("sender candidate ordinals must be contiguous and canonical")
        require_nonempty_text(candidate["path_id"], f"sender candidate[{ordinal}].path_id")
        require_nonempty_text(candidate["bank"], f"sender candidate[{ordinal}].bank")
        require_nonempty_text(candidate["motor_group"], f"sender candidate[{ordinal}].motor_group")
        require_int(candidate["selector_position"], f"sender candidate[{ordinal}].selector_position", minimum=0, maximum=4096)

    if register is not None:
        request = _request_from_register(register)
        if not isinstance(register, dict):
            raise ValueError("sender register binding requires an object")
        if item["digit_register_sha256"] != register.get("sha256"):
            raise ValueError("sender programme is not bound to the sealed register")
        if item["payload_sha256"] != request.payload_sha256:
            raise ValueError("sender programme payload binding mismatch")
        if item["destination"] != request.destination:
            raise ValueError("sender programme destination binding mismatch")

    if (topology is None) != (table is None):
        raise ValueError("sender program control validation requires topology and table together")
    if topology is not None and table is not None:
        if not isinstance(register, dict):
            raise ValueError("sender program control validation requires the sealed register")
        expected_program = compile_sender_program(topology, table, register)
        if item != expected_program:
            raise ValueError("sender programme does not match canonical compiled control intent")
    return item


def build_sender_register_receipt(
    register: dict[str, object],
    program: dict[str, object],
) -> dict[str, object]:
    request = _request_from_register(register)
    validated_program = _validate_sender_program(program, register=register)
    if validated_program["payload_sha256"] != request.payload_sha256:
        raise ValueError("sender/register receipt payload binding mismatch")
    unsigned = {
        "schema": SENDER_REGISTER_RECEIPT_SCHEMA,
        "contract_version": "171.1",
        "digit_register_sha256": register["sha256"],
        "sender_program_sha256": validated_program["sha256"],
        "payload_sha256": register["payload_sha256"],
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


@dataclass(frozen=True)
class PanelRouteResult:
    outcome: str
    receipt: dict[str, object]


class PanelExchange:
    def __init__(self, topology: PanelTopology, translation_table: TranslationTable) -> None:
        self.topology = topology
        self.translation_table = translation_table

    def _validate_faults(self, faults: PanelFaultPlan) -> None:
        bank_names = {bank.name for bank in self.topology.banks}
        group_names = {group.name for group in self.topology.motor_groups}
        path_ids = {path.path_id for path in self.topology.paths}
        if not set(faults.busy_banks) <= bank_names:
            raise ValueError("busy_banks references unknown bank")
        if not set(faults.stalled_motor_groups) <= group_names:
            raise ValueError("stalled_motor_groups references unknown motor group")
        if not set(faults.unavailable_paths) <= path_ids:
            raise ValueError("unavailable_paths references unknown path")

    def route(self, request: PanelRequest, *, faults: PanelFaultPlan | None = None) -> PanelRouteResult:
        fault_plan = PanelFaultPlan() if faults is None else faults
        self._validate_faults(fault_plan)
        payload_before = request.payload
        log = PanelEventLog()
        register = seal_digit_register(request)
        log.append("register", "digit_register_sealed", {"digit_register_sha256": register["sha256"]})
        program = compile_sender_program(self.topology, self.translation_table, register)
        sender_receipt = build_sender_register_receipt(register, program)
        log.append("control", "sender_program_sealed", {
            "sender_program_sha256": program["sha256"],
            "panel_sender_register_receipt_hash": sender_receipt["sha256"],
        })

        selected: dict[str, object] | None = None
        verification: dict[str, object] = {"verified": False}
        outcome = "control_rejected"

        if fault_plan.translation_corruption:
            observed = canonical_sha256({"expected": self.translation_table.sha256(), "fault": "translation_corruption"})
            log.append("control_verification", "translation_hash_mismatch", {
                "expected": self.translation_table.sha256(), "observed": observed,
            })
            outcome = "translation_corruption"
        elif fault_plan.sender_disagreement:
            observed = canonical_sha256({"expected": program["sha256"], "fault": "sender_disagreement"})
            log.append("control_verification", "sender_disagreement", {
                "expected": program["sha256"], "observed": observed,
            })
            outcome = "sender_disagreement"
        else:
            candidates = program.get("candidate_paths")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("sealed sender programme has no candidates")
            fallback_rule = program.get("fallback_rule")
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    raise ValueError("sender candidate must be an object")
                path_id = candidate.get("path_id")
                if path_id in fault_plan.unavailable_paths:
                    log.append("path", "path_unavailable", {
                        "path_id": path_id,
                        "bank": candidate.get("bank"),
                        "candidate_ordinal": index,
                    })
                    if index == 0 and fallback_rule == "none":
                        break
                    continue
                bank_name = candidate.get("bank")
                if bank_name in fault_plan.busy_banks:
                    log.append("capacity", "bank_busy", {"bank": bank_name, "candidate_ordinal": index})
                    if index == 0 and fallback_rule == "none":
                        break
                    continue
                selected = dict(candidate)
                break
            if selected is None:
                outcome = "capacity_exhausted"
                log.append("capacity", "no_admissible_path", {
                    "fallback_rule": fallback_rule,
                    "unavailable_paths": list(fault_plan.unavailable_paths),
                })
            else:
                motor_group = selected["motor_group"]
                if motor_group in fault_plan.stalled_motor_groups:
                    outcome = "motor_stall"
                    log.append("actuation", "motor_stall", {
                        "motor_group": motor_group,
                        "path_id": selected["path_id"],
                    })
                else:
                    log.append("actuation", "selector_move", {
                        "motor_group": motor_group,
                        "bank": selected["bank"],
                        "from": 0,
                        "to": selected["selector_position"],
                    })
                    log.append("actuation", "path_connected", {
                        "path_id": selected["path_id"],
                        "destination": request.destination,
                    })
                    path = self.topology.path(selected["path_id"])  # type: ignore[arg-type]
                    bank = self.topology.bank(path.bank)
                    payload_after_sha256 = _payload_sha256(request.payload)
                    checks = {
                        "sender_sealed": program.get("sealed") is True,
                        "path_is_compiled_candidate": any(
                            isinstance(candidate, dict) and candidate.get("path_id") == path.path_id
                            for candidate in candidates
                        ),
                        "destination_matches": path.destination == request.destination,
                        "selector_within_bounds": bank.selector_min <= path.selector_position <= bank.selector_max,
                        "payload_unchanged": payload_before == request.payload and payload_after_sha256 == request.payload_sha256,
                    }
                    verification = {"checks": checks, "verified": all(checks.values())}
                    log.append("verification", "independent_route_verification", verification)
                    if verification["verified"]:
                        outcome = "committed"
                        log.append("commit", "commit_verified_route", {"destination": request.destination})
                    else:
                        outcome = "verification_failed"
                        log.append("commit", "reject_unverified_route", {"destination": request.destination})

        receipt_unsigned = {
            "schema": ROUTE_RECEIPT_SCHEMA,
            "qec_version": QEC_VERSION,
            "contract_version": "171.2",
            "topology": self.topology.as_dict(),
            "translation_table": self.translation_table.as_dict(),
            "digit_register": register,
            "sender_program": program,
            "sender_register_receipt": sender_receipt,
            "panel_sender_register_receipt_hash": sender_receipt["sha256"],
            "payload_identity": {
                "before_sha256": request.payload_sha256,
                "after_sha256": _payload_sha256(request.payload),
                "length": len(request.payload),
            },
            "fault_plan": fault_plan.as_dict(),
            "route": selected,
            "verification": verification,
            "outcome": outcome,
            "events": log.events,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
        receipt = {**receipt_unsigned, "sha256": canonical_sha256(receipt_unsigned)}
        return PanelRouteResult(outcome, receipt)


def _validate_event_chain(events: object) -> None:
    values = _exact_list(events, label="panel events")
    previous: str | None = None
    sender_index: int | None = None
    first_actuation: int | None = None
    for index, event in enumerate(values):
        if not isinstance(event, dict):
            raise ValueError("panel event must be an object")
        expected_fields = {"sequence", "tick", "phase", "action", "details", "previous_event_sha256", "event_sha256"}
        if set(event) != expected_fields or event["sequence"] != index or event["tick"] != index:
            raise ValueError("panel event sequence/tick is not canonical")
        observed = validate_sha256(event["event_sha256"], f"events[{index}].event_sha256")
        unsigned = dict(event)
        unsigned.pop("event_sha256")
        if event["previous_event_sha256"] != previous or canonical_sha256(unsigned) != observed:
            raise ValueError("panel event hash chain mismatch")
        if event["action"] == "sender_program_sealed":
            sender_index = index
        if event["phase"] == "actuation" and first_actuation is None:
            first_actuation = index
        previous = observed
    if first_actuation is not None and (sender_index is None or sender_index >= first_actuation):
        raise ValueError("route actuation began before sender sealing")


def validate_route_receipt(receipt: dict[str, object]) -> dict[str, object]:
    item = _validate_hashed_artifact(receipt, schema=ROUTE_RECEIPT_SCHEMA, label="panel route receipt")
    if item.get("qec_version") != QEC_VERSION or item.get("contract_version") != "171.2":
        raise ValueError("unexpected Panel route receipt version")
    if item.get("claim_boundary") != dict(CLAIM_BOUNDARY):
        raise ValueError("Panel claim boundary mismatch")
    topology = PanelTopology.from_dict(item.get("topology"))
    table = TranslationTable.from_dict(item.get("translation_table"))
    register = item.get("digit_register")
    request = _request_from_register(register)
    program = _validate_sender_program(
        item.get("sender_program"),
        register=register,
        topology=topology,
        table=table,
    )
    sender_receipt = _validate_hashed_artifact(
        item.get("sender_register_receipt"), schema=SENDER_REGISTER_RECEIPT_SCHEMA, label="sender register receipt"
    )
    expected_sender_receipt_fields = {
        "schema", "contract_version", "digit_register_sha256", "sender_program_sha256", "payload_sha256", "sha256"
    }
    if set(sender_receipt) != expected_sender_receipt_fields or sender_receipt.get("contract_version") != "171.1":
        raise ValueError("sender/register receipt fields are not canonical")
    if item.get("panel_sender_register_receipt_hash") != sender_receipt.get("sha256"):
        raise ValueError("panel sender/register primary hash mismatch")
    if sender_receipt.get("digit_register_sha256") != program.get("digit_register_sha256"):
        raise ValueError("sender/register receipt register mismatch")
    if sender_receipt.get("sender_program_sha256") != program.get("sha256"):
        raise ValueError("sender/register receipt programme mismatch")
    if sender_receipt.get("payload_sha256") != request.payload_sha256:
        raise ValueError("sender/register receipt payload mismatch")
    _validate_event_chain(item.get("events"))
    faults = PanelFaultPlan.from_dict(item.get("fault_plan"))
    replay = PanelExchange(topology, table).route(request, faults=faults)
    if replay.receipt != receipt:
        raise ValueError("Panel route receipt replay mismatch")
    return {
        "valid": True,
        "replayed": True,
        "panel_sender_register_receipt_hash": sender_receipt["sha256"],
        "panel_separated_control_receipt_hash": receipt["sha256"],
        "outcome": receipt["outcome"],
        "events": len(receipt["events"]),
    }


def build_claim_validation(receipt: dict[str, object]) -> dict[str, object]:
    validation = validate_route_receipt(receipt)
    topology = PanelTopology.from_dict(receipt["topology"])
    table = TranslationTable.from_dict(receipt["translation_table"])
    register = receipt["digit_register"]
    program_a = compile_sender_program(topology, table, register)
    program_b = compile_sender_program(topology, table, register)
    changed_table = TranslationTable(table.table_id, table.version + "+identity-check", table.fallback_rule, table.entries)
    changed_program = compile_sender_program(topology, changed_table, register)
    events = receipt["events"]
    sender_index = next(i for i, event in enumerate(events) if event["action"] == "sender_program_sealed")
    actuation_indices = [i for i, event in enumerate(events) if event["phase"] == "actuation"]
    payload_identity = receipt["payload_identity"]
    checks = {
        "deterministic_sender_program": program_a["sha256"] == program_b["sha256"] == receipt["sender_program"]["sha256"],
        "sender_sealed_before_actuation": not actuation_indices or sender_index < min(actuation_indices),
        "payload_bytes_unchanged": payload_identity["before_sha256"] == payload_identity["after_sha256"],
        "translation_identity_bound": receipt["sender_program"]["translation_table_sha256"] == table.sha256(),
        "changed_translation_changes_program_hash": changed_program["sha256"] != program_a["sha256"],
        "fault_plan_explicit": isinstance(receipt["fault_plan"], dict),
        "path_fault_plan_explicit": "unavailable_paths" in receipt["fault_plan"],
        "fallback_rule_declared": receipt["sender_program"]["fallback_rule"] in {"none", "first_declared_free_path"},
        "replay_validation": validation["valid"] is True and validation["replayed"] is True,
    }
    unsigned = {
        "schema": CLAIM_VALIDATION_SCHEMA,
        "contract_version": "171.4",
        "panel_separated_control_receipt_hash": receipt["sha256"],
        "checks": checks,
        "all_passed": all(checks.values()),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "cross_era_note": "destination/outcome equivalence is checked by compare_strowger_panel for a shared corpus",
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def build_fault_battery(exchange: PanelExchange, request: PanelRequest) -> dict[str, object]:
    entry = exchange.translation_table.lookup(request.digits, request.destination)
    primary_path = entry.path_candidates[0]
    primary_bank = exchange.topology.path(primary_path).bank
    secondary_candidates = entry.path_candidates[1:]
    all_candidate_banks = tuple(sorted({exchange.topology.path(path_id).bank for path_id in entry.path_candidates}))
    primary_group = exchange.topology.bank(primary_bank).motor_group
    cases = [
        ("clean", PanelFaultPlan()),
        ("primary_path_unavailable", PanelFaultPlan(unavailable_paths=(primary_path,))),
        ("busy_primary_bank", PanelFaultPlan(busy_banks=(primary_bank,))),
        ("all_candidate_banks_busy", PanelFaultPlan(busy_banks=all_candidate_banks)),
        ("motor_stall", PanelFaultPlan(stalled_motor_groups=(primary_group,))),
        ("translation_corruption", PanelFaultPlan(translation_corruption=True)),
        ("sender_disagreement", PanelFaultPlan(sender_disagreement=True)),
    ]
    results: list[dict[str, object]] = []
    for name, faults in cases:
        result = exchange.route(request, faults=faults)
        results.append({
            "case": name,
            "fault_plan": faults.as_dict(),
            "outcome": result.outcome,
            "selected_path": result.receipt["route"]["path_id"] if isinstance(result.receipt["route"], dict) else None,
            "panel_separated_control_receipt_hash": result.receipt["sha256"],
        })
    unsigned = {
        "schema": FAULT_BATTERY_SCHEMA,
        "contract_version": "171.4",
        "topology_sha256": exchange.topology.sha256(),
        "translation_table_sha256": exchange.translation_table.sha256(),
        "request_register_sha256": seal_digit_register(request)["sha256"],
        "cases": results,
        "reproducible": True,
        "secondary_candidate_present": bool(secondary_candidates),
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def compare_strowger_panel(
    strowger_receipt: dict[str, object],
    panel_receipt: dict[str, object],
) -> dict[str, object]:
    strowger_validation = validate_strowger_receipt(strowger_receipt)
    panel_validation = validate_route_receipt(panel_receipt)
    request = strowger_receipt.get("request")
    register = panel_receipt.get("digit_register")
    if not isinstance(request, dict) or not isinstance(register, dict):
        raise ValueError("cross-era receipts are missing request records")
    strowger_destination = request.get("destination")
    panel_destination = register.get("destination")
    strowger_outcome = strowger_receipt.get("outcome")
    panel_outcome = panel_receipt.get("outcome")
    checks = {
        "destination_equal": strowger_destination == panel_destination,
        "outcome_equal": strowger_outcome == panel_outcome,
    }
    unsigned = {
        "schema": EQUIVALENCE_SCHEMA,
        "contract_version": "171.2",
        "strowger_receipt_sha256": strowger_validation["sha256"],
        "panel_receipt_sha256": panel_validation["panel_separated_control_receipt_hash"],
        "checks": checks,
        "equivalent": all(checks.values()),
        "trace_identity_required": False,
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def demo_topology(destination: str) -> PanelTopology:
    require_nonempty_text(destination, "destination")
    return PanelTopology(
        topology_id="panel-demo-v171",
        motor_groups=(
            MotorGroup("motor-a", ("bank-a",)),
            MotorGroup("motor-b", ("bank-b",)),
        ),
        banks=(
            PanelBank("bank-a", "motor-a", 1, 0, 9),
            PanelBank("bank-b", "motor-b", 1, 0, 9),
        ),
        paths=(
            PanelPath("path-a", "bank-a", 4, destination),
            PanelPath("path-b", "bank-b", 5, destination),
        ),
    )


def demo_translation(digits: tuple[int, ...], destination: str, *, version: str = "1") -> TranslationTable:
    return TranslationTable(
        table_id="panel-demo-translation",
        version=version,
        fallback_rule="first_declared_free_path",
        entries=(TranslationEntry(digits, destination, ("path-a", "path-b")),),
    )
