# SPDX-License-Identifier: MPL-2.0
"""v172.2 immutable layered fabrics and bounded reverse-reachability selection.

Inputs are adjacent-stage wires, v172.0 matrices and a sealed endpoint request.
Reverse dynamic programming computes viable suffixes. Forward reconstruction
chooses the first complete idle path in declared coordinate/wire order; it never
reserves resources. Full replay validates every search observation and outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from qec.sonify.canonical import canonical_sha256, validate_sha256

from .core import CrossbarMatrix, LINK_STATES, demo_matrix
from .marker import CrossbarRequest, _exact, _matrix, _request, _text

PATH_CONTRACT_VERSION = "172.2"
FABRIC_SCHEMA = "qec.crossbar-fabric-manifest.v1"
PATH_REGISTER_SCHEMA = "qec.crossbar-path-input-register.v1"
PATH_PROGRAM_SCHEMA = "qec.crossbar-path-search-program.v1"
PATH_RECEIPT_SCHEMA = "qec.crossbar-path-search-receipt.v1"
PATH_VALIDATION_SCHEMA = "qec.crossbar-path-search-validation.v1"
PATH_POLICY = "reverse-reachability-first-complete-idle-path.v1"
MAX_STAGES = 16
MAX_MATRICES = 64
MAX_INTERSECTIONS = 65536
MAX_INTERSTAGE_LINKS = 4096
MAX_SEARCH_EVALUATIONS = 131072
PATH_CLAIM_BOUNDARY = MappingProxyType({
    "classical_software_model_only": True,
    "marker_authority": "compute_first_complete_admissible_path_plan",
    "payload_mutation_permitted": False,
    "decoder_output_mutation_permitted": False,
    "decoder_output_identity_is_caller_declared": True,
    "reservation_present": False,
    "connection_commit_present": False,
    "continuity_receipt_present": False,
    "receipt_proves": "bounded_deterministic_multistage_selection_under_declared_state",
    "receipt_does_not_prove": "actuated_connection_or_authenticated_provenance",
})


def _seal(schema: str, **fields: object) -> dict[str, object]:
    unsigned = {"schema": schema, "contract_version": PATH_CONTRACT_VERSION, **fields}
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _sequence(value: object, minimum: int, maximum: int) -> tuple:
    if type(value) not in (tuple, list) or not minimum <= len(value) <= maximum:
        raise ValueError("INVALID_INPUT")
    return tuple(value)


def _budget(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_SEARCH_EVALUATIONS:
        raise ValueError("INVALID_INPUT")
    return value


@dataclass(frozen=True)
class InterstageLink:
    """A declared directed wire; adjacency and endpoint axes are checked by the fabric."""

    link_id: str
    source_matrix_id: str
    source_vertical_link_id: str
    target_matrix_id: str
    target_horizontal_link_id: str
    state: str = "idle"

    def __post_init__(self) -> None:
        for value in (self.link_id, self.source_matrix_id, self.source_vertical_link_id,
                      self.target_matrix_id, self.target_horizontal_link_id):
            _text(value)
        if type(self.state) is not str or self.state not in LINK_STATES:
            raise ValueError("INVALID_INPUT")

    def as_dict(self) -> dict[str, object]:
        return {
            "link_id": self.link_id,
            "source_matrix_id": self.source_matrix_id,
            "source_vertical_link_id": self.source_vertical_link_id,
            "target_matrix_id": self.target_matrix_id,
            "target_horizontal_link_id": self.target_horizontal_link_id,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, value: object) -> InterstageLink:
        if not isinstance(value, dict):
            raise ValueError("INVALID_INPUT")
        try:
            link = cls(**value)
            _exact(value, link.as_dict())
            return link
        except (TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class CrossbarFabric:
    """Ordered stages; matrices within each stage are sorted by unique matrix ID."""

    fabric_id: str
    stages: tuple[tuple[CrossbarMatrix, ...], ...]
    interstage_links: tuple[InterstageLink, ...]

    def __post_init__(self) -> None:
        _text(self.fabric_id)
        stages = tuple(_sequence(stage, 1, MAX_MATRICES)
                       for stage in _sequence(self.stages, 2, MAX_STAGES))
        if sum(map(len, stages)) > MAX_MATRICES:
            raise ValueError("INVALID_INPUT")
        if any(type(matrix) is not CrossbarMatrix for stage in stages for matrix in stage):
            raise ValueError("INVALID_INPUT")
        if sum(len(m.horizontal_links) * len(m.vertical_links)
               for stage in stages for m in stage) > MAX_INTERSECTIONS:
            raise ValueError("INVALID_INPUT")
        # Copy through the published validator, retaining no caller-owned arrays.
        stages = tuple(tuple(_matrix(m.as_dict()) for m in stage) for stage in stages)
        locations = {}
        for stage_index, stage in enumerate(stages):
            ids = tuple(m.matrix_id for m in stage)
            if ids != tuple(sorted(set(ids))):
                raise ValueError("INVALID_INPUT")
            for matrix_index, matrix in enumerate(stage):
                _text(matrix.matrix_id)
                if matrix.matrix_id in locations:
                    raise ValueError("INVALID_INPUT")
                locations[matrix.matrix_id] = (stage_index, matrix_index, matrix)
        links = _sequence(self.interstage_links, 0, MAX_INTERSTAGE_LINKS)
        if any(type(link) is not InterstageLink for link in links):
            raise ValueError("INVALID_INPUT")
        links = tuple(InterstageLink.from_dict(link.as_dict()) for link in links)
        keys = []
        names = set()
        endpoints = set()
        for link in links:
            if link.link_id in names:
                raise ValueError("INVALID_INPUT")
            names.add(link.link_id)
            source = locations.get(link.source_matrix_id)
            target = locations.get(link.target_matrix_id)
            if source is None or target is None or target[0] != source[0] + 1:
                raise ValueError("INVALID_INPUT")
            vertical = next((v for v in source[2].vertical_links
                             if v.link_id == link.source_vertical_link_id), None)
            horizontal = next((h for h in target[2].horizontal_links
                               if h.link_id == link.target_horizontal_link_id), None)
            if vertical is None or horizontal is None:
                raise ValueError("INVALID_INPUT")
            pair = (link.source_matrix_id, vertical.link_id, link.target_matrix_id, horizontal.link_id)
            if pair in endpoints:
                raise ValueError("INVALID_INPUT")
            endpoints.add(pair)
            keys.append((source[0], source[1], vertical.ordinal, target[1], horizontal.ordinal, link.link_id))
        if keys != sorted(keys):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "interstage_links", links)

    def as_dict(self) -> dict[str, object]:
        return _seal(
            FABRIC_SCHEMA, fabric_id=self.fabric_id,
            stages=[[m.as_dict() for m in stage] for stage in self.stages],
            interstage_links=[link.as_dict() for link in self.interstage_links],
        )

    @classmethod
    def from_dict(cls, value: object) -> CrossbarFabric:
        if not isinstance(value, dict):
            raise ValueError("INVALID_INPUT")
        try:
            raw_stages = _sequence(value["stages"], 2, MAX_STAGES)
            raw_stages = tuple(_sequence(stage, 1, MAX_MATRICES) for stage in raw_stages)
            if sum(map(len, raw_stages)) > MAX_MATRICES:
                raise ValueError("INVALID_INPUT")
            raw_links = _sequence(value["interstage_links"], 0, MAX_INTERSTAGE_LINKS)
            fabric = cls(
                value["fabric_id"],
                tuple(tuple(_matrix(m) for m in stage) for stage in raw_stages),
                tuple(InterstageLink.from_dict(link) for link in raw_links),
            )
            _exact(value, fabric.as_dict())
            return fabric
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class MultiStageRequest:
    """Qualify the v172.1 request's horizontal/vertical endpoints by matrix ID."""

    source_matrix_id: str
    destination_matrix_id: str
    request: CrossbarRequest

    def __post_init__(self) -> None:
        _text(self.source_matrix_id)
        _text(self.destination_matrix_id)
        _request(self.request)

    def as_dict(self) -> dict[str, object]:
        return _seal(
            PATH_REGISTER_SCHEMA,
            source_matrix_id=self.source_matrix_id,
            destination_matrix_id=self.destination_matrix_id,
            request=self.request.as_dict(),
        )

    @classmethod
    def from_dict(cls, value: object) -> MultiStageRequest:
        if not isinstance(value, dict):
            raise ValueError("INVALID_INPUT")
        try:
            request = cls(value["source_matrix_id"], value["destination_matrix_id"],
                          CrossbarRequest.from_dict(value["request"]))
            _exact(value, request.as_dict())
            return request
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def _register(request: MultiStageRequest) -> dict[str, object]:
    if type(request) is not MultiStageRequest:
        raise ValueError("INVALID_INPUT")
    request.__post_init__()
    return request.as_dict()


def _program(fabric_hash: str, register: dict[str, object], marker_id: str,
             max_search_evaluations: int) -> dict[str, object]:
    _text(marker_id)
    return _seal(
        PATH_PROGRAM_SCHEMA, marker_id=marker_id,
        fabric_sha256=fabric_hash, input_register_sha256=register["sha256"],
        policy=PATH_POLICY, max_search_evaluations=_budget(max_search_evaluations),
        claim_boundary=dict(PATH_CLAIM_BOUNDARY),
    )


def compile_path_program(fabric_manifest: object, request: MultiStageRequest, *,
                         marker_id: str = "marker-0",
                         max_search_evaluations: int = MAX_SEARCH_EVALUATIONS) -> dict[str, object]:
    """Bind complete intent, fabric state, fixed policy and a caller-declared budget."""
    fabric = CrossbarFabric.from_dict(fabric_manifest)
    return _program(fabric.as_dict()["sha256"], _register(request), marker_id, max_search_evaluations)


class _BudgetExhausted(Exception):
    pass


def execute_path_program(fabric_manifest: object, request: MultiStageRequest,
                         program: object) -> dict[str, object]:
    """Compute suffix viability, then reconstruct the canonical first complete path."""
    fabric = CrossbarFabric.from_dict(fabric_manifest)
    manifest = fabric.as_dict()
    register = _register(request)
    if not isinstance(program, dict):
        raise ValueError("INVALID_INPUT")
    expected_program = _program(manifest["sha256"], register, program.get("marker_id"),
                                program.get("max_search_evaluations"))
    _exact(program, expected_program)
    root = canonical_sha256({
        "schema": "qec.crossbar-path-search-event-root.v1",
        "contract_version": PATH_CONTRACT_VERSION,
        "fabric_sha256": manifest["sha256"],
        "register_sha256": register["sha256"], "program_sha256": expected_program["sha256"],
    })
    events = []
    evaluations = 0

    def event(kind: str, **details: object) -> None:
        unsigned = {"sequence": len(events), "kind": kind, "details": details,
                    "previous_sha256": events[-1]["sha256"] if events else root}
        events.append({**unsigned, "sha256": canonical_sha256(unsigned)})

    def consume() -> None:
        nonlocal evaluations
        if evaluations == expected_program["max_search_evaluations"]:
            raise _BudgetExhausted
        evaluations += 1

    event("register_sealed", input_register_sha256=register["sha256"])
    event("program_verified", program_sha256=expected_program["sha256"])
    source = next((m for m in fabric.stages[0] if m.matrix_id == request.source_matrix_id), None)
    destination = next((m for m in fabric.stages[-1]
                        if m.matrix_id == request.destination_matrix_id), None)
    reason = ""
    if source is None:
        reason = "unknown_source_matrix"
    elif destination is None:
        reason = "unknown_destination_matrix"
    elif request.request.horizontal_link_id not in {h.link_id for h in source.horizontal_links}:
        reason = "unknown_source_horizontal_link"
    elif request.request.vertical_link_id not in {v.link_id for v in destination.vertical_links}:
        reason = "unknown_destination_vertical_link"
    event("endpoints_checked", reason=reason or "endpoints_known")
    path_plan = None
    search_complete = False
    if not reason:
        # Canonical input order is retained in each adjacency list. Dictionary
        # lookup never supplies a traversal or tie-breaking order.
        outgoing = {}
        wires = {}
        for wire in fabric.interstage_links:
            outgoing.setdefault((wire.source_matrix_id, wire.source_vertical_link_id), []).append(wire)
            wires[wire.link_id] = wire
        choices = {}
        wire_choices = {}
        try:
            for stage_index in range(len(fabric.stages) - 1, -1, -1):
                for matrix in fabric.stages[stage_index]:
                    viable_verticals = []
                    for vertical in matrix.vertical_links:
                        key = (matrix.matrix_id, vertical.link_id)
                        first_wire = None
                        for wire in outgoing.get(key, ()):
                            consume()
                            suffix = choices.get((wire.target_matrix_id, wire.target_horizontal_link_id))
                            if vertical.state != "idle":
                                blocked = "source_vertical_" + vertical.state
                            elif wire.state != "idle":
                                blocked = "interstage_" + wire.state
                            elif suffix is None:
                                blocked = "suffix_unreachable"
                            else:
                                blocked = "admissible_suffix"
                            event("interstage_evaluated", stage=stage_index,
                                      link_id=wire.link_id, reason=blocked)
                            if blocked == "admissible_suffix" and first_wire is None:
                                first_wire = wire.link_id
                        consume()
                        if stage_index == len(fabric.stages) - 1:
                            viable = (matrix.matrix_id == request.destination_matrix_id
                                      and vertical.link_id == request.request.vertical_link_id
                                      and vertical.state == "idle")
                        else:
                            viable = first_wire is not None
                        event("vertical_evaluated", stage=stage_index, matrix_id=matrix.matrix_id,
                                  vertical_link_id=vertical.link_id, state=vertical.state,
                                  suffix_reachable=viable, first_interstage_link_id=first_wire)
                        wire_choices[key] = first_wire
                        if viable:
                            viable_verticals.append(vertical.link_id)
                    for horizontal in matrix.horizontal_links:
                        consume()
                        first_vertical = (viable_verticals[0]
                                          if horizontal.state == "idle" and viable_verticals else None)
                        event("horizontal_evaluated", stage=stage_index, matrix_id=matrix.matrix_id,
                                  horizontal_link_id=horizontal.link_id, state=horizontal.state,
                                  first_vertical_link_id=first_vertical)
                        choices[(matrix.matrix_id, horizontal.link_id)] = first_vertical
            search_complete = True
            current = (request.source_matrix_id, request.request.horizontal_link_id)
            if choices[current] is None:
                reason = "no_admissible_complete_path"
            else:
                coordinates = []
                selected_wires = []
                for stage_index, stage in enumerate(fabric.stages):
                    matrix = next(m for m in stage if m.matrix_id == current[0])
                    vertical_id = choices[current]
                    coordinates.append({"stage": stage_index, "matrix_id": matrix.matrix_id,
                                        "coordinate": matrix.coordinate(current[1], vertical_id).as_dict()})
                    if stage_index < len(fabric.stages) - 1:
                        wire_id = wire_choices[(matrix.matrix_id, vertical_id)]
                        selected_wires.append(wire_id)
                        wire = wires[wire_id]
                        current = (wire.target_matrix_id, wire.target_horizontal_link_id)
                path_plan = {"coordinates": coordinates, "interstage_link_ids": selected_wires}
                reason = "first_admissible_complete_path"
        except _BudgetExhausted:
            reason = "search_budget_exhausted"
    outcome = "plan_selected" if path_plan is not None else "rejected"
    event(outcome, reason=reason, path_plan=path_plan)
    event("marker_released", marker_id=expected_program["marker_id"])
    return _seal(
        PATH_RECEIPT_SCHEMA, fabric_manifest=manifest, input_register=register,
        path_program=expected_program, fabric_sha256=manifest["sha256"],
        payload_sha256=register["request"]["payload_sha256"],
        decoder_output_sha256=request.request.decoder_output_sha256,
        event_root_sha256=root, events=events,
        search_evaluations=evaluations, search_complete=search_complete,
        outcome=outcome, reason=reason, path_plan=path_plan,
        fabric_after_sha256=manifest["sha256"], marker_released=True,
        claim_boundary=dict(PATH_CLAIM_BOUNDARY),
    )


def validate_path_search_receipt(receipt: object, *, expected_fabric_sha256: str | None = None,
                                 expected_request_sha256: str | None = None,
                                 expected_program_sha256: str | None = None) -> dict[str, object]:
    """Replay all decisions; optional trusted hashes pin evidence to an intended run."""
    if not isinstance(receipt, dict):
        raise ValueError("INVALID_INPUT")
    try:
        request = MultiStageRequest.from_dict(receipt["input_register"])
        replay = execute_path_program(receipt["fabric_manifest"], request, receipt["path_program"])
        _exact(receipt, replay)
        bindings = {
            "fabric": (expected_fabric_sha256, replay["fabric_sha256"]),
            "request": (expected_request_sha256, replay["input_register"]["sha256"]),
            "program": (expected_program_sha256, replay["path_program"]["sha256"]),
        }
        for expected, actual in bindings.values():
            if expected is not None and validate_sha256(expected) != actual:
                raise ValueError("INVALID_INPUT")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
    return _seal(
        PATH_VALIDATION_SCHEMA,
        crossbar_path_search_receipt_hash=replay["sha256"],
        fabric_sha256=replay["fabric_sha256"],
        input_register_sha256=replay["input_register"]["sha256"],
        path_program_sha256=replay["path_program"]["sha256"],
        outcome=replay["outcome"], reason=replay["reason"],
        search_complete=replay["search_complete"], replay_verified=True,
        first_complete_path_verified=replay["outcome"] == "plan_selected",
        payload_identity_verified=True, bounded_authority_verified=True,
        marker_release_verified=True,
        external_bindings_verified={key: expected is not None for key, (expected, _) in bindings.items()},
        all_passed=True,
    )


def demo_fabric(*, dead_end_first: bool = True) -> CrossbarFabric:
    """Three stages: a locally free first branch may have an unavailable exit wire."""
    if type(dead_end_first) is not bool:
        raise ValueError("INVALID_INPUT")
    def matrix(name):
        return demo_matrix(name, horizontal_count=2, vertical_count=2)
    return CrossbarFabric(
        "multistage-demo",
        ((matrix("ingress"),), (matrix("middle-a"), matrix("middle-b")), (matrix("egress"),)),
        (
            InterstageLink("entry-a", "ingress", "V000", "middle-a", "H000"),
            InterstageLink("entry-b", "ingress", "V001", "middle-b", "H000"),
            InterstageLink("exit-a", "middle-a", "V000", "egress", "H000",
                           "unavailable" if dead_end_first else "idle"),
            InterstageLink("exit-b", "middle-b", "V000", "egress", "H001"),
        ),
    )
