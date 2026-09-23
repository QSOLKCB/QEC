# SPDX-License-Identifier: MPL-2.0
"""v172.4 independent forward continuity witnesses over replayed selections.

A walk checks every coordinate and adjacent wire against the bound snapshot,
without consulting selector choices. Receipts cover all reserve attempts in a
closed contention batch, or one path-search result. They prove continuity at
selection, never physical actuation or a new reservation/connection transition.
"""
from __future__ import annotations

import json
from types import MappingProxyType

from qec.sonify.canonical import canonical_json, canonical_sha256, validate_sha256
from .contention import validate_contention_receipt
from .marker import _exact, _text
from .multistage import (
    PATH_RECEIPT_SCHEMA, CrossbarFabric, MultiStageRequest,
    validate_path_search_receipt,
)

CONTINUITY_CONTRACT_VERSION = "172.4"
CONTINUITY_POLICY = "exact-forward-stage-and-wire-walk.v1"
CONTINUITY_CLAIM_BOUNDARY = MappingProxyType({
    "classical_software_model_only": True,
    "continuity_scope": "selected_path_in_bound_pre_selection_snapshot",
    "source_replay_required": True,
    "all_selected_coordinates_covered": True,
    "new_reservation_transition_present": False,
    "connection_commit_present": False,
    "physical_actuation_present": False,
    "payload_mutation_permitted": False,
    "decoder_output_mutation_permitted": False,
    "decoder_output_identity_is_caller_declared": True,
    "receipt_proves": "every_selected_coordinate_and_wire_forms_one_endpoint_bound_route",
    "receipt_does_not_prove": "current_liveness_or_authenticated_provenance_or_decoder_correctness",
})


def _seal(name: str, **fields: object) -> dict[str, object]:
    unsigned = {"schema": "qec.crossbar-" + name + ".v1",
                "contract_version": CONTINUITY_CONTRACT_VERSION, **fields}
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def verify_path_continuity(fabric_manifest: object, request: MultiStageRequest,
                           path_plan: object) -> dict[str, object]:
    """Check one complete idle route; no search, selection or reservation claim.

    Invalid, partial, branched, reordered, disconnected or blocked plans raise
    INVALID_INPUT. The walk consumes exactly N coordinates and N-1 wires, bounded
    by the published 2..16-stage fabric. Input validation is additional work.
    """
    try:
        fabric = CrossbarFabric.from_dict(fabric_manifest)
        if type(request) is not MultiStageRequest:
            raise ValueError("INVALID_INPUT")
        register = MultiStageRequest.from_dict(request.as_dict()).as_dict()
        if type(path_plan) is not dict or set(path_plan) != {"coordinates", "interstage_link_ids"}:
            raise ValueError("INVALID_INPUT")
        coordinates, wire_ids = path_plan["coordinates"], path_plan["interstage_link_ids"]
        count = len(fabric.stages)
        if (type(coordinates) is not list or len(coordinates) != count
                or type(wire_ids) is not list or len(wire_ids) != count - 1):
            raise ValueError("INVALID_INPUT")
        wires = {w.link_id: w for w in fabric.interstage_links}
        current_matrix = request.source_matrix_id
        current_horizontal = request.request.horizontal_link_id
        expected_coordinates, expected_wires, steps = [], [], []
        for stage_index, stage in enumerate(fabric.stages):
            row = coordinates[stage_index]
            if type(row) is not dict or type(row.get("coordinate")) is not dict:
                raise ValueError("INVALID_INPUT")
            matrix = next((m for m in stage if m.matrix_id == current_matrix), None)
            if matrix is None:
                raise ValueError("INVALID_INPUT")
            vertical_id = row["coordinate"]["vertical_link_id"]
            _text(vertical_id)
            coordinate = matrix.coordinate(current_horizontal, vertical_id)
            expected_row = {"stage": stage_index, "matrix_id": matrix.matrix_id,
                            "coordinate": coordinate.as_dict()}
            _exact(row, expected_row)
            horizontal = matrix.horizontal_links[coordinate.horizontal_ordinal]
            vertical = matrix.vertical_links[coordinate.vertical_ordinal]
            if horizontal.state != "idle" or vertical.state != "idle":
                raise ValueError("INVALID_INPUT")
            expected_coordinates.append(expected_row)
            steps.append({"sequence": len(steps), "kind": "coordinate", **expected_row,
                          "horizontal_state": horizontal.state, "vertical_state": vertical.state})
            if stage_index == count - 1:
                if (matrix.matrix_id != request.destination_matrix_id
                        or vertical_id != request.request.vertical_link_id):
                    raise ValueError("INVALID_INPUT")
            else:
                wire_id = wire_ids[stage_index]
                _text(wire_id)
                wire = wires.get(wire_id)
                if (wire is None or wire.state != "idle"
                        or wire.source_matrix_id != matrix.matrix_id
                        or wire.source_vertical_link_id != vertical_id):
                    raise ValueError("INVALID_INPUT")
                expected_wires.append(wire_id)
                steps.append({"sequence": len(steps), "kind": "interstage",
                              "source_stage": stage_index, "target_stage": stage_index + 1,
                              "wire": wire.as_dict()})
                current_matrix = wire.target_matrix_id
                current_horizontal = wire.target_horizontal_link_id
        plan = {"coordinates": expected_coordinates, "interstage_link_ids": expected_wires}
        _exact(path_plan, plan)
        return _seal("continuity-witness", policy=CONTINUITY_POLICY,
                     fabric_sha256=fabric.as_dict()["sha256"], input_register_sha256=register["sha256"],
                     payload_sha256=register["request"]["payload_sha256"],
                     decoder_output_sha256=register["request"]["decoder_output_sha256"],
                     path_plan=plan, steps=steps, coordinate_count=count,
                     interstage_link_count=count - 1, continuity_verified=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _bindings(source: dict, expected_source_sha256, expected_fabric_sha256,
              expected_input_sha256, expected_program_sha256) -> dict:
    if source["schema"] == PATH_RECEIPT_SCHEMA:
        fabric_hash = source["fabric_sha256"]
        input_hash = source["input_register"]["sha256"]
        program_hash = source["path_program"]["sha256"]
    else:
        fabric_hash = source["input"]["fabric_manifest"]["sha256"]
        input_hash = source["input"]["sha256"]
        program_hash = source["program"]["sha256"]
    pairs = {"source": (expected_source_sha256, source["sha256"]),
             "fabric": (expected_fabric_sha256, fabric_hash),
             "input": (expected_input_sha256, input_hash),
             "program": (expected_program_sha256, program_hash)}
    for expected, actual in pairs.values():
        if expected is not None and validate_sha256(expected) != actual:
            raise ValueError("INVALID_INPUT")
    return {key: expected is not None for key, (expected, _) in pairs.items()}


def create_continuity_receipt(source_receipt: object, *, expected_source_sha256: str | None = None,
                               expected_fabric_sha256: str | None = None,
                               expected_input_sha256: str | None = None,
                               expected_program_sha256: str | None = None) -> dict[str, object]:
    """Replay a v172.2/.3 source and witness every selection in source order.

    Rejected reserve attempts are retained with no witness. Release/quarantine
    results remain in the embedded source. Active-at-end is a historical batch
    observation, not live availability. No caller-owned objects are retained.
    """
    if type(source_receipt) is not dict:
        raise ValueError("INVALID_INPUT")
    try:
        schema = source_receipt.get("schema")
        if schema == PATH_RECEIPT_SCHEMA:
            validate_path_search_receipt(source_receipt)
        elif schema == "qec.crossbar-contention-receipt.v1":
            validate_contention_receipt(source_receipt)
        else:
            raise ValueError("INVALID_INPUT")
        source = json.loads(canonical_json(source_receipt))
        _bindings(source, expected_source_sha256, expected_fabric_sha256,
                  expected_input_sha256, expected_program_sha256)
        routes, events = [], []
        root = _seal("continuity-event-root", source_receipt_sha256=source["sha256"],
                     policy=CONTINUITY_POLICY)["sha256"]

        def event(kind, **details):
            unsigned = {"sequence": len(events), "kind": kind, "details": details,
                        "previous_sha256": events[-1]["sha256"] if events else root}
            events.append({**unsigned, "sha256": canonical_sha256(unsigned)})

        event("source_replay_verified", source_receipt_sha256=source["sha256"])
        if schema == PATH_RECEIPT_SCHEMA:
            attempts = [(None, source["input_register"], source, None, None, None, None)]
        else:
            active = {r["reservation_id"] for r in source["final_state"]["active_reservations"]}
            attempts = []
            for command, result in zip(source["input"]["commands"], source["results"]):
                if command["kind"] == "reserve":
                    reservation = result["reservation"]
                    attempts.append((command["command_id"], command["request"],
                                     result["path_search_receipt"], reservation,
                                     command["command_id"] in active if reservation else None,
                                     result["state_before_sha256"], result["reason"]))
        for command_id, register, search, reservation, active_at_end, before, batch_reason in attempts:
            witness = None
            if search is not None and search["outcome"] == "plan_selected":
                witness = verify_path_continuity(search["fabric_manifest"],
                    MultiStageRequest.from_dict(register), search["path_plan"])
                if reservation is not None:
                    _exact(reservation["path_plan"], witness["path_plan"])
            row = {"index": len(routes), "command_id": command_id,
                   "input_register_sha256": register["sha256"],
                   "payload_sha256": register["request"]["payload_sha256"],
                   "decoder_output_sha256": register["request"]["decoder_output_sha256"],
                   "source_reason": batch_reason if batch_reason is not None else search["reason"],
                   "path_search_receipt_sha256": search["sha256"] if search is not None else None,
                   "state_before_sha256": before,
                   "reservation_sha256": reservation["sha256"] if reservation is not None else None,
                   "reservation_active_at_batch_end": active_at_end,
                   "outcome": "continuous" if witness is not None else "not_selected",
                   "witness": witness}
            routes.append(row)
            event("route_checked", index=row["index"], outcome=row["outcome"],
                  route_sha256=canonical_sha256(row))
        selected = sum(row["witness"] is not None for row in routes)
        outcome = "continuity_verified" if selected else "no_selected_route"
        event("verification_completed", outcome=outcome, selected_route_count=selected)
        return _seal("continuity-receipt", policy=CONTINUITY_POLICY, source_receipt=source,
                     source_receipt_sha256=source["sha256"], routes=routes,
                     selected_route_count=selected, unselected_attempt_count=len(routes) - selected,
                     outcome=outcome, event_root_sha256=root, events=events,
                     claim_boundary=dict(CONTINUITY_CLAIM_BOUNDARY))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def validate_continuity_receipt(receipt: object, *, expected_source_sha256: str | None = None,
                                 expected_fabric_sha256: str | None = None,
                                 expected_input_sha256: str | None = None,
                                 expected_program_sha256: str | None = None) -> dict[str, object]:
    """Rebuild all evidence; a valid no-selection receipt is not a route proof."""
    if type(receipt) is not dict:
        raise ValueError("INVALID_INPUT")
    try:
        replay = create_continuity_receipt(receipt["source_receipt"])
        _exact(receipt, replay)
        bindings = _bindings(replay["source_receipt"], expected_source_sha256,
                             expected_fabric_sha256, expected_input_sha256, expected_program_sha256)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
    return _seal("continuity-validation", crossbar_continuity_receipt_hash=replay["sha256"],
                 source_receipt_sha256=replay["source_receipt_sha256"], outcome=replay["outcome"],
                 selected_route_count=replay["selected_route_count"],
                 continuity_verified=replay["selected_route_count"] > 0,
                 source_replay_verified=True, complete_selection_coverage_verified=True,
                 payload_identity_verified=True, replay_verified=True,
                 external_bindings_verified=bindings, all_passed=True)
