# SPDX-License-Identifier: MPL-2.0
"""v172.5 bounded Strowger/Panel/Crossbar route-decision equivalence battery.

A shared corpus drives fixed, explicit adapters into the unchanged native models.
The report retains complete native evidence and compares destination, admission
and first available lane, not event traces, commit semantics or unsupported
payload capabilities. Replay reconstructs both adapters and every source byte.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from itertools import product
from types import MappingProxyType

from qec.sonify.canonical import canonical_sha256, validate_sha256
from . import panel, strowger
from .crossbar.core import demo_matrix
from .crossbar.marker import CrossbarRequest, _exact, _text
from .crossbar.multistage import (
    CrossbarFabric, InterstageLink, MultiStageRequest, compile_path_program,
    execute_path_program,
)
from .crossbar.continuity import create_continuity_receipt

EQUIVALENCE_CONTRACT_VERSION = "172.5"
MAX_CASES = 64
MAX_CASE_PAYLOAD_BYTES = 4096
MAX_CORPUS_PAYLOAD_BYTES = 65536
LANE_STATES = ("idle", "busy", "quarantined")
FAULTS = ("none", "strowger_selector_fault", "strowger_tone_mismatch",
          "panel_motor_stall", "panel_sender_disagreement", "crossbar_search_budget")
ARCHITECTURES = ("strowger", "panel", "crossbar")
COMPARISON_FIELDS = ("requested_destination", "routing_outcome", "selected_lane", "reached_destination")
EQUIVALENCE_CLAIM_BOUNDARY = MappingProxyType({
    "classical_software_model_only": True,
    "comparison_scope": "fixed_four_destination_two_lane_route_decisions",
    "trace_identity_required": False,
    "commit_semantics_equivalent": False,
    "reservation_lifecycles_equivalent": False,
    "three_way_native_payload_equivalence": False,
    "strowger_payload_scope": "case_envelope_only",
    "native_payload_comparison": "panel_and_crossbar_only",
    "native_decoder_reference_scope": "crossbar_only_caller_declared",
    "receipt_proves": "declared_route_decision_equivalence_and_expected_fault_differences_for_this_corpus",
    "receipt_does_not_prove": "universal_equivalence_or_migration_or_physical_fidelity_or_decoder_correctness",
})


def _seal(name: str, **fields: object) -> dict[str, object]:
    unsigned = {"schema": "qec.switch-equivalence-" + name + ".v1",
                "contract_version": EQUIVALENCE_CONTRACT_VERSION, **fields}
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _destination(index: int) -> str:
    return f"correction/destination-{index}"


def _digits(index: int) -> tuple[int, int, int]:
    return (0, index // 2, index % 2)


@dataclass(frozen=True)
class EquivalenceCase:
    """One independent initial-state experiment, never an imported live ledger."""
    case_id: str
    destination: int
    lane_states: tuple[str, str]
    payload: bytes = b""
    decoder_output_sha256: str = "a" * 64
    fault: str = "none"

    def __post_init__(self) -> None:
        _text(self.case_id)
        if len(self.case_id) > 128 or type(self.destination) is not int or not 0 <= self.destination < 4:
            raise ValueError("INVALID_INPUT")
        if type(self.lane_states) not in (list, tuple) or len(self.lane_states) != 2:
            raise ValueError("INVALID_INPUT")
        if any(type(s) is not str or s not in LANE_STATES for s in self.lane_states):
            raise ValueError("INVALID_INPUT")
        if type(self.payload) is not bytes or len(self.payload) > MAX_CASE_PAYLOAD_BYTES:
            raise ValueError("INVALID_INPUT")
        try:
            _text(self.decoder_output_sha256)
            validate_sha256(self.decoder_output_sha256)
        except (TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc
        if type(self.fault) is not str or self.fault not in FAULTS:
            raise ValueError("INVALID_INPUT")
        # Native fault controls require admitted lanes so capacity cannot mask
        # the fault. Arbitrary combinations belong to a future broader corpus.
        if self.fault != "none" and tuple(self.lane_states) != ("idle", "idle"):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "lane_states", tuple(self.lane_states))

    def as_dict(self) -> dict[str, object]:
        return _seal("case", case_id=self.case_id, destination=self.destination,
                     lane_states=list(self.lane_states), payload_hex=self.payload.hex(),
                     payload_sha256=hashlib.sha256(self.payload).hexdigest(),
                     decoder_output_sha256=self.decoder_output_sha256, fault=self.fault)

    @classmethod
    def from_dict(cls, value: object) -> EquivalenceCase:
        if type(value) is not dict:
            raise ValueError("INVALID_INPUT")
        try:
            hex_payload = value["payload_hex"]
            if type(hex_payload) is not str or len(hex_payload) > 2 * MAX_CASE_PAYLOAD_BYTES:
                raise ValueError("INVALID_INPUT")
            case = cls(value["case_id"], value["destination"], value["lane_states"],
                       bytes.fromhex(hex_payload), value["decoder_output_sha256"], value["fault"])
            _exact(value, case.as_dict())
            return case
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class EquivalenceCorpus:
    cases: tuple[EquivalenceCase, ...]

    def __post_init__(self) -> None:
        if type(self.cases) not in (list, tuple) or not 1 <= len(self.cases) <= MAX_CASES:
            raise ValueError("INVALID_INPUT")
        if any(type(case) is not EquivalenceCase for case in self.cases):
            raise ValueError("INVALID_INPUT")
        cases = tuple(EquivalenceCase.from_dict(case.as_dict()) for case in self.cases)
        names = tuple(case.case_id for case in cases)
        if names != tuple(sorted(set(names))) or sum(len(c.payload) for c in cases) > MAX_CORPUS_PAYLOAD_BYTES:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "cases", cases)

    def as_dict(self) -> dict[str, object]:
        return _seal("corpus", cases=[case.as_dict() for case in self.cases])

    @classmethod
    def from_dict(cls, value: object) -> EquivalenceCorpus:
        if type(value) is not dict:
            raise ValueError("INVALID_INPUT")
        try:
            rows = value["cases"]
            if type(rows) is not list or not 1 <= len(rows) <= MAX_CASES:
                raise ValueError("INVALID_INPUT")
            corpus = cls(tuple(EquivalenceCase.from_dict(row) for row in rows))
            _exact(value, corpus.as_dict())
            return corpus
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def _models():
    config = strowger.ExchangeConfig(1, (strowger.StageConfig("lane", 2, trunks=2),), 2, 2)
    topology = panel.PanelTopology("equivalence-panel",
        tuple(panel.MotorGroup(f"motor-{lane}", (f"bank-{lane}",)) for lane in range(2)),
        tuple(panel.PanelBank(f"bank-{lane}", f"motor-{lane}", 4, 0, 3) for lane in range(2)),
        tuple(panel.PanelPath(f"lane-{lane}-destination-{dest}", f"bank-{lane}", dest, _destination(dest))
              for lane in range(2) for dest in range(4)))
    translation = panel.TranslationTable("equivalence-translation", "1", "first_declared_free_path",
        tuple(panel.TranslationEntry(_digits(dest), _destination(dest),
              tuple(f"lane-{lane}-destination-{dest}" for lane in range(2))) for dest in range(4)))
    fabric = CrossbarFabric("equivalence-crossbar", (
        (demo_matrix("ingress", horizontal_count=1, vertical_count=2),),
        (demo_matrix("egress", horizontal_count=2, vertical_count=4),)),
        tuple(InterstageLink(f"lane-{lane}", "ingress", f"V{lane:03d}", "egress", f"H{lane:03d}")
              for lane in range(2)))
    return config, topology, translation, fabric


def equivalence_adapter_manifest() -> dict[str, object]:
    """Executable adapter contract: fixed native topology and explicit projections."""
    config, topology, translation, fabric = _models()
    return _seal("adapters", architectures=list(ARCHITECTURES),
        strowger_config=config.as_dict(), panel_topology=topology.as_dict(),
        panel_translation_table=translation.as_dict(), crossbar_fabric=fabric.as_dict(),
        destinations=[{"destination": dest, "label": _destination(dest), "digits": list(_digits(dest)),
                       "crossbar_vertical_link_id": f"V{dest:03d}"} for dest in range(4)],
        lane_mapping=[{"lane": lane, "strowger_selector_trunk": lane,
                       "panel_bank": f"bank-{lane}", "crossbar_wire": f"lane-{lane}"} for lane in range(2)],
        state_mapping={"strowger": {"idle": "free", "busy": "busy", "quarantined": "quarantined"},
                       "panel": {"idle": "available", "busy": "busy_bank", "quarantined": "unavailable_paths"},
                       "crossbar": {state: state for state in LANE_STATES}},
        comparison_fields=list(COMPARISON_FIELDS),
        success_projection={"strowger": "committed", "panel": "committed", "crossbar": "plan_selected"},
        capacity_projection={"strowger": "all_trunks_busy", "panel": "capacity_exhausted",
                             "crossbar": "no_admissible_complete_path"},
        fault_controls=list(FAULTS), claim_boundary=dict(EQUIVALENCE_CLAIM_BOUNDARY))


def _execute(case: EquivalenceCase) -> dict:
    config, topology, translation, fabric = _models()
    label, digits = _destination(case.destination), _digits(case.destination)
    exchange = strowger.StrowgerExchange(config)
    exchange.trunk_states["lane"] = [strowger.TrunkState("free" if s == "idle" else s) for s in case.lane_states]
    strowger_faults = strowger.FaultPlan(
        stuck_selectors=(0,) if case.fault == "strowger_selector_fault" else (),
        tone_offsets_hz=(1, 0, 0) if case.fault == "strowger_tone_mismatch" else (0, 0, 0))
    sr = exchange.route(strowger.RouteRequest(case.case_id, digits, 0, label), faults=strowger_faults).receipt
    panel_faults = panel.PanelFaultPlan(
        busy_banks=tuple(f"bank-{i}" for i, state in enumerate(case.lane_states) if state == "busy"),
        unavailable_paths=tuple(path.path_id for path in topology.paths
                                if case.lane_states[int(path.bank[-1])] == "quarantined"),
        stalled_motor_groups=("motor-0", "motor-1") if case.fault == "panel_motor_stall" else (),
        sender_disagreement=case.fault == "panel_sender_disagreement")
    pr = panel.PanelExchange(topology, translation).route(
        panel.PanelRequest(case.case_id, digits, 0, label, case.payload), faults=panel_faults).receipt
    effective = replace(fabric, interstage_links=tuple(replace(wire, state=state)
                        for wire, state in zip(fabric.interstage_links, case.lane_states))).as_dict()
    cr_request = MultiStageRequest("ingress", "egress", CrossbarRequest(
        case.case_id, "H000", f"V{case.destination:03d}", case.payload, case.decoder_output_sha256))
    program = compile_path_program(effective, cr_request,
        max_search_evaluations=1 if case.fault == "crossbar_search_budget" else 131072)
    cr = execute_path_program(effective, cr_request, program)
    return {"strowger": sr, "panel": pr, "crossbar": create_continuity_receipt(cr)}


def _projections(case: EquivalenceCase, evidence: dict) -> list[dict]:
    """Read achieved route coordinates, not merely the requested destination label."""
    sr, pr = evidence["strowger"], evidence["panel"]
    cr = evidence["crossbar"]["source_receipt"]
    _, topology, _, _ = _models()
    result = []
    for architecture, receipt in (("strowger", sr), ("panel", pr), ("crossbar", cr)):
        native = receipt["outcome"]
        reason = receipt["reason"] if architecture == "crossbar" else native
        lane = reached = None
        if (architecture in ("strowger", "panel") and native == "committed") or (
                architecture == "crossbar" and native == "plan_selected"):
            status = "route_available"
            if architecture == "strowger":
                lane = receipt["route"]["selector_trunks"][0]
                vertical, rotary = receipt["route"]["connector"]
                reached = _destination(2 * vertical + rotary)
            elif architecture == "panel":
                path = topology.path(receipt["route"]["path_id"])
                lane, reached = int(path.bank[-1]), path.destination
            else:
                witness = evidence["crossbar"]["routes"][0]["witness"]
                lane = int(witness["path_plan"]["interstage_link_ids"][0][-1])
                reached = _destination(witness["path_plan"]["coordinates"][-1]["coordinate"]["vertical_ordinal"])
        elif (architecture == "strowger" and native == "all_trunks_busy") or (
                architecture == "panel" and native == "capacity_exhausted") or (
                architecture == "crossbar" and reason == "no_admissible_complete_path"):
            status = "capacity_blocked"
        else:
            status = architecture + "." + reason
        requested_destination = (sr["request"]["destination"] if architecture == "strowger" else
            pr["digit_register"]["destination"] if architecture == "panel" else
            _destination(int(cr["input_register"]["request"]["vertical_link_id"][1:])))
        result.append({"architecture": architecture, "requested_destination": requested_destination,
            "routing_outcome": status, "selected_lane": lane, "reached_destination": reached,
            "native_outcome": native, "native_reason": reason,
            "source_receipt_sha256": evidence[architecture]["sha256"],
            "native_commit_present": architecture != "crossbar" and native == "committed",
            "native_payload_sha256": (None if architecture == "strowger" else
                pr["payload_identity"]["after_sha256"] if architecture == "panel" else cr["payload_sha256"]),
            "native_decoder_output_sha256": cr["decoder_output_sha256"] if architecture == "crossbar" else None})
    return result


def _reference(case: EquivalenceCase) -> dict:
    """Small admission oracle; no native selector or event stream is consulted."""
    first_idle = next((i for i, state in enumerate(case.lane_states) if state == "idle"), None)
    available = first_idle is not None
    raw = {"strowger": "committed" if available else "all_trunks_busy",
           "panel": "committed" if available else "capacity_exhausted",
           "crossbar": "first_admissible_complete_path" if available else "no_admissible_complete_path"}
    overrides = {"strowger_selector_fault": ("strowger", "selector_fault"),
                 "strowger_tone_mismatch": ("strowger", "tone_mismatch"),
                 "panel_motor_stall": ("panel", "motor_stall"),
                 "panel_sender_disagreement": ("panel", "sender_disagreement"),
                 "crossbar_search_budget": ("crossbar", "search_budget_exhausted")}
    affected = None
    if case.fault != "none":
        affected, reason = overrides[case.fault]
        raw[affected] = reason
    return {architecture: {"routing_outcome": architecture + "." + raw[architecture] if architecture == affected else
                            "route_available" if available else "capacity_blocked",
                          "selected_lane": first_idle if architecture != affected else None,
                          "reached_destination": _destination(case.destination) if available and architecture != affected else None,
                          "native_reason": raw[architecture]} for architecture in ARCHITECTURES}


def _row(case: EquivalenceCase, adapters_hash: str) -> dict:
    evidence = _execute(case)
    strowger.validate_receipt(evidence["strowger"])
    panel.validate_route_receipt(evidence["panel"])
    projections = _projections(case, evidence)
    reference = _reference(case)
    pairs = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        a, b = projections[left], projections[right]
        checks = {field: a[field] == b[field] for field in COMPARISON_FIELDS}
        pairs.append({"left": a["architecture"], "right": b["architecture"],
                      "checks": checks, "equivalent": all(checks.values())})
    equivalent = all(pair["equivalent"] for pair in pairs)
    oracle_checks = {p["architecture"]: all(p[key] == value for key, value in reference[p["architecture"]].items())
                     for p in projections}
    case_record = case.as_dict()
    pr = evidence["panel"]
    cr = evidence["crossbar"]["source_receipt"]
    sr_request = evidence["strowger"]["request"]
    panel_register = pr["digit_register"]
    crossbar_register = cr["input_register"]
    expected_request = {"request_id": case.case_id, "digits": list(_digits(case.destination)),
                        "epoch": 0, "destination": _destination(case.destination)}
    request_checks = {
        "strowger": sr_request == expected_request,
        "panel": all(panel_register[key] == value for key, value in expected_request.items()),
        "crossbar": crossbar_register["source_matrix_id"] == "ingress"
            and crossbar_register["destination_matrix_id"] == "egress"
            and crossbar_register["request"]["request_id"] == case.case_id
            and crossbar_register["request"]["horizontal_link_id"] == "H000"
            and crossbar_register["request"]["vertical_link_id"] == f"V{case.destination:03d}",
    }
    payload_checks = {
        "panel_bytes_preserved": pr["digit_register"]["payload_hex"] == case.payload.hex()
            and pr["payload_identity"]["before_sha256"] == case_record["payload_sha256"]
            and pr["payload_identity"]["after_sha256"] == case_record["payload_sha256"],
        "crossbar_bytes_preserved": cr["input_register"]["request"]["payload_hex"] == case.payload.hex()
            and cr["payload_sha256"] == case_record["payload_sha256"],
        "crossbar_decoder_reference_preserved": cr["decoder_output_sha256"] == case.decoder_output_sha256,
    }
    expected_equivalent = case.fault == "none"
    return _seal("case-result", case_sha256=case_record["sha256"], case_id=case.case_id,
        adapter_manifest_sha256=adapters_hash, evidence=evidence, projections=projections,
        pairwise=pairs, equivalent=equivalent, expected_equivalent=expected_equivalent,
        reference=reference, reference_checks=oracle_checks, request_checks=request_checks, payload_checks=payload_checks,
        passed=all(oracle_checks.values()) and all(request_checks.values())
            and all(payload_checks.values()) and equivalent == expected_equivalent)


def run_equivalence_battery(corpus: EquivalenceCorpus) -> dict[str, object]:
    """Run independent cases in canonical ID order; preserve all native evidence."""
    if type(corpus) is not EquivalenceCorpus:
        raise ValueError("INVALID_INPUT")
    corpus = EquivalenceCorpus.from_dict(corpus.as_dict())
    adapters = equivalence_adapter_manifest()
    results = [_row(case, adapters["sha256"]) for case in corpus.cases]
    return _seal("matrix", corpus=corpus.as_dict(), adapter_manifest=adapters,
        results=results, case_count=len(results),
        default_corpus_used=corpus.as_dict()["sha256"] == demo_equivalence_corpus().as_dict()["sha256"],
        equivalent_case_count=sum(row["equivalent"] for row in results),
        expected_difference_count=sum(not row["expected_equivalent"] for row in results),
        all_cases_equivalent=all(row["equivalent"] for row in results),
        battery_passed=all(row["passed"] for row in results),
        claim_boundary=dict(EQUIVALENCE_CLAIM_BOUNDARY))


def validate_equivalence_matrix(matrix: object, *, expected_corpus_sha256: str | None = None,
                                expected_adapter_sha256: str | None = None) -> dict[str, object]:
    """Rebuild every native source and derived field; never trust summary flags."""
    if type(matrix) is not dict:
        raise ValueError("INVALID_INPUT")
    try:
        corpus = EquivalenceCorpus.from_dict(matrix["corpus"])
        replay = run_equivalence_battery(corpus)
        _exact(matrix, replay)
        bindings = {"corpus": (expected_corpus_sha256, replay["corpus"]["sha256"]),
                    "adapters": (expected_adapter_sha256, replay["adapter_manifest"]["sha256"])}
        for expected, actual in bindings.values():
            if expected is not None and validate_sha256(expected) != actual:
                raise ValueError("INVALID_INPUT")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
    return _seal("validation", switch_equivalence_matrix_hash=replay["sha256"],
        corpus_sha256=replay["corpus"]["sha256"], adapter_manifest_sha256=replay["adapter_manifest"]["sha256"],
        case_count=replay["case_count"], equivalent_case_count=replay["equivalent_case_count"],
        expected_difference_count=replay["expected_difference_count"],
        default_corpus_used=replay["default_corpus_used"],
        replay_verified=True, native_sources_reconstructed=True,
        all_cases_equivalent=replay["all_cases_equivalent"], all_passed=replay["battery_passed"],
        external_bindings_verified={key: expected is not None for key, (expected, _) in bindings.items()})


def demo_equivalence_corpus() -> EquivalenceCorpus:
    """36 shared availability cases and five deliberately unequal fault controls."""
    cases = [EquivalenceCase(f"destination-{dest}-{left}-{right}", dest, (left, right), b"\x00opaque\xff")
             for dest in range(4) for left, right in product(LANE_STATES, repeat=2)]
    cases.extend(EquivalenceCase("fault-" + fault, 0, ("idle", "idle"), b"\x00opaque\xff", fault=fault)
                 for fault in FAULTS if fault != "none")
    return EquivalenceCorpus(tuple(sorted(cases, key=lambda case: case.case_id)))
