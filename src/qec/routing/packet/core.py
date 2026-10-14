# SPDX-License-Identifier: MPL-2.0
"""Bounded, offline v175.0–v175.6 packet switching; no network authority.

All time is logical. Packet identities are immutable; mutable transport state is
held only in hash-linked events. Receipts are replayed in full, never trusted on
the strength of their self-hash. The published v173.0 source API is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import re
from types import MappingProxyType

from qec.sonify.canonical import canonical_json, canonical_sha256, validate_sha256

CONTRACT_VERSION = "175.6"
MAX_NODES = 16
MAX_ROUTES = 128
MAX_REQUESTS = 16
MAX_PAYLOAD_BYTES = 4096
MAX_FRAGMENT_BYTES = 256
MAX_PACKETS = 256
MAX_FRAGMENTS = 64
MAX_HOPS = 16
MAX_QUEUE_CAPACITY = 64
MAX_FAULTS = 128
MAX_LOGICAL_TICK = 1023
MAX_EVENTS = 131072
CLAIM_BOUNDARY = MappingProxyType({
    "classical_software_model_only": True,
    "implementation_status": "implemented_unreleased",
    "released_source_package_version": "173.0.0",
    "logical_time_only": True,
    "payload_mutation_permitted": False,
    "declared_transport_corruption_is_fault_evidence_only": True,
    "decoder_output_mutation_permitted": False,
    "provenance_identity_is_caller_declared": True,
    "external_observations_are_authoritative": False,
    "live_network_activation_present": False,
    "physical_network_fidelity": False,
    "hardware_or_quantum_evidence": False,
    "browser_demo_is_canonical_evidence": False,
    "receipt_proves": "bounded_offline_packet_execution_under_declared_inputs_and_faults",
    "receipt_does_not_prove": "authenticated_provenance_decoder_correctness_or_carrier_reliability",
})


def _seal(name: str, **fields: object) -> dict:
    value = {"schema": "qec.packet-" + name + ".v1", "contract_version": CONTRACT_VERSION, **fields}
    return {**value, "sha256": canonical_sha256(value)}


def _exact(actual: object, expected: dict) -> None:
    try:
        if type(actual) is not dict or canonical_json(actual) != canonical_json(expected):
            raise ValueError("INVALID_INPUT")
    except (TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _int(value: object, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("INVALID_INPUT")


def _text(value: object, maximum: int = 64) -> None:
    if type(value) is not str or not 1 <= len(value) <= maximum:
        raise ValueError("INVALID_INPUT")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("INVALID_INPUT") from exc


def _name(value: object) -> None:
    _text(value)
    if re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError("INVALID_INPUT")


def _address(value: object, *, prefix: bool = False) -> None:
    if type(value) is not str or not (0 if prefix else 1) <= len(value) <= 32:
        raise ValueError("INVALID_INPUT")
    if any(bit not in "01" for bit in value):
        raise ValueError("INVALID_INPUT")


def _hash(value: object) -> None:
    try:
        if type(value) is not str:
            raise ValueError("INVALID_INPUT")
        validate_sha256(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _seq(value: object, maximum: int) -> tuple:
    if type(value) not in (tuple, list) or len(value) > maximum:
        raise ValueError("INVALID_INPUT")
    return tuple(value)


def _bytes(value: object, maximum: int) -> None:
    if type(value) is not bytes or len(value) > maximum:
        raise ValueError("INVALID_INPUT")


def _unhex(value: object, maximum: int) -> bytes:
    if type(value) is not str or len(value) > maximum * 2 or re.fullmatch(r"(?:[0-9a-f]{2})*", value) is None:
        raise ValueError("INVALID_INPUT")
    return bytes.fromhex(value)


def _read(cls, value: object, fields: tuple[str, ...], transforms=None):
    try:
        if type(value) is not dict:
            raise ValueError("INVALID_INPUT")
        transforms = transforms or {}
        args = [transforms.get(field, lambda item: item)(value[field]) for field in fields]
        result = cls(*args)
        _exact(value, result.as_dict())
        return result
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class PacketRequest:
    request_id: str
    source: str
    destination: str
    payload: bytes
    provenance_sha256: str
    priority: int = 0
    logical_tick: int = 0

    def __post_init__(self):
        _name(self.request_id)
        _name(self.source)
        _address(self.destination)
        _bytes(self.payload, MAX_PAYLOAD_BYTES)
        _hash(self.provenance_sha256)
        _int(self.priority, 7)
        _int(self.logical_tick, MAX_LOGICAL_TICK)

    def as_dict(self) -> dict:
        return _seal("request", request_id=self.request_id, source=self.source,
                     destination=self.destination, payload_hex=self.payload.hex(),
                     payload_sha256=hashlib.sha256(self.payload).hexdigest(),
                     provenance_sha256=self.provenance_sha256, priority=self.priority,
                     logical_tick=self.logical_tick)

    @classmethod
    def from_dict(cls, value):
        return _read(cls, value, ("request_id", "source", "destination", "payload_hex",
                                 "provenance_sha256", "priority", "logical_tick"),
                     {"payload_hex": lambda v: _unhex(v, MAX_PAYLOAD_BYTES)})


@dataclass(frozen=True)
class PacketEnvelope:
    """Immutable header and fragment. Hop count and wire corruption are excluded."""
    request_sha256: str
    request_id: str
    source: str
    destination: str
    original_payload_sha256: str
    provenance_sha256: str
    sequence: int
    fragment_count: int
    offset: int
    payload: bytes
    priority: int = 0

    def __post_init__(self):
        for digest in (self.request_sha256, self.original_payload_sha256, self.provenance_sha256):
            _hash(digest)
        _name(self.request_id)
        _name(self.source)
        _address(self.destination)
        _int(self.fragment_count, MAX_FRAGMENTS, 1)
        _int(self.sequence, self.fragment_count - 1)
        _int(self.offset, MAX_PAYLOAD_BYTES)
        _bytes(self.payload, MAX_FRAGMENT_BYTES)
        _int(self.priority, 7)
        if self.offset + len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ValueError("INVALID_INPUT")

    def as_dict(self) -> dict:
        sequence_identity = _seal("sequence-identity", request_sha256=self.request_sha256,
                                  sequence=self.sequence, fragment_count=self.fragment_count)
        return _seal("envelope", request_sha256=self.request_sha256, request_id=self.request_id,
                     source=self.source, destination=self.destination,
                     original_payload_sha256=self.original_payload_sha256,
                     provenance_sha256=self.provenance_sha256, sequence=self.sequence,
                     fragment_count=self.fragment_count, offset=self.offset,
                     payload_hex=self.payload.hex(), fragment_sha256=hashlib.sha256(self.payload).hexdigest(),
                     sequence_identity=sequence_identity, priority=self.priority,
                     transport_metadata_in_identity=False)

    @classmethod
    def from_dict(cls, value):
        return _read(cls, value, ("request_sha256", "request_id", "source", "destination",
                                 "original_payload_sha256", "provenance_sha256", "sequence",
                                 "fragment_count", "offset", "payload_hex", "priority"),
                     {"payload_hex": lambda v: _unhex(v, MAX_FRAGMENT_BYTES)})


def canonical_packet_schema() -> dict:
    return _seal("canonical-schema", envelope_schema="qec.packet-envelope.v1",
                 encoding="utf8_sorted_keys_compact_json_no_floats", address_encoding="binary_string_1_to_32_bits",
                 payload_encoding="lowercase_hex", payload_hash="sha256_of_original_bytes",
                 max_payload_bytes=MAX_PAYLOAD_BYTES, max_fragment_bytes=MAX_FRAGMENT_BYTES,
                 max_fragments=MAX_FRAGMENTS, max_packets=MAX_PACKETS,
                 sequence_policy="zero_based_contiguous_fragment_indices",
                 provenance_policy="caller_declared_sha256_not_authenticated",
                 transport_metadata_policy="mutable_hop_queue_and_wire_state_excluded_from_envelope",
                 claim_boundary=dict(CLAIM_BOUNDARY))


def packetize(request: PacketRequest, fragment_bytes: int = MAX_FRAGMENT_BYTES) -> tuple[PacketEnvelope, ...]:
    if type(request) is not PacketRequest:
        raise ValueError("INVALID_INPUT")
    _int(fragment_bytes, MAX_FRAGMENT_BYTES, 1)
    snapshot = request.as_dict()
    count = max(1, (len(request.payload) + fragment_bytes - 1) // fragment_bytes)
    _int(count, MAX_FRAGMENTS, 1)
    return tuple(PacketEnvelope(snapshot["sha256"], request.request_id, request.source,
                                request.destination, snapshot["payload_sha256"], request.provenance_sha256,
                                sequence, count, sequence * fragment_bytes,
                                request.payload[sequence * fragment_bytes:(sequence + 1) * fragment_bytes],
                                request.priority) for sequence in range(count))


@dataclass(frozen=True)
class PacketEndpoint:
    address: str
    node: str

    def __post_init__(self):
        _address(self.address)
        _name(self.node)

    def as_dict(self):
        return _seal("endpoint", address=self.address, node=self.node)

    @classmethod
    def from_dict(cls, value):
        return _read(cls, value, ("address", "node"))


@dataclass(frozen=True)
class Route:
    node: str
    prefix: str
    next_hops: tuple[str, ...]
    metric: int = 0
    route_id: str = "route"

    def __post_init__(self):
        _name(self.node)
        _address(self.prefix, prefix=True)
        _int(self.metric, 65535)
        _name(self.route_id)
        hops = _seq(self.next_hops, MAX_NODES)
        for hop in hops:
            _name(hop)
        if not hops or len(hops) != len(set(hops)) or self.node in hops:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "next_hops", hops)

    def as_dict(self):
        return _seal("route", node=self.node, prefix=self.prefix, next_hops=list(self.next_hops),
                     metric=self.metric, route_id=self.route_id)

    @classmethod
    def from_dict(cls, value):
        return _read(cls, value, ("node", "prefix", "next_hops", "metric", "route_id"))


def _route_key(route):
    return (route.node, -len(route.prefix), route.metric, route.route_id, route.prefix)


@dataclass(frozen=True)
class PacketTopology:
    nodes: tuple[str, ...]
    destinations: tuple[PacketEndpoint, ...]
    routes: tuple[Route, ...]

    def __post_init__(self):
        nodes = _seq(self.nodes, MAX_NODES)
        for node in nodes:
            _name(node)
        if not nodes or len(nodes) != len(set(nodes)):
            raise ValueError("INVALID_INPUT")
        endpoints = _seq(self.destinations, MAX_ROUTES)
        routes = _seq(self.routes, MAX_ROUTES)
        if any(type(e) is not PacketEndpoint or e.node not in nodes for e in endpoints):
            raise ValueError("INVALID_INPUT")
        if len({e.address for e in endpoints}) != len(endpoints):
            raise ValueError("INVALID_INPUT")
        if any(type(r) is not Route or r.node not in nodes or any(n not in nodes for n in r.next_hops)
               for r in routes):
            raise ValueError("INVALID_INPUT")
        if len({r.route_id for r in routes}) != len(routes):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "nodes", tuple(sorted(nodes)))
        object.__setattr__(self, "destinations", tuple(sorted(endpoints, key=lambda e: e.address)))
        object.__setattr__(self, "routes", tuple(sorted(routes, key=_route_key)))

    def as_dict(self):
        return _seal("route-table-manifest", nodes=list(self.nodes),
                     destinations=[e.as_dict() for e in self.destinations],
                     routes=[r.as_dict() for r in self.routes],
                     route_order="node_then_descending_prefix_length_then_metric_then_route_id_then_prefix",
                     lookup_policy="exact_binary_prefix_longest_match_then_lowest_metric_then_route_id",
                     alternate_policy="declared_next_hop_order_only_no_less_specific_fallback",
                     destination_policy="exact_registered_address_at_registered_node")

    @classmethod
    def from_dict(cls, value):
        return _read(cls, value, ("nodes", "destinations", "routes"),
                     {"destinations": lambda v: tuple(PacketEndpoint.from_dict(e) for e in _seq(v, MAX_ROUTES)),
                      "routes": lambda v: tuple(Route.from_dict(r) for r in _seq(v, MAX_ROUTES))})


def lookup_route(topology: PacketTopology, node: str, destination: str) -> Route | None:
    if type(topology) is not PacketTopology:
        raise ValueError("INVALID_INPUT")
    _name(node)
    _address(destination)
    if node not in topology.nodes:
        raise ValueError("INVALID_INPUT")
    return next((r for r in topology.routes if r.node == node and destination.startswith(r.prefix)), None)


@dataclass(frozen=True)
class QueuePolicy:
    capacity: int = 16
    max_hops: int = 8

    def __post_init__(self):
        _int(self.capacity, MAX_QUEUE_CAPACITY, 1)
        _int(self.max_hops, MAX_HOPS, 1)

    def as_dict(self):
        return _seal("queue-policy-manifest", capacity=self.capacity, max_hops=self.max_hops,
                     service_per_node_per_tick=1, overflow="drop_arriving_packet_no_eviction",
                     arrival_order="priority_then_request_id_then_sequence",
                     service_order="priority_then_enqueue_tick_then_request_id_then_sequence",
                     tick_phases="admit_all_arrivals_then_service_nodes_in_name_order",
                     forwarding_delay_ticks=1, initial_state="all_node_queues_empty")

    @classmethod
    def from_dict(cls, value):
        return _read(cls, value, ("capacity", "max_hops"))


@dataclass(frozen=True)
class PacketFault:
    logical_tick: int
    node: str
    next_hop: str
    kind: str = "link_down"
    duration: int = 1
    source_state: str = "declared_fault"

    def __post_init__(self):
        _int(self.logical_tick, MAX_LOGICAL_TICK)
        _name(self.node)
        _name(self.next_hop)
        if type(self.kind) is not str or self.kind not in ("link_down", "drop", "corrupt"):
            raise ValueError("INVALID_INPUT")
        if type(self.source_state) is not str or self.source_state not in ("declared_fault", "busy", "quarantined", "unavailable"):
            raise ValueError("INVALID_INPUT")
        _int(self.duration, MAX_LOGICAL_TICK + 1, 1)
        if self.logical_tick + self.duration > MAX_LOGICAL_TICK + 1 or self.node == self.next_hop:
            raise ValueError("INVALID_INPUT")

    def as_dict(self):
        return _seal("fault", logical_tick=self.logical_tick, node=self.node, next_hop=self.next_hop,
                     kind=self.kind, duration=self.duration, source_state=self.source_state)

    @classmethod
    def from_dict(cls, value):
        return _read(cls, value, ("logical_tick", "node", "next_hop", "kind", "duration", "source_state"))


@dataclass(frozen=True)
class FaultPlan:
    faults: tuple[PacketFault, ...] = ()

    def __post_init__(self):
        faults = _seq(self.faults, MAX_FAULTS)
        if any(type(f) is not PacketFault for f in faults):
            raise ValueError("INVALID_INPUT")
        faults = tuple(sorted(faults, key=lambda f: (f.logical_tick, f.node, f.next_hop, f.kind, f.duration)))
        for i, fault in enumerate(faults):
            for earlier in faults[:i]:
                if ((fault.node, fault.next_hop, fault.kind) == (earlier.node, earlier.next_hop, earlier.kind)
                        and fault.logical_tick < earlier.logical_tick + earlier.duration):
                    raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "faults", faults)

    def as_dict(self):
        return _seal("fault-plan", faults=[f.as_dict() for f in self.faults],
                     active_interval="start_inclusive_end_exclusive",
                     precedence=["link_down", "drop", "corrupt"],
                     corruption_rule="xor_first_wire_byte_with_1_or_append_01_to_empty",
                     recovery_rule="link_available_again_after_declared_interval_no_retransmission")

    @classmethod
    def from_dict(cls, value):
        return _read(cls, value, ("faults",),
                     {"faults": lambda v: tuple(PacketFault.from_dict(f) for f in _seq(v, MAX_FAULTS))})

    def active(self, tick: int, node: str, next_hop: str) -> tuple[PacketFault, ...]:
        _int(tick, MAX_LOGICAL_TICK + MAX_PACKETS * (MAX_HOPS + 1) + 1)
        _name(node)
        _name(next_hop)
        return tuple(f for f in self.faults if f.node == node and f.next_hop == next_hop
                     and f.logical_tick <= tick < f.logical_tick + f.duration)


@dataclass(frozen=True)
class PacketInput:
    requests: tuple[PacketRequest, ...]
    topology: PacketTopology
    queue_policy: QueuePolicy = QueuePolicy()
    fault_plan: FaultPlan = FaultPlan()
    fragment_bytes: int = MAX_FRAGMENT_BYTES

    def __post_init__(self):
        requests = _seq(self.requests, MAX_REQUESTS)
        if any(type(r) is not PacketRequest for r in requests):
            raise ValueError("INVALID_INPUT")
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("INVALID_INPUT")
        if type(self.topology) is not PacketTopology or type(self.queue_policy) is not QueuePolicy or type(self.fault_plan) is not FaultPlan:
            raise ValueError("INVALID_INPUT")
        if any(r.source not in self.topology.nodes for r in requests):
            raise ValueError("INVALID_INPUT")
        _int(self.fragment_bytes, MAX_FRAGMENT_BYTES, 1)
        if sum(len(packetize(r, self.fragment_bytes)) for r in requests) > MAX_PACKETS:
            raise ValueError("INVALID_INPUT")
        edges = {(r.node, hop) for r in self.topology.routes for hop in r.next_hops}
        if any((f.node, f.next_hop) not in edges for f in self.fault_plan.faults):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "requests", tuple(sorted(requests, key=lambda r: r.request_id)))

    def as_dict(self):
        return _seal("fabric-input", requests=[r.as_dict() for r in self.requests],
                     topology=self.topology.as_dict(), queue_policy=self.queue_policy.as_dict(),
                     fault_plan=self.fault_plan.as_dict(), fragment_bytes=self.fragment_bytes,
                     canonical_packet_schema_sha256=canonical_packet_schema()["sha256"])

    @classmethod
    def from_dict(cls, value):
        return _read(cls, value, ("requests", "topology", "queue_policy", "fault_plan", "fragment_bytes"),
                     {"requests": lambda v: tuple(PacketRequest.from_dict(r) for r in _seq(v, MAX_REQUESTS)),
                      "topology": PacketTopology.from_dict, "queue_policy": QueuePolicy.from_dict,
                      "fault_plan": FaultPlan.from_dict})


def execute_packet_fabric(source: PacketInput) -> dict:
    """Replay a closed batch through bounded node queues and declared link faults."""
    if type(source) is not PacketInput:
        raise ValueError("INVALID_INPUT")
    source = PacketInput.from_dict(source.as_dict())
    snapshot = source.as_dict()
    envelopes = tuple(p for request in source.requests for p in packetize(request, source.fragment_bytes))
    packets = {p.as_dict()["sha256"]: p for p in envelopes}
    packet_dicts = {key: p.as_dict() for key, p in packets.items()}
    previous = snapshot["sha256"]
    packet_previous = {key: key for key in packets}
    events = []

    def event(kind, tick, node, packet_id, **details):
        nonlocal previous
        if len(events) >= MAX_EVENTS:
            raise ValueError("INVALID_INPUT")
        row = _seal("hop-event", index=len(events), logical_tick=tick, node=node, kind=kind,
                    packet_sha256=packet_id, previous_event_sha256=previous,
                    previous_packet_event_sha256=packet_previous.get(packet_id), details=details)
        events.append(row)
        previous = row["sha256"]
        if packet_id is not None:
            packet_previous[packet_id] = row["sha256"]
        return row

    # Mutable wire bytes and hop counters never mutate an envelope identity.
    arrivals = {}
    wire = {key: p.payload for key, p in packets.items()}
    hops = {key: 0 for key in packets}
    paths = {key: [p.source] for key, p in packets.items()}
    delivered = {r.request_id: [] for r in source.requests}
    drops = {r.request_id: [] for r in source.requests}
    queues = {node: [] for node in source.topology.nodes}
    endpoints = {e.address: e.node for e in source.topology.destinations}
    requests = {r.request_id: r for r in source.requests}
    for key, packet in packets.items():
        tick = requests[packet.request_id].logical_tick
        arrivals.setdefault(tick, []).append((packet.source, key))

    def drop(key, tick, node, reason):
        row = event("drop", tick, node, key, reason=reason, hop_count=hops[key])
        drops[packets[key].request_id].append({"packet_sha256": key, "reason": reason, "event_sha256": row["sha256"]})

    def order(key):
        packet = packets[key]
        return packet.priority, packet.request_id, packet.sequence

    tick = 0
    # One packet leaves each nonempty node every tick; every forwarded packet
    # consumes a finite hop. This bound covers gaps, queues, forwarding, delivery.
    stop_tick = max((r.logical_tick for r in source.requests), default=0) + len(packets) * (source.queue_policy.max_hops + 1) + 1
    while arrivals or any(queues.values()):
        if tick > stop_tick:
            raise ValueError("INVALID_INPUT")
        for node, key in sorted(arrivals.pop(tick, ()), key=lambda pair: (pair[0], *order(pair[1]))):
            packet = packets[key]
            if hops[key] == 0:
                event("packetize", tick, node, key, request_sha256=packet.request_sha256,
                      sequence_identity_sha256=packet_dicts[key]["sequence_identity"]["sha256"],
                      offset=packet.offset, fragment_sha256=packet_dicts[key]["fragment_sha256"])
            queue = queues[node]
            if len(queue) == source.queue_policy.capacity:
                event("queue_overflow", tick, node, key, queue_before=[entry[0] for entry in queue],
                      capacity=source.queue_policy.capacity, decision="drop_arriving_packet_no_eviction")
                drop(key, tick, node, "queue_overflow")
            else:
                before = [entry[0] for entry in queue]
                queue.append((key, tick))
                queue.sort(key=lambda entry: (packets[entry[0]].priority, entry[1], packets[entry[0]].request_id, packets[entry[0]].sequence))
                event("enqueue", tick, node, key, queue_before=before, queue_after=[entry[0] for entry in queue])
        for node in source.topology.nodes:
            queue = queues[node]
            if not queue:
                continue
            before = [entry[0] for entry in queue]
            key, enqueued_tick = queue.pop(0)
            packet = packets[key]
            event("dequeue", tick, node, key, queue_before=before, queue_after=[entry[0] for entry in queue],
                  enqueued_tick=enqueued_tick, hop_count=hops[key])
            if endpoints.get(packet.destination) == node:
                row = event("deliver_fragment", tick, node, key, destination=packet.destination,
                            received_payload_hex=wire[key].hex(), received_payload_sha256=hashlib.sha256(wire[key]).hexdigest(),
                            hop_count=hops[key])
                delivered[packet.request_id].append({"packet_sha256": key, "sequence": packet.sequence,
                    "offset": packet.offset, "node": node, "payload_hex": wire[key].hex(), "event_sha256": row["sha256"]})
                continue
            if packet.destination not in endpoints:
                drop(key, tick, node, "unknown_destination")
                continue
            if hops[key] >= source.queue_policy.max_hops:
                drop(key, tick, node, "hop_limit")
                continue
            route = lookup_route(source.topology, node, packet.destination)
            event("route_lookup", tick, node, key, destination=packet.destination,
                  route_sha256=route.as_dict()["sha256"] if route else None,
                  matching_route_sha256=[r.as_dict()["sha256"] for r in source.topology.routes
                                         if r.node == node and packet.destination.startswith(r.prefix)])
            if route is None:
                drop(key, tick, node, "no_route")
                continue
            selected = None
            for alternate_index, next_hop in enumerate(route.next_hops):
                active = source.fault_plan.active(tick, node, next_hop)
                unavailable = any(f.kind == "link_down" for f in active)
                event("route_attempt", tick, node, key, route_sha256=route.as_dict()["sha256"],
                      next_hop=next_hop, alternate_index=alternate_index,
                      active_fault_sha256=[f.as_dict()["sha256"] for f in active],
                      decision="unavailable" if unavailable else "selected")
                if not unavailable:
                    selected = (next_hop, alternate_index, active)
                    break
            if selected is None:
                drop(key, tick, node, "all_next_hops_unavailable")
                continue
            next_hop, alternate_index, active = selected
            if alternate_index:
                event("failover", tick, node, key, route_sha256=route.as_dict()["sha256"],
                      selected_next_hop=next_hop, alternate_index=alternate_index,
                      skipped_next_hops=list(route.next_hops[:alternate_index]))
            if any(f.kind == "drop" for f in active):
                event("fault_drop", tick, node, key, next_hop=next_hop,
                      fault_sha256=[f.as_dict()["sha256"] for f in active if f.kind == "drop"])
                drop(key, tick, node, "declared_link_drop")
                continue
            old = wire[key]
            if any(f.kind == "corrupt" for f in active):
                wire[key] = bytes([old[0] ^ 1]) + old[1:] if old else b"\x01"
                event("fault_corrupt", tick, node, key, next_hop=next_hop,
                      before_payload_hex=old.hex(), after_payload_hex=wire[key].hex(),
                      fault_sha256=[f.as_dict()["sha256"] for f in active if f.kind == "corrupt"])
            hops[key] += 1
            paths[key].append(next_hop)
            event("forward", tick, node, key, next_hop=next_hop, route_sha256=route.as_dict()["sha256"],
                  arrival_tick=tick + 1, hop_count=hops[key], wire_payload_sha256=hashlib.sha256(wire[key]).hexdigest())
            arrivals.setdefault(tick + 1, []).append((next_hop, key))
        tick += 1
    results = []
    for request in source.requests:
        fragments = sorted(delivered[request.request_id], key=lambda row: row["sequence"])
        expected_packets = [p for p in envelopes if p.request_id == request.request_id]
        payload = b"".join(bytes.fromhex(row["payload_hex"]) for row in fragments)
        complete = [row["sequence"] for row in fragments] == list(range(len(expected_packets)))
        offsets_valid = complete and [row["offset"] for row in fragments] == [p.offset for p in expected_packets]
        identity_match = complete and offsets_valid and hashlib.sha256(payload).hexdigest() == request.as_dict()["payload_sha256"] and payload == request.payload
        reason = "payload_verified" if identity_match else ("payload_mismatch" if complete else "missing_fragments")
        row = event("reassembly", tick, endpoints.get(request.destination), None, request_id=request.request_id,
                    request_sha256=request.as_dict()["sha256"], received_fragment_count=len(fragments),
                    expected_fragment_count=len(expected_packets), outcome="accepted" if identity_match else "rejected",
                    reason=reason, recomputed_payload_sha256=hashlib.sha256(payload).hexdigest())
        results.append(_seal("reassembly-receipt", request_id=request.request_id, request_sha256=request.as_dict()["sha256"],
                            destination=request.destination, destination_node=endpoints.get(request.destination),
                            outcome="accepted" if identity_match else "rejected", reason=reason,
                            payload_sha256=request.as_dict()["payload_sha256"], provenance_sha256=request.provenance_sha256,
                            recomputed_payload_sha256=hashlib.sha256(payload).hexdigest(), reassembled_payload_hex=payload.hex(),
                            accepted_payload_sha256=hashlib.sha256(payload).hexdigest() if identity_match else None,
                            expected_packet_sha256=[p.as_dict()["sha256"] for p in expected_packets],
                            fragments=fragments, drops=drops[request.request_id], complete=complete,
                            packet_paths=[{"packet_sha256": p.as_dict()["sha256"], "nodes": paths[p.as_dict()["sha256"]]} for p in expected_packets],
                            reassembly_event_sha256=row["sha256"]))
    chain = _seal("hop-receipt-chain", input_sha256=snapshot["sha256"], events=events,
                  chain_head_sha256=previous, packet_tails=[{"packet_sha256": key, "tail_sha256": packet_previous[key]}
                                                           for key in sorted(packet_previous)])
    failover = _seal("failover-receipt", input_sha256=snapshot["sha256"], fault_plan_sha256=snapshot["fault_plan"]["sha256"],
                     hop_receipt_chain_sha256=chain["sha256"],
                     attempt_event_sha256=[e["sha256"] for e in events if e["kind"] == "route_attempt"],
                     failover_event_sha256=[e["sha256"] for e in events if e["kind"] == "failover"],
                     fault_event_sha256=[e["sha256"] for e in events if e["kind"] in ("fault_drop", "fault_corrupt")],
                     failover_count=sum(e["kind"] == "failover" for e in events),
                     replay_safe=True, recovery_policy=snapshot["fault_plan"]["recovery_rule"])
    return _seal("fabric-receipt", input=snapshot, canonical_packet_schema=canonical_packet_schema(),
                 envelopes=[packet_dicts[key] for key in packets], hop_receipt_chain=chain,
                 failover_receipt=failover, results=results, packet_count=len(packets),
                 accepted_count=sum(r["outcome"] == "accepted" for r in results),
                 rejected_count=sum(r["outcome"] == "rejected" for r in results),
                 final_queues=[{"node": node, "packets": []} for node in source.topology.nodes],
                 last_logical_tick=tick, claim_boundary=dict(CLAIM_BOUNDARY))


def validate_packet_receipt(receipt: object, *, expected_input_sha256=None, expected_topology_sha256=None,
                            expected_queue_policy_sha256=None, expected_fault_plan_sha256=None,
                            expected_receipt_sha256=None) -> dict:
    """Rebuild every envelope, queue, hop, fault and reassembly; optionally pin origin."""
    try:
        if type(receipt) is not dict:
            raise ValueError("INVALID_INPUT")
        source = PacketInput.from_dict(receipt["input"])
        snapshot = source.as_dict()
        pins = {"input": (expected_input_sha256, snapshot["sha256"]),
                "topology": (expected_topology_sha256, snapshot["topology"]["sha256"]),
                "queue_policy": (expected_queue_policy_sha256, snapshot["queue_policy"]["sha256"]),
                "fault_plan": (expected_fault_plan_sha256, snapshot["fault_plan"]["sha256"]),
                "receipt": (expected_receipt_sha256, receipt.get("sha256"))}
        for expected, actual in pins.values():
            if expected is not None:
                _hash(expected)
                if expected != actual:
                    raise ValueError("INVALID_INPUT")
        replay = execute_packet_fabric(source)
        _exact(receipt, replay)
        return _seal("fabric-validation", canonical_packet_fabric_receipt_hash=replay["sha256"],
                     deterministic_failover_receipt_hash=replay["failover_receipt"]["sha256"],
                     complete_replay_verified=True, all_packet_chains_verified=True,
                     accepted_reassemblies_rehashed=True, packet_count=replay["packet_count"],
                     accepted_count=replay["accepted_count"], rejected_count=replay["rejected_count"],
                     nonempty_evidence=bool(replay["packet_count"]),
                     external_bindings_verified={name: pair[0] is not None for name, pair in pins.items()},
                     all_passed=True)
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def create_packet_report(receipt: object) -> dict:
    validation = validate_packet_receipt(receipt)
    receipt = deepcopy(receipt)
    return _seal("fabric-report", source_receipt=receipt, validation=validation,
                 canonical_packet_fabric_receipt_hash=receipt["sha256"],
                 accepted_count=receipt["accepted_count"], rejected_count=receipt["rejected_count"],
                 queue_drop_count=sum(e["kind"] == "queue_overflow" for e in receipt["hop_receipt_chain"]["events"]),
                 failover_count=receipt["failover_receipt"]["failover_count"],
                 claim_boundary=dict(CLAIM_BOUNDARY))


def validate_packet_report(report: object) -> dict:
    try:
        expected = create_packet_report(report["source_receipt"])
        _exact(report, expected)
        return _seal("report-validation", packet_fabric_report_sha256=expected["sha256"], all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def demo_packet_input(scenario: str = "failover") -> PacketInput:
    if type(scenario) is not str or scenario not in ("success", "failover", "overflow", "corrupt", "drop", "recovery"):
        raise ValueError("INVALID_INPUT")
    topology = PacketTopology(("a", "b", "c", "d"), (PacketEndpoint("1010", "d"),), (
        Route("a", "10", ("b", "c"), route_id="a-10"), Route("b", "10", ("d",), route_id="b-10"),
        Route("c", "10", ("d",), route_id="c-10")))
    requests = (PacketRequest("call-a", "a", "1010", b"opaque-correction", "a" * 64),)
    faults = ()
    if scenario in ("failover", "recovery"):
        faults = (PacketFault(0, "a", "b", duration=2),)
    elif scenario in ("corrupt", "drop"):
        faults = (PacketFault(0, "a", "b", scenario, 2),)
    if scenario == "recovery":
        requests += (PacketRequest("call-b", "a", "1010", b"recovered", "b" * 64, logical_tick=3),)
    return PacketInput(requests, topology, QueuePolicy(1 if scenario == "overflow" else 8, 8),
                       FaultPlan(faults), 8)


def validate_packet_artifact(artifact: object, *, fabric_receipt: object, expected_input_sha256=None,
                             expected_receipt_sha256=None) -> dict:
    """Validate a separated registered artifact against complete native replay.

    A chain or failover view alone omits the source snapshots needed to replay.
    Therefore this API requires the full fabric receipt, even for empty views.
    """
    try:
        validation = validate_packet_receipt(fabric_receipt, expected_input_sha256=expected_input_sha256,
                                              expected_receipt_sha256=expected_receipt_sha256)
        if type(artifact) is not dict:
            raise ValueError("INVALID_INPUT")
        candidates = [fabric_receipt["canonical_packet_schema"], fabric_receipt["input"]["topology"],
                      fabric_receipt["input"]["queue_policy"], fabric_receipt["hop_receipt_chain"],
                      fabric_receipt["failover_receipt"], *fabric_receipt["results"]]
        candidate = next((item for item in candidates if item["sha256"] == artifact.get("sha256")), None)
        if candidate is None:
            raise ValueError("INVALID_INPUT")
        _exact(artifact, candidate)
        return _seal("artifact-validation", artifact_schema=candidate["schema"], artifact_sha256=candidate["sha256"],
                     canonical_packet_fabric_receipt_hash=fabric_receipt["sha256"],
                     complete_source_replay_verified=True,
                     external_bindings_verified=validation["external_bindings_verified"], all_passed=True)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("INVALID_INPUT") from exc
