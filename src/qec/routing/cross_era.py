# SPDX-License-Identifier: MPL-2.0
"""v176.0–.5 bounded six-era migration and replay, with visible capability gaps.

Every execution is a declared independent initial-state experiment. This module
never pretends that native commits, reservations, frames and queues are the same
lifecycle, or that Strowger has a payload capability absent from its frozen API.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from itertools import combinations, product
import re
from types import MappingProxyType

from qec.sonify.canonical import canonical_bytes, canonical_json, canonical_sha256, validate_sha256
from .equivalence import (EquivalenceCase, EquivalenceCorpus, FAULTS as LEGACY_FAULTS,
                          equivalence_adapter_manifest, run_equivalence_battery)
from .crossbar.multistage import CrossbarFabric
from .ess.programme import ProgramManifest, TranslationTable
from .ess.lifecycle import Call, FabricAdapter, LifecycleInput, execute_call_processing, validate_call_processing_receipt
from .digital import (DigitalRequest, DigitalSwitchInput, FrameFault, FrameSchema,
                      execute_digital_switch, validate_digital_receipt)
from .packet import (FaultPlan, PacketEndpoint, PacketFault, PacketInput, PacketRequest,
                     PacketTopology, QueuePolicy, Route, execute_packet_fabric, validate_packet_receipt)

CONTRACT_VERSION = '176.5'
ARCHITECTURES = ('strowger', 'panel', 'crossbar', 'ess', 'digital', 'packet')
LANE_STATES = ('idle', 'busy', 'quarantined')
DIGITAL_FAULTS = ('bit_flip', 'frame_loss', 'duplicate_frame', 'reorder_frames', 'alignment_slip', 'slot_collision')
FAULTS = (*LEGACY_FAULTS, 'ess_search_budget', *(f'digital_{x}' for x in DIGITAL_FAULTS),
          'packet_corrupt', 'packet_drop', 'packet_link_down')
MAX_CASES = 64
MAX_PAYLOAD = 256
MAX_CORPUS_PAYLOAD = 8192
CAPACITY_REASONS = MappingProxyType({'strowger': 'all_trunks_busy', 'panel': 'capacity_exhausted',
    'crossbar': 'no_admissible_complete_path', 'ess': 'no_admissible_complete_path',
    'digital': 'no_capacity', 'packet': 'all_next_hops_unavailable'})
SUCCESS_REASONS = MappingProxyType({'strowger': 'committed', 'panel': 'committed',
    'crossbar': 'first_admissible_complete_path', 'ess': 'complete_path_reserved',
    'digital': 'delivered', 'packet': 'payload_verified'})
FAULT_REASONS = MappingProxyType({'strowger_selector_fault': 'selector_fault', 'strowger_tone_mismatch': 'tone_mismatch',
    'panel_motor_stall': 'motor_stall', 'panel_sender_disagreement': 'sender_disagreement',
    'crossbar_search_budget': 'search_budget_exhausted', 'ess_search_budget': 'search_budget_exhausted',
    'digital_bit_flip': 'payload_hash_mismatch', 'digital_frame_loss': 'frame_count_mismatch',
    'digital_duplicate_frame': 'duplicate_frame', 'digital_reorder_frames': 'reordered_frames',
    'digital_alignment_slip': 'alignment_slip', 'digital_slot_collision': 'invalid_symbol',
    'packet_corrupt': 'payload_mismatch', 'packet_drop': 'declared_link_drop', 'packet_link_down': 'all_next_hops_unavailable'})
COMPARISON_FIELDS = ('requested_destination', 'routing_outcome', 'selected_lane', 'reached_destination', 'failure_class')
CLAIM_BOUNDARY = MappingProxyType({
    'classical_software_model_only': True,
    'implementation_status': 'implemented_unreleased',
    'comparison_scope': 'four_destinations_two_lane_independent_initial_state_experiments',
    'all_six_native_payload_equivalence': False,
    'strowger_payload_scope': 'canonical_case_envelope_only',
    'native_payload_scope': 'other_five_models_with_separate_transport_and_input_preservation_semantics',
    'trace_identity_required': False,
    'native_lifecycle_equivalence': False,
    'live_state_migration_performed': False,
    'migration_scope': 'hash_bound_input_topology_and_policy_translation',
    'external_provenance_authenticated': False,
    'decoder_correctness_proven': False,
    'hardware_performance_claims': False,
    'universal_architecture_superiority': False,
    'architecture_specific_faults_preserved': True,
})
CAPABILITIES = MappingProxyType({
    'strowger': {'native_payload': False, 'owned_reservations': False, 'fixed_frames': False, 'bounded_queues': False, 'source_bound_programme': False},
    'panel': {'native_payload': True, 'owned_reservations': False, 'fixed_frames': False, 'bounded_queues': False, 'source_bound_programme': False},
    'crossbar': {'native_payload': True, 'owned_reservations': True, 'fixed_frames': False, 'bounded_queues': False, 'source_bound_programme': False},
    'ess': {'native_payload': True, 'owned_reservations': True, 'fixed_frames': False, 'bounded_queues': False, 'source_bound_programme': True},
    'digital': {'native_payload': True, 'owned_reservations': True, 'fixed_frames': True, 'bounded_queues': False, 'source_bound_programme': False},
    'packet': {'native_payload': True, 'owned_reservations': False, 'fixed_frames': False, 'bounded_queues': True, 'source_bound_programme': False},
})

CAPABILITIES = MappingProxyType({key: MappingProxyType(value) for key, value in CAPABILITIES.items()})


def _seal(name, **fields):
    # A downstream receipt owns its serialized snapshot. Never retain aliases to
    # caller-owned source receipts that could mutate another artifact in place.
    value = {'schema': 'qec.cross-era-' + name + '.v1', 'contract_version': CONTRACT_VERSION, **deepcopy(fields)}
    return {**value, 'sha256': canonical_sha256(value)}


def _exact(actual, expected):
    try:
        if type(actual) is not dict or canonical_json(actual) != canonical_json(expected):
            raise ValueError('INVALID_INPUT')
    except (TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError('INVALID_INPUT') from exc


def _bounded(value):
    stack, count, chars = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > 2_000_000 or depth > 80:
            raise ValueError('INVALID_INPUT')
        if type(item) is dict:
            if len(item) > 128:
                raise ValueError('INVALID_INPUT')
            for key, child in item.items():
                if type(key) is not str or len(key) > 128:
                    raise ValueError('INVALID_INPUT')
                stack.append((child, depth + 1))
        elif type(item) is list:
            if len(item) > 8192:
                raise ValueError('INVALID_INPUT')
            stack.extend((x, depth + 1) for x in item)
        elif type(item) is str:
            chars += len(item)
            if len(item) > 32768 or chars > 64_000_000:
                raise ValueError('INVALID_INPUT')
        elif item is not None and type(item) not in (int, bool):
            raise ValueError('INVALID_INPUT')


def _pins(values):
    for expected, actual in values.values():
        if expected is not None and validate_sha256(expected) != actual:
            raise ValueError('INVALID_INPUT')
    return {key: expected is not None for key, (expected, _) in values.items()}


def _destination(index):
    return f'correction/destination-{index}'


@dataclass(frozen=True)
class CompatibilityCase:
    case_id: str
    destination: int
    lane_states: tuple[str, str]
    payload: bytes = b'\x00opaque\xff'
    decoder_output_sha256: str = 'a' * 64
    fault: str = 'none'

    def __post_init__(self):
        if type(self.case_id) is not str or re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', self.case_id) is None:
            raise ValueError('INVALID_INPUT')
        if type(self.destination) is not int or not 0 <= self.destination < 4:
            raise ValueError('INVALID_INPUT')
        if type(self.lane_states) not in (tuple, list) or len(self.lane_states) != 2:
            raise ValueError('INVALID_INPUT')
        if any(type(x) is not str or x not in LANE_STATES for x in self.lane_states):
            raise ValueError('INVALID_INPUT')
        if type(self.payload) is not bytes or len(self.payload) > MAX_PAYLOAD:
            raise ValueError('INVALID_INPUT')
        if type(self.decoder_output_sha256) is not str:
            raise ValueError('INVALID_INPUT')
        validate_sha256(self.decoder_output_sha256)
        if type(self.fault) is not str or self.fault not in FAULTS:
            raise ValueError('INVALID_INPUT')
        if self.fault != 'none' and (tuple(self.lane_states) != ('idle', 'idle') or len(self.payload) < 2):
            raise ValueError('INVALID_INPUT')
        object.__setattr__(self, 'lane_states', tuple(self.lane_states))

    def as_dict(self):
        return _seal('compatibility-case', case_id=self.case_id, destination=self.destination,
                     destination_label=_destination(self.destination), lane_states=list(self.lane_states),
                     payload_hex=self.payload.hex(), payload_sha256=hashlib.sha256(self.payload).hexdigest(),
                     decoder_output_sha256=self.decoder_output_sha256, fault=self.fault,
                     state_scope='independent_initial_snapshot', operator_commands=['admit', 'validate', 'release_where_native'])

    @classmethod
    def from_dict(cls, value):
        try:
            if type(value) is not dict or type(value['payload_hex']) is not str or len(value['payload_hex']) > MAX_PAYLOAD * 2:
                raise ValueError('INVALID_INPUT')
            result = cls(value['case_id'], value['destination'], value['lane_states'], bytes.fromhex(value['payload_hex']),
                         value['decoder_output_sha256'], value['fault'])
            _exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('INVALID_INPUT') from exc


@dataclass(frozen=True)
class CompatibilityCorpus:
    cases: tuple[CompatibilityCase, ...]

    def __post_init__(self):
        if type(self.cases) not in (tuple, list) or not 1 <= len(self.cases) <= MAX_CASES:
            raise ValueError('INVALID_INPUT')
        if any(type(x) is not CompatibilityCase for x in self.cases):
            raise ValueError('INVALID_INPUT')
        cases = tuple(CompatibilityCase.from_dict(x.as_dict()) for x in self.cases)
        names = tuple(x.case_id for x in cases)
        if names != tuple(sorted(set(names))) or sum(len(x.payload) for x in cases) > MAX_CORPUS_PAYLOAD:
            raise ValueError('INVALID_INPUT')
        object.__setattr__(self, 'cases', cases)

    def as_dict(self):
        return _seal('compatibility-corpus-manifest', cases=[x.as_dict() for x in self.cases],
                     architectures=list(ARCHITECTURES), common_payload_bound=MAX_PAYLOAD,
                     state_scope='independent_initial_snapshots_not_universal_concurrent_capacity')

    @classmethod
    def from_dict(cls, value):
        try:
            rows = value['cases']
            if type(rows) is not list or not 1 <= len(rows) <= MAX_CASES:
                raise ValueError('INVALID_INPUT')
            result = cls(tuple(CompatibilityCase.from_dict(x) for x in rows))
            _exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('INVALID_INPUT') from exc


def default_compatibility_corpus():
    rows = [CompatibilityCase(f'destination-{dest}-{left}-{right}', dest, (left, right))
            for dest in range(4) for left, right in product(LANE_STATES, repeat=2)]
    rows += [CompatibilityCase('fault-' + fault, 0, ('idle', 'idle'), fault=fault)
             for fault in FAULTS if fault != 'none']
    return CompatibilityCorpus(tuple(sorted(rows, key=lambda c: c.case_id)))


def _legacy_case(case):
    return EquivalenceCase(case.case_id, case.destination, case.lane_states, case.payload,
                           case.decoder_output_sha256, case.fault if case.fault in LEGACY_FAULTS else 'none')


def _native_inputs(case):
    original = equivalence_adapter_manifest()
    fabric = CrossbarFabric.from_dict(original['crossbar_fabric'])
    fabric = replace(fabric, interstage_links=tuple(replace(w, state=s) for w, s in zip(fabric.interstage_links, case.lane_states)))
    destinations = tuple(_destination(i) for i in range(4))
    programme = ProgramManifest(TranslationTable(tuple((d, (d,)) for d in destinations)),
                               max_search_evaluations=1 if case.fault == 'ess_search_budget' else 65536)
    adapter = FabricAdapter(fabric, (('source', 'ingress', 'H000'),),
                            tuple((d, 'egress', f'V{i:03d}') for i, d in enumerate(destinations)))
    ess = LifecycleInput(programme, (Call(case.case_id, 'source', _destination(case.destination), case.payload,
                                        case.decoder_output_sha256),), adapter)
    faults = ()
    if case.fault.startswith('digital_'):
        kind = case.fault.removeprefix('digital_')
        faults = (FrameFault(kind, 0, slot=0 if kind in ('bit_flip', 'slot_collision') else None),)
    digital = DigitalSwitchInput(DigitalRequest(case.case_id, 'source', _destination(case.destination),
                                               case.payload, case.decoder_output_sha256), FrameSchema(slot_count=2),
                                 occupied_slots=tuple(i for i, s in enumerate(case.lane_states) if s == 'busy'),
                                 quarantined_slots=tuple(i for i, s in enumerate(case.lane_states) if s == 'quarantined'),
                                 destinations=destinations, faults=faults)
    topology = PacketTopology(('ingress', 'lane0', 'lane1', 'egress'),
        tuple(PacketEndpoint(f'{i:02b}', 'egress') for i in range(4)),
        (Route('ingress', '', ('lane0', 'lane1'), route_id='ingress-route'),
         Route('lane0', '', ('egress',), route_id='lane0-route'),
         Route('lane1', '', ('egress',), route_id='lane1-route')))
    packet_faults = [PacketFault(0, 'ingress', f'lane{i}', 'link_down', 1024, source_state=state)
                     for i, state in enumerate(case.lane_states) if state != 'idle']
    if case.fault.startswith('packet_'):
        kind = case.fault.removeprefix('packet_')
        lanes = (0, 1) if kind == 'link_down' else (0,)
        packet_faults += [PacketFault(0, 'ingress', f'lane{i}', kind, 1024) for i in lanes]
    packet = PacketInput((PacketRequest(case.case_id, 'ingress', f'{case.destination:02b}', case.payload,
                                       case.decoder_output_sha256),), topology, QueuePolicy(capacity=16),
                         FaultPlan(tuple(packet_faults)), fragment_bytes=256)
    return {'ess': ess.as_dict(), 'digital': digital.as_dict(), 'packet': packet.as_dict()}


def create_switch_migration_contract(corpus):
    if type(corpus) is not CompatibilityCorpus:
        raise ValueError('INVALID_INPUT')
    corpus = CompatibilityCorpus.from_dict(corpus.as_dict())
    older = equivalence_adapter_manifest()
    return _seal('switch-migration-contract', corpus=corpus.as_dict(), corpus_sha256=corpus.as_dict()['sha256'],
        architectures=list(ARCHITECTURES), historical_adapter_manifest=older,
        historical_adapter_manifest_sha256=older['sha256'],
        destination_map=[{'semantic': _destination(i), 'ordinal': i, 'packet_address': f'{i:02b}'} for i in range(4)],
        availability_map={'strowger': 'selector_trunk_state', 'panel': 'bank_or_path_fault',
                          'crossbar': 'interstage_link_state', 'ess': 'native_interstage_link_state',
                          'digital': 'initial_output_slot_state', 'packet': 'declared_first_hop_link_down_with_source_state'},
        policy='first_available_declared_lane;no_undeclared_fallback;independent_cases',
        initial_owned_reservations=[], logical_start_tick=0,
        native_inputs=[{'case_sha256': case.as_dict()['sha256'], **_native_inputs(case)} for case in corpus.cases],
        claim_boundary=dict(CLAIM_BOUNDARY))


def validate_migration_contract(value, *, expected_corpus_sha256=None, expected_migration_sha256=None):
    try:
        _bounded(value)
        corpus = CompatibilityCorpus.from_dict(value['corpus'])
        rebuilt = create_switch_migration_contract(corpus)
        _exact(value, rebuilt)
        pins = _pins({'corpus': (expected_corpus_sha256, rebuilt['corpus_sha256']),
                      'migration': (expected_migration_sha256, rebuilt['sha256'])})
        return _seal('migration-validation', switch_migration_receipt_hash=rebuilt['sha256'],
                     all_passed=True, external_bindings_verified=pins)
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError('INVALID_INPUT') from exc


def _projection(case, architecture, accepted, reason, lane, destination, payload_hash, receipt_hash):
    affected = case.fault != 'none' and case.fault.startswith(architecture + '_')
    blocked = (not accepted and not affected and reason == CAPACITY_REASONS[architecture]
               and all(state != 'idle' for state in case.lane_states))
    status = 'route_available' if accepted else 'capacity_blocked' if blocked else architecture + '.' + reason
    return {'architecture': architecture, 'requested_destination': _destination(case.destination),
            'routing_outcome': status, 'selected_lane': lane if accepted else None,
            'reached_destination': destination if accepted else None,
            'failure_class': None if accepted else 'capacity' if blocked else architecture + '.' + reason,
            'native_outcome': 'accepted' if accepted else 'rejected', 'native_reason': reason,
            'source_receipt_sha256': receipt_hash, 'native_payload_sha256': payload_hash,
            'payload_capability': 'none_case_envelope_only' if architecture == 'strowger' else 'native',
            'input_payload_sha256': case.as_dict()['payload_sha256']}


def _new_evidence(case, inputs):
    ess_input = LifecycleInput.from_dict(inputs['ess'])
    ess = execute_call_processing(ess_input)
    validate_call_processing_receipt(ess, expected_input_sha256=inputs['ess']['sha256'])
    digital = execute_digital_switch(DigitalSwitchInput.from_dict(inputs['digital']))
    validate_digital_receipt(digital, expected_input_sha256=inputs['digital']['sha256'])
    packet = execute_packet_fabric(PacketInput.from_dict(inputs['packet']))
    validate_packet_receipt(packet, expected_input_sha256=inputs['packet']['sha256'])
    attempt = ess['calls'][0]['attempts'][0]
    accepted = ess['calls'][0]['outcome'] == 'released'
    plan = attempt.get('path_plan')
    lane = int(plan['interstage_link_ids'][0].rsplit('-', 1)[1]) if accepted else None
    achieved = _destination(plan['coordinates'][-1]['coordinate']['vertical_ordinal']) if accepted else None
    ep = _projection(case, 'ess', accepted, attempt['native_reason'], lane, achieved,
                     attempt['payload_sha256'], ess['sha256'])
    ep['native_outcome'] = ess['calls'][0]['outcome']
    dp = _projection(case, 'digital', digital['accepted'], digital['reason'], digital['selected_slot'],
                     digital['destination'], digital['reassembled_payload_sha256'], digital['sha256'])
    dp['native_outcome'] = digital['outcome']
    result = packet['results'][0]
    accepted = result['outcome'] == 'accepted'
    path = result['packet_paths'][0]['nodes']
    lane = int(path[1][-1]) if accepted and len(path) >= 3 else None
    achieved = _destination(int(result['destination'], 2)) if accepted and result['destination_node'] == 'egress' else None
    reason = result['drops'][0]['reason'] if result['drops'] else result['reason']
    pp = _projection(case, 'packet', accepted, reason, lane, achieved, result['accepted_payload_sha256'], packet['sha256'])
    return {'ess': ess, 'digital': digital, 'packet': packet}, [ep, dp, pp]


def _reference(case):
    available = next((i for i, state in enumerate(case.lane_states) if state == 'idle'), None)
    affected = case.fault.split('_', 1)[0] if case.fault != 'none' else None
    result = {}
    for architecture in ARCHITECTURES:
        faulted = architecture == affected
        accepted = available is not None and not faulted
        reason = FAULT_REASONS[case.fault] if faulted else SUCCESS_REASONS[architecture] if accepted else CAPACITY_REASONS[architecture]
        failure = architecture + '.' + reason if faulted else None if accepted else 'capacity'
        result[architecture] = {'accepted': accepted,
            'selected_lane': available if not faulted else None,
            'reached_destination': _destination(case.destination) if accepted else None,
            'native_reason': reason, 'failure_class': failure}
    return result


def execute_cross_era_replay(corpus):
    """Reconstruct translated sources and invoke all six actual native engines."""
    migration = create_switch_migration_contract(corpus)
    corpus = CompatibilityCorpus.from_dict(migration['corpus'])
    legacy = run_equivalence_battery(EquivalenceCorpus(tuple(_legacy_case(case) for case in corpus.cases)))
    rows = []
    for case, old, native_inputs in zip(corpus.cases, legacy['results'], migration['native_inputs']):
        new_evidence, projections = _new_evidence(case, native_inputs)
        older = []
        for p in old['projections']:
            q = dict(p)
            q['failure_class'] = None if p['routing_outcome'] == 'route_available' else 'capacity' if p['routing_outcome'] == 'capacity_blocked' else p['routing_outcome']
            q['input_payload_sha256'] = case.as_dict()['payload_sha256']
            q['payload_capability'] = 'none_case_envelope_only' if p['architecture'] == 'strowger' else 'native'
            older.append(q)
        projections = older + projections
        reference = _reference(case)
        reference_checks = {p['architecture']: reference[p['architecture']] == {
            'accepted': p['routing_outcome'] == 'route_available', 'selected_lane': p['selected_lane'],
            'reached_destination': p['reached_destination'], 'native_reason': p['native_reason'],
            'failure_class': p['failure_class']} for p in projections}
        payload_checks = {p['architecture']: p['native_payload_sha256'] == case.as_dict()['payload_sha256']
                          for p in projections if p['architecture'] != 'strowger' and p['routing_outcome'] == 'route_available'}
        evidence = {**old['evidence'], **new_evidence}
        rows.append(_seal('replay-case', case=case.as_dict(), evidence=evidence, projections=projections,
            reference=reference, reference_checks=reference_checks, payload_checks=payload_checks,
            native_input_sha256={key: native_inputs[key]['sha256'] for key in ('ess', 'digital', 'packet')},
            historical_case_result_sha256=old['sha256'], native_sources_validated=True,
            passed=all(reference_checks.values()) and all(payload_checks.values()) and old['passed']))
    return _seal('replay-receipt', migration_contract=migration, switch_migration_receipt_hash=migration['sha256'],
        corpus_sha256=corpus.as_dict()['sha256'], historical_matrix_sha256=legacy['sha256'], results=rows,
        case_count=len(rows), native_execution_count=6 * len(rows), all_passed=all(row['passed'] for row in rows),
        default_corpus_used=corpus.as_dict()['sha256'] == default_compatibility_corpus().as_dict()['sha256'],
        claim_boundary=dict(CLAIM_BOUNDARY))


def validate_cross_era_replay(receipt, *, expected_corpus_sha256=None, expected_migration_sha256=None, expected_receipt_sha256=None):
    try:
        _bounded(receipt)
        corpus = CompatibilityCorpus.from_dict(receipt['migration_contract']['corpus'])
        replay = execute_cross_era_replay(corpus)
        _exact(receipt, replay)
        pins = _pins({'corpus': (expected_corpus_sha256, replay['corpus_sha256']),
                      'migration': (expected_migration_sha256, replay['switch_migration_receipt_hash']),
                      'receipt': (expected_receipt_sha256, replay['sha256'])})
        return _seal('replay-validation', cross_era_replay_receipt_hash=replay['sha256'],
                     case_count=replay['case_count'], native_execution_count=replay['native_execution_count'],
                     all_passed=replay['all_passed'], default_corpus_used=replay['default_corpus_used'],
                     external_bindings_verified=pins)
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError('INVALID_INPUT') from exc


def create_cross_era_equivalence_receipt(replay):
    validation = validate_cross_era_replay(replay)
    rows = []
    for row in replay['results']:
        pairs = []
        for left, right in combinations(row['projections'], 2):
            checks = {name: left[name] == right[name] for name in COMPARISON_FIELDS}
            payload_supported = left['architecture'] != 'strowger' and right['architecture'] != 'strowger'
            both_accepted = left['routing_outcome'] == right['routing_outcome'] == 'route_available'
            payload = (left['native_payload_sha256'] == right['native_payload_sha256'] == row['case']['payload_sha256']) if payload_supported and both_accepted else None
            pairs.append({'left': left['architecture'], 'right': right['architecture'], 'checks': checks,
                          'route_equivalent': all(checks.values()), 'native_payload_equivalent': payload,
                          'payload_comparison_scope': 'native_verified' if payload is not None else 'unsupported_strowger' if not payload_supported else 'no_common_successful_delivery'})
        equivalent = all(pair['route_equivalent'] for pair in pairs)
        expected = row['case']['fault'] == 'none'
        rows.append(_seal('equivalence-case', case_sha256=row['case']['sha256'], source_replay_case_sha256=row['sha256'],
                          pairwise=pairs, route_equivalent=equivalent, expected_equivalent=expected,
                          passed=row['passed'] and equivalent == expected))
    return _seal('equivalence-receipt', source_replay=replay, replay_validation=validation, comparisons=rows,
                 case_count=len(rows), equivalent_case_count=sum(row['route_equivalent'] for row in rows),
                 expected_difference_count=sum(not row['expected_equivalent'] for row in rows),
                 all_cases_equivalent=all(row['route_equivalent'] for row in rows),
                 all_passed=all(row['passed'] for row in rows), claim_boundary=dict(CLAIM_BOUNDARY))


def validate_cross_era_equivalence(receipt, *, expected_corpus_sha256=None, expected_receipt_sha256=None):
    try:
        _bounded(receipt)
        rebuilt = create_cross_era_equivalence_receipt(receipt['source_replay'])
        _exact(receipt, rebuilt)
        pins = _pins({'corpus': (expected_corpus_sha256, rebuilt['source_replay']['corpus_sha256']),
                      'receipt': (expected_receipt_sha256, rebuilt['sha256'])})
        return _seal('equivalence-validation', cross_era_switch_equivalence_receipt_hash=rebuilt['sha256'],
                     all_passed=rebuilt['all_passed'], case_count=rebuilt['case_count'],
                     equivalent_case_count=rebuilt['equivalent_case_count'], expected_difference_count=rebuilt['expected_difference_count'],
                     external_bindings_verified=pins)
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError('INVALID_INPUT') from exc


@dataclass(frozen=True)
class ArchitectureSelectionPolicy:
    required_capabilities: tuple[str, ...] = ()
    preference_order: tuple[str, ...] = ARCHITECTURES

    def __post_init__(self):
        requirements, order = self.required_capabilities, self.preference_order
        if type(requirements) not in (list, tuple) or len(requirements) > len(CAPABILITIES['strowger']):
            raise ValueError('INVALID_INPUT')
        if any(type(x) is not str or x not in CAPABILITIES['strowger'] for x in requirements):
            raise ValueError('INVALID_INPUT')
        if tuple(requirements) != tuple(sorted(set(requirements))):
            raise ValueError('INVALID_INPUT')
        if type(order) not in (list, tuple) or len(order) != len(ARCHITECTURES) or any(type(x) is not str for x in order) or set(order) != set(ARCHITECTURES):
            raise ValueError('INVALID_INPUT')
        object.__setattr__(self, 'required_capabilities', tuple(requirements))
        object.__setattr__(self, 'preference_order', tuple(order))

    def as_dict(self):
        return _seal('architecture-selection-policy', required_capabilities=list(self.required_capabilities),
                     preference_order=list(self.preference_order), ranking='filter_requirements_then_explicit_preference',
                     capability_scope='implemented_model_apis_not_hardware_or_universal_quality')

    @classmethod
    def from_dict(cls, value):
        try:
            result = cls(value['required_capabilities'], value['preference_order'])
            _exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('INVALID_INPUT') from exc


def select_architecture(policy):
    if type(policy) is not ArchitectureSelectionPolicy:
        raise ValueError('INVALID_INPUT')
    policy = ArchitectureSelectionPolicy.from_dict(policy.as_dict())
    rows = [{'architecture': architecture, 'capabilities': dict(CAPABILITIES[architecture]),
             'unmet_requirements': [key for key in policy.required_capabilities if not CAPABILITIES[architecture][key]]}
            for architecture in policy.preference_order]
    eligible = [row['architecture'] for row in rows if not row['unmet_requirements']]
    return _seal('architecture-selection-receipt', policy=policy.as_dict(), candidates=rows, ranking=eligible,
                 selected=eligible[0] if eligible else None, outcome='selected' if eligible else 'no_eligible_architecture',
                 hardware_performance_considered=False, universal_superiority_claimed=False)


def validate_architecture_selection(receipt, *, expected_policy_sha256=None):
    try:
        _bounded(receipt)
        rebuilt = select_architecture(ArchitectureSelectionPolicy.from_dict(receipt['policy']))
        _exact(receipt, rebuilt)
        pins = _pins({'policy': (expected_policy_sha256, rebuilt['policy']['sha256'])})
        return _seal('architecture-selection-validation', source_receipt_sha256=rebuilt['sha256'], all_passed=True,
                     external_bindings_verified=pins)
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError('INVALID_INPUT') from exc


def _event_count(architecture, receipt):
    if architecture == 'crossbar':
        return len(receipt['source_receipt']['events'])
    if architecture == 'digital':
        return len(receipt['allocation_receipt']['events'])
    if architecture == 'packet':
        return len(receipt['hop_receipt_chain']['events'])
    return len(receipt['events'])


def create_switching_evolution_report(equivalence, policy=None):
    validation = validate_cross_era_equivalence(equivalence)
    policy = ArchitectureSelectionPolicy() if policy is None else policy
    selection = select_architecture(policy)
    rows = []
    complexities = {'strowger': 'sequential_mixed_radix_pulses', 'panel': 'compiled_translation_then_motor_path',
                    'crossbar': 'bounded_layered_path_search', 'ess': 'logical_scheduler_and_native_reservations',
                    'digital': 'fixed_width_frames_and_slot_memory', 'packet': 'prefix_routes_and_bounded_hop_queues'}
    capacity = {'strowger': 'initial_trunk_state', 'panel': 'declared_bank_and_path_faults',
                'crossbar': 'independent_snapshot_planning_here;owned_reservations_in_native_contention_api',
                'ess': 'owned_whole_path_reservation_then_release', 'digital': 'explicit_slot_allocation_and_release',
                'packet': 'per_node_bounded_queues_and_declared_link_availability'}
    for architecture in ARCHITECTURES:
        native = [row['evidence'][architecture] for row in equivalence['source_replay']['results']]
        rows.append({'architecture': architecture, 'complexity_model': complexities[architecture],
                     'capacity_model': capacity[architecture], 'native_api_capabilities': dict(CAPABILITIES[architecture]),
                     'event_count': sum(_event_count(architecture, source) for source in native),
                     'event_count_scope': 'slot_allocation_events_only_frames_counted_separately' if architecture == 'digital' else 'top_level_native_model_events_excludes_nested_validation',
                     'frame_count': sum(len(source['frames']) for source in native) if architecture == 'digital' else None,
                     'canonical_receipt_bytes': sum(len(canonical_bytes(source)) for source in native),
                     'receipt_count': len(native),
                     'fault_controls': [case.fault for case in CompatibilityCorpus.from_dict(
                         equivalence['source_replay']['migration_contract']['corpus']).cases if case.fault.startswith(architecture + '_')]})
    return _seal('switching-evolution-report', source_equivalence=equivalence, validation=validation,
                 architecture_selection=selection, observations=rows, metrics_are_correctness_proofs=False,
                 metrics_are_hardware_benchmarks=False, all_passed=validation['all_passed'],
                 claim_boundary=dict(CLAIM_BOUNDARY))


def validate_switching_evolution_report(receipt, *, expected_equivalence_sha256=None):
    try:
        _bounded(receipt)
        policy = ArchitectureSelectionPolicy.from_dict(receipt['architecture_selection']['policy'])
        rebuilt = create_switching_evolution_report(receipt['source_equivalence'], policy)
        _exact(receipt, rebuilt)
        pins = _pins({'equivalence': (expected_equivalence_sha256, rebuilt['source_equivalence']['sha256'])})
        return _seal('switching-evolution-validation', report_sha256=rebuilt['sha256'],
                     all_passed=rebuilt['all_passed'], external_bindings_verified=pins)
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError('INVALID_INPUT') from exc
