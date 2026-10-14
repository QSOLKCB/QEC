# SPDX-License-Identifier: MPL-2.0
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qec.routing.cross_era import (
    ARCHITECTURES, CAPABILITIES, FAULTS, ArchitectureSelectionPolicy, CompatibilityCase,
    CompatibilityCorpus, create_cross_era_equivalence_receipt, create_switch_migration_contract,
    create_switching_evolution_report, default_compatibility_corpus, execute_cross_era_replay,
    select_architecture, validate_architecture_selection, validate_cross_era_equivalence,
    validate_cross_era_replay, validate_migration_contract, validate_switching_evolution_report,
)
from qec.sonify.canonical import canonical_bytes, canonical_json, canonical_sha256


def corpus():
    return CompatibilityCorpus((CompatibilityCase('a',0,('idle','idle')),
        CompatibilityCase('b',3,('busy','idle')), CompatibilityCase('c',2,('quarantined','busy'))))


@pytest.fixture(scope='module')
def replay():
    return execute_cross_era_replay(corpus())


@pytest.fixture(scope='module')
def equivalence(replay):
    return create_cross_era_equivalence_receipt(replay)


def resign(value):
    if type(value) is list:
        for child in value: resign(child)
    elif type(value) is dict:
        for child in value.values(): resign(child)
        for field in ('sha256', 'event_sha256'):
            if field in value:
                value[field] = canonical_sha256({k:v for k,v in value.items() if k != field})
    return value


def test_six_native_models_preserve_common_case_semantics(replay):
    assert replay['all_passed']
    assert replay['native_execution_count'] == 18
    assert not replay['default_corpus_used']
    expected = [(0, 'correction/destination-0'), (1, 'correction/destination-3'), (None,None)]
    for row, (lane,dest) in zip(replay['results'],expected):
        assert list(row['evidence']) == list(ARCHITECTURES)
        for p in row['projections']:
            assert p['selected_lane'] == lane
            assert p['reached_destination'] == dest
            assert p['routing_outcome'] == ('capacity_blocked' if lane is None else 'route_available')
            assert p['failure_class'] == ('capacity' if lane is None else None)
            assert p['source_receipt_sha256'] == row['evidence'][p['architecture']]['sha256']
        assert all(row['reference_checks'].values()) and all(row['payload_checks'].values())
    assert validate_cross_era_replay(replay, expected_corpus_sha256=corpus().as_dict()['sha256'])['all_passed']


def test_capability_gaps_and_lifecycles_remain_explicit(equivalence):
    assert equivalence['all_passed'] and equivalence['equivalent_case_count'] == 3
    assert equivalence['claim_boundary']['all_six_native_payload_equivalence'] is False
    assert equivalence['claim_boundary']['native_lifecycle_equivalence'] is False
    first = equivalence['comparisons'][0]
    assert len(first['pairwise']) == 15
    assert sum(pair['native_payload_equivalent'] is None for pair in first['pairwise']) == 5
    assert sum(pair['native_payload_equivalent'] is True for pair in first['pairwise']) == 10
    for pair in first['pairwise']:
        if pair['left'] == 'strowger':
            assert pair['payload_comparison_scope'] == 'unsupported_strowger'
    native = equivalence['source_replay']['results'][0]['evidence']
    assert native['strowger']['qec_version'] == '170.3.0'
    assert native['panel']['contract_version'] == '171.2'
    assert native['crossbar']['contract_version'] == '172.4'
    assert native['ess']['calls'][0]['outcome'] == 'released'
    assert native['digital']['initial_state_restored']
    assert native['packet']['results'][0]['accepted_payload_sha256'] == corpus().cases[0].as_dict()['payload_sha256']


def test_complete_grid_and_architecture_fault_controls():
    frozen=json.loads((Path(__file__).with_name('fixtures')/'cross-era-hashes-v1.json').read_text())
    source = default_compatibility_corpus()
    assert len(source.cases) == 51
    receipt = execute_cross_era_replay(source)
    assert receipt['all_passed'] and receipt['default_corpus_used']
    assert receipt['sha256']==frozen['default_replay_sha256']
    assert source.as_dict()['sha256']==frozen['default_corpus_sha256']
    assert len({(case.destination,case.lane_states) for case in source.cases if case.fault == 'none'}) == 36
    for row in receipt['results']:
        fault = row['case']['fault']
        if fault == 'none': continue
        affected = fault.split('_',1)[0]
        rejected = [p['architecture'] for p in row['projections'] if p['routing_outcome'] != 'route_available']
        assert rejected == [affected]
        assert row['projections'][ARCHITECTURES.index(affected)]['failure_class'].startswith(affected + '.')
    compared = create_cross_era_equivalence_receipt(receipt)
    assert compared['all_passed']
    assert compared['sha256']==frozen['default_equivalence_sha256']
    assert compared['equivalent_case_count'] == 36
    assert compared['expected_difference_count'] == 15
    assert not compared['all_cases_equivalent']


def test_migration_has_complete_translated_inputs(replay):
    m = replay['migration_contract']
    assert m['corpus_sha256'] == corpus().as_dict()['sha256']
    assert m['historical_adapter_manifest']['sha256'] == m['historical_adapter_manifest_sha256']
    for case, native in zip(corpus().cases,m['native_inputs']):
        assert native['case_sha256'] == case.as_dict()['sha256']
        assert all(len(native[arch]['sha256']) == 64 for arch in ('ess','digital','packet'))
    blocked = m['native_inputs'][2]
    assert blocked['digital']['occupied_slots'] == [1]
    assert blocked['digital']['quarantined_slots'] == [0]
    assert [f['source_state'] for f in blocked['packet']['fault_plan']['faults']] == ['quarantined','busy']
    assert validate_migration_contract(m,expected_migration_sha256=m['sha256'])['all_passed']


@pytest.mark.parametrize('mutation',[
    lambda r:r['results'].pop(),
    lambda r:r['results'].reverse(),
    lambda r:r.__setitem__('all_passed',False),
    lambda r:r.__setitem__('native_execution_count',True),
    lambda r:r.__setitem__('default_corpus_used',True),
    lambda r:r['claim_boundary'].__setitem__('all_six_native_payload_equivalence',True),
    lambda r:r['results'][0]['projections'][0].__setitem__('native_payload_sha256','a'*64),
    lambda r:r['results'][0]['projections'][4].__setitem__('selected_lane',1),
    lambda r:r['results'][0]['evidence']['ess']['events'].pop(),
    lambda r:r['results'][0]['evidence']['digital'].__setitem__('accepted',False),
    lambda r:r['results'][0]['evidence']['packet']['hop_receipt_chain']['events'].pop(),
    lambda r:r['migration_contract']['destination_map'][0].__setitem__('packet_address','11'),
    lambda r:r['migration_contract']['native_inputs'][0]['packet']['queue_policy'].__setitem__('capacity',1),
    lambda r:r['results'][0]['reference']['strowger'].__setitem__('accepted',False),
])
def test_resigned_replay_tampering_rejected(replay,mutation):
    value=deepcopy(replay);mutation(value)
    with pytest.raises(ValueError,match='INVALID_INPUT'): validate_cross_era_replay(resign(value))


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_valid_unrelated_native_evidence_rejected(replay,architecture):
    value=deepcopy(replay)
    value['results'][0]['evidence'][architecture]=deepcopy(replay['results'][1]['evidence'][architecture])
    with pytest.raises(ValueError,match='INVALID_INPUT'): validate_cross_era_replay(resign(value))


@pytest.mark.parametrize('pin', ['expected_corpus_sha256','expected_migration_sha256','expected_receipt_sha256'])
def test_trusted_replay_pins(replay,pin):
    with pytest.raises(ValueError): validate_cross_era_replay(replay,**{pin:'0'*64})


def test_rehashed_equivalence_and_report_rejected(equivalence):
    value=deepcopy(equivalence)
    value['comparisons'][0]['pairwise'][0]['native_payload_equivalent']=True
    with pytest.raises(ValueError,match='INVALID_INPUT'): validate_cross_era_equivalence(resign(value))
    report=create_switching_evolution_report(equivalence)
    assert report['all_passed']
    assert not report['metrics_are_correctness_proofs'] and not report['metrics_are_hardware_benchmarks']
    for row in report['observations']:
        assert row['receipt_count']==3 and row['canonical_receipt_bytes']>0
    assert validate_switching_evolution_report(report)['all_passed']
    report['observations'][0]['canonical_receipt_bytes']+=1
    with pytest.raises(ValueError,match='INVALID_INPUT'): validate_switching_evolution_report(resign(report))


@pytest.mark.parametrize('required,selected', [
    ((), 'strowger'), (('native_payload',),'panel'), (('source_bound_programme',),'ess'),
    (('bounded_queues',),'packet'), (('fixed_frames',),'digital'),
    (('bounded_queues','fixed_frames'),None),
])
def test_selection_filters_declared_requirements(required,selected):
    policy=ArchitectureSelectionPolicy(required)
    receipt=select_architecture(policy)
    assert receipt['selected']==selected
    assert validate_architecture_selection(receipt,expected_policy_sha256=policy.as_dict()['sha256'])['all_passed']
    assert not receipt['universal_superiority_claimed']


def test_selection_preference_is_explicit_and_immutable():
    receipt=select_architecture(ArchitectureSelectionPolicy((),tuple(reversed(ARCHITECTURES))))
    assert receipt['selected']=='packet'
    with pytest.raises(TypeError): CAPABILITIES['strowger']['native_payload']=True
    receipt['ranking'].reverse()
    with pytest.raises(ValueError): validate_architecture_selection(resign(receipt))


@pytest.mark.parametrize('kwargs',[
    {'case_id':''}, {'case_id':'a/b'}, {'case_id':'x'*65}, {'destination':True}, {'destination':4},
    {'destination':1.0}, {'lane_states':('idle','bogus')}, {'lane_states':('idle',)},
    {'payload':b'x'*257}, {'payload':'x'}, {'decoder_output_sha256':'A'*64}, {'fault':'force_accept'},
    {'fault':'digital_bit_flip','lane_states':('busy','idle')}, {'fault':'digital_bit_flip','payload':b''},
])
def test_case_bounds(kwargs):
    fields={'case_id':'a','destination':0,'lane_states':('idle','idle')};fields.update(kwargs)
    with pytest.raises((ValueError,TypeError)): CompatibilityCase(**fields)


def test_corpus_bounds_and_detached_serialization():
    case=CompatibilityCase('a',0,('idle','idle'))
    for values in ((),(case,case),[CompatibilityCase('b',0,('idle','idle')),case]):
        with pytest.raises(ValueError): CompatibilityCorpus(values)
    value=corpus(); serialized=value.as_dict();serialized['cases'].clear()
    assert len(value.cases)==3
    with pytest.raises(FrozenInstanceError): value.cases=()
    with pytest.raises(ValueError): CompatibilityCorpus(tuple(CompatibilityCase(f'c{i:02d}',0,('idle','idle'),b'x'*256) for i in range(33)))


def test_empty_payload_is_supported_by_all_native_models():
    source=CompatibilityCorpus((CompatibilityCase('empty',0,('idle','idle'),b''),))
    receipt=execute_cross_era_replay(source)
    assert receipt['all_passed']
    for p in receipt['results'][0]['projections'][1:]:
        assert p['native_payload_sha256']==hashlib.sha256(b'').hexdigest()


def test_hashseed_determinism():
    code="from qec.routing.cross_era import *; print(execute_cross_era_replay(CompatibilityCorpus((CompatibilityCase('hashseed',1,('busy','idle')),)))['sha256'])"
    results=[subprocess.check_output([sys.executable,'-c',code],env={**os.environ,'PYTHONHASHSEED':seed},text=True).strip() for seed in ('1','7','91')]
    assert len(set(results))==1


def test_cli_roundtrip_and_invalid_json(tmp_path,capsys):
    from qec.routing.cross_era_cli import main
    path=tmp_path/'corpus.json';path.write_text(canonical_json(CompatibilityCorpus((corpus().cases[0],)).as_dict()))
    assert main(['run','--input',str(path),'--output-dir',str(tmp_path)])==0
    report=tmp_path/'switching_evolution_report.json'
    assert main(['validate','--kind','report','--receipt',str(report)])==0
    assert main(['validate','--kind','report','--receipt',str(report),'--expected-sha256','0'*64])==1
    assert main(['select','--required','bounded_queues','--output-dir',str(tmp_path)])==0
    assert json.loads((tmp_path/'architecture_selection_receipt.json').read_text())['selected']=='packet'
    path.write_text('{"cases":[],"cases":[]}')
    assert main(['run','--input',str(path),'--output-dir',str(tmp_path)])==1
    path.write_text('{"x":NaN}')
    assert main(['run','--input',str(path),'--output-dir',str(tmp_path)])==1


@pytest.mark.parametrize('fault,states',[('none',('busy','busy')),('digital_bit_flip',('idle','idle'))])
def test_reference_rejects_wrong_failure_class_without_hiding_evidence(monkeypatch,fault,states):
    # Simulate an adapter/projection implementation defect after native replay.
    # The independent semantic oracle must still reject this reason substitution.
    import qec.routing.cross_era as ce
    original=ce._new_evidence
    def unexpected(case,inputs):
        evidence, projections=original(case,inputs)
        evidence['digital']['reason']='unexpected_verification_fault'
        projections[1]=ce._projection(case,'digital',False,'unexpected_verification_fault',None,None,None,evidence['digital']['sha256'])
        return evidence,projections
    monkeypatch.setattr(ce,'_new_evidence',unexpected)
    receipt=ce.execute_cross_era_replay(ce.CompatibilityCorpus((ce.CompatibilityCase('unexpected',0,states,fault=fault),)))
    row=receipt['results'][0]
    assert not receipt['all_passed'] and not row['reference_checks']['digital']
    assert row['evidence']['digital']['reason']=='unexpected_verification_fault'
    assert row['projections'][4]['failure_class']=='digital.unexpected_verification_fault'
    assert row['projections'][4]['routing_outcome']!='capacity_blocked'


def test_frozen_corpus_and_all_small_artifact_identities(replay,equivalence):
    directory=Path(__file__).with_name('fixtures')
    expected=json.loads((directory/'cross-era-hashes-v1.json').read_text())
    source=CompatibilityCorpus.from_dict(json.loads((directory/'compatibility-corpus-v1.json').read_text()))
    assert source==corpus()
    assert source.as_dict()['sha256']==expected['corpus_sha256']
    assert replay['switch_migration_receipt_hash']==expected['migration_sha256']
    assert replay['sha256']==expected['replay_sha256']
    assert equivalence['sha256']==expected['equivalence_sha256']
    assert create_switching_evolution_report(equivalence)['sha256']==expected['report_sha256']


def test_downstream_artifacts_detach_all_nested_source_snapshots(replay,equivalence):
    before_replay=canonical_json(replay)
    before_equivalence=canonical_json(equivalence)
    comparison=create_cross_era_equivalence_receipt(replay)
    comparison['source_replay']['results'][0]['evidence']['ess']['input'].clear()
    report=create_switching_evolution_report(equivalence)
    report['source_equivalence']['source_replay']['migration_contract'].clear()
    assert canonical_json(replay)==before_replay
    assert canonical_json(equivalence)==before_equivalence
