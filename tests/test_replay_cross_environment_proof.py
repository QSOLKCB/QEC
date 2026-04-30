import json
import pytest
from dataclasses import FrozenInstanceError
from qec.analysis.replay_cross_environment_proof import *

h=lambda s: __import__('hashlib').sha256(s.encode()).hexdigest()

def mk_ext(**kw):
    d=dict(evidence_id='e1',environment_hash=h('env1'),raw_bytes_hash=h('raw'),extraction_config_hash=h('ec'),schema_hash=h('s'),query_fields_hash=h('q'),locale_hash=h('l'),backend_config_hash=h('b'),canonicalization_rules_hash=h('c'),numeric_profile_hash=h('n'),extraction_hash=h('x'),canonical_hash=h('can'))
    d.update(kw); d['evidence_hash']=h(json.dumps({k:v for k,v in d.items() if k!='evidence_hash'},sort_keys=True,separators=(',',':'),ensure_ascii=True)); return ExtractionReplayEvidence(**d)

def mk_res(**kw):
    d=dict(evidence_id='r1',environment_hash=h('env1'),canonical_hash=h('can'),res_hash=h('res'),rag_hash=h('rag'),semantic_field_hash=h('sf'),res_rag_mapping_hash=h('map'),governance_context_hash=h('gc'),resonance_classifier_hash=h('rc'),tolerance_hash=h('tol'),resonance_receipt_hash=h('rr'),aggregate_resonance_class='ALIGNED')
    d.update(kw); d['evidence_hash']=h(json.dumps({k:v for k,v in d.items() if k!='evidence_hash'},sort_keys=True,separators=(',',':'),ensure_ascii=True)); return ResonanceReplayEvidence(**d)

def mk_rw(**kw):
    d=dict(evidence_id='w1',environment_hash=h('env1'),canonical_hash=h('can'),semantic_field_hash=h('sf'),resonance_receipt_hash=h('rr'),validation_hash=h('val'),governance_hash=h('gov'),local_proof_hash=h('lp'),distributed_convergence_hash=h('dc'),final_proof_hash=h('fp'),governance_decision='ACCEPT')
    d.update(kw); d['evidence_hash']=h(json.dumps({k:v for k,v in d.items() if k!='evidence_hash'},sort_keys=True,separators=(',',':'),ensure_ascii=True)); return RealWorldReplayEvidence(**d)

def test_extraction_validated_and_cross_environment():
    a=mk_ext(); b=mk_ext(environment_hash=h('env2'))
    r=run_extraction_replay_validation(a,b)
    assert r.status=='EXTRACTION_REPLAY_VALIDATED' and r.result_count==0

def test_extraction_divergence_types():
    a=mk_ext(); b=mk_ext(environment_hash=h('env2'),numeric_profile_hash=h('n2'),extraction_hash=h('x2'),canonical_hash=h('can2'))
    r=run_extraction_replay_validation(a,b)
    assert {x.divergence_type for x in r.results}=={'FLOATING_POINT_DRIFT','BACKEND_INCONSISTENCY','CANONICALIZATION_DRIFT','ENVIRONMENT_DIVERGENCE'}

def test_resonance_validated_and_divergence():
    a=mk_res(); b=mk_res()
    assert run_resonance_replay_validation(a,b).status=='RESONANCE_REPLAY_VALIDATED'
    c=mk_res(semantic_field_hash=h('sf2'),res_hash=h('res2'),rag_hash=h('rag2'),aggregate_resonance_class='DIVERGENT',resonance_receipt_hash=h('rr2'))
    r=run_resonance_replay_validation(a,c)
    assert any(x.divergence_type=='SEMANTIC_FIELD_DRIFT' for x in r.results)
    assert any(x.divergence_type=='RES_STATE_DRIFT' for x in r.results)
    assert any(x.divergence_type=='RAG_STATE_DRIFT' for x in r.results)
    assert sum(1 for x in r.results if x.divergence_type=='RESONANCE_CLASSIFICATION_DRIFT')==2

def test_real_world_validated_and_divergence():
    e=run_extraction_replay_validation(mk_ext(),mk_ext())
    s=run_resonance_replay_validation(mk_res(),mk_res())
    a=mk_rw(); b=mk_rw()
    assert run_real_world_replay_proof(a,b,e,s).status=='REAL_WORLD_REPLAY_VALIDATED'
    c=mk_rw(validation_hash=h('v2'),governance_hash=h('g2'),governance_decision='REJECT',local_proof_hash=h('lp2'),distributed_convergence_hash=h('dc2'),final_proof_hash=h('fp2'))
    rr=run_real_world_replay_proof(a,c,e,s)
    assert rr.status=='REAL_WORLD_REPLAY_DIVERGENCE_DETECTED'

def test_invalid_comparison_and_message():
    with pytest.raises(ValueError,match='^INVALID_INPUT$'): run_extraction_replay_validation(mk_ext(),mk_ext(raw_bytes_hash=h('other')))
    with pytest.raises(ValueError,match='^INVALID_INPUT$'): run_resonance_replay_validation(mk_res(),mk_res(canonical_hash=h('other')))
    e=run_extraction_replay_validation(mk_ext(),mk_ext()); s=run_resonance_replay_validation(mk_res(),mk_res())
    with pytest.raises(ValueError,match='^INVALID_INPUT$'): run_real_world_replay_proof(mk_rw(),mk_rw(canonical_hash=h('other')),e,s)

def test_determinism_immutability_json():
    a=mk_ext(); b=mk_ext(environment_hash=h('env2'))
    r1=run_extraction_replay_validation(a,b); r2=run_extraction_replay_validation(a,b)
    assert r1.stable_hash==r2.stable_hash
    with pytest.raises(FrozenInstanceError): a.evidence_id='x'
    json.dumps(r1.to_dict())

def test_receipt_integrity_checks():
    a=mk_ext(); b=mk_ext(); ok=run_extraction_replay_validation(a,b)
    with pytest.raises(ValueError,match='^INVALID_INPUT$'):
        ExtractionReplayReceipt(ok.version,ok.baseline_evidence_hash,ok.observed_evidence_hash,ok.raw_bytes_hash,ok.results,1,ok.reject_count,ok.flag_count,ok.status,ok.stable_hash)
