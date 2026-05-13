from dataclasses import FrozenInstanceError, replace
import pytest

from qec.analysis.loop_termination_contract import build_loop_termination_contract, TERMINATION_POLICY_MAX_DEPTH_ONLY
from qec.analysis.recursive_proof_receipt import build_loop_iteration_record, build_recursive_proof_receipt
from qec.analysis.loop_termination_proof import build_loop_termination_proof, build_ouroboric_convergence_receipt
from qec.analysis.subsystem_loop_receipts import *

H1='1'*64;H2='2'*64;H3='3'*64

def _art(loop_label='ROUTER_FLOW', source='RouterArtifact', in_field='router_input_hash', out_field='router_output_hash'):
    c=build_loop_termination_contract(source_artifact_type=source,source_artifact_hash=H1,loop_label=loop_label,max_depth=3,input_receipt_hash_field=in_field,output_receipt_hash_field=out_field,termination_policy=TERMINATION_POLICY_MAX_DEPTH_ONLY)
    r1=build_loop_iteration_record(c,0,H1,H2,'ITERATION_CONTINUED')
    r2=build_loop_iteration_record(c,1,H2,H3,'ITERATION_CONTINUED')
    r3=build_loop_iteration_record(c,2,H3,H3,'ITERATION_MAX_DEPTH_REACHED')
    rr=build_recursive_proof_receipt(c,[r1,r2,r3])
    p=build_loop_termination_proof(c,rr)
    o=build_ouroboric_convergence_receipt(c,rr,p)
    return c,rr,p,o

def test_basic_and_hash_stability():
    c,rr,p,o=_art()
    a=build_router_loop_receipt(c,rr,p,o); b=build_router_loop_receipt(c,rr,p,o)
    assert a.router_loop_receipt_hash==b.router_loop_receipt_hash
    assert a.to_canonical_json()==b.to_canonical_json()
    assert a.to_canonical_bytes()==b.to_canonical_bytes()
    with pytest.raises(FrozenInstanceError): a.loop_label='X'
    assert validate_router_loop_receipt(a)

def test_empty_and_mismatch_and_complete_validators():
    c,rr,p,o=_art(loop_label='GENERIC_LOOP',source='Generic',in_field='input_hash',out_field='output_hash')
    r=build_router_loop_receipt(c,rr,p,o)
    assert r.subsystem_iteration_count==0 and r.subsystem_loop_stability_class=='SUBSYSTEM_LOOP_EMPTY'
    rc,rrc,pc,oc=_art(loop_label='READOUT_LOOP',source='ReadoutArtifact',in_field='input_hash',out_field='output_hash')
    with pytest.raises(ValueError, match='SUBSYSTEM_LOOP_MISMATCH'):
        build_markov_loop_stability_receipt(rc,rrc,pc,oc)
    assert validate_router_loop_receipt_with_artifacts(r,c,rr,p,o)

def test_invalid_cases():
    c,rr,p,o=_art()
    r=build_router_loop_receipt(c,rr,p,o)
    with pytest.raises(ValueError, match='INVALID_HASH_FORMAT'):
        validate_router_loop_receipt(replace(r, router_loop_receipt_hash=r.router_loop_receipt_hash.upper()))
    wrong='f'*64
    with pytest.raises(ValueError, match='HASH_MISMATCH'):
        validate_router_loop_receipt(replace(r, router_loop_receipt_hash=wrong))


def test_scope_scan():
    txt=open('src/qec/analysis/subsystem_loop_receipts.py',encoding='utf-8').read()
    for bad in ['RealityLoopProofReceipt','GlobalTruthReceipt','CrossArcIdentityLink','RealityLoopCompositionSpec','global_validation','global_truth','while True','recursive_execution','gameplay','render','step_world','execute_action','run_game','importlib','__import__(','subprocess','exec(','eval(','random','time.time','datetime.now','probability','probabilistic','neural','learned_policy']:
        assert bad not in txt
