import json
import pytest

from qec.analysis.functional_kernel_readout_shell import *

H='a'*64
B='b'*64
C='c'*64
D='d'*64

def build_all():
    core=build_core_kernel_spec('k1','READOUT_CORE',H,B,{"x":1})
    derived=build_derived_kernel_spec(core,'dk1','IDENTITY_DERIVATION',{"y":[1,2]},H,C)
    dr=build_kernel_derivation_receipt(core,derived,'dr1')
    comp=build_kernel_compatibility_receipt(core,derived,'cr1',())
    s1=build_readout_shell('s1','IDENTITY_SHELL',H,B,{"p":1})
    s2=build_readout_shell('s2','ORDER_BINDING_SHELL',B,C,{"q":2})
    stack=build_readout_shell_stack('st1',[s1,s2])
    order=build_readout_order_receipt(stack,'or1')
    rc=build_readout_composition_receipt('cmp1',core,derived,dr,comp,stack,order,D)
    return core,derived,dr,comp,s1,s2,stack,order,rc

def test_determinism():
    a=build_all(); b=build_all()
    assert a[0].kernel_hash==b[0].kernel_hash
    assert a[1].derived_kernel_hash==b[1].derived_kernel_hash
    assert a[4].shell_hash==b[4].shell_hash
    assert a[6].stack_hash==b[6].stack_hash
    assert a[8].composed_readout_identity_hash==b[8].composed_readout_identity_hash
    assert a[8].receipt_hash==b[8].receipt_hash
    assert json.dumps(a[8].to_dict(), sort_keys=True)==json.dumps(b[8].to_dict(), sort_keys=True)

def test_ordering_identity_bearing():
    _,_,_,_,s1,s2,stack,order,_=build_all()
    stack2=build_readout_shell_stack('st2',[s2,s1])
    assert stack.stack_order_hash!=stack2.stack_order_hash
    rc1=build_readout_composition_receipt('a',*build_all()[:4],stack,order,D)
    order2=build_readout_order_receipt(stack2,'o2')
    rc2=build_readout_composition_receipt('b',*build_all()[:4],stack2,order2,D)
    assert rc1.composition_hash!=rc2.composition_hash

def test_failures_and_self_validation():
    core,derived,dr,comp,_,_,stack,order,rc=build_all()
    with pytest.raises(ValueError): build_derived_kernel_spec(core,'x','IDENTITY_DERIVATION',{},H,C,parent_kernel_hash='f'*64)
    with pytest.raises(ValueError): build_readout_shell_stack('x',[])
    with pytest.raises(ValueError): build_readout_shell('x','BAD',H,B,{})
    with pytest.raises(ValueError): build_readout_shell('x','IDENTITY_SHELL',H,B,{"f":lambda x:x})
    with pytest.raises(ValueError): build_kernel_compatibility_receipt(core,derived,'id',('nothex',))

    with pytest.raises(ValueError): KernelDerivationReceipt(**{**dr.to_dict(),'derivation_hash':'f'*64})
    with pytest.raises(ValueError): KernelDerivationReceipt(**{**dr.to_dict(),'receipt_hash':'f'*64})
    with pytest.raises(ValueError): KernelCompatibilityReceipt(**{**comp.to_dict(),'compatibility_hash':'f'*64})
    with pytest.raises(ValueError): KernelCompatibilityReceipt(**{**comp.to_dict(),'receipt_hash':'f'*64})
    with pytest.raises(ValueError): ReadoutOrderReceipt(**{**order.to_dict(),'order_hash':'f'*64})
    with pytest.raises(ValueError): ReadoutOrderReceipt(**{**order.to_dict(),'receipt_hash':'f'*64})
    with pytest.raises(ValueError): ReadoutCompositionReceipt(**{**rc.to_dict(),'composition_hash':'f'*64})
    with pytest.raises(ValueError): ReadoutCompositionReceipt(**{**rc.to_dict(),'receipt_hash':'f'*64})

def test_order_binding_and_tuple_coercion():
    _,_,_,_,_,_,stack,order,_=build_all()
    with pytest.raises(ValueError):
        bad=ReadoutOrderReceipt(**{**order.to_dict(),'ordered_shell_hashes':('f'*64,order.ordered_shell_hashes[1]),'ordered_shell_ids':order.ordered_shell_ids})
        validate_readout_order_receipt(stack,bad)
    with pytest.raises(ValueError):
        bad=ReadoutOrderReceipt(**{**order.to_dict(),'ordered_shell_ids':('x',order.ordered_shell_ids[1]),'ordered_shell_hashes':order.ordered_shell_hashes})
        validate_readout_order_receipt(stack,bad)
    with pytest.raises(ValueError):
        bad=ReadoutOrderReceipt(**{**order.to_dict(),'stack_hash':'f'*64})
        validate_readout_order_receipt(stack,bad)

    hashes=list(order.ordered_shell_hashes)
    ids=list(order.ordered_shell_ids)
    r=ReadoutOrderReceipt(**{**order.to_dict(),'ordered_shell_hashes':hashes,'ordered_shell_ids':ids})
    assert isinstance(r.ordered_shell_hashes, tuple)
    assert isinstance(r.ordered_shell_ids, tuple)
    hashes[0]='f'*64
    ids[0]='changed'
    assert r.ordered_shell_hashes[0]==order.ordered_shell_hashes[0]
    assert r.ordered_shell_ids[0]==order.ordered_shell_ids[0]

def test_immutability_json_and_scope():
    core,derived,_,_,s1,_,stack,_,_=build_all()
    with pytest.raises(TypeError): core.kernel_parameters['x']=2
    with pytest.raises(TypeError): derived.derivation_parameters['y']=3
    with pytest.raises(TypeError): s1.shell_parameters['p']=4
    d=core.to_dict(); d['kernel_parameters']['x']=9
    assert core.kernel_parameters['x']==1
    json.dumps(stack.to_dict(), sort_keys=True)

    forbidden_exports=["ReadoutCombinationMatrix","ReadoutMatrixReceipt","MarkovBasisReceipt","ReadoutTransitionReceipt","LatticeDriftReceipt","RouterReplayReceipt","ReadoutReplayReceipt","MaskReplayReceipt","ShiftReplayReceipt","LatticeReplayProofReceipt"]
    for name in forbidden_exports:
        assert name not in globals()
    forbidden={"apply","execute","run","dispatch","route","traverse","pathfind","resolve","project","search","filter","shift","matrix","markov","sample","random"}
    for cls in (CoreKernelSpec,DerivedKernelSpec,KernelDerivationReceipt,KernelCompatibilityReceipt,ReadoutShell,ReadoutShellStack,ReadoutOrderReceipt,ReadoutCompositionReceipt):
        for n in forbidden:
            assert not hasattr(cls,n)
