# SPDX-License-Identifier: MPL-2.0
from copy import deepcopy
from dataclasses import replace
import pytest

from qec.experiments.acoustic.channel import ChannelConfig, PCM, SINE48, encode_symbols, symbol_block
from qec.experiments.acoustic.receiver import ReceiverPolicy, receive_pcm, validate_observation
from qec.sonify.canonical import canonical_sha256


def rehash(value):
    if isinstance(value,list):
        return [rehash(v) for v in value]
    if isinstance(value,dict):
        result={k:rehash(v) for k,v in value.items() if k!='sha256'}
        if 'sha256' in value:
            result['sha256']=canonical_sha256(result)
        return result
    return value


@pytest.mark.parametrize('d',[3,4])
@pytest.mark.parametrize('fec',['none','unanimous','majority'])
@pytest.mark.parametrize('rate',[24000,48000])
def test_clean_all_states(d,fec,rate):
    config=ChannelConfig(d,sample_rate=rate)
    pcm=encode_symbols(tuple(range(d)),config)
    policy=ReceiverPolicy(fec)
    result=receive_pcm(pcm,config,d,policy)
    assert result['accepted'] and result['estimated_syndrome']==list(range(d))
    assert validate_observation(result,expected_pcm=pcm,config=config,symbol_count=d,policy=policy)['all_passed']


@pytest.mark.parametrize('d',[3,4])
def test_coherent_phase_substitution_is_raw_false_accept_not_authentication(d):
    config=ChannelConfig(d)
    original=(0,1)
    shifted=encode_symbols(tuple((s+1)%d for s in original),config)
    result=receive_pcm(shifted,config,len(original))
    assert result['accepted'] and result['estimated_syndrome'] != list(original)
    assert result['receiver_false_accept_possible']
    assert not result['claim_boundary']['receiver_acceptance_is_independent_truth']


@pytest.mark.parametrize('d',[3,4])
def test_one_erased_repetition_corrects_but_two_do_not(d):
    config=ChannelConfig(d)
    pcm=encode_symbols((1,),config)
    n=config.block_samples
    damaged=PCM((0,)*n+pcm.samples[n:])
    assert receive_pcm(damaged,config,1)['estimated_syndrome']==[1]
    assert not receive_pcm(damaged,config,1,ReceiverPolicy('none'))['accepted']
    assert not receive_pcm(damaged,config,1,ReceiverPolicy('unanimous'))['accepted']
    assert not receive_pcm(PCM((0,)*(2*n)+pcm.samples[2*n:]),config,1)['accepted']


@pytest.mark.parametrize('d',[3,4])
def test_one_wrong_repetition_majority_and_two_wrong_false_accept(d):
    config=ChannelConfig(d)
    good=symbol_block(0,config); wrong=symbol_block(1,config)
    assert receive_pcm(PCM(wrong+good+good),config,1)['estimated_syndrome']==[0]
    result=receive_pcm(PCM(wrong+wrong+good),config,1)
    assert result['accepted'] and result['estimated_syndrome']==[1]


@pytest.mark.parametrize('d,harmonics',[(3,(3,)),(3,(1,)),(3,(1,3)),(4,(2,4)),(4,(1,3,4))])
def test_dark_parity_or_incomplete_bank_never_trusted(d,harmonics):
    config=ChannelConfig(d,harmonics=harmonics)
    result=receive_pcm(encode_symbols((0,),config),config,1)
    assert not result['accepted'] and result['reasons']==['incomplete_harmonic_classes']


@pytest.mark.parametrize('d',[3,4])
def test_dark_corruption_and_discordant_independent_orders_reject(d):
    config=ChannelConfig(d)
    n=config.block_samples
    for harmonic in (d,1):
        samples=tuple(sum(config.amplitude*SINE48[(h*(i+(48//d if h==harmonic else 0)))%48]//32767
                          for h in config.harmonics) for i in range(n))
        # A symbol-sized shift cannot change a state-dark order; invert it instead.
        if harmonic==d:
            samples=tuple(sum(((-1 if h==d else 1)*config.amplitude*SINE48[(h*i)%48])//32767
                              for h in config.harmonics) for i in range(n))
        result=receive_pcm(PCM(samples*3),config,1)
        assert not result['accepted']


@pytest.mark.parametrize('d',[3,4])
def test_no_tone_length_and_clock_reject(d):
    config=ChannelConfig(d)
    pcm=encode_symbols((0,1),config)
    for bad in (PCM((0,)*len(pcm.samples)), PCM(pcm.samples[:-1]),PCM(pcm.samples,24000)):
        assert not receive_pcm(bad,config,2)['accepted']


def test_rehashed_forgery_foreign_pcm_and_policy_fail():
    config=ChannelConfig()
    pcm=encode_symbols((0,1),config)
    result=receive_pcm(pcm,config,2)
    for mutate in (lambda x:x.update(estimated_syndrome=[2,2]),lambda x:x.update(accepted=False),
                   lambda x:x.update(extra=0),lambda x:x['per_symbol'][0]['blocks'][0].update(accepted=False)):
        forged=deepcopy(result); mutate(forged)
        with pytest.raises(ValueError,match='INVALID_INPUT'):
            validate_observation(rehash(forged),expected_pcm=pcm,config=config,symbol_count=2)
    with pytest.raises(ValueError,match='INVALID_INPUT'):
        validate_observation(result,expected_pcm=encode_symbols((2,1),config),config=config,symbol_count=2)
    with pytest.raises(ValueError,match='INVALID_INPUT'):
        validate_observation(result,expected_pcm=pcm,config=config,symbol_count=2,policy=ReceiverPolicy('none'))


@pytest.mark.parametrize('changes',[{'fec':'auto'},{'tolerance_per_mille':True},{'tolerance_per_mille':1000},
 {'max_residual_per_mille':0},{'minimum_margin_per_mille':0},{'minimum_margin_per_mille':1.0}])
def test_policy_strict(changes):
    with pytest.raises(ValueError,match='INVALID_INPUT'):
        ReceiverPolicy(**changes)


@pytest.mark.parametrize('d',[3,4])
def test_robust_period_estimator_has_public_fixed_policy_and_replays(d):
    config=ChannelConfig(d)
    pcm=encode_symbols(tuple(range(d)),config)
    policy=ReceiverPolicy(estimator='median_period')
    result=receive_pcm(pcm,config,d,policy)
    assert result['estimated_syndrome']==list(range(d))
    assert validate_observation(result,expected_pcm=pcm,config=config,symbol_count=d,policy=policy)['all_passed']
    assert ReceiverPolicy.from_dict(policy.as_dict())==policy
    with pytest.raises(ValueError,match='INVALID_INPUT'):
        validate_observation(result,expected_pcm=pcm,config=config,symbol_count=d)


def test_robust_estimator_median_discards_one_period_projection_outlier():
    from qec.experiments.acoustic.receiver import _features
    config=ChannelConfig()
    block=symbol_block(0,config)
    outlier=tuple(3*x for x in block[:48])+block[48:]
    assert _features(outlier,1,'median_period')==_features(block,1,'median_period')
    assert _features(outlier,1)!=_features(block,1)
    # The separate residual-energy guard remains active; robustness is not
    # permission to discard adverse evidence or force trust.
    policy=ReceiverPolicy(estimator='median_period')
    result=receive_pcm(PCM(outlier*3),config,1,policy)
    assert not result['accepted']
    assert 'excessive_residual' in result['per_symbol'][0]['blocks'][0]['reasons']
