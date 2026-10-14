# SPDX-License-Identifier: MPL-2.0
"""Sample-level and adversarial replay checks for the v177.2 noise boundary."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import inspect
import os
import subprocess
import sys

import pytest

from qec.experiments.acoustic.channel import ChannelConfig, MAX_SAMPLES, PCM, encode_symbols
from qec.experiments.acoustic.noise import (
    CancellationConfig, NoiseFault, NoisePlan, apply_noise, cancel_reference,
    independent_noise_reference, validate_cancellation_receipt, validate_noise_observation,
)
from qec.sonify.canonical import canonical_json, canonical_sha256

SOURCE = "a" * 64
REFERENCE_SOURCE = "b" * 64


def fault(kind, **parameters):
    return NoiseFault.make(kind, **parameters)


def rehash(value):
    value["sha256"] = canonical_sha256({key: val for key, val in value.items() if key != "sha256"})
    return value


def noise_pins(original, plan):
    return {"expected_source_sha256": SOURCE, "expected_original_sha256": original.as_dict()["sha256"],
            "expected_plan_sha256": plan.as_dict()["sha256"]}


def permission(**kwargs):
    return CancellationConfig(REFERENCE_SOURCE, "privileged_synthetic_reference", "noise_cancellation",
                              True, True, **kwargs)


def cancellation_pins(observed, reference, config, observation_hash=SOURCE):
    return {"expected_observed_source_sha256": observation_hash,
            "expected_reference_source_sha256": REFERENCE_SOURCE,
            "expected_observed_sha256": observed.as_dict()["sha256"],
            "expected_reference_sha256": reference.as_dict()["sha256"],
            "expected_config_sha256": config.as_dict()["sha256"]}


ALL_FAULTS = (
    fault("broadband", amplitude=512),
    fault("colored", amplitude=512, memory=7),
    fault("correlated", amplitude=512, period=17),
    fault("speech_like", amplitude=512, fundamental_period=48, envelope_period=193),
    fault("impulses", amplitude=2000, interval=31, offset=7),
    fault("clipping", limit=500),
    fault("timing_slip", at_sample=13, slip_samples=2),
    fault("missing_samples", start=13, count=2),
    fault("phase_drift", period=48, sample_step=3),
    fault("nonlinear", gain_numerator=4, gain_denominator=1),
    fault("echo", delay=9, gain_numerator=1, gain_denominator=2),
    fault("aliasing", factor=3),
)


@pytest.mark.parametrize("item", ALL_FAULTS, ids=lambda item: item.kind)
def test_each_family_transforms_real_pcm_and_replays(item):
    original = encode_symbols((0, 1, 2), ChannelConfig(amplitude=2048))
    plan = NoisePlan((item,), seed=41)
    receipt = apply_noise(original, plan, source_sha256=SOURCE)
    observed = PCM.from_dict(receipt["observed"])
    assert observed.samples != original.samples
    assert receipt["original"] == original.as_dict()
    assert receipt["cancellation_applied"] is False
    assert len(receipt["stages"]) == 1
    assert receipt["stages"][0]["changed_aligned_samples"] > 0
    assert canonical_json(apply_noise(original, plan, source_sha256=SOURCE)) == canonical_json(receipt)
    assert validate_noise_observation(receipt, **noise_pins(original, plan))["all_passed"] is True
    assert NoisePlan.from_dict(plan.as_dict()) == plan
    assert all(type(sample) is int and -32768 <= sample <= 32767 for sample in observed.samples)


def test_fixed_seed_algorithm_pcm_known_answer():
    original = PCM((0,) * 12)
    plan = NoisePlan((fault("broadband", amplitude=1000),), seed=1)
    receipt = apply_noise(original, plan, source_sha256=SOURCE)
    assert PCM.from_dict(receipt["observed"]).samples == (-528, -262, 8, 409, -899, -261, 549, 112, -968, 278, -500, -156)
    assert receipt["observed"]["pcm_hex"] == "f0fdfafe080099017dfcfbfe2502700038fc16010cfe64ff"
    assert plan.as_dict()["sha256"] == "0c41b28a6b0e8b44a5f2a30c034eb631ae0eb8774d2ba9d061fd24f7eaa8af1e"
    assert receipt["sha256"] == "84c769f5e08a4a0e2fd9252b719a71243086be5e75e01569701c3e5f41886e76"


def test_cross_process_seed_and_hash_seed_invariance():
    script = """
from qec.experiments.acoustic.channel import PCM
from qec.experiments.acoustic.noise import NoiseFault, NoisePlan, apply_noise
p=NoisePlan((NoiseFault.make('broadband', amplitude=1000),), seed=1)
print(apply_noise(PCM((0,)*12), p, source_sha256='a'*64)['sha256'])
"""
    results = [subprocess.check_output([sys.executable, "-c", script],
                                      env={**os.environ, "PYTHONHASHSEED": seed}, text=True).strip()
               for seed in ("0", "937")]
    assert results == ["84c769f5e08a4a0e2fd9252b719a71243086be5e75e01569701c3e5f41886e76"] * 2


def test_noop_empty_plan_and_immutable_inputs():
    samples = [1, -2, 3]
    original = PCM(samples)
    faults = []
    plan = NoisePlan(faults)
    samples[0] = 100
    faults.append(fault("clipping", limit=1))
    receipt = apply_noise(original, plan, source_sha256=SOURCE)
    assert original.samples == (1, -2, 3)
    assert receipt["original"] == receipt["observed"]
    assert receipt["stages"] == []
    with pytest.raises(FrozenInstanceError):
        plan.seed = 2
    with pytest.raises(FrozenInstanceError):
        ALL_FAULTS[0].kind = "echo"


def test_declared_order_changes_result_and_rng_is_local():
    original = PCM((100,) * 64)
    a, b = fault("broadband", amplitude=1000), fault("clipping", limit=100)
    left, right = NoisePlan((a, b), seed=7), NoisePlan((b, a), seed=7)
    assert apply_noise(original, left, source_sha256=SOURCE)["observed"] != apply_noise(original, right, source_sha256=SOURCE)["observed"]
    first = apply_noise(original, left, source_sha256=SOURCE)
    apply_noise(original, NoisePlan((a,), seed=123), source_sha256=SOURCE)
    assert apply_noise(original, left, source_sha256=SOURCE) == first


def observed_for(samples, item):
    return PCM.from_dict(apply_noise(PCM(samples), NoisePlan((item,)), source_sha256=SOURCE)["observed"]).samples


def test_timing_loss_phase_echo_aliasing_exact_semantics():
    samples = (100, 200, 300, 400, 500, 600)
    assert observed_for(samples, fault("timing_slip", at_sample=2, slip_samples=2)) == (100, 200, 0, 0, 300, 400)
    assert observed_for(samples, fault("timing_slip", at_sample=2, slip_samples=-2)) == (100, 200, 500, 600, 0, 0)
    assert observed_for(samples, fault("missing_samples", start=2, count=2)) == (100, 200, 500, 600)
    assert observed_for(samples, fault("phase_drift", period=2, sample_step=1)) == (100, 200, 400, 500, 0, 0)
    assert observed_for(samples, fault("echo", delay=2, gain_numerator=1, gain_denominator=2)) == (100, 200, 350, 500, 650, 800)
    assert observed_for(samples, fault("aliasing", factor=2)) == (100, 100, 300, 300, 500, 500)
    assert observed_for(samples, fault("clipping", limit=250)) == (100, 200, 250, 250, 250, 250)
    assert observed_for((0,) * 8, fault("impulses", amplitude=50, interval=3, offset=1)) == (0, 50, 0, 0, -50, 0, 0, 50)


def test_saturation_and_irrecoverable_reference_mixing_clipping_are_visible():
    original = PCM((32767,) * 10)
    plan = NoisePlan((fault("impulses", amplitude=1000, interval=1, offset=0),))
    observation = apply_noise(original, plan, source_sha256=SOURCE)
    assert observation["stages"][0]["saturated_sample_count"] == 5
    reference = independent_noise_reference(len(original.samples), plan)
    receipt = cancel_reference(PCM.from_dict(observation["observed"]), reference, permission(),
                               observed_source_sha256=observation["sha256"])
    assert PCM.from_dict(receipt["filtered"]).samples != original.samples
    assert receipt["improvement_claimed"] is False


@pytest.mark.parametrize("item", [fault("missing_samples", start=5, count=2),
                                   fault("timing_slip", at_sample=7, slip_samples=1),
                                   fault("timing_slip", at_sample=5, slip_samples=-2)])
def test_sample_dependent_out_of_bounds_faults_fail(item):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        apply_noise(PCM((0,) * 6), NoisePlan((item,)), source_sha256=SOURCE)


@pytest.mark.parametrize("construct", [
    lambda: NoisePlan(seed=True), lambda: NoisePlan(seed=-1), lambda: NoisePlan(seed=2 ** 32),
    lambda: NoisePlan((ALL_FAULTS[0],) * 17), lambda: NoisePlan(({},)),
    lambda: fault("unknown", amplitude=1), lambda: fault("broadband", amplitude=1.0),
    lambda: fault("broadband", amplitude=True), lambda: fault("broadband", amplitude=16385),
    lambda: fault("broadband"), lambda: fault("broadband", amplitude=1, ignored=2),
    lambda: NoiseFault("colored", (("memory", 1), ("amplitude", 1))),
    lambda: NoiseFault("broadband", (("amplitude", 1), ("amplitude", 2))),
    lambda: fault("impulses", amplitude=1, interval=1, offset=1),
    lambda: fault("echo", delay=1, gain_numerator=5, gain_denominator=1),
    lambda: fault("nonlinear", gain_numerator=9, gain_denominator=1),
    lambda: fault("aliasing", factor=1),
])
def test_fault_input_bounds_and_strict_types(construct):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        construct()


def test_noise_rejects_rehashed_output_stage_claim_and_upstream_tampering():
    original, plan = PCM((1, 2, 3, 4)), NoisePlan((fault("broadband", amplitude=100),))
    receipt = apply_noise(original, plan, source_sha256=SOURCE)
    for key, val in (("source_sha256", "c" * 64), ("cancellation_applied", True), ("unknown", 1)):
        changed = deepcopy(receipt)
        changed[key] = val
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            validate_noise_observation(rehash(changed), **noise_pins(original, plan))
    changed = deepcopy(receipt)
    changed["observed"] = PCM((5, 6, 7, 8)).as_dict()
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_noise_observation(rehash(changed), **noise_pins(original, plan))
    changed = deepcopy(receipt)
    changed["stages"][0]["rng_state_after"] += 1
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_noise_observation(rehash(changed), **noise_pins(original, plan))
    for replacement_original, replacement_plan in ((PCM((5, 6, 7, 8)), plan), (original, replace(plan, seed=2))):
        changed = apply_noise(replacement_original, replacement_plan, source_sha256=SOURCE)
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            validate_noise_observation(changed, **noise_pins(original, plan))


def test_required_external_pins_cannot_be_omitted():
    with pytest.raises(TypeError):
        validate_noise_observation({})
    with pytest.raises(TypeError):
        validate_cancellation_receipt({})
    for name in inspect.signature(validate_noise_observation).parameters:
        if name.startswith("expected_"):
            assert inspect.signature(validate_noise_observation).parameters[name].default is inspect.Parameter.empty
    for name in inspect.signature(validate_cancellation_receipt).parameters:
        if name.startswith("expected_"):
            assert inspect.signature(validate_cancellation_receipt).parameters[name].default is inspect.Parameter.empty


def squared_error(a, b):
    assert len(a.samples) == len(b.samples)
    return sum((x - y) ** 2 for x, y in zip(a.samples, b.samples))


def test_aligned_reference_recovers_and_wrong_phase_delay_unrelated_are_negative_controls():
    original = encode_symbols((0, 1, 2))
    plan = NoisePlan((fault("broadband", amplitude=700),), seed=23)
    observation = apply_noise(original, plan, source_sha256=SOURCE)
    observed = PCM.from_dict(observation["observed"])
    reference = independent_noise_reference(len(original.samples), plan, sample_rate=original.sample_rate)
    unrelated = independent_noise_reference(len(original.samples), replace(plan, seed=987))
    baseline_error = squared_error(original, observed)
    assert baseline_error > 0
    for config, ref, improved in ((permission(), reference, True),
                                  (permission(phase_degrees=180), reference, False),
                                  (permission(delay_samples=7), reference, False),
                                  (permission(), unrelated, False)):
        receipt = cancel_reference(observed, ref, config, observed_source_sha256=observation["sha256"])
        filtered = PCM.from_dict(receipt["filtered"])
        if improved:
            assert filtered == original
            assert squared_error(original, filtered) == 0
        else:
            assert squared_error(original, filtered) > baseline_error
        assert receipt["observed"] == observation["observed"]
        assert receipt["reference"] == ref.as_dict()
        assert receipt["config"]["reference_role"] == "privileged_synthetic_reference"
        assert validate_cancellation_receipt(receipt, **cancellation_pins(observed, ref, config, observation["sha256"]))["all_passed"]
        assert canonical_json(receipt) == canonical_json(cancel_reference(observed, ref, config,
                                                                         observed_source_sha256=observation["sha256"]))


def test_exact_delay_gain_phase_no_implicit_parameter_search():
    observed, reference = PCM((10, 20, 30, 40)), PCM((3, 5, 7))
    receipt = cancel_reference(observed, reference, permission(delay_samples=1, gain_numerator=3,
                               gain_denominator=2), observed_source_sha256=SOURCE)
    assert PCM.from_dict(receipt["filtered"]).samples == (10, 16, 23, 30)
    assert receipt["reference_used_samples"] == 3
    receipt = cancel_reference(observed, reference, permission(delay_samples=-1, phase_degrees=180),
                               observed_source_sha256=SOURCE)
    assert PCM.from_dict(receipt["filtered"]).samples == (15, 27, 30, 40)
    assert "original" not in inspect.signature(cancel_reference).parameters
    assert "symbols" not in inspect.signature(cancel_reference).parameters
    assert receipt["clean_target_access"] is False
    assert receipt["ideal_symbol_access"] is False
    assert receipt["automatic_reference_inference"] is False


@pytest.mark.parametrize("field,value", [
    ("permission_granted", False), ("permission_granted", 1),
    ("independent_reference", False), ("independent_reference", 1),
    ("permission_purpose", "voice_separation"), ("reference_role", "arbitrary_second_voice"),
    ("reference_source_sha256", "not-a-hash"), ("phase_degrees", 90), ("phase_degrees", False),
    ("gain_numerator", 9), ("gain_denominator", 0), ("gain_numerator", 1.0),
    ("delay_samples", 4097), ("delay_samples", True),
])
def test_reference_requires_explicit_permission_independence_and_valid_transform(field, value):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        replace(permission(), **{field: value})


def test_synthetic_reference_cannot_read_target_or_include_target_dependent_faults():
    for item in ALL_FAULTS[5:]:
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            independent_noise_reference(128, NoisePlan((item,)))
    saturated = NoisePlan((fault("impulses", amplitude=32767, interval=1, offset=0),) * 2)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        independent_noise_reference(128, saturated)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        independent_noise_reference(MAX_SAMPLES + 1, NoisePlan())
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        cancel_reference(PCM((1,)), PCM(()), permission(), observed_source_sha256=SOURCE)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        cancel_reference(PCM((1,)), PCM((1,), 24000), permission(), observed_source_sha256=SOURCE)


def test_cancellation_rejects_fully_rehashed_tampering_and_changed_pinned_input():
    observed, reference, config = PCM((100, 200, 300)), PCM((10, 20, 30)), permission()
    receipt = cancel_reference(observed, reference, config, observed_source_sha256=SOURCE)
    pins = cancellation_pins(observed, reference, config)
    for key, value in (("filtered", PCM((1, 2, 3)).as_dict()), ("clean_target_access", True),
                        ("changed_samples", 2), ("reference_used_samples", 2),
                        ("reference_source_sha256", "c" * 64), ("observed_source_sha256", "c" * 64),
                        ("ideal_symbols", [0])):
        changed = deepcopy(receipt)
        changed[key] = value
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            validate_cancellation_receipt(rehash(changed), **pins)
    for obs, ref, conf in ((PCM((101, 200, 300)), reference, config),
                           (observed, PCM((11, 20, 30)), config),
                           (observed, reference, replace(config, phase_degrees=180)),
                           (observed, reference, replace(config, reference_source_sha256="c" * 64))):
        alternate = cancel_reference(obs, ref, conf, observed_source_sha256=SOURCE)
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            validate_cancellation_receipt(alternate, **pins)


@pytest.mark.parametrize("bad", [None, [], "bad", 0, True, {}, {"original": {}}])
def test_malformed_artifacts_fail_closed(bad):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_noise_observation(bad, **noise_pins(PCM((0,)), NoisePlan()))
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_cancellation_receipt(bad, **cancellation_pins(PCM((0,)), PCM((0,)), permission()))
