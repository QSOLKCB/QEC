# SPDX-License-Identifier: MPL-2.0
"""Bounded comparative classical PCM evidence, with independent-reference gates.

The finite battery is a set of synthetic controls, not a population estimate.
Source bundles retain every input; replay recomputes samples, observations,
reference vetoes, exact finite-Pauli outcomes, and integer/rational metrics.
Runtime and host memory are deliberately absent from canonical evidence.
"""
from __future__ import annotations

from qec.decoder.qutrit.codes import cyclic_five_qutrit_code
from qec.decoder.qutrit.exact import ExactDecoder as QutritDecoder
from qec.decoder.qutrit.stabilizer import Pauli
from qec.decoder.ququart.codes import packed_five_ququart_code
from qec.decoder.ququart.exact import ExactDecoder as QuquartDecoder
from qec.decoder.ququart.packed import PackedPauli
from qec.decoder.ququart.syndrome import pack_binary_syndrome, unpack_ququart_syndrome
from qec.sonify.canonical import canonical_sha256

from ._common import CLAIM_BOUNDARY, digest, exact, integer, seal, sequence
from .bridge import MeasuredSyndromeReference, evaluate_logical_outcome, run_pcm_bridge
from .channel import AcousticSource, ChannelConfig, PCM, encode_symbols
from .noise import CancellationConfig, NoiseFault, NoisePlan, apply_noise, cancel_reference, independent_noise_reference
from .receiver import ReceiverPolicy, receive_pcm

CONTRACT_VERSION = "177.5"
CASE_NAMES = (
    "clean", "broadband", "colored", "correlated", "speech_like", "impulses",
    "clipping", "timing_slip", "missing_samples", "phase_drift", "nonlinear", "echo", "aliasing",
    "wrong_phase_reference", "unrelated_reference", "no_tone", "coherent_substitution",
    "first_repeat_substitution", "period_impulse",
)
CHANNEL_METHODS = ("unfiltered", "cancellation_only", "fec_only", "combined", "robust_median_period")
METHOD_NAMES = CHANNEL_METHODS + ("direct_syndrome_reference", "exact_clean_reference")
MAX_TRIALS = 2 * len(CASE_NAMES)


def _seal(name, **fields):
    return seal(name, contract_version=CONTRACT_VERSION, **fields)


def _ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "defined": denominator != 0}


def _fixture(alphabet_size):
    x, z = (1, 0, 0, 0, 0), (0,) * 5
    if alphabet_size == 3:
        return cyclic_five_qutrit_code(), Pauli(x, z), QutritDecoder
    return packed_five_ququart_code(), PackedPauli(x, z, z, x), QuquartDecoder


def _pauli(value):
    if type(value) is Pauli:
        return {"algebra": "GF(3)", "x": list(value.x), "z": list(value.z)}
    return {"algebra": "packed_GF(2)_x_GF(2)", **{
        k: list(getattr(value, k)) for k in ("lane0_x", "lane0_z", "lane1_x", "lane1_z")}}


def _plan(case_name, seed):
    fault = NoiseFault.make
    faults = {
        "broadband": fault("broadband", amplitude=2400),
        "colored": fault("colored", amplitude=2400, memory=7),
        "correlated": fault("correlated", amplitude=2400, period=17),
        "speech_like": fault("speech_like", amplitude=2400, fundamental_period=48, envelope_period=193),
        "impulses": fault("impulses", amplitude=16000, interval=31, offset=7),
        "clipping": fault("clipping", limit=500),
        "timing_slip": fault("timing_slip", at_sample=13, slip_samples=2),
        "missing_samples": fault("missing_samples", start=13, count=2),
        "phase_drift": fault("phase_drift", period=48, sample_step=1),
        "nonlinear": fault("nonlinear", gain_numerator=8, gain_denominator=1),
        "echo": fault("echo", delay=9, gain_numerator=1, gain_denominator=2),
        "aliasing": fault("aliasing", factor=3),
    }
    key = "broadband" if case_name in ("wrong_phase_reference", "unrelated_reference") else case_name
    return NoisePlan((faults[key],) if key in faults else (), seed=seed)


def build_trial_source(case_name: str, *, alphabet_size: int = 3, seed: int = 1775) -> dict:
    """Construct one fully specified synthetic control from the finite catalogue.

    The separately generated additive reference knows its own generator/seed,
    never observed-minus-clean samples. That knowledge is a privileged resource.
    Neither this function nor a hash authenticates a physical measurement.
    """
    if type(case_name) is not str or case_name not in CASE_NAMES:
        raise ValueError("INVALID_INPUT")
    integer(alphabet_size, 4, 3)
    integer(seed, 0xFFFFFFFF)
    config = ChannelConfig(alphabet_size=alphabet_size, cycles=3)
    code, error, _ = _fixture(alphabet_size)
    syndrome = code.syndrome(error)
    symbols = syndrome if alphabet_size == 3 else pack_binary_syndrome(syndrome)
    reference = MeasuredSyndromeReference(alphabet_size, tuple(symbols),
                                          "independent-synthetic-measurement-fixture", f"q{alphabet_size}-single-site")
    source = AcousticSource(config, tuple(symbols), canonical_sha256({
        "fixture": "single_site_finite_pauli", "error": _pauli(error), "reference": reference.as_dict()}))
    plan = _plan(case_name, seed)
    additive = all(f.kind in ("broadband", "colored", "correlated", "speech_like", "impulses") for f in plan.faults)
    reference_plan = plan if additive else NoisePlan((), seed=seed)
    if case_name == "unrelated_reference":
        reference_plan = NoisePlan(reference_plan.faults, seed=(seed + 1) & 0xFFFFFFFF)
    original = encode_symbols(symbols, config)
    noise_reference = independent_noise_reference(len(original.samples), reference_plan,
                                                  sample_rate=config.sample_rate)
    reference_identity = canonical_sha256({"plan": reference_plan.as_dict(), "sample_count": len(original.samples),
                                          "sample_rate": config.sample_rate, "scope": "independently_generated_noise"})
    cancel = CancellationConfig(reference_identity, "privileged_synthetic_reference", "noise_cancellation",
                                True, True, phase_degrees=180 if case_name == "wrong_phase_reference" else 0)
    transform = (case_name if case_name in ("no_tone", "coherent_substitution", "first_repeat_substitution", "period_impulse")
                 else "none")
    return _seal("comparative-source", case_name=case_name, alphabet_size=alphabet_size, seed=seed,
                 acoustic_source=source.as_dict(), measured_reference=reference.as_dict(),
                 noise_plan=plan.as_dict(), post_noise_control=transform,
                 noise_reference_plan=reference_plan.as_dict(), noise_reference=noise_reference.as_dict(),
                 cancellation_config=cancel.as_dict(), error_fixture=_pauli(error),
                 logical_scope="single_site_error_exact_finite_pauli_stabilizer_model_only",
                 generator_truth_access="fixture_generator_and_post_decode_evaluator_only",
                 claim_boundary=dict(CLAIM_BOUNDARY))


def _read_source(value):
    try:
        if type(value) is not dict:
            raise ValueError("INVALID_INPUT")
        expected = build_trial_source(value["case_name"], alphabet_size=value["alphabet_size"], seed=value["seed"])
        exact(value, expected)
        return expected
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _control(pcm, config, mode):
    samples = list(pcm.samples)
    if mode == "no_tone":
        samples = [0] * len(samples)
    elif mode in ("coherent_substitution", "first_repeat_substitution"):
        for block_index, start in enumerate(range(0, len(samples), config.block_samples)):
            if mode == "coherent_substitution" or block_index % config.repetitions == 0:
                block = samples[start:start + config.block_samples]
                shift = 48 // config.alphabet_size
                samples[start:start + config.block_samples] = block[shift:] + block[:shift]
    elif mode == "period_impulse":
        # One corrupted period in the first block of each symbol. Saturation is
        # explicit and reproducible; this is not a model of microphone impulse response.
        for start in range(0, len(samples), config.block_samples * config.repetitions):
            samples[start:start + 48] = [max(-32768, min(32767, 3 * x)) for x in samples[start:start + 48]]
    elif mode != "none":
        raise ValueError("INVALID_INPUT")
    return PCM(tuple(samples), pcm.sample_rate)


def _samples(source):
    config = ChannelConfig.from_dict(source["acoustic_source"]["config"])
    original = PCM.from_dict(source["acoustic_source"]["encoded_sound"])
    noise = apply_noise(original, NoisePlan.from_dict(source["noise_plan"]), source_sha256=source["sha256"])
    observed = _control(PCM.from_dict(noise["observed"]), config, source["post_noise_control"])
    control = _seal("control-observation", input_noise_sha256=noise["sha256"],
                    control=source["post_noise_control"], observed=observed.as_dict())
    cancellation = cancel_reference(observed, PCM.from_dict(source["noise_reference"]),
                                    CancellationConfig.from_dict(source["cancellation_config"]),
                                    observed_source_sha256=control["sha256"])
    return config, original, observed, PCM.from_dict(cancellation["filtered"]), noise, control, cancellation


def _metrics(predictions, measured, *, raw_accepted, verified, attempted, decoder_accepted, logical_success, raw_receiver_applicable=True):
    erasures = sum(s is None for s in predictions)
    decoded = len(measured) - erasures
    wrong = sum(a is not None and a != b for a, b in zip(predictions, measured))
    bit_errors = sum((a ^ b).bit_count() for a, b in zip(predictions, measured) if a is not None)
    return {"symbols_total": len(measured), "symbols_decoded": decoded, "symbol_errors": wrong,
            "symbol_erasures": erasures, "ser_decoded_symbols": _ratio(wrong, decoded),
            "symbol_error_or_erasure_rate": _ratio(wrong + erasures, len(measured)),
            "symbol_erasure_rate": _ratio(erasures, len(measured)),
            "bit_errors": bit_errors, "bits_decoded": 2 * decoded, "bit_erasures": 2 * erasures,
            "ber_decoded_labels": _ratio(bit_errors, 2 * decoded),
            "ber_scope": "two_bit_natural_binary_symbol_labels_conditional_on_non_erasure; qutrit_label_11_unused",
            "raw_receiver_metrics_applicable": raw_receiver_applicable,
            "recovery_verification_metrics_applicable": raw_receiver_applicable,
            "raw_receiver_frame_denominator": int(raw_receiver_applicable),
            "raw_receiver_accepted_frames": int(raw_accepted),
            "raw_receiver_false_accept_frames": int(raw_accepted and list(predictions) != list(measured)),
            "raw_false_accept_rate_all_frames": _ratio(int(raw_accepted and list(predictions) != list(measured)), int(raw_receiver_applicable)),
            "raw_false_accept_rate_accepted_frames": _ratio(int(raw_accepted and list(predictions) != list(measured)), int(raw_accepted)),
            "independently_verified_frames": int(verified), "decoder_attempted_frames": int(attempted),
            "decoder_accepted_frames": int(decoder_accepted), "exact_model_success_frames": int(logical_success),
            "frame_denominator": 1,
            "verification_is_caller_declared_fixture_authority": True}


def _budget(source, original, *, method):
    count, rate = len(original.samples), original.sample_rate
    config = source["acoustic_source"]["config"]
    channel = method in CHANNEL_METHODS
    cancellation = method in ("cancellation_only", "combined")
    return {"equal_channel_resource_comparison": channel,
            "transmitted_pcm_sha256": original.as_dict()["pcm_sha256"] if channel else None,
            "transmitted_sample_count": count if channel else 0,
            "nominal_carrier_bandwidth_hz": max(config["harmonics"]) * config["carrier_hz"] if channel else 0,
            "bandwidth_scope": "nominal_harmonic_band; quantized_pcm_is_not_strictly_bandlimited",
            "transmitted_energy_sum_squares": sum(s * s for s in original.samples) if channel else 0,
            "mean_transmitted_sample_power": _ratio(sum(s * s for s in original.samples), count) if channel else _ratio(0, 0),
            "whole_frame_latency_budget_seconds": _ratio(count, rate) if channel else _ratio(0, 0),
            "latency_scope": "sample_frame_budget; excludes_processing_and_reference_acquisition; not_hardware_latency",
            "repetitions_transmitted": config["repetitions"] if channel else 0,
            "unused_repetitions_remain_in_budget": method in ("unfiltered", "cancellation_only"),
            "extra_noise_reference_pcm_bytes": 2 * count if cancellation else 0,
            "extra_clean_reference_pcm_bytes": 2 * count if method == "exact_clean_reference" else 0,
            "extra_independent_measured_reference_symbols": 4,
            "reference_acquisition_cost": "not_measured; independent_reference_is_privileged",
            "reference_generator_seed_knowledge": cancellation,
            "equal_total_resources_claimed": False}


def _receiver_method(source, method, pcm, config, original, reference):
    policy = ReceiverPolicy(fec="majority" if method in ("fec_only", "combined", "robust_median_period") else "none",
                            estimator="median_period" if method == "robust_median_period" else "matched_filter")
    observation = receive_pcm(pcm, config, 4, policy)
    inputs = {"pcm": pcm, "config": config, "observation": observation, "reference": reference,
              "expected_reference_sha256": reference.as_dict()["sha256"], "policy": policy}
    bridge = run_pcm_bridge(**inputs)
    _, error, _ = _fixture(config.alphabet_size)
    outcome = evaluate_logical_outcome(bridge, error, **inputs)
    predictions = ([row["estimated_symbol"] for row in observation["per_symbol"]]
                   if observation["per_symbol"] else [None] * 4)
    metrics = _metrics(predictions, reference.measured_syndrome, raw_accepted=observation["accepted"],
                       verified=bridge["verified_recovery"], attempted=bridge["decoder_decision"]["attempted"],
                       decoder_accepted=bridge["decoder_decision"]["accepted"],
                       logical_success=outcome["logical_outcome"]["success"])
    row = _seal("comparative-method", method=method, source_sha256=source["sha256"],
                input_pcm_sha256=pcm.as_dict()["sha256"], policy=policy.as_dict(),
                predicted_symbols=predictions, observation_sha256=observation["sha256"],
                bridge_sha256=bridge["sha256"], logical_evaluation_sha256=outcome["sha256"],
                decoder_decision=bridge["decoder_decision"], logical_outcome=outcome["logical_outcome"],
                metrics=metrics, resources=_budget(source, original, method=method),
                waveform_squared_error=(sum((a - b) ** 2 for a, b in zip(pcm.samples, original.samples))
                                        if len(pcm.samples) == len(original.samples) else None),
                waveform_error_scope="aligned_sample_grid_only; undefined_for_length_mismatch",
                receiver_reasons=observation["reasons"], claim_boundary=dict(CLAIM_BOUNDARY))
    return row, observation, bridge, outcome


def _direct_method(source, original, reference):
    code, error, decoder_type = _fixture(source["alphabet_size"])
    syndrome = (reference.measured_syndrome if source["alphabet_size"] == 3 else
                unpack_ququart_syndrome(reference.measured_syndrome))
    correction = decoder_type(code).decode(syndrome)
    residual = error.compose(correction)
    success = code.is_stabilizer(residual)
    return _seal("comparative-method", method="direct_syndrome_reference", source_sha256=source["sha256"],
                 input_reference_sha256=reference.as_dict()["sha256"],
                 predicted_symbols=list(reference.measured_syndrome),
                 decoder_decision={"attempted": True, "accepted": True, "syndrome": list(syndrome),
                                   "correction": _pauli(correction)},
                 logical_outcome={"success": success, "residual": _pauli(residual),
                                  "residual_is_stabilizer": success, "status": "verified_exact_model_correction"},
                 metrics=_metrics(reference.measured_syndrome, reference.measured_syndrome,
                                  raw_accepted=False, verified=False, attempted=True,
                                  decoder_accepted=True, logical_success=success, raw_receiver_applicable=False),
                 raw_receiver_metrics_applicable=False,
                 resources=_budget(source, original, method="direct_syndrome_reference"),
                 claim_boundary=dict(CLAIM_BOUNDARY))


def run_comparative_trial(source: dict) -> dict:
    """Run all seven methods against one source; retain pre-veto false accepts."""
    source = _read_source(source)
    config, original, observed, filtered, noise, control, cancellation = _samples(source)
    reference = MeasuredSyndromeReference.from_dict(source["measured_reference"])
    methods = []
    for name in METHOD_NAMES:
        if name == "direct_syndrome_reference":
            methods.append(_direct_method(source, original, reference))
        else:
            pcm = (filtered if name in ("cancellation_only", "combined") else
                   original if name == "exact_clean_reference" else observed)
            methods.append(_receiver_method(source, name, pcm, config, original, reference)[0])
    return _seal("comparative-trial", source=source, noise_observation_sha256=noise["sha256"],
                 control_observation_sha256=control["sha256"], cancellation_sha256=cancellation["sha256"],
                 original=original.as_dict(), observed=observed.as_dict(), filtered=filtered.as_dict(),
                 methods=methods, comparison_scope="paired_fixed_pcm_seed_and_channel_budget; reference_costs_separate",
                 method_evidence="full_observations_and_gates_recomputed_from_complete_source_bundle",
                 claim_boundary=dict(CLAIM_BOUNDARY))


def validate_comparative_trial(receipt, *, expected_source_sha256: str, expected_reference_sha256: str) -> dict:
    """Require external source AND measured-reference pins; replay all fields."""
    try:
        digest(expected_source_sha256)
        digest(expected_reference_sha256)
        if type(receipt) is not dict:
            raise ValueError("INVALID_INPUT")
        source = _read_source(receipt["source"])
        if (source["sha256"] != expected_source_sha256 or
                source["measured_reference"]["sha256"] != expected_reference_sha256):
            raise ValueError("INVALID_INPUT")
        expected = run_comparative_trial(source)
        exact(receipt, expected)
        return _seal("comparative-trial-validation", trial_sha256=expected["sha256"], all_passed=True,
                     sample_and_receiver_and_decoder_replayed=True, external_bindings_verified=True,
                     source_provenance_authenticated=False)
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def build_battery_source(*, cases=CASE_NAMES, alphabets=(3, 4), seed=1775) -> dict:
    cases, alphabets = sequence(cases, len(CASE_NAMES)), sequence(alphabets, 2)
    if (not cases or not alphabets or any(type(c) is not str or c not in CASE_NAMES for c in cases)
            or any(type(a) is not int or a not in (3, 4) for a in alphabets)
            or len(set(cases)) != len(cases) or len(set(alphabets)) != len(alphabets)):
        raise ValueError("INVALID_INPUT")
    integer(seed, 0xFFFFFFFF)
    return _seal("comparative-battery-source", cases=list(cases), alphabets=list(alphabets), seed=seed,
                 trials=[build_trial_source(c, alphabet_size=a, seed=seed) for a in alphabets for c in cases])


def _read_battery_source(source):
    try:
        if type(source) is not dict:
            raise ValueError("INVALID_INPUT")
        expected = build_battery_source(cases=source["cases"], alphabets=source["alphabets"], seed=source["seed"])
        exact(source, expected)
        return expected
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def run_comparative_battery(source: dict) -> dict:
    source = _read_battery_source(source)
    trials = [run_comparative_trial(s) for s in source["trials"]]
    summary = []
    for name in METHOD_NAMES:
        rows = [next(m for m in t["methods"] if m["method"] == name)["metrics"] for t in trials]
        counters = ("symbols_total", "symbols_decoded", "symbol_errors", "symbol_erasures", "bit_errors", "bits_decoded",
                    "bit_erasures", "raw_receiver_frame_denominator", "raw_receiver_accepted_frames", "raw_receiver_false_accept_frames",
                    "independently_verified_frames", "decoder_attempted_frames", "decoder_accepted_frames", "exact_model_success_frames")
        totals = {key: sum(m[key] for m in rows) for key in counters}
        summary.append({"method": name, **totals, "frame_denominator": len(rows),
                        "raw_receiver_metrics_applicable": name != "direct_syndrome_reference",
                        "recovery_verification_metrics_applicable": name != "direct_syndrome_reference",
                        "ser_decoded_symbols": _ratio(totals["symbol_errors"], totals["symbols_decoded"]),
                        "ber_decoded_labels": _ratio(totals["bit_errors"], totals["bits_decoded"]),
                        "symbol_erasure_rate": _ratio(totals["symbol_erasures"], totals["symbols_total"]),
                        "raw_false_accept_rate_all_frames": _ratio(totals["raw_receiver_false_accept_frames"], totals["raw_receiver_frame_denominator"]),
                        "raw_false_accept_rate_accepted_frames": _ratio(totals["raw_receiver_false_accept_frames"], totals["raw_receiver_accepted_frames"])})
    return _seal("comparative-battery", source=source, trials=trials, summary=summary,
                 bounded_trial_count=len(trials), maximum_trial_count=MAX_TRIALS,
                 inference_scope="finite_named_controls_only; no_confidence_intervals_or_population_generalization",
                 transport_evidence="separate_explicit_demonstration_not_embedded_per_trial",
                 claim_boundary=dict(CLAIM_BOUNDARY))


def validate_comparative_battery(receipt, *, expected_source_sha256: str) -> dict:
    try:
        digest(expected_source_sha256)
        if type(receipt) is not dict:
            raise ValueError("INVALID_INPUT")
        source = _read_battery_source(receipt["source"])
        if source["sha256"] != expected_source_sha256:
            raise ValueError("INVALID_INPUT")
        expected = run_comparative_battery(source)
        exact(receipt, expected)
        return _seal("comparative-battery-validation", battery_sha256=expected["sha256"],
                     bounded_trial_count=len(expected["trials"]), all_passed=True,
                     full_comparative_replay=True, external_source_binding_verified=True)
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def build_demonstration(source: dict) -> dict:
    """Six actual role payloads from one explicit broadband trial, not a battery.

    Rejected is a real rejected raw observation; verified includes the complete
    independent gate. JSON waveform payloads contain the actual PCM bytes.
    """
    source = _read_source(source)
    if source["case_name"] != "broadband":
        raise ValueError("INVALID_INPUT")
    config, original, observed, filtered, noise, control, cancellation = _samples(source)
    reference = MeasuredSyndromeReference.from_dict(source["measured_reference"])
    _, rejected, rejected_bridge, _ = _receiver_method(source, "unfiltered", observed, config, original, reference)
    _, candidate, verified, outcome = _receiver_method(source, "combined", filtered, config, original, reference)
    if rejected["accepted"] or not verified["verified_recovery"]:
        raise ValueError("INVALID_INPUT")
    roles = {"original": original.as_dict(), "observed": observed.as_dict(), "filtered": filtered.as_dict(),
             "candidate": candidate, "rejected": rejected, "verified": verified}
    return _seal("lab-demonstration", source=source, roles=roles, noise_observation=noise,
                 control_observation=control, cancellation=cancellation, rejected_bridge=rejected_bridge,
                 logical_evaluation=outcome, external_import="ordinary_PCM16_WAV_and_JSON_only",
                 spectral_repository_or_ui_integration=False, claim_boundary=dict(CLAIM_BOUNDARY))


def validate_demonstration(value, *, expected_source_sha256: str, expected_reference_sha256: str) -> dict:
    try:
        digest(expected_source_sha256)
        digest(expected_reference_sha256)
        if type(value) is not dict:
            raise ValueError("INVALID_INPUT")
        source = _read_source(value["source"])
        if source["sha256"] != expected_source_sha256 or source["measured_reference"]["sha256"] != expected_reference_sha256:
            raise ValueError("INVALID_INPUT")
        expected = build_demonstration(source)
        exact(value, expected)
        return _seal("lab-demonstration-validation", demonstration_sha256=expected["sha256"],
                     all_passed=True, full_sample_receiver_gate_and_model_replay=True)
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc
