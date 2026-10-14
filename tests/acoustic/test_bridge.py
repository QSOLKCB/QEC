# SPDX-License-Identifier: MPL-2.0
"""PCM-only recovery, independent reference veto, and exact-model outcomes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from qec.decoder.qutrit.codes import cyclic_five_qutrit_code
from qec.decoder.qutrit.stabilizer import Pauli, paulis_of_weight as qutrit_errors
from qec.decoder.ququart.codes import packed_five_ququart_code
from qec.decoder.ququart.packed import PackedPauli, paulis_of_weight as ququart_errors
from qec.decoder.ququart.syndrome import pack_binary_syndrome, unpack_ququart_syndrome
from qec.experiments.acoustic import bridge
from qec.experiments.acoustic.bridge import (
    MeasuredSyndromeReference,
    evaluate_logical_outcome,
    run_pcm_bridge,
    validate_bridge_receipt,
)
from qec.experiments.acoustic.channel import ChannelConfig, PCM, encode_symbols
from qec.experiments.acoustic.receiver import ReceiverPolicy, receive_pcm
from qec.sonify.canonical import canonical_json, canonical_sha256


def _model(alphabet):
    if alphabet == 3:
        return cyclic_five_qutrit_code(), qutrit_errors
    return packed_five_ququart_code(), ququart_errors


def _symbols(code, error, alphabet):
    actual = code.syndrome(error)
    return actual if alphabet == 3 else pack_binary_syndrome(actual)


def _reference(alphabet, measured=(0, 0, 0, 0), source="fixture-0"):
    return MeasuredSyndromeReference(alphabet, tuple(measured), "independent-test-fixture", source)


def _inputs(pcm, config, reference, policy=ReceiverPolicy()):
    observation = receive_pcm(pcm, config, 4, policy=policy)
    return {"pcm": pcm, "config": config, "observation": observation,
            "reference": reference, "expected_reference_sha256": reference.as_dict()["sha256"],
            "policy": policy}


def _rehash(value):
    result = deepcopy(value)
    result["sha256"] = canonical_sha256({key: val for key, val in result.items() if key != "sha256"})
    return result


def _no_decode(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unverified observation reached the exact decoder")
    monkeypatch.setattr(bridge.QutritDecoder, "decode", forbidden)
    monkeypatch.setattr(bridge.QuquartDecoder, "decode", forbidden)


@pytest.mark.parametrize(("alphabet", "expected_count"), [(3, 41), (4, 76)])
def test_all_bounded_five_site_errors_from_noisy_pcm(alphabet, expected_count):
    code, errors = _model(alphabet)
    # Fixed 48-sample blocks reduce test cost without changing the alphabet.
    config = ChannelConfig(alphabet_size=alphabet, cycles=1)
    count = 0
    for weight in (0, 1):
        for error in errors(5, weight):
            measured = _symbols(code, error, alphabet)
            encoded_sound = encode_symbols(measured, config)
            # Real sample perturbations, not ideal harmonic-domain phasors.
            received_waveform = PCM(tuple(s + (i % 3 - 1) for i, s in enumerate(encoded_sound.samples)), config.sample_rate)
            assert received_waveform != encoded_sound
            reference = _reference(alphabet, measured, f"bounded-error-{count}")
            inputs = _inputs(received_waveform, config, reference)
            result = run_pcm_bridge(**inputs)
            assert result["candidate_accepted"]
            assert result["verified_recovery"]
            assert not result["candidate_false_accept"]
            assert result["decoder_decision"]["attempted"]
            assert result["decoder_decision"]["accepted"]
            expected_syndrome = measured if alphabet == 3 else unpack_ququart_syndrome(measured)
            assert result["decoder_decision"]["syndrome"] == list(expected_syndrome)
            outcome = evaluate_logical_outcome(result, error, **inputs)
            assert outcome["measured_matches_ideal"]
            assert outcome["logical_outcome"]["success"]
            assert outcome["logical_outcome"]["residual_is_stabilizer"]
            assert outcome["claim_boundary"]["quantum_state_evolution"] is False
            count += 1
    assert count == expected_count


@pytest.mark.parametrize("alphabet", [3, 4])
def test_coherent_phase_shift_is_raw_false_accept_then_reference_veto(monkeypatch, alphabet):
    config = ChannelConfig(alphabet_size=alphabet)
    measured = (0, 1, 2, 0)
    clean = encode_symbols(measured, config)
    shift = 48 // alphabet
    shifted = []
    for offset in range(0, len(clean.samples), config.block_samples):
        block = clean.samples[offset:offset + config.block_samples]
        shifted.extend(block[shift:] + block[:shift])
    pcm = PCM(tuple(shifted), config.sample_rate)
    inputs = _inputs(pcm, config, _reference(alphabet, measured))
    assert inputs["observation"]["accepted"]
    assert inputs["observation"]["estimated_syndrome"] == [(s + 1) % alphabet for s in measured]
    _no_decode(monkeypatch)
    result = run_pcm_bridge(**inputs)
    assert result["candidate_accepted"] and result["candidate_false_accept"]
    assert not result["verified_recovery"]
    assert result["decoder_decision"]["reason"] == "reference_mismatch"
    assert not result["decoder_decision"]["attempted"]
    assert result["decoder_decision"]["correction"] is None


@pytest.mark.parametrize("alphabet", [3, 4])
def test_no_tone_is_rejected_before_decoder(monkeypatch, alphabet):
    config = ChannelConfig(alphabet_size=alphabet)
    pcm = PCM((0,) * (4 * config.repetitions * config.block_samples), config.sample_rate)
    inputs = _inputs(pcm, config, _reference(alphabet))
    assert not inputs["observation"]["accepted"]
    _no_decode(monkeypatch)
    result = run_pcm_bridge(**inputs)
    assert not result["candidate_accepted"]
    assert not result["candidate_false_accept"]
    assert result["decoder_decision"]["reason"] == "receiver_rejected"
    assert not result["decoder_decision"]["attempted"]


@pytest.mark.parametrize("alphabet", [3, 4])
def test_unknown_but_verified_syndrome_fails_closed(alphabet):
    code, errors = _model(alphabet)
    bounded = {_symbols(code, error, alphabet) for weight in (0, 1) for error in errors(5, weight)}
    unknown = next(s for s in product(range(alphabet), repeat=4) if s not in bounded)
    config = ChannelConfig(alphabet_size=alphabet)
    inputs = _inputs(encode_symbols(unknown, config), config, _reference(alphabet, unknown))
    result = run_pcm_bridge(**inputs)
    assert result["verified_recovery"]
    assert result["decoder_decision"]["attempted"]
    assert not result["decoder_decision"]["accepted"]
    assert result["decoder_decision"]["correction"] is None
    assert result["decoder_decision"]["reason"] == "outside_bounded_decoder_table"


@pytest.mark.parametrize("alphabet", [3, 4])
def test_resealed_forged_receiver_receipt_never_reaches_decoder(monkeypatch, alphabet):
    config = ChannelConfig(alphabet_size=alphabet)
    inputs = _inputs(encode_symbols((1, 1, 1, 1), config), config, _reference(alphabet))
    forged = deepcopy(inputs["observation"])
    forged["accepted"] = True
    forged["estimated_syndrome"] = [0, 0, 0, 0]
    inputs["observation"] = _rehash(forged)
    _no_decode(monkeypatch)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_pcm_bridge(**inputs)


@pytest.mark.parametrize("kind", ["missing", "serialized", "foreign-source", "foreign-symbols", "foreign-authority", "wrong-alphabet"])
def test_missing_or_foreign_reference_rejected_before_decoder(monkeypatch, kind):
    config = ChannelConfig()
    ref = _reference(3)
    inputs = _inputs(encode_symbols(ref.measured_syndrome, config), config, ref)
    changes = {"missing": None, "serialized": ref.as_dict(),
               "foreign-source": replace(ref, source_id="different-fixture"),
               "foreign-symbols": replace(ref, measured_syndrome=(1, 1, 1, 1)),
               "foreign-authority": replace(ref, authority="different-authority"),
               "wrong-alphabet": replace(ref, alphabet_size=4)}
    inputs["reference"] = changes[kind]
    _no_decode(monkeypatch)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_pcm_bridge(**inputs)


@pytest.mark.parametrize("field", ["independent_reference", "expected_reference_sha256", "decoder_decision", "received_waveform_sha256", "estimated_syndrome"])
def test_resealed_bridge_mutations_rejected_against_external_inputs(field):
    config = ChannelConfig()
    ref = _reference(3)
    inputs = _inputs(encode_symbols(ref.measured_syndrome, config), config, ref)
    result = run_pcm_bridge(**inputs)
    forged = deepcopy(result)
    if field == "independent_reference":
        forged[field] = replace(ref, source_id="foreign").as_dict()
    elif field in ("expected_reference_sha256", "received_waveform_sha256"):
        forged[field] = "0" * 64
    elif field == "decoder_decision":
        forged[field]["correction"]["x"][0] = 1
    else:
        forged[field] = [1, 0, 0, 0]
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_bridge_receipt(_rehash(forged), **inputs)


@pytest.mark.parametrize("alphabet", [3, 4])
def test_self_consistent_false_authority_is_not_physical_truth(alphabet):
    code, errors = _model(alphabet)
    error = next(errors(5, 1))
    # The caller declares a zero measured syndrome, although the fixture's
    # ideal syndrome is nonzero. A hash pin cannot detect a false authority.
    config = ChannelConfig(alphabet_size=alphabet)
    ref = _reference(alphabet)
    inputs = _inputs(encode_symbols(ref.measured_syndrome, config), config, ref)
    result = run_pcm_bridge(**inputs)
    assert result["verified_recovery"]
    assert result["decoder_decision"]["accepted"]
    outcome = evaluate_logical_outcome(result, error, **inputs)
    assert not outcome["measured_matches_ideal"]
    assert outcome["ideal_syndrome"] != outcome["measured_syndrome"]
    assert not outcome["logical_outcome"]["success"]
    assert outcome["logical_outcome"]["status"] == "exact_model_failure"
    assert outcome["claim_boundary"]["source_hashes_authenticate_physical_provenance"] is False


@pytest.mark.parametrize("alphabet", [3, 4])
def test_zero_syndrome_logical_operator_is_not_success(alphabet):
    code, errors = _model(alphabet)
    logical = next(error for error in errors(5, 3)
                   if not any(code.syndrome(error)) and not code.is_stabilizer(error))
    config = ChannelConfig(alphabet_size=alphabet)
    ref = _reference(alphabet)
    inputs = _inputs(encode_symbols(ref.measured_syndrome, config), config, ref)
    result = run_pcm_bridge(**inputs)
    outcome = evaluate_logical_outcome(result, logical, **inputs)
    assert outcome["ideal_syndrome"] == [0, 0, 0, 0]
    assert outcome["measured_matches_ideal"]
    assert not outcome["logical_outcome"]["residual_is_stabilizer"]
    assert not outcome["logical_outcome"]["success"]


def test_packed_ququart_decodes_binary_lanes(monkeypatch):
    config = ChannelConfig(alphabet_size=4)
    code, errors = _model(4)
    error = next(e for e in errors(5, 1) if any(e.lane0_x) and any(e.lane1_z))
    symbols = _symbols(code, error, 4)
    inputs = _inputs(encode_symbols(symbols, config), config, _reference(4, symbols))
    calls = []
    original = bridge.QuquartDecoder.decode
    def record(decoder, syndrome):
        calls.append(syndrome)
        return original(decoder, syndrome)
    monkeypatch.setattr(bridge.QuquartDecoder, "decode", record)
    result = run_pcm_bridge(**inputs)
    assert calls == [unpack_ququart_syndrome(symbols)]
    assert len(calls[0]) == 8 and set(calls[0]) <= {0, 1}
    assert result["decoder_decision"]["correction"]["algebra"] == "packed_GF(2)_x_GF(2)"


def test_replay_is_canonical_and_reference_round_trips():
    config = ChannelConfig()
    reference = _reference(3)
    assert MeasuredSyndromeReference.from_dict(reference.as_dict()) == reference
    inputs = _inputs(encode_symbols(reference.measured_syndrome, config), config, reference)
    first, second = run_pcm_bridge(**inputs), run_pcm_bridge(**inputs)
    assert canonical_json(first) == canonical_json(second)
    assert validate_bridge_receipt(first, **inputs) == first


@pytest.mark.parametrize("symbols", [(0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 3), (True, 0, 0, 0), (0.0, 0, 0, 0)])
def test_reference_rejects_wrong_width_or_symbols(symbols):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        _reference(3, symbols)


@pytest.mark.parametrize("label", ["", " ", "bad\nsource", "nonascii-λ", "a" * 161, 1])
def test_reference_rejects_invalid_authority_or_source(label):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        MeasuredSyndromeReference(3, (0, 0, 0, 0), label, "source")
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        MeasuredSyndromeReference(3, (0, 0, 0, 0), "authority", label)


@pytest.mark.parametrize("changed", ["pcm", "config", "policy"])
def test_source_or_policy_substitution_rejected_before_decoder(monkeypatch, changed):
    config = ChannelConfig()
    ref = _reference(3)
    inputs = _inputs(encode_symbols(ref.measured_syndrome, config), config, ref)
    if changed == "pcm":
        inputs["pcm"] = PCM(tuple(s + 1 for s in inputs["pcm"].samples), config.sample_rate)
    elif changed == "config":
        inputs["config"] = replace(config, amplitude=config.amplitude + 1)
    else:
        inputs["policy"] = ReceiverPolicy(fec="unanimous")
    _no_decode(monkeypatch)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_pcm_bridge(**inputs)


@pytest.mark.parametrize("malformed", [None, [], {}, {"accepted": True}, {"accepted": True, "estimated_syndrome": [0, 0, 0, 0]}])
def test_malformed_observation_cannot_authorize_decode(monkeypatch, malformed):
    config = ChannelConfig()
    ref = _reference(3)
    inputs = _inputs(encode_symbols(ref.measured_syndrome, config), config, ref)
    inputs["observation"] = malformed
    _no_decode(monkeypatch)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_pcm_bridge(**inputs)


def test_receiver_and_bridge_signatures_have_no_hidden_truth_inputs():
    from inspect import signature
    forbidden = {"error", "ideal_syndrome", "measured_syndrome", "expected_correction", "reference"}
    assert not forbidden.intersection(signature(receive_pcm).parameters)
    assert not (forbidden - {"reference"}).intersection(signature(run_pcm_bridge).parameters)
    assert "error" in signature(evaluate_logical_outcome).parameters


def test_reference_frozen_instance_corruption_is_revalidated(monkeypatch):
    config = ChannelConfig()
    ref = _reference(3)
    object.__setattr__(ref, "measured_syndrome", (True, 0, 0, 0))
    inputs = _inputs(encode_symbols((0, 0, 0, 0), config), config, ref)
    _no_decode(monkeypatch)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_pcm_bridge(**inputs)


@pytest.mark.parametrize("field", ["ideal", "measured", "correction", "residual", "status", "success", "stabilizer", "claim", "error", "bridge"])
def test_logical_outcome_replays_all_fields_against_external_fixture(field):
    config = ChannelConfig()
    ref = _reference(3)
    inputs = _inputs(encode_symbols(ref.measured_syndrome, config), config, ref)
    bridge_receipt = run_pcm_bridge(**inputs)
    error = Pauli((0,) * 5, (0,) * 5)
    outcome = evaluate_logical_outcome(bridge_receipt, error, **inputs)
    assert bridge.validate_logical_outcome(outcome, bridge_receipt=bridge_receipt, error=error, **inputs) == outcome
    forged = deepcopy(outcome)
    if field in ("ideal", "measured"):
        forged[field + "_syndrome"][0] = 1
    elif field == "correction":
        forged["decoder_decision"]["correction"]["x"][0] = 1
    elif field == "residual":
        forged["logical_outcome"]["residual"]["x"][0] = 1
    elif field == "status":
        forged["logical_outcome"]["status"] = "physical_quantum_protection"
    elif field == "success":
        forged["logical_outcome"]["success"] = False
    elif field == "stabilizer":
        forged["logical_outcome"]["residual_is_stabilizer"] = False
    elif field == "claim":
        forged["claim_boundary"]["physical_quantum_protection"] = True
    elif field == "error":
        forged["error"]["x"][0] = 1
    else:
        forged["bridge_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        bridge.validate_logical_outcome(_rehash(forged), bridge_receipt=bridge_receipt, error=error, **inputs)


def test_bridge_reference_and_outcome_receipts_use_v177_3():
    config = ChannelConfig()
    reference = _reference(3)
    inputs = _inputs(encode_symbols(reference.measured_syndrome, config), config, reference)
    result = run_pcm_bridge(**inputs)
    error = Pauli((0,) * 5, (0,) * 5)
    outcome = evaluate_logical_outcome(result, error, **inputs)
    assert reference.as_dict()["contract_version"] == "177.3"
    assert result["contract_version"] == "177.3"
    assert outcome["contract_version"] == "177.3"
    assert validate_bridge_receipt(result, **inputs)["contract_version"] == "177.3"
    assert bridge.validate_logical_outcome(outcome, bridge_receipt=result, error=error, **inputs)["contract_version"] == "177.3"
    # Existing channel contracts remain their own versioned objects.
    assert config.as_dict()["contract_version"] == "177.0"


@pytest.mark.parametrize("level", ["quantum_simulation_verified", "physical_quantum_verified"])
def test_public_validators_reject_evidence_level_promotion(level):
    config = ChannelConfig()
    reference = _reference(3)
    inputs = _inputs(encode_symbols(reference.measured_syndrome, config), config, reference)
    result = run_pcm_bridge(**inputs)
    error = Pauli((0,) * 5, (0,) * 5)
    outcome = evaluate_logical_outcome(result, error, **inputs)
    for actual in (result, outcome):
        assert actual["claim_boundary"] == dict(bridge.CLAIM_BOUNDARY)
        assert actual["claim_boundary"]["quantum_state_evolution"] is False
        assert actual["claim_boundary"]["physical_quantum_protection"] is False
    forged_bridge = deepcopy(result)
    forged_bridge["claim_boundary"]["evidence_level"] = level
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_bridge_receipt(_rehash(forged_bridge), **inputs)
    forged_outcome = deepcopy(outcome)
    forged_outcome["claim_boundary"]["evidence_level"] = level
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        bridge.validate_logical_outcome(_rehash(forged_outcome), bridge_receipt=result, error=error, **inputs)


def test_corrupted_qutrit_multiple_of_three_cannot_be_graded_or_replayed():
    """Regression: GF(3) normalization previously hid the invalid coordinate."""
    config = ChannelConfig()
    reference = _reference(3)
    inputs = _inputs(encode_symbols(reference.measured_syndrome, config), config, reference)
    bridge_receipt = run_pcm_bridge(**inputs)
    error = Pauli((0,) * 5, (0,) * 5)
    outcome = evaluate_logical_outcome(bridge_receipt, error, **inputs)
    object.__setattr__(error, "x", (3, 0, 0, 0, 0))
    forged = deepcopy(outcome)
    forged["error"]["x"] = [3, 0, 0, 0, 0]
    # Before the fix, both the grade and this self-consistent rehashed result
    # were accepted with success=True and the invalid error coordinate intact.
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        evaluate_logical_outcome(bridge_receipt, error, **inputs)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        bridge.validate_logical_outcome(_rehash(forged), bridge_receipt=bridge_receipt, error=error, **inputs)


@pytest.mark.parametrize(("alphabet", "field"), [
    (3, "x"), (3, "z"),
    (4, "lane0_x"), (4, "lane0_z"), (4, "lane1_x"), (4, "lane1_z"),
])
@pytest.mark.parametrize("mutation", [
    "above", "negative", "bool_true", "bool_false", "float", "numpy_integer", "string_coordinate", "null_coordinate",
    "short", "long", "empty", "mutable_list", "null_lane", "string_lane", "missing_lane",
])
def test_all_corrupted_pauli_coordinates_reject_in_evaluator_and_validator(alphabet, field, mutation):
    config = ChannelConfig(alphabet_size=alphabet)
    reference = _reference(alphabet)
    inputs = _inputs(encode_symbols(reference.measured_syndrome, config), config, reference)
    bridge_receipt = run_pcm_bridge(**inputs)
    error = Pauli((0,) * 5, (0,) * 5) if alphabet == 3 else PackedPauli(*((0,) * 5,) * 4)
    outcome = evaluate_logical_outcome(bridge_receipt, error, **inputs)
    mutations = {
        "above": (3 if alphabet == 3 else 2, 0, 0, 0, 0),
        "negative": (-1, 0, 0, 0, 0),
        "bool_true": (True, 0, 0, 0, 0), "bool_false": (False, 0, 0, 0, 0),
        "float": (0.0, 0, 0, 0, 0), "numpy_integer": (np.int64(0), 0, 0, 0, 0),
        "string_coordinate": ("0", 0, 0, 0, 0),
        "null_coordinate": (None, 0, 0, 0, 0),
        "short": (0,) * 4, "long": (0,) * 6, "empty": (),
        "mutable_list": [0] * 5, "null_lane": None, "string_lane": "00000",
    }
    if mutation == "missing_lane":
        object.__delattr__(error, field)
    else:
        object.__setattr__(error, field, mutations[mutation])
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        evaluate_logical_outcome(bridge_receipt, error, **inputs)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        bridge.validate_logical_outcome(outcome, bridge_receipt=bridge_receipt, error=error, **inputs)
