# SPDX-License-Identifier: MPL-2.0
"""Independently reference-gated PCM recovery into unchanged exact decoders.

Only the fixture evaluator below sees a Pauli error or computes its ideal
syndrome. The bridge replays a PCM-only receiver, checks the recovered measured
syndrome against caller-declared test authority, then invokes exact decoding.
A separately supplied reference digest pins fixture identity, not real-world
truth. This module neither evolves a quantum state nor establishes physical
quantum protection.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from qec.decoder.qutrit.codes import cyclic_five_qutrit_code
from qec.decoder.qutrit.exact import ExactDecoder as QutritDecoder
from qec.decoder.qutrit.exact import UncorrectableSyndrome as QutritUncorrectable
from qec.decoder.qutrit.stabilizer import Pauli
from qec.decoder.ququart.codes import packed_five_ququart_code
from qec.decoder.ququart.exact import ExactDecoder as QuquartDecoder
from qec.decoder.ququart.exact import UncorrectableSyndrome as QuquartUncorrectable
from qec.decoder.ququart.packed import PackedPauli
from qec.decoder.ququart.syndrome import pack_binary_syndrome, unpack_ququart_syndrome

from ._common import CLAIM_BOUNDARY, digest, exact, integer, read, seal, sequence
from .channel import ChannelConfig, PCM
from .receiver import ReceiverPolicy, validate_observation


def _seal(name: str, **fields) -> dict:
    return seal(name, contract_version="177.3", **fields)


@dataclass(frozen=True)
class MeasuredSyndromeReference:
    """Four measured symbols declared authoritative by an independent caller.

    ``authority`` and ``source_id`` are test-fixture labels, not authenticated
    identities. The expected digest must come from outside the artifact under
    validation. Neither a digest nor these labels establish physical provenance.
    """

    alphabet_size: int
    measured_syndrome: tuple[int, ...]
    authority: str
    source_id: str

    def __post_init__(self):
        integer(self.alphabet_size, 4, 3)
        symbols = sequence(self.measured_syndrome, 4)
        if len(symbols) != 4:
            raise ValueError("INVALID_INPUT")
        for symbol in symbols:
            integer(symbol, self.alphabet_size - 1)
        for label in (self.authority, self.source_id):
            if type(label) is not str or not 1 <= len(label) <= 160 or not label.isascii() or not all(32 <= ord(c) < 127 for c in label) or not label.strip():
                raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "measured_syndrome", symbols)

    def as_dict(self):
        return _seal("measured-syndrome-reference", alphabet_size=self.alphabet_size,
                    measured_syndrome=list(self.measured_syndrome), authority=self.authority,
                    source_id=self.source_id, authority_scope="caller_declared_test_fixture_only")

    @classmethod
    def from_dict(cls, value):
        return read(cls, value, ("alphabet_size", "measured_syndrome", "authority", "source_id"))


def _reference(reference, config, expected_reference_sha256):
    if type(reference) is not MeasuredSyndromeReference or type(config) is not ChannelConfig:
        raise ValueError("INVALID_INPUT")
    digest(expected_reference_sha256)
    if reference.alphabet_size != config.alphabet_size or reference.as_dict()["sha256"] != expected_reference_sha256:
        raise ValueError("INVALID_INPUT")
    return MeasuredSyndromeReference.from_dict(reference.as_dict())


def _code(alphabet_size):
    return cyclic_five_qutrit_code() if alphabet_size == 3 else packed_five_ququart_code()


def _pauli_dict(pauli):
    if type(pauli) is Pauli:
        return {"algebra": "GF(3)", "x": list(pauli.x), "z": list(pauli.z)}
    if type(pauli) is PackedPauli:
        return {"algebra": "packed_GF(2)_x_GF(2)", "lane0_x": list(pauli.lane0_x),
                "lane0_z": list(pauli.lane0_z), "lane1_x": list(pauli.lane1_x),
                "lane1_z": list(pauli.lane1_z)}
    raise ValueError("INVALID_INPUT")


def _pauli_from_dict(value, alphabet_size):
    # Only called after the complete bridge receipt has been replay-validated.
    if alphabet_size == 3:
        return Pauli(tuple(value["x"]), tuple(value["z"]))
    return PackedPauli(*(tuple(value[key]) for key in ("lane0_x", "lane0_z", "lane1_x", "lane1_z")))


def run_pcm_bridge(
    pcm: PCM,
    config: ChannelConfig,
    observation: dict,
    reference: MeasuredSyndromeReference,
    *,
    expected_reference_sha256: str,
    policy: ReceiverPolicy = ReceiverPolicy(),
) -> dict:
    """Verify PCM recovery and independent reference before any decoder action.

    Inputs contain no hidden error, ideal syndrome, or expected correction.
    The caller must supply the separately pinned measured-syndrome reference.
    Arbitrary ``accepted``/hash fields are insufficient: receiver evidence is
    recomputed from ``pcm`` with the explicit configuration and policy.
    """
    reference = _reference(reference, config, expected_reference_sha256)
    try:
        recovered = deepcopy(observation)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc
    validation = validate_observation(recovered, expected_pcm=pcm, config=config,
                                      symbol_count=4, policy=policy)
    estimated = recovered["estimated_syndrome"]
    candidate_accepted = recovered["accepted"]
    reference_matches = candidate_accepted and estimated == list(reference.measured_syndrome)
    code = _code(config.alphabet_size)
    decision = {"code": code.name, "max_error_weight": 1, "attempted": False,
                "accepted": False, "syndrome": None, "correction": None,
                "reason": "receiver_rejected" if not candidate_accepted else "reference_mismatch"}
    if reference_matches:
        # Ququart acoustic digits label paired binary checks. They are never
        # interpreted as exponents in a mod-4 Pauli algebra.
        syndrome = (tuple(estimated) if config.alphabet_size == 3
                    else unpack_ququart_syndrome(estimated))
        decoder = QutritDecoder(code) if config.alphabet_size == 3 else QuquartDecoder(code)
        decision.update(attempted=True, syndrome=list(syndrome))
        try:
            correction = decoder.decode(syndrome)
        except (QutritUncorrectable, QuquartUncorrectable):
            decision["reason"] = "outside_bounded_decoder_table"
        else:
            decision.update(accepted=True, correction=_pauli_dict(correction), reason="exact_correction")
    return _seal("syndrome-bridge", config=config.as_dict(), receiver_observation=recovered,
                observation_validation=validation,
                received_waveform_sha256=pcm.as_dict()["sha256"],
                independent_reference=reference.as_dict(),
                expected_reference_sha256=expected_reference_sha256,
                measured_syndrome=list(reference.measured_syndrome), estimated_syndrome=estimated,
                candidate_accepted=candidate_accepted,
                candidate_false_accept=candidate_accepted and not reference_matches,
                verified_recovery=reference_matches, decoder_decision=decision,
                claim_boundary=dict(CLAIM_BOUNDARY))


def validate_bridge_receipt(
    receipt: dict,
    *,
    pcm: PCM,
    config: ChannelConfig,
    observation: dict,
    reference: MeasuredSyndromeReference,
    expected_reference_sha256: str,
    policy: ReceiverPolicy = ReceiverPolicy(),
) -> dict:
    """Recompute the full result against external inputs, not embedded hashes."""
    expected = run_pcm_bridge(pcm, config, observation, reference,
                              expected_reference_sha256=expected_reference_sha256, policy=policy)
    exact(receipt, expected)
    return expected


def evaluate_logical_outcome(
    receipt: dict,
    error: Pauli | PackedPauli,
    *,
    pcm: PCM,
    config: ChannelConfig,
    observation: dict,
    reference: MeasuredSyndromeReference,
    expected_reference_sha256: str,
    policy: ReceiverPolicy = ReceiverPolicy(),
) -> dict:
    """Grade a verified bridge in a Pauli/stabilizer fixture after decoding.

    Only this evaluator gets hidden error truth. It never passes it to the
    receiver or bridge. ``success`` requires a decoder-accepted correction and
    stabilizer membership of the residual, not merely its zero syndrome.
    A bad measured reference can recover faithfully yet fail this model test.
    """
    bridge = validate_bridge_receipt(receipt, pcm=pcm, config=config, observation=observation,
                                     reference=reference, expected_reference_sha256=expected_reference_sha256,
                                     policy=policy)
    expected_type = Pauli if config.alphabet_size == 3 else PackedPauli
    if type(error) is not expected_type or error.width != 5:
        raise ValueError("INVALID_INPUT")
    code = _code(config.alphabet_size)
    ideal = code.syndrome(error)
    acoustic_ideal = ideal if config.alphabet_size == 3 else pack_binary_syndrome(ideal)
    decision = bridge["decoder_decision"]
    residual = error
    if decision["accepted"]:
        residual = error.compose(_pauli_from_dict(decision["correction"], config.alphabet_size))
    stabilizer = code.is_stabilizer(residual)
    success = decision["accepted"] and stabilizer
    logical_outcome = {"success": success, "residual": _pauli_dict(residual),
                       "residual_is_stabilizer": stabilizer,
                       "status": ("verified_exact_model_correction" if success else
                                  "exact_model_failure" if decision["accepted"] else "no_correction")}
    return _seal("logical-fixture-evaluation", bridge_sha256=bridge["sha256"],
                error=_pauli_dict(error), ideal_syndrome=list(acoustic_ideal),
                ideal_decoder_syndrome=list(ideal), measured_syndrome=list(reference.measured_syndrome),
                measured_matches_ideal=list(acoustic_ideal) == list(reference.measured_syndrome),
                estimated_syndrome=bridge["estimated_syndrome"], decoder_decision=decision,
                logical_outcome=logical_outcome, claim_boundary=dict(CLAIM_BOUNDARY))


def validate_logical_outcome(
    receipt: dict,
    *,
    bridge_receipt: dict,
    error: Pauli | PackedPauli,
    pcm: PCM,
    config: ChannelConfig,
    observation: dict,
    reference: MeasuredSyndromeReference,
    expected_reference_sha256: str,
    policy: ReceiverPolicy = ReceiverPolicy(),
) -> dict:
    """Replay the model-only outcome against independently supplied inputs."""
    expected = evaluate_logical_outcome(
        bridge_receipt, error, pcm=pcm, config=config, observation=observation,
        reference=reference, expected_reference_sha256=expected_reference_sha256,
        policy=policy,
    )
    exact(receipt, expected)
    return expected
