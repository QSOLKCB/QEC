# SPDX-License-Identifier: MPL-2.0
"""External SIP/RTP/SDP observations, captured offline without network activation.

A capture's hash is caller-declared provenance, never an authentication claim.
This module does not import a protocol stack, start a listener, or send traffic.
External observations cannot enter PacketInput or alter native execution.
"""
from dataclasses import dataclass
from .core import _exact, _hash, _int, _name, _read, _seal, _seq, _text, validate_packet_receipt

MAX_OBSERVATIONS = 256


@dataclass(frozen=True)
class SoftswitchObservation:
    logical_index: int
    protocol: str
    source_label: str
    capture_sha256: str
    observation: str

    def __post_init__(self):
        _int(self.logical_index, MAX_OBSERVATIONS - 1)
        if type(self.protocol) is not str or self.protocol not in ("sip", "rtp", "sdp"):
            raise ValueError("INVALID_INPUT")
        _name(self.source_label)
        _hash(self.capture_sha256)
        _text(self.observation, 256)

    def as_dict(self):
        return _seal("softswitch-observation", logical_index=self.logical_index,
                     protocol=self.protocol, source_label=self.source_label,
                     capture_sha256=self.capture_sha256, observation=self.observation,
                     external=True, authoritative=False, captured_offline=True,
                     capture_identity_is_caller_declared=True)

    @classmethod
    def from_dict(cls, value):
        return _read(cls, value, ("logical_index", "protocol", "source_label", "capture_sha256", "observation"))


def create_softswitch_adapter_receipt(observations, *, packet_receipt=None) -> dict:
    """Bind a bounded ordered capture list, optionally referencing verified software."""
    observations = _seq(observations, MAX_OBSERVATIONS)
    if any(type(o) is not SoftswitchObservation for o in observations):
        raise ValueError("INVALID_INPUT")
    if tuple(o.logical_index for o in observations) != tuple(range(len(observations))):
        raise ValueError("INVALID_INPUT")
    if packet_receipt is not None:
        validate_packet_receipt(packet_receipt)
    return _seal("softswitch-adapter-receipt", observations=[o.as_dict() for o in observations],
                 referenced_software_receipt_sha256=packet_receipt["sha256"] if packet_receipt is not None else None,
                 external=True, authoritative=False, capture_authenticated=False,
                 receipt_proves="offline_observation_record_integrity_only",
                 network_activation_present=False, protocol_conformance_verified=False,
                 packet_outcome_override_permitted=False)


def validate_softswitch_adapter_receipt(receipt, *, packet_receipt=None, expected_receipt_sha256=None) -> dict:
    try:
        if type(receipt) is not dict:
            raise ValueError("INVALID_INPUT")
        observations = tuple(SoftswitchObservation.from_dict(o) for o in _seq(receipt["observations"], MAX_OBSERVATIONS))
        # A reference to native software evidence requires that full evidence at validation.
        expected = create_softswitch_adapter_receipt(observations, packet_receipt=packet_receipt)
        _exact(receipt, expected)
        if expected_receipt_sha256 is not None:
            _hash(expected_receipt_sha256)
            if expected_receipt_sha256 != expected["sha256"]:
                raise ValueError("INVALID_INPUT")
        return _seal("softswitch-adapter-validation", softswitch_adapter_receipt_sha256=expected["sha256"],
                     offline_record_replay_verified=True, external=True, authoritative=False,
                     referenced_software_receipt_verified=packet_receipt is not None,
                     external_receipt_binding_verified=expected_receipt_sha256 is not None, all_passed=True)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("INVALID_INPUT") from exc
