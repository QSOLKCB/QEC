from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.layer_spec_contract import _ensure_json_safe

QAM_COMPATIBILITY_VERSION = "v153.4"
MAX_QAM_FEATURES = 128
MAX_QAM_LINEAGE_REFERENCES = 32
MAX_QAM_COMPATIBILITY_NOTES = 128

_ALLOWED_FEATURE_STATUSES = {"SUPPORTED_METADATA_ONLY", "DEFERRED", "UNSUPPORTED", "REJECTED"}
_ALLOWED_VALIDATION_STATUS = {"VALIDATED_METADATA_ONLY", "VALIDATED_WITH_DEFERRED_FEATURES", "REJECTED"}
_ALLOWED_VALIDATION_REASON = {
    "ALL_FEATURES_METADATA_ONLY",
    "DEFERRED_OR_UNSUPPORTED_FEATURES_PRESENT",
    "REJECTED_FEATURES_PRESENT",
}


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(value[k]) for k in sorted(value)})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_thaw(value[k]) for k in sorted(value)}
    if isinstance(value, tuple):
        return [_deep_thaw(v) for v in value]
    if isinstance(value, list):
        return [_deep_thaw(v) for v in value]
    return value


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return _deep_freeze(dict(mapping))


@dataclass(frozen=True)
class QAMCompatibilityProfile:
    profile_id: str
    profile_version: str
    qam_version: str
    source_name: str
    source_orcid: str
    lineage_references: tuple[Mapping[str, Any], ...]
    compatibility_features: tuple[Mapping[str, Any], ...]
    compatibility_notes: tuple[Any, ...]
    profile_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineage_references", tuple(_freeze_mapping(x) for x in self.lineage_references))
        object.__setattr__(self, "compatibility_features", tuple(_freeze_mapping(x) for x in self.compatibility_features))
        object.__setattr__(self, "compatibility_notes", tuple(_deep_freeze(x) for x in self.compatibility_notes))

        if not all((self.profile_id, self.profile_version, self.qam_version, self.source_name)):
            raise ValueError("INVALID_INPUT")
        if self.source_orcid is not None and (not isinstance(self.source_orcid, str) or not self.source_orcid):
            raise ValueError("INVALID_INPUT")
        if len(self.compatibility_features) > MAX_QAM_FEATURES:
            raise ValueError("INVALID_INPUT")
        if len(self.lineage_references) > MAX_QAM_LINEAGE_REFERENCES:
            raise ValueError("INVALID_INPUT")
        if len(self.compatibility_notes) > MAX_QAM_COMPATIBILITY_NOTES:
            raise ValueError("INVALID_INPUT")

        feature_ids: list[str] = []
        for feature in self.compatibility_features:
            feature_id = feature.get("feature_id")
            status = feature.get("compatibility_status")
            if not isinstance(feature_id, str) or not feature_id:
                raise ValueError("INVALID_INPUT")
            if status not in _ALLOWED_FEATURE_STATUSES:
                raise ValueError("INVALID_INPUT")
            feature_ids.append(feature_id)
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("INVALID_INPUT")

        lineage_ids: list[str] = []
        for ref in self.lineage_references:
            ref_id = ref.get("reference_id")
            if not isinstance(ref_id, str) or not ref_id:
                raise ValueError("INVALID_INPUT")
            lineage_ids.append(ref_id)
        if len(set(lineage_ids)) != len(lineage_ids):
            raise ValueError("INVALID_INPUT")

        if tuple(sorted(self.compatibility_features, key=lambda x: (x["feature_id"], x.get("feature_name", ""), x.get("qec_target", "")))) != self.compatibility_features:
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.lineage_references, key=lambda x: (x["reference_id"], x.get("reference_uri", "")))) != self.lineage_references:
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.compatibility_notes, key=lambda x: str(_deep_thaw(x)))) != self.compatibility_notes:
            raise ValueError("INVALID_INPUT")

        _ensure_json_safe(self._canonical_payload())
        if self.profile_hash and self.profile_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "qam_version": self.qam_version,
            "source_name": self.source_name,
            "source_orcid": self.source_orcid,
            "lineage_references": [_deep_thaw(x) for x in self.lineage_references],
            "compatibility_features": [_deep_thaw(x) for x in self.compatibility_features],
            "compatibility_notes": [_deep_thaw(x) for x in self.compatibility_notes],
        }

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), profile_hash=self.profile_hash)


@dataclass(frozen=True)
class QAMSpecReceipt:
    profile_id: str
    profile_version: str
    qam_version: str
    source_name: str
    source_orcid: str
    profile_hash: str
    spec_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        _ensure_json_safe(self._canonical_payload())
        if self.spec_hash and self.spec_hash != self._spec_hash():
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _spec_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "qam_version": self.qam_version,
            "source_name": self.source_name,
            "source_orcid": self.source_orcid,
            "profile_hash": self.profile_hash,
        }

    def _spec_hash(self) -> str:
        return sha256_hex(self._spec_payload())

    def _canonical_payload(self) -> dict[str, Any]:
        return dict(self._spec_payload(), spec_hash=self.spec_hash)

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), receipt_hash=self.receipt_hash)


@dataclass(frozen=True)
class QAMCompatibilityValidationReceipt:
    profile_id: str
    profile_version: str
    qam_version: str
    qam_spec_receipt_hash: str
    profile_hash: str
    accepted_feature_ids: tuple[str, ...]
    deferred_feature_ids: tuple[str, ...]
    unsupported_feature_ids: tuple[str, ...]
    rejected_feature_ids: tuple[str, ...]
    validation_status: str
    validation_reason: str
    validation_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_feature_ids", tuple(self.accepted_feature_ids))
        object.__setattr__(self, "deferred_feature_ids", tuple(self.deferred_feature_ids))
        object.__setattr__(self, "unsupported_feature_ids", tuple(self.unsupported_feature_ids))
        object.__setattr__(self, "rejected_feature_ids", tuple(self.rejected_feature_ids))
        for bucket in (self.accepted_feature_ids, self.deferred_feature_ids, self.unsupported_feature_ids, self.rejected_feature_ids):
            if tuple(sorted(bucket)) != bucket:
                raise ValueError("INVALID_INPUT")
        all_ids = self.accepted_feature_ids + self.deferred_feature_ids + self.unsupported_feature_ids + self.rejected_feature_ids
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("INVALID_INPUT")
        if self.validation_status not in _ALLOWED_VALIDATION_STATUS or self.validation_reason not in _ALLOWED_VALIDATION_REASON:
            raise ValueError("INVALID_INPUT")
        _ensure_json_safe(self._canonical_payload())
        if self.validation_hash and self.validation_hash != self._validation_hash():
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _validation_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "qam_version": self.qam_version,
            "qam_spec_receipt_hash": self.qam_spec_receipt_hash,
            "profile_hash": self.profile_hash,
            "accepted_feature_ids": list(self.accepted_feature_ids),
            "deferred_feature_ids": list(self.deferred_feature_ids),
            "unsupported_feature_ids": list(self.unsupported_feature_ids),
            "rejected_feature_ids": list(self.rejected_feature_ids),
            "validation_status": self.validation_status,
            "validation_reason": self.validation_reason,
        }

    def _validation_hash(self) -> str:
        return sha256_hex(self._validation_payload())

    def _canonical_payload(self) -> dict[str, Any]:
        return dict(self._validation_payload(), validation_hash=self.validation_hash)

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), receipt_hash=self.receipt_hash)



def _qam_spec_payload(profile: QAMCompatibilityProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "qam_version": profile.qam_version,
        "source_name": profile.source_name,
        "source_orcid": profile.source_orcid,
        "profile_hash": profile.profile_hash,
    }


def _compute_spec_hash(spec_payload: Mapping[str, Any]) -> str:
    return sha256_hex(dict(spec_payload))


def _qam_validation_payload(
    profile: QAMCompatibilityProfile,
    spec_receipt_hash: str,
    accepted_feature_ids: tuple[str, ...],
    deferred_feature_ids: tuple[str, ...],
    unsupported_feature_ids: tuple[str, ...],
    rejected_feature_ids: tuple[str, ...],
    validation_status: str,
    validation_reason: str,
) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "qam_version": profile.qam_version,
        "qam_spec_receipt_hash": spec_receipt_hash,
        "profile_hash": profile.profile_hash,
        "accepted_feature_ids": list(accepted_feature_ids),
        "deferred_feature_ids": list(deferred_feature_ids),
        "unsupported_feature_ids": list(unsupported_feature_ids),
        "rejected_feature_ids": list(rejected_feature_ids),
        "validation_status": validation_status,
        "validation_reason": validation_reason,
    }


def _compute_validation_hash(validation_payload: Mapping[str, Any]) -> str:
    return sha256_hex(dict(validation_payload))


def _compute_receipt_hash(payload: Mapping[str, Any]) -> str:
    return sha256_hex(dict(payload))

def build_qam_compatibility_profile(
    profile_id: str,
    profile_version: str = QAM_COMPATIBILITY_VERSION,
    qam_version: str = "v4.1.0",
    source_name: str = "Marc Brendecke QAM",
    source_orcid: str = "https://orcid.org/0009-0009-4034-598X",
    lineage_references: tuple[Mapping[str, Any], ...] = (),
    compatibility_features: tuple[Mapping[str, Any], ...] = (),
    compatibility_notes: tuple[Any, ...] = (),
) -> QAMCompatibilityProfile:
    ordered_refs = tuple(sorted((dict(r) for r in lineage_references), key=lambda x: (x["reference_id"], x.get("reference_uri", ""))))
    ordered_features = tuple(sorted((dict(f) for f in compatibility_features), key=lambda x: (x["feature_id"], x.get("feature_name", ""), x.get("qec_target", ""))))
    ordered_notes = tuple(sorted((_deep_thaw(_deep_freeze(n)) for n in compatibility_notes), key=lambda x: str(x)))
    p = QAMCompatibilityProfile(profile_id, profile_version, qam_version, source_name, source_orcid, ordered_refs, ordered_features, ordered_notes, "")
    return QAMCompatibilityProfile(**{**p.__dict__, "profile_hash": p.stable_hash()})


def build_qam_spec_receipt(profile: QAMCompatibilityProfile) -> QAMSpecReceipt:
    if profile.profile_hash != profile.stable_hash():
        raise ValueError("INVALID_INPUT")
    spec_payload = _qam_spec_payload(profile)
    spec_hash = _compute_spec_hash(spec_payload)
    receipt_payload = dict(spec_payload, spec_hash=spec_hash)
    receipt_hash = _compute_receipt_hash(receipt_payload)
    return QAMSpecReceipt(
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        qam_version=profile.qam_version,
        source_name=profile.source_name,
        source_orcid=profile.source_orcid,
        profile_hash=profile.profile_hash,
        spec_hash=spec_hash,
        receipt_hash=receipt_hash,
    )


def build_qam_compatibility_validation_receipt(profile: QAMCompatibilityProfile, spec_receipt: QAMSpecReceipt) -> QAMCompatibilityValidationReceipt:
    if profile.profile_hash != profile.stable_hash():
        raise ValueError("INVALID_INPUT")
    if spec_receipt.profile_hash != profile.profile_hash or spec_receipt.spec_hash != spec_receipt._spec_hash() or spec_receipt.receipt_hash != spec_receipt.stable_hash():
        raise ValueError("INVALID_INPUT")
    accepted = sorted(f["feature_id"] for f in profile.compatibility_features if f["compatibility_status"] == "SUPPORTED_METADATA_ONLY")
    deferred = sorted(f["feature_id"] for f in profile.compatibility_features if f["compatibility_status"] == "DEFERRED")
    unsupported = sorted(f["feature_id"] for f in profile.compatibility_features if f["compatibility_status"] == "UNSUPPORTED")
    rejected = sorted(f["feature_id"] for f in profile.compatibility_features if f["compatibility_status"] == "REJECTED")
    if rejected:
        status, reason = "REJECTED", "REJECTED_FEATURES_PRESENT"
    elif deferred or unsupported:
        status, reason = "VALIDATED_WITH_DEFERRED_FEATURES", "DEFERRED_OR_UNSUPPORTED_FEATURES_PRESENT"
    else:
        status, reason = "VALIDATED_METADATA_ONLY", "ALL_FEATURES_METADATA_ONLY"
    accepted_ids = tuple(accepted)
    deferred_ids = tuple(deferred)
    unsupported_ids = tuple(unsupported)
    rejected_ids = tuple(rejected)
    validation_payload = _qam_validation_payload(
        profile,
        spec_receipt.receipt_hash,
        accepted_ids,
        deferred_ids,
        unsupported_ids,
        rejected_ids,
        status,
        reason,
    )
    validation_hash = _compute_validation_hash(validation_payload)
    receipt_payload = dict(validation_payload, validation_hash=validation_hash)
    receipt_hash = _compute_receipt_hash(receipt_payload)
    return QAMCompatibilityValidationReceipt(
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        qam_version=profile.qam_version,
        qam_spec_receipt_hash=spec_receipt.receipt_hash,
        profile_hash=profile.profile_hash,
        accepted_feature_ids=accepted_ids,
        deferred_feature_ids=deferred_ids,
        unsupported_feature_ids=unsupported_ids,
        rejected_feature_ids=rejected_ids,
        validation_status=status,
        validation_reason=reason,
        validation_hash=validation_hash,
        receipt_hash=receipt_hash,
    )


def validate_qam_compatibility_validation_receipt(
    validation_receipt: QAMCompatibilityValidationReceipt,
    profile: QAMCompatibilityProfile,
    spec_receipt: QAMSpecReceipt,
) -> None:
    if profile.profile_hash != profile.stable_hash():
        raise ValueError("INVALID_INPUT")
    if spec_receipt.profile_hash != profile.profile_hash:
        raise ValueError("INVALID_INPUT")
    if spec_receipt.spec_hash != spec_receipt._spec_hash() or spec_receipt.receipt_hash != spec_receipt.stable_hash():
        raise ValueError("INVALID_INPUT")
    expected = build_qam_compatibility_validation_receipt(profile, spec_receipt)
    if validation_receipt.to_dict() != expected.to_dict():
        raise ValueError("INVALID_INPUT")


for _name in ("apply", "execute", "run", "traverse", "pathfind", "resolve", "project", "readout", "search", "mask", "reduce", "collide", "shift", "hilber", "hilbert", "shell", "matrix", "markov"):
    for _cls in (QAMCompatibilityProfile, QAMSpecReceipt, QAMCompatibilityValidationReceipt):
        if hasattr(_cls, _name):
            raise RuntimeError("INVALID_STATE")


__all__ = [
    "QAM_COMPATIBILITY_VERSION",
    "MAX_QAM_FEATURES",
    "MAX_QAM_LINEAGE_REFERENCES",
    "MAX_QAM_COMPATIBILITY_NOTES",
    "QAMCompatibilityProfile",
    "QAMSpecReceipt",
    "QAMCompatibilityValidationReceipt",
    "build_qam_compatibility_profile",
    "build_qam_spec_receipt",
    "build_qam_compatibility_validation_receipt",
    "validate_qam_compatibility_validation_receipt",
]
