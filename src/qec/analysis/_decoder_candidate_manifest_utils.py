from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class DecoderCandidateManifestErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_DECODER_CANDIDATE = "INVALID_DECODER_CANDIDATE"
    INVALID_HASH = "INVALID_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"


class DecoderCandidateManifestError(ValueError):
    def __init__(self, code: DecoderCandidateManifestErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}:{detail}")


def _error(
    code: DecoderCandidateManifestErrorCode, detail: str
) -> DecoderCandidateManifestError:
    return DecoderCandidateManifestError(code, detail)


def _invalid_input(detail: str = "GENERIC") -> DecoderCandidateManifestError:
    return _error(DecoderCandidateManifestErrorCode.INVALID_INPUT, detail)


def _invalid_candidate(detail: str = "GENERIC") -> DecoderCandidateManifestError:
    return _error(DecoderCandidateManifestErrorCode.INVALID_DECODER_CANDIDATE, detail)


def _invalid_hash(detail: str = "FORMAT") -> DecoderCandidateManifestError:
    return _error(DecoderCandidateManifestErrorCode.INVALID_HASH, detail)


def _hash_mismatch(detail: str) -> DecoderCandidateManifestError:
    return _error(DecoderCandidateManifestErrorCode.HASH_MISMATCH, detail)


def _to_canonical_obj(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_canonical_obj(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _to_canonical_obj(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _base_payload(value: Any, hash_key: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise _invalid_input("payload:DATACLASS_OR_MAPPING")
    payload.pop(hash_key, None)
    return payload


def _validate_hash_format(value: str, field_name: str = "sha256") -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _invalid_hash(f"{field_name}:FORMAT")


def _assert_hash_matches(obj: Any, field_name: str, payload_fn: Any) -> None:
    expected_hash = getattr(obj, field_name)
    _validate_hash_format(expected_hash, field_name)
    if _hash_payload(payload_fn(obj)) != expected_hash:
        raise _hash_mismatch(field_name)


def _require_exact_bool(value: Any, field_name: str = "bool") -> None:
    if type(value) is not bool:
        raise _invalid_input(f"{field_name}:BOOL")


def _require_policy_flags(
    obj: Any, expected_flags: Mapping[str, bool], unsafe_detail: str
) -> None:
    for field_name, expected in expected_flags.items():
        _require_exact_bool(getattr(obj, field_name), field_name)
        if getattr(obj, field_name) is not expected:
            raise _invalid_candidate(unsafe_detail)


def _policy_flags_satisfied(obj: Any, expected_flags: Mapping[str, bool]) -> bool:
    return all(
        getattr(obj, field_name, None) is expected
        for field_name, expected in expected_flags.items()
    )


def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls or not is_dataclass(value):
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    expected_field_names = tuple(field.name for field in fields(cls))
    actual_field_names = tuple(field.name for field in fields(value))
    if actual_field_names != expected_field_names:
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    post_init = getattr(value, "__post_init__", None)
    if callable(post_init):
        post_init()
