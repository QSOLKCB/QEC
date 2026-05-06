from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .game_world_adapter_contract import (
    WorldAdapterContractReceipt,
    WorldAdapterSpec,
    validate_world_adapter_contract_receipt,
    validate_world_adapter_spec,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_CHANNEL_TYPE = "INVALID_CHANNEL_TYPE"
_ERR_INVALID_OBSERVATION_PAYLOAD = "INVALID_OBSERVATION_PAYLOAD"
_ERR_PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
_ERR_OBSERVATION_INDEX_OUT_OF_BOUNDS = "OBSERVATION_INDEX_OUT_OF_BOUNDS"
_ERR_ADAPTER_SPEC_NOT_IN_CONTRACT = "ADAPTER_SPEC_NOT_IN_CONTRACT"
_ERR_ADAPTER_CONTRACT_MISMATCH = "ADAPTER_CONTRACT_MISMATCH"
_ERR_ACTION_MASK_UNKNOWN_ACTION = "ACTION_MASK_UNKNOWN_ACTION"
_ERR_ACTION_MASK_ORDER_MISMATCH = "ACTION_MASK_ORDER_MISMATCH"
_ERR_DUPLICATE_OBSERVATION = "DUPLICATE_OBSERVATION"
_ERR_OBSERVATION_COUNT_MISMATCH = "OBSERVATION_COUNT_MISMATCH"
_ERR_SNAPSHOT_ADAPTER_MISMATCH = "SNAPSHOT_ADAPTER_MISMATCH"

_MAX_OBSERVATION_INDEX = 1_000_000
_MAX_PAYLOAD_BYTES = 8192
_MAX_CHANNEL_LABEL_LENGTH = 64
_MAX_POSITION_DIMENSIONS = 8
_MAX_ABS_POSITION_COORDINATE = 1_000_000
_MAX_ABS_SCORE_VALUE = 1_000_000_000
_MAX_OBSERVATION_SNAPSHOTS = 1_000

_ALLOWED_OBSERVATION_CHANNEL_TYPES = {
    "SYMBOLIC_STATE",
    "PIXEL_HASH",
    "TEXT_EVENT",
    "SCORE_VALUE",
    "POSITION_VECTOR",
    "ACTION_MASK",
    "TERMINAL_FLAG",
    "UNKNOWN",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHANNEL_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def get_allowed_observation_channel_types() -> frozenset[str]:
    return frozenset(_ALLOWED_OBSERVATION_CHANNEL_TYPES)


def _validate_hash_string(value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)


def _validate_json_safe_no_floats(value: object) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, (float, bytes, tuple, set)):
        raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
    if isinstance(value, list):
        for item in value:
            _validate_json_safe_no_floats(item)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str) or not k:
                raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
            _validate_json_safe_no_floats(v)
        return
    raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)


def _observation_channel_spec_payload(
    channel_type: str, channel_label: str, max_payload_bytes: int
) -> dict[str, Any]:
    return {
        "channel_type": channel_type,
        "channel_label": channel_label,
        "max_payload_bytes": max_payload_bytes,
    }


def _observation_snapshot_payload(
    adapter_contract_receipt_hash: str,
    adapter_spec_hash: str,
    observation_index: int,
    observation_channel: "ObservationChannelSpec",
    canonical_observation_payload: str,
    observation_payload_hash: str,
) -> dict[str, Any]:
    return {
        "adapter_contract_receipt_hash": adapter_contract_receipt_hash,
        "adapter_spec_hash": adapter_spec_hash,
        "observation_index": observation_index,
        "observation_channel": observation_channel.to_dict(),
        "canonical_observation_payload": canonical_observation_payload,
        "observation_payload_hash": observation_payload_hash,
    }


def _observation_snapshot_receipt_payload(
    adapter_contract_receipt_hash: str,
    adapter_spec_hash: str,
    observation_snapshot: "ObservationSnapshot",
) -> dict[str, Any]:
    return {
        "adapter_contract_receipt_hash": adapter_contract_receipt_hash,
        "adapter_spec_hash": adapter_spec_hash,
        "observation_snapshot": observation_snapshot.to_dict(),
    }


def _observation_snapshot_set_payload(
    adapter_contract_receipt_hash: str,
    adapter_spec_hash: str,
    observation_snapshot_receipts: tuple["ObservationSnapshotReceipt", ...],
    observation_count: int,
) -> dict[str, Any]:
    return {
        "adapter_contract_receipt_hash": adapter_contract_receipt_hash,
        "adapter_spec_hash": adapter_spec_hash,
        "observation_snapshot_receipts": [
            r.to_dict() for r in observation_snapshot_receipts
        ],
        "observation_count": observation_count,
    }


def _snapshot_order_key(receipt: "ObservationSnapshotReceipt") -> tuple[int, str, str]:
    """Return canonical ordering key for snapshot receipts: (index, channel_hash, snapshot_hash)."""
    return (
        receipt.observation_snapshot.observation_index,
        receipt.observation_snapshot.observation_channel.observation_channel_hash,
        receipt.observation_snapshot.observation_snapshot_hash,
    )


def _validate_channel_payload(
    channel: "ObservationChannelSpec",
    payload: object,
    adapter_spec: WorldAdapterSpec | None = None,
) -> None:
    ct = channel.channel_type
    if ct == "SYMBOLIC_STATE":
        if not isinstance(payload, dict):
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
        _validate_json_safe_no_floats(payload)
    elif ct == "PIXEL_HASH":
        if not isinstance(payload, str):
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
        _validate_hash_string(payload)
    elif ct == "TEXT_EVENT":
        if not isinstance(payload, str):
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
    elif ct == "SCORE_VALUE":
        if not isinstance(payload, int) or isinstance(payload, bool):
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
        if abs(payload) > _MAX_ABS_SCORE_VALUE:
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
    elif ct == "POSITION_VECTOR":
        if not isinstance(payload, list):
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
        if len(payload) < 1 or len(payload) > _MAX_POSITION_DIMENSIONS:
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
        for x in payload:
            if (
                not isinstance(x, int)
                or isinstance(x, bool)
                or abs(x) > _MAX_ABS_POSITION_COORDINATE
            ):
                raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
    elif ct == "ACTION_MASK":
        if not isinstance(payload, list):
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
        seen: set[str] = set()
        for x in payload:
            if not isinstance(x, str):
                raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
            if x in seen:
                raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
            seen.add(x)
        if adapter_spec is not None:
            alphabet = [a.action_code for a in adapter_spec.action_alphabet.actions]
            allowed = set(alphabet)
            for x in payload:
                if x not in allowed:
                    raise ValueError(_ERR_ACTION_MASK_UNKNOWN_ACTION)
            exp = [x for x in alphabet if x in seen]
            if payload != exp:
                raise ValueError(_ERR_ACTION_MASK_ORDER_MISMATCH)
    elif ct == "TERMINAL_FLAG":
        if not isinstance(payload, bool):
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
    elif ct == "UNKNOWN":
        _validate_json_safe_no_floats(payload)
    else:
        raise ValueError(_ERR_INVALID_CHANNEL_TYPE)


@dataclass(frozen=True)
class ObservationChannelSpec:
    channel_type: str
    channel_label: str
    max_payload_bytes: int
    observation_channel_hash: str

    def __post_init__(self) -> None:
        validate_observation_channel_spec(self)

    def _hash_payload(self) -> dict[str, Any]:
        return _observation_channel_spec_payload(
            self.channel_type, self.channel_label, self.max_payload_bytes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_type": self.channel_type,
            "channel_label": self.channel_label,
            "max_payload_bytes": self.max_payload_bytes,
            "observation_channel_hash": self.observation_channel_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class ObservationSnapshot:
    adapter_contract_receipt_hash: str
    adapter_spec_hash: str
    observation_index: int
    observation_channel: ObservationChannelSpec
    canonical_observation_payload: str
    observation_payload_hash: str
    observation_snapshot_hash: str

    def __post_init__(self) -> None:
        validate_observation_snapshot(self)

    def _hash_payload(self) -> dict[str, Any]:
        return _observation_snapshot_payload(
            self.adapter_contract_receipt_hash,
            self.adapter_spec_hash,
            self.observation_index,
            self.observation_channel,
            self.canonical_observation_payload,
            self.observation_payload_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_contract_receipt_hash": self.adapter_contract_receipt_hash,
            "adapter_spec_hash": self.adapter_spec_hash,
            "observation_index": self.observation_index,
            "observation_channel": self.observation_channel.to_dict(),
            "canonical_observation_payload": self.canonical_observation_payload,
            "observation_payload_hash": self.observation_payload_hash,
            "observation_snapshot_hash": self.observation_snapshot_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class ObservationSnapshotReceipt:
    adapter_contract_receipt_hash: str
    adapter_spec_hash: str
    observation_snapshot: ObservationSnapshot
    observation_snapshot_receipt_hash: str

    def __post_init__(self) -> None:
        validate_observation_snapshot_receipt(self)

    def _hash_payload(self) -> dict[str, Any]:
        return _observation_snapshot_receipt_payload(
            self.adapter_contract_receipt_hash,
            self.adapter_spec_hash,
            self.observation_snapshot,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_contract_receipt_hash": self.adapter_contract_receipt_hash,
            "adapter_spec_hash": self.adapter_spec_hash,
            "observation_snapshot": self.observation_snapshot.to_dict(),
            "observation_snapshot_receipt_hash": self.observation_snapshot_receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class ObservationSnapshotSet:
    adapter_contract_receipt_hash: str
    adapter_spec_hash: str
    observation_snapshot_receipts: tuple[ObservationSnapshotReceipt, ...]
    observation_count: int
    observation_snapshot_set_hash: str

    def __post_init__(self) -> None:
        validate_observation_snapshot_set(self)

    def _hash_payload(self) -> dict[str, Any]:
        return _observation_snapshot_set_payload(
            self.adapter_contract_receipt_hash,
            self.adapter_spec_hash,
            self.observation_snapshot_receipts,
            self.observation_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_contract_receipt_hash": self.adapter_contract_receipt_hash,
            "adapter_spec_hash": self.adapter_spec_hash,
            "observation_snapshot_receipts": [
                r.to_dict() for r in self.observation_snapshot_receipts
            ],
            "observation_count": self.observation_count,
            "observation_snapshot_set_hash": self.observation_snapshot_set_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def validate_observation_channel_spec(spec: ObservationChannelSpec) -> bool:
    if not isinstance(spec, ObservationChannelSpec):
        raise ValueError(_ERR_INVALID_INPUT)
    if (
        not isinstance(spec.channel_type, str)
        or spec.channel_type not in _ALLOWED_OBSERVATION_CHANNEL_TYPES
    ):
        raise ValueError(_ERR_INVALID_CHANNEL_TYPE)
    if (
        not isinstance(spec.channel_label, str)
        or _CHANNEL_LABEL_RE.fullmatch(spec.channel_label) is None
        or len(spec.channel_label) > _MAX_CHANNEL_LABEL_LENGTH
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    if (
        not isinstance(spec.max_payload_bytes, int)
        or isinstance(spec.max_payload_bytes, bool)
        or not (1 <= spec.max_payload_bytes <= _MAX_PAYLOAD_BYTES)
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(spec.observation_channel_hash)
    if spec.observation_channel_hash != sha256_hex(spec._hash_payload()):
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_observation_snapshot(snapshot: ObservationSnapshot) -> bool:
    """Validate intrinsic snapshot integrity only.

    ACTION_MASK membership/order against adapter alphabet requires
    validate_observation_snapshot_with_adapter(...).
    """
    if not isinstance(snapshot, ObservationSnapshot):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(snapshot.adapter_contract_receipt_hash)
    _validate_hash_string(snapshot.adapter_spec_hash)
    if not isinstance(snapshot.observation_index, int) or isinstance(
        snapshot.observation_index, bool
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    if (
        snapshot.observation_index < 0
        or snapshot.observation_index > _MAX_OBSERVATION_INDEX
    ):
        raise ValueError(_ERR_OBSERVATION_INDEX_OUT_OF_BOUNDS)
    validate_observation_channel_spec(snapshot.observation_channel)
    if not isinstance(snapshot.canonical_observation_payload, str):
        raise ValueError(_ERR_INVALID_INPUT)
    # For string-typed channels, the size limit applies to the raw payload, not canonical JSON
    # For structured types, the canonical JSON size is checked
    if snapshot.observation_channel.channel_type in ("TEXT_EVENT", "PIXEL_HASH"):
        try:
            parsed = json.loads(snapshot.canonical_observation_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD) from exc
        if (
            isinstance(parsed, str)
            and len(parsed.encode("utf-8"))
            > snapshot.observation_channel.max_payload_bytes
        ):
            raise ValueError(_ERR_PAYLOAD_TOO_LARGE)
    else:
        if (
            len(snapshot.canonical_observation_payload.encode("utf-8"))
            > snapshot.observation_channel.max_payload_bytes
        ):
            raise ValueError(_ERR_PAYLOAD_TOO_LARGE)
        try:
            parsed = json.loads(snapshot.canonical_observation_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD) from exc
    if canonical_json(parsed) != snapshot.canonical_observation_payload:
        raise ValueError(_ERR_INVALID_OBSERVATION_PAYLOAD)
    _validate_channel_payload(snapshot.observation_channel, parsed)
    _validate_hash_string(snapshot.observation_payload_hash)
    _validate_hash_string(snapshot.observation_snapshot_hash)
    if snapshot.observation_payload_hash != sha256_hex(
        snapshot.canonical_observation_payload
    ):
        raise ValueError(_ERR_HASH_MISMATCH)
    if snapshot.observation_snapshot_hash != sha256_hex(snapshot._hash_payload()):
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_observation_snapshot_receipt(receipt: ObservationSnapshotReceipt) -> bool:
    """Validate intrinsic receipt integrity only.

    Adapter/spec contract binding and ACTION_MASK alphabet checks require
    validate_observation_snapshot_receipt_with_adapter(...).
    """
    if not isinstance(receipt, ObservationSnapshotReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(receipt.adapter_contract_receipt_hash)
    _validate_hash_string(receipt.adapter_spec_hash)
    validate_observation_snapshot(receipt.observation_snapshot)
    if (
        receipt.adapter_contract_receipt_hash
        != receipt.observation_snapshot.adapter_contract_receipt_hash
        or receipt.adapter_spec_hash != receipt.observation_snapshot.adapter_spec_hash
    ):
        raise ValueError(_ERR_SNAPSHOT_ADAPTER_MISMATCH)
    _validate_hash_string(receipt.observation_snapshot_receipt_hash)
    if receipt.observation_snapshot_receipt_hash != sha256_hex(receipt._hash_payload()):
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_observation_snapshot_set(snapshot_set: ObservationSnapshotSet) -> bool:
    """Validate intrinsic set integrity only.

    Complete adapter/spec validation requires
    validate_observation_snapshot_set_with_adapter(...).
    """
    if not isinstance(snapshot_set, ObservationSnapshotSet):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(snapshot_set.adapter_contract_receipt_hash)
    _validate_hash_string(snapshot_set.adapter_spec_hash)
    if not isinstance(snapshot_set.observation_snapshot_receipts, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    if (
        not isinstance(snapshot_set.observation_count, int)
        or isinstance(snapshot_set.observation_count, bool)
        or snapshot_set.observation_count < 0
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    if len(snapshot_set.observation_snapshot_receipts) > _MAX_OBSERVATION_SNAPSHOTS:
        raise ValueError(_ERR_INVALID_INPUT)
    for r in snapshot_set.observation_snapshot_receipts:
        validate_observation_snapshot_receipt(r)
        if (
            r.adapter_contract_receipt_hash
            != snapshot_set.adapter_contract_receipt_hash
            or r.observation_snapshot.adapter_contract_receipt_hash
            != snapshot_set.adapter_contract_receipt_hash
        ):
            raise ValueError(_ERR_SNAPSHOT_ADAPTER_MISMATCH)
        if (
            r.adapter_spec_hash != snapshot_set.adapter_spec_hash
            or r.observation_snapshot.adapter_spec_hash
            != snapshot_set.adapter_spec_hash
        ):
            raise ValueError(_ERR_SNAPSHOT_ADAPTER_MISMATCH)
    obs_keys = [
        (
            r.observation_snapshot.observation_index,
            r.observation_snapshot.observation_channel.observation_channel_hash,
        )
        for r in snapshot_set.observation_snapshot_receipts
    ]
    if len(set(obs_keys)) != len(obs_keys):
        raise ValueError(_ERR_DUPLICATE_OBSERVATION)
    if (
        tuple(
            sorted(snapshot_set.observation_snapshot_receipts, key=_snapshot_order_key)
        )
        != snapshot_set.observation_snapshot_receipts
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    if snapshot_set.observation_count != len(
        snapshot_set.observation_snapshot_receipts
    ):
        raise ValueError(_ERR_OBSERVATION_COUNT_MISMATCH)
    _validate_hash_string(snapshot_set.observation_snapshot_set_hash)
    if snapshot_set.observation_snapshot_set_hash != sha256_hex(
        snapshot_set._hash_payload()
    ):
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_spec_in_contract(
    adapter_contract_receipt: WorldAdapterContractReceipt,
    adapter_spec: WorldAdapterSpec,
) -> bool:
    validate_world_adapter_contract_receipt(adapter_contract_receipt)
    validate_world_adapter_spec(adapter_spec)
    matches = [
        s
        for s in adapter_contract_receipt.adapter_specs
        if s.adapter_spec_hash == adapter_spec.adapter_spec_hash
    ]
    if len(matches) != 1:
        raise ValueError(_ERR_ADAPTER_SPEC_NOT_IN_CONTRACT)
    return True


def validate_observation_snapshot_with_adapter(
    snapshot: ObservationSnapshot,
    adapter_contract_receipt: WorldAdapterContractReceipt,
    adapter_spec: WorldAdapterSpec,
) -> bool:
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    validate_observation_snapshot(snapshot)
    if (
        snapshot.adapter_contract_receipt_hash
        != adapter_contract_receipt.adapter_contract_receipt_hash
    ):
        raise ValueError(_ERR_ADAPTER_CONTRACT_MISMATCH)
    if snapshot.adapter_spec_hash != adapter_spec.adapter_spec_hash:
        raise ValueError(_ERR_SNAPSHOT_ADAPTER_MISMATCH)
    parsed = json.loads(snapshot.canonical_observation_payload)
    _validate_channel_payload(
        snapshot.observation_channel, parsed, adapter_spec=adapter_spec
    )
    return True


def validate_observation_snapshot_receipt_with_adapter(
    receipt: ObservationSnapshotReceipt,
    adapter_contract_receipt: WorldAdapterContractReceipt,
    adapter_spec: WorldAdapterSpec,
) -> bool:
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    validate_observation_snapshot_receipt(receipt)
    if (
        receipt.adapter_contract_receipt_hash
        != adapter_contract_receipt.adapter_contract_receipt_hash
    ):
        raise ValueError(_ERR_ADAPTER_CONTRACT_MISMATCH)
    if receipt.adapter_spec_hash != adapter_spec.adapter_spec_hash:
        raise ValueError(_ERR_SNAPSHOT_ADAPTER_MISMATCH)
    validate_observation_snapshot_with_adapter(
        receipt.observation_snapshot, adapter_contract_receipt, adapter_spec
    )
    return True


def validate_observation_snapshot_set_with_adapter(
    snapshot_set: ObservationSnapshotSet,
    adapter_contract_receipt: WorldAdapterContractReceipt,
    adapter_spec: WorldAdapterSpec,
) -> bool:
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    validate_observation_snapshot_set(snapshot_set)
    if (
        snapshot_set.adapter_contract_receipt_hash
        != adapter_contract_receipt.adapter_contract_receipt_hash
    ):
        raise ValueError(_ERR_ADAPTER_CONTRACT_MISMATCH)
    if snapshot_set.adapter_spec_hash != adapter_spec.adapter_spec_hash:
        raise ValueError(_ERR_SNAPSHOT_ADAPTER_MISMATCH)
    for receipt in snapshot_set.observation_snapshot_receipts:
        validate_observation_snapshot_receipt_with_adapter(
            receipt, adapter_contract_receipt, adapter_spec
        )
    return True


def build_observation_channel_spec(
    channel_type: str,
    channel_label: str = "DEFAULT",
    max_payload_bytes: int = _MAX_PAYLOAD_BYTES,
) -> ObservationChannelSpec:
    if (
        not isinstance(channel_type, str)
        or channel_type not in _ALLOWED_OBSERVATION_CHANNEL_TYPES
    ):
        raise ValueError(_ERR_INVALID_CHANNEL_TYPE)
    if (
        not isinstance(channel_label, str)
        or _CHANNEL_LABEL_RE.fullmatch(channel_label) is None
        or len(channel_label) > _MAX_CHANNEL_LABEL_LENGTH
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    if (
        not isinstance(max_payload_bytes, int)
        or isinstance(max_payload_bytes, bool)
        or not (1 <= max_payload_bytes <= _MAX_PAYLOAD_BYTES)
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    return ObservationChannelSpec(
        channel_type,
        channel_label,
        max_payload_bytes,
        sha256_hex(
            _observation_channel_spec_payload(
                channel_type, channel_label, max_payload_bytes
            )
        ),
    )


def build_observation_snapshot(
    adapter_contract_receipt: WorldAdapterContractReceipt,
    adapter_spec: WorldAdapterSpec,
    observation_channel: ObservationChannelSpec,
    observation_index: int,
    observation_payload: object,
) -> ObservationSnapshot:
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    validate_observation_channel_spec(observation_channel)
    if not isinstance(observation_index, int) or isinstance(observation_index, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    if observation_index < 0 or observation_index > _MAX_OBSERVATION_INDEX:
        raise ValueError(_ERR_OBSERVATION_INDEX_OUT_OF_BOUNDS)
    _validate_channel_payload(
        observation_channel, observation_payload, adapter_spec=adapter_spec
    )
    # Check raw payload size for string-typed channels (TEXT_EVENT, PIXEL_HASH)
    # to avoid rejecting valid payloads due to JSON encoding overhead
    if observation_channel.channel_type in ("TEXT_EVENT", "PIXEL_HASH"):
        if (
            isinstance(observation_payload, str)
            and len(observation_payload.encode("utf-8"))
            > observation_channel.max_payload_bytes
        ):
            raise ValueError(_ERR_PAYLOAD_TOO_LARGE)
    canonical_payload = canonical_json(observation_payload)
    payload_hash = sha256_hex(canonical_payload)
    snapshot_hash = sha256_hex(
        _observation_snapshot_payload(
            adapter_contract_receipt.adapter_contract_receipt_hash,
            adapter_spec.adapter_spec_hash,
            observation_index,
            observation_channel,
            canonical_payload,
            payload_hash,
        )
    )
    snapshot = ObservationSnapshot(
        adapter_contract_receipt.adapter_contract_receipt_hash,
        adapter_spec.adapter_spec_hash,
        observation_index,
        observation_channel,
        canonical_payload,
        payload_hash,
        snapshot_hash,
    )
    validate_observation_snapshot_with_adapter(
        snapshot, adapter_contract_receipt, adapter_spec
    )
    return snapshot


def build_observation_snapshot_receipt(
    adapter_contract_receipt: WorldAdapterContractReceipt,
    adapter_spec: WorldAdapterSpec,
    observation_channel: ObservationChannelSpec,
    observation_index: int,
    observation_payload: object,
) -> ObservationSnapshotReceipt:
    snapshot = build_observation_snapshot(
        adapter_contract_receipt,
        adapter_spec,
        observation_channel,
        observation_index,
        observation_payload,
    )
    rh = sha256_hex(
        _observation_snapshot_receipt_payload(
            adapter_contract_receipt.adapter_contract_receipt_hash,
            adapter_spec.adapter_spec_hash,
            snapshot,
        )
    )
    receipt = ObservationSnapshotReceipt(
        adapter_contract_receipt.adapter_contract_receipt_hash,
        adapter_spec.adapter_spec_hash,
        snapshot,
        rh,
    )
    validate_observation_snapshot_receipt_with_adapter(
        receipt, adapter_contract_receipt, adapter_spec
    )
    return receipt


def build_observation_snapshot_set(
    adapter_contract_receipt: WorldAdapterContractReceipt,
    adapter_spec: WorldAdapterSpec,
    observation_snapshot_receipts: (
        list[ObservationSnapshotReceipt] | tuple[ObservationSnapshotReceipt, ...]
    ),
) -> ObservationSnapshotSet:
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    if not isinstance(observation_snapshot_receipts, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    if len(observation_snapshot_receipts) > _MAX_OBSERVATION_SNAPSHOTS:
        raise ValueError(_ERR_INVALID_INPUT)
    for r in observation_snapshot_receipts:
        if not isinstance(r, ObservationSnapshotReceipt):
            raise ValueError(_ERR_INVALID_INPUT)
        validate_observation_snapshot_receipt_with_adapter(
            r, adapter_contract_receipt, adapter_spec
        )
    ordered = tuple(sorted(observation_snapshot_receipts, key=_snapshot_order_key))
    obs_keys = [
        (
            r.observation_snapshot.observation_index,
            r.observation_snapshot.observation_channel.observation_channel_hash,
        )
        for r in ordered
    ]
    if len(set(obs_keys)) != len(obs_keys):
        raise ValueError(_ERR_DUPLICATE_OBSERVATION)
    count = len(ordered)
    sh = sha256_hex(
        _observation_snapshot_set_payload(
            adapter_contract_receipt.adapter_contract_receipt_hash,
            adapter_spec.adapter_spec_hash,
            ordered,
            count,
        )
    )
    snapshot_set = ObservationSnapshotSet(
        adapter_contract_receipt.adapter_contract_receipt_hash,
        adapter_spec.adapter_spec_hash,
        ordered,
        count,
        sh,
    )
    validate_observation_snapshot_set_with_adapter(
        snapshot_set, adapter_contract_receipt, adapter_spec
    )
    return snapshot_set
