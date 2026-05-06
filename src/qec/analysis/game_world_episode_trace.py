from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .game_world_adapter_contract import (
    ActionAtom,
    WorldAdapterContractReceipt,
    WorldAdapterSpec,
    validate_action_atom,
    validate_world_adapter_contract_receipt,
    validate_world_adapter_spec,
)
from .game_world_observation_snapshot import (
    ObservationSnapshotReceipt,
    validate_observation_snapshot_receipt,
    validate_observation_snapshot_receipt_with_adapter,
    validate_spec_in_contract,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_STEP_INDEX_OUT_OF_BOUNDS = "STEP_INDEX_OUT_OF_BOUNDS"
_ERR_DUPLICATE_STEP = "DUPLICATE_STEP"
_ERR_STEP_COUNT_MISMATCH = "STEP_COUNT_MISMATCH"
_ERR_STEP_ORDER_MISMATCH = "STEP_ORDER_MISMATCH"
_ERR_ACTION_NOT_IN_ALPHABET = "ACTION_NOT_IN_ALPHABET"
_ERR_OBSERVATION_ADAPTER_MISMATCH = "OBSERVATION_ADAPTER_MISMATCH"
_ERR_TRACE_ADAPTER_MISMATCH = "TRACE_ADAPTER_MISMATCH"
_ERR_TERMINAL_STEP_MISMATCH = "TERMINAL_STEP_MISMATCH"
_ERR_MULTIPLE_TERMINAL_STEPS = "MULTIPLE_TERMINAL_STEPS"
_ERR_POST_TERMINAL_STEP = "POST_TERMINAL_STEP"
_ERR_ADAPTER_SPEC_NOT_IN_CONTRACT = "ADAPTER_SPEC_NOT_IN_CONTRACT"

_MAX_EPISODE_STEPS = 1_000
_MAX_STEP_INDEX = 1_000_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_hash_string(value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)


def _snapshot_or_step_order_key(step: "EpisodeStep") -> tuple[int, str]:
    return (step.step_index, step.episode_step_hash)


def _episode_step_payload(
    adapter_contract_receipt_hash: str,
    adapter_spec_hash: str,
    step_index: int,
    observation_snapshot_receipt: ObservationSnapshotReceipt,
    action_atom: ActionAtom,
    terminal_flag: bool,
) -> dict[str, Any]:
    return {
        "adapter_contract_receipt_hash": adapter_contract_receipt_hash,
        "adapter_spec_hash": adapter_spec_hash,
        "step_index": step_index,
        "observation_snapshot_receipt": observation_snapshot_receipt.to_dict(),
        "action_atom": action_atom.to_dict(),
        "terminal_flag": terminal_flag,
    }


def _episode_trace_payload(
    adapter_contract_receipt_hash: str,
    adapter_spec_hash: str,
    episode_steps: tuple["EpisodeStep", ...],
    step_count: int,
    terminal_step_index: int | None,
) -> dict[str, Any]:
    return {
        "adapter_contract_receipt_hash": adapter_contract_receipt_hash,
        "adapter_spec_hash": adapter_spec_hash,
        "episode_steps": [s.to_dict() for s in episode_steps],
        "step_count": step_count,
        "terminal_step_index": terminal_step_index,
    }


def _episode_trace_receipt_payload(
    adapter_contract_receipt_hash: str,
    adapter_spec_hash: str,
    episode_trace: "EpisodeTrace",
) -> dict[str, Any]:
    return {
        "adapter_contract_receipt_hash": adapter_contract_receipt_hash,
        "adapter_spec_hash": adapter_spec_hash,
        "episode_trace": episode_trace.to_dict(),
    }


def _validate_action_in_adapter(action_atom: ActionAtom, adapter_spec: WorldAdapterSpec) -> None:
    validate_action_atom(action_atom)
    matches = [a for a in adapter_spec.action_alphabet.actions if a.action_code == action_atom.action_code]
    if len(matches) != 1 or matches[0].action_atom_hash != action_atom.action_atom_hash:
        raise ValueError(_ERR_ACTION_NOT_IN_ALPHABET)


@dataclass(frozen=True)
class EpisodeStep:
    adapter_contract_receipt_hash: str
    adapter_spec_hash: str
    step_index: int
    observation_snapshot_receipt: ObservationSnapshotReceipt
    action_atom: ActionAtom
    terminal_flag: bool
    episode_step_hash: str

    def __post_init__(self) -> None:
        validate_episode_step(self)

    def _hash_payload(self) -> dict[str, Any]:
        return _episode_step_payload(
            self.adapter_contract_receipt_hash,
            self.adapter_spec_hash,
            self.step_index,
            self.observation_snapshot_receipt,
            self.action_atom,
            self.terminal_flag,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_contract_receipt_hash": self.adapter_contract_receipt_hash,
            "adapter_spec_hash": self.adapter_spec_hash,
            "step_index": self.step_index,
            "observation_snapshot_receipt": self.observation_snapshot_receipt.to_dict(),
            "action_atom": self.action_atom.to_dict(),
            "terminal_flag": self.terminal_flag,
            "episode_step_hash": self.episode_step_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class EpisodeTrace:
    adapter_contract_receipt_hash: str
    adapter_spec_hash: str
    episode_steps: tuple[EpisodeStep, ...]
    step_count: int
    terminal_step_index: int | None
    episode_trace_hash: str

    def __post_init__(self) -> None:
        validate_episode_trace(self)

    def _hash_payload(self) -> dict[str, Any]:
        return _episode_trace_payload(self.adapter_contract_receipt_hash, self.adapter_spec_hash, self.episode_steps, self.step_count, self.terminal_step_index)

    def to_dict(self) -> dict[str, Any]:
        return {"adapter_contract_receipt_hash": self.adapter_contract_receipt_hash, "adapter_spec_hash": self.adapter_spec_hash, "episode_steps": [s.to_dict() for s in self.episode_steps], "step_count": self.step_count, "terminal_step_index": self.terminal_step_index, "episode_trace_hash": self.episode_trace_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class EpisodeTraceReceipt:
    adapter_contract_receipt_hash: str
    adapter_spec_hash: str
    episode_trace: EpisodeTrace
    episode_trace_receipt_hash: str

    def __post_init__(self) -> None:
        validate_episode_trace_receipt(self)

    def _hash_payload(self) -> dict[str, Any]:
        return _episode_trace_receipt_payload(self.adapter_contract_receipt_hash, self.adapter_spec_hash, self.episode_trace)

    def to_dict(self) -> dict[str, Any]:
        return {"adapter_contract_receipt_hash": self.adapter_contract_receipt_hash, "adapter_spec_hash": self.adapter_spec_hash, "episode_trace": self.episode_trace.to_dict(), "episode_trace_receipt_hash": self.episode_trace_receipt_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def validate_episode_step(step: EpisodeStep) -> bool:
    if not isinstance(step, EpisodeStep):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(step.adapter_contract_receipt_hash)
    _validate_hash_string(step.adapter_spec_hash)
    if not isinstance(step.step_index, int) or isinstance(step.step_index, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    if step.step_index < 0 or step.step_index > _MAX_STEP_INDEX:
        raise ValueError(_ERR_STEP_INDEX_OUT_OF_BOUNDS)
    if not isinstance(step.terminal_flag, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    validate_observation_snapshot_receipt(step.observation_snapshot_receipt)
    if step.observation_snapshot_receipt.adapter_contract_receipt_hash != step.adapter_contract_receipt_hash or step.observation_snapshot_receipt.adapter_spec_hash != step.adapter_spec_hash:
        raise ValueError(_ERR_OBSERVATION_ADAPTER_MISMATCH)
    validate_action_atom(step.action_atom)
    _validate_hash_string(step.episode_step_hash)
    if step.episode_step_hash != sha256_hex(step._hash_payload()):
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_episode_trace(trace: EpisodeTrace) -> bool:
    if not isinstance(trace, EpisodeTrace):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(trace.adapter_contract_receipt_hash)
    _validate_hash_string(trace.adapter_spec_hash)
    if not isinstance(trace.episode_steps, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    if len(trace.episode_steps) < 1 or len(trace.episode_steps) > _MAX_EPISODE_STEPS:
        raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(trace.step_count, int) or isinstance(trace.step_count, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    if trace.step_count != len(trace.episode_steps):
        raise ValueError(_ERR_STEP_COUNT_MISMATCH)
    if trace.terminal_step_index is not None and (not isinstance(trace.terminal_step_index, int) or isinstance(trace.terminal_step_index, bool)):
        raise ValueError(_ERR_INVALID_INPUT)
    prev = -1
    terminal_indices: list[int] = []
    for i, step in enumerate(trace.episode_steps):
        validate_episode_step(step)
        if step.adapter_contract_receipt_hash != trace.adapter_contract_receipt_hash or step.adapter_spec_hash != trace.adapter_spec_hash:
            raise ValueError(_ERR_TRACE_ADAPTER_MISMATCH)
        if i > 0 and _snapshot_or_step_order_key(trace.episode_steps[i - 1]) > _snapshot_or_step_order_key(step):
            raise ValueError(_ERR_INVALID_INPUT)
        if step.step_index == prev:
            raise ValueError(_ERR_DUPLICATE_STEP)
        if step.step_index != i:
            raise ValueError(_ERR_STEP_ORDER_MISMATCH)
        prev = step.step_index
        if step.terminal_flag:
            terminal_indices.append(step.step_index)
            if i != len(trace.episode_steps) - 1:
                raise ValueError(_ERR_POST_TERMINAL_STEP)
    if len(terminal_indices) > 1:
        raise ValueError(_ERR_MULTIPLE_TERMINAL_STEPS)
    expected_terminal = terminal_indices[0] if terminal_indices else None
    if trace.terminal_step_index != expected_terminal:
        raise ValueError(_ERR_TERMINAL_STEP_MISMATCH)
    _validate_hash_string(trace.episode_trace_hash)
    if trace.episode_trace_hash != sha256_hex(trace._hash_payload()):
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_episode_trace_receipt(receipt: EpisodeTraceReceipt) -> bool:
    if not isinstance(receipt, EpisodeTraceReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(receipt.adapter_contract_receipt_hash)
    _validate_hash_string(receipt.adapter_spec_hash)
    validate_episode_trace(receipt.episode_trace)
    if receipt.episode_trace.adapter_contract_receipt_hash != receipt.adapter_contract_receipt_hash or receipt.episode_trace.adapter_spec_hash != receipt.adapter_spec_hash:
        raise ValueError(_ERR_TRACE_ADAPTER_MISMATCH)
    _validate_hash_string(receipt.episode_trace_receipt_hash)
    if receipt.episode_trace_receipt_hash != sha256_hex(receipt._hash_payload()):
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_episode_step_with_adapter(step: EpisodeStep, adapter_contract_receipt: WorldAdapterContractReceipt, adapter_spec: WorldAdapterSpec) -> bool:
    validate_world_adapter_contract_receipt(adapter_contract_receipt)
    validate_world_adapter_spec(adapter_spec)
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    validate_episode_step(step)
    if step.adapter_contract_receipt_hash != adapter_contract_receipt.adapter_contract_receipt_hash or step.adapter_spec_hash != adapter_spec.adapter_spec_hash:
        raise ValueError(_ERR_TRACE_ADAPTER_MISMATCH)
    validate_observation_snapshot_receipt_with_adapter(step.observation_snapshot_receipt, adapter_contract_receipt, adapter_spec)
    _validate_action_in_adapter(step.action_atom, adapter_spec)
    return True


def validate_episode_trace_with_adapter(trace: EpisodeTrace, adapter_contract_receipt: WorldAdapterContractReceipt, adapter_spec: WorldAdapterSpec) -> bool:
    validate_world_adapter_contract_receipt(adapter_contract_receipt)
    validate_world_adapter_spec(adapter_spec)
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    validate_episode_trace(trace)
    if trace.adapter_contract_receipt_hash != adapter_contract_receipt.adapter_contract_receipt_hash or trace.adapter_spec_hash != adapter_spec.adapter_spec_hash:
        raise ValueError(_ERR_TRACE_ADAPTER_MISMATCH)
    for step in trace.episode_steps:
        validate_episode_step_with_adapter(step, adapter_contract_receipt, adapter_spec)
    return True


def validate_episode_trace_receipt_with_adapter(receipt: EpisodeTraceReceipt, adapter_contract_receipt: WorldAdapterContractReceipt, adapter_spec: WorldAdapterSpec) -> bool:
    validate_world_adapter_contract_receipt(adapter_contract_receipt)
    validate_world_adapter_spec(adapter_spec)
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    validate_episode_trace_receipt(receipt)
    if receipt.adapter_contract_receipt_hash != adapter_contract_receipt.adapter_contract_receipt_hash or receipt.adapter_spec_hash != adapter_spec.adapter_spec_hash:
        raise ValueError(_ERR_TRACE_ADAPTER_MISMATCH)
    validate_episode_trace_with_adapter(receipt.episode_trace, adapter_contract_receipt, adapter_spec)
    return True


def build_episode_step(adapter_contract_receipt: WorldAdapterContractReceipt, adapter_spec: WorldAdapterSpec, step_index: int, observation_snapshot_receipt: ObservationSnapshotReceipt, action_atom: ActionAtom, terminal_flag: bool = False) -> EpisodeStep:
    validate_world_adapter_contract_receipt(adapter_contract_receipt)
    validate_world_adapter_spec(adapter_spec)
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    validate_observation_snapshot_receipt_with_adapter(observation_snapshot_receipt, adapter_contract_receipt, adapter_spec)
    _validate_action_in_adapter(action_atom, adapter_spec)
    if not isinstance(step_index, int) or isinstance(step_index, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    if step_index < 0 or step_index > _MAX_STEP_INDEX:
        raise ValueError(_ERR_STEP_INDEX_OUT_OF_BOUNDS)
    if not isinstance(terminal_flag, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    h = sha256_hex(_episode_step_payload(adapter_contract_receipt.adapter_contract_receipt_hash, adapter_spec.adapter_spec_hash, step_index, observation_snapshot_receipt, action_atom, terminal_flag))
    return EpisodeStep(adapter_contract_receipt.adapter_contract_receipt_hash, adapter_spec.adapter_spec_hash, step_index, observation_snapshot_receipt, action_atom, terminal_flag, h)


def build_episode_trace(adapter_contract_receipt: WorldAdapterContractReceipt, adapter_spec: WorldAdapterSpec, episode_steps: list[EpisodeStep] | tuple[EpisodeStep, ...]) -> EpisodeTrace:
    validate_world_adapter_contract_receipt(adapter_contract_receipt)
    validate_world_adapter_spec(adapter_spec)
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    if not isinstance(episode_steps, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    if len(episode_steps) < 1 or len(episode_steps) > _MAX_EPISODE_STEPS:
        raise ValueError(_ERR_INVALID_INPUT)
    for s in episode_steps:
        validate_episode_step_with_adapter(s, adapter_contract_receipt, adapter_spec)
    ordered = tuple(sorted(episode_steps, key=lambda s: s.step_index))
    seen: set[int] = set()
    for i, s in enumerate(ordered):
        if s.step_index in seen:
            raise ValueError(_ERR_DUPLICATE_STEP)
        seen.add(s.step_index)
        if s.step_index != i:
            raise ValueError(_ERR_STEP_ORDER_MISMATCH)
    terminal_steps = [s.step_index for s in ordered if s.terminal_flag]
    if len(terminal_steps) > 1:
        raise ValueError(_ERR_MULTIPLE_TERMINAL_STEPS)
    if terminal_steps and terminal_steps[0] != ordered[-1].step_index:
        raise ValueError(_ERR_POST_TERMINAL_STEP)
    tsi = terminal_steps[0] if terminal_steps else None
    h = sha256_hex(_episode_trace_payload(adapter_contract_receipt.adapter_contract_receipt_hash, adapter_spec.adapter_spec_hash, ordered, len(ordered), tsi))
    return EpisodeTrace(adapter_contract_receipt.adapter_contract_receipt_hash, adapter_spec.adapter_spec_hash, ordered, len(ordered), tsi, h)


def build_episode_trace_receipt(adapter_contract_receipt: WorldAdapterContractReceipt, adapter_spec: WorldAdapterSpec, episode_trace: EpisodeTrace) -> EpisodeTraceReceipt:
    validate_world_adapter_contract_receipt(adapter_contract_receipt)
    validate_world_adapter_spec(adapter_spec)
    validate_spec_in_contract(adapter_contract_receipt, adapter_spec)
    validate_episode_trace_with_adapter(episode_trace, adapter_contract_receipt, adapter_spec)
    h = sha256_hex(_episode_trace_receipt_payload(adapter_contract_receipt.adapter_contract_receipt_hash, adapter_spec.adapter_spec_hash, episode_trace))
    return EpisodeTraceReceipt(adapter_contract_receipt.adapter_contract_receipt_hash, adapter_spec.adapter_spec_hash, episode_trace, h)
