from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .game_world_intake_contract import (
    GameWorldArchive,
    GameWorldCorpusManifest,
    GameWorldIntakeReceipt,
    _ALLOWED_WORLD_FAMILIES,
    validate_game_world_archive,
    validate_game_world_corpus_manifest,
    validate_game_world_intake_receipt,
)

# ---------------------------------------------------------------------------
# Error code constants – centralised so all raises and tests reference the
# same literal values and refactoring only needs to touch one place.
# ---------------------------------------------------------------------------
_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_DUPLICATE_PARAMETER_SLOT = "DUPLICATE_PARAMETER_SLOT"
_ERR_DUPLICATE_ACTION = "DUPLICATE_ACTION"
_ERR_ACTION_COUNT_MISMATCH = "ACTION_COUNT_MISMATCH"
_ERR_ADAPTER_FAMILY_MISMATCH = "ADAPTER_FAMILY_MISMATCH"
_ERR_CORPUS_RECEIPT_MISMATCH = "CORPUS_RECEIPT_MISMATCH"
_ERR_DUPLICATE_ADAPTER = "DUPLICATE_ADAPTER"
_ERR_ADAPTER_COUNT_MISMATCH = "ADAPTER_COUNT_MISMATCH"

_ALLOWED_ADAPTER_MODES = {"STATIC_ACTION_ALPHABET_ONLY"}
_ALLOWED_ACTION_KINDS = {
    "META",
    "MOVEMENT",
    "COMBAT",
    "MENU",
    "PIXEL_BUTTON",
    "STRATEGY",
    "FRAMEWORK_CONTROL",
    "RHYTHM_EDIT",
    "BOARD_ACTION",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTION_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


# ---------------------------------------------------------------------------
# Public getter for allowed world families to avoid tests relying on private
# constants directly.
# ---------------------------------------------------------------------------
def get_allowed_world_families() -> frozenset[str]:
    """Return the set of allowed world families."""
    return frozenset(_ALLOWED_WORLD_FAMILIES)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_hash_string(value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)


def _validate_world_family(value: object) -> None:
    if not isinstance(value, str) or value not in _ALLOWED_WORLD_FAMILIES:
        raise ValueError(_ERR_INVALID_INPUT)


# ---------------------------------------------------------------------------
# Centralized hash payload helpers – single source of truth for preimage
# formulas to avoid drift between builders and validators.
# ---------------------------------------------------------------------------
def _action_atom_payload(
    action_code: str,
    action_kind: str,
    parameter_slots: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "action_code": action_code,
        "action_kind": action_kind,
        "parameter_slots": list(parameter_slots),
    }


def _action_alphabet_payload(
    world_family: str,
    actions: tuple[ActionAtom, ...],
    action_count: int,
) -> dict[str, Any]:
    return {
        "world_family": world_family,
        "actions": [a.to_dict() for a in actions],
        "action_count": action_count,
    }


def _world_adapter_spec_payload(
    archive_manifest_hash: str,
    world_family: str,
    adapter_mode: str,
    action_alphabet: ActionAlphabet,
) -> dict[str, Any]:
    return {
        "archive_manifest_hash": archive_manifest_hash,
        "world_family": world_family,
        "adapter_mode": adapter_mode,
        "action_alphabet": action_alphabet.to_dict(),
    }


def _world_adapter_contract_receipt_payload(
    corpus_manifest_hash: str,
    intake_receipt_hash: str,
    adapter_specs: tuple[WorldAdapterSpec, ...],
    total_adapters: int,
) -> dict[str, Any]:
    return {
        "corpus_manifest_hash": corpus_manifest_hash,
        "intake_receipt_hash": intake_receipt_hash,
        "adapter_specs": [s.to_dict() for s in adapter_specs],
        "total_adapters": total_adapters,
    }


# ---------------------------------------------------------------------------
# Integrity validation functions
# ---------------------------------------------------------------------------
def _validate_action_atom_integrity(atom: ActionAtom) -> None:
    if (
        not isinstance(atom.action_code, str)
        or _ACTION_CODE_RE.fullmatch(atom.action_code) is None
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    if (
        not isinstance(atom.action_kind, str)
        or atom.action_kind not in _ALLOWED_ACTION_KINDS
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(atom.parameter_slots, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    if any(not isinstance(slot, str) for slot in atom.parameter_slots):
        raise ValueError(_ERR_INVALID_INPUT)
    if tuple(sorted(atom.parameter_slots)) != atom.parameter_slots:
        raise ValueError(_ERR_INVALID_INPUT)
    if len(set(atom.parameter_slots)) != len(atom.parameter_slots):
        raise ValueError(_ERR_DUPLICATE_PARAMETER_SLOT)
    _validate_hash_string(atom.action_atom_hash)
    payload = _action_atom_payload(
        atom.action_code, atom.action_kind, atom.parameter_slots
    )
    if atom.action_atom_hash != sha256_hex(payload):
        raise ValueError(_ERR_HASH_MISMATCH)


def _validate_action_alphabet_integrity(alphabet: ActionAlphabet) -> None:
    _validate_world_family(alphabet.world_family)
    if not isinstance(alphabet.actions, tuple) or len(alphabet.actions) == 0:
        raise ValueError(_ERR_INVALID_INPUT)
    for action in alphabet.actions:
        if not isinstance(action, ActionAtom):
            raise ValueError(_ERR_INVALID_INPUT)
        _validate_action_atom_integrity(action)
    ordered = tuple(
        sorted(alphabet.actions, key=lambda a: (a.action_code, a.action_atom_hash))
    )
    if alphabet.actions != ordered:
        raise ValueError(_ERR_INVALID_INPUT)
    codes = [a.action_code for a in alphabet.actions]
    if len(set(codes)) != len(codes):
        raise ValueError(_ERR_DUPLICATE_ACTION)
    if (
        not isinstance(alphabet.action_count, int)
        or isinstance(alphabet.action_count, bool)
        or alphabet.action_count < 0
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    if alphabet.action_count != len(alphabet.actions):
        raise ValueError(_ERR_ACTION_COUNT_MISMATCH)
    _validate_hash_string(alphabet.action_alphabet_hash)
    payload = _action_alphabet_payload(
        alphabet.world_family, alphabet.actions, alphabet.action_count
    )
    if alphabet.action_alphabet_hash != sha256_hex(payload):
        raise ValueError(_ERR_HASH_MISMATCH)


def _validate_world_adapter_spec_integrity(spec: WorldAdapterSpec) -> None:
    _validate_hash_string(spec.archive_manifest_hash)
    _validate_world_family(spec.world_family)
    if (
        not isinstance(spec.adapter_mode, str)
        or spec.adapter_mode not in _ALLOWED_ADAPTER_MODES
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(spec.action_alphabet, ActionAlphabet):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_action_alphabet_integrity(spec.action_alphabet)
    if spec.action_alphabet.world_family != spec.world_family:
        raise ValueError(_ERR_ADAPTER_FAMILY_MISMATCH)
    _validate_hash_string(spec.adapter_spec_hash)
    payload = _world_adapter_spec_payload(
        spec.archive_manifest_hash,
        spec.world_family,
        spec.adapter_mode,
        spec.action_alphabet,
    )
    if spec.adapter_spec_hash != sha256_hex(payload):
        raise ValueError(_ERR_HASH_MISMATCH)


def _validate_world_adapter_contract_receipt_integrity(
    receipt: WorldAdapterContractReceipt,
) -> None:
    _validate_hash_string(receipt.corpus_manifest_hash)
    _validate_hash_string(receipt.intake_receipt_hash)
    if not isinstance(receipt.adapter_specs, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    for spec in receipt.adapter_specs:
        if not isinstance(spec, WorldAdapterSpec):
            raise ValueError(_ERR_INVALID_INPUT)
        _validate_world_adapter_spec_integrity(spec)
    ordered = tuple(
        sorted(
            receipt.adapter_specs,
            key=lambda s: (s.archive_manifest_hash, s.adapter_spec_hash),
        )
    )
    if ordered != receipt.adapter_specs:
        raise ValueError(_ERR_INVALID_INPUT)
    hashes = [s.archive_manifest_hash for s in receipt.adapter_specs]
    if len(set(hashes)) != len(hashes):
        raise ValueError(_ERR_DUPLICATE_ADAPTER)
    if (
        not isinstance(receipt.total_adapters, int)
        or isinstance(receipt.total_adapters, bool)
        or receipt.total_adapters < 0
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    if receipt.total_adapters != len(receipt.adapter_specs):
        raise ValueError(_ERR_ADAPTER_COUNT_MISMATCH)
    _validate_hash_string(receipt.adapter_contract_receipt_hash)
    payload = _world_adapter_contract_receipt_payload(
        receipt.corpus_manifest_hash,
        receipt.intake_receipt_hash,
        receipt.adapter_specs,
        receipt.total_adapters,
    )
    if receipt.adapter_contract_receipt_hash != sha256_hex(payload):
        raise ValueError(_ERR_HASH_MISMATCH)


# ---------------------------------------------------------------------------
# Artifact dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionAtom:
    """A single action definition in a game world action alphabet."""

    action_code: str
    action_kind: str
    parameter_slots: tuple[str, ...]
    action_atom_hash: str

    def __post_init__(self) -> None:
        _validate_action_atom_integrity(self)

    def _hash_payload(self) -> dict[str, Any]:
        """Return the payload used for hash computation (excludes self-hash)."""
        return _action_atom_payload(
            self.action_code, self.action_kind, self.parameter_slots
        )

    def to_dict(self) -> dict[str, Any]:
        """Export full artifact dict including hash."""
        return {
            "action_code": self.action_code,
            "action_kind": self.action_kind,
            "parameter_slots": list(self.parameter_slots),
            "action_atom_hash": self.action_atom_hash,
        }

    def to_canonical_json(self) -> str:
        """Export full canonical artifact JSON."""
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        """Export full canonical artifact bytes."""
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class ActionAlphabet:
    """A collection of actions for a specific world family."""

    world_family: str
    actions: tuple[ActionAtom, ...]
    action_count: int
    action_alphabet_hash: str

    def __post_init__(self) -> None:
        _validate_action_alphabet_integrity(self)

    def _hash_payload(self) -> dict[str, Any]:
        """Return the payload used for hash computation (excludes self-hash)."""
        return _action_alphabet_payload(
            self.world_family, self.actions, self.action_count
        )

    def to_dict(self) -> dict[str, Any]:
        """Export full artifact dict including hash."""
        return {
            "world_family": self.world_family,
            "actions": [a.to_dict() for a in self.actions],
            "action_count": self.action_count,
            "action_alphabet_hash": self.action_alphabet_hash,
        }

    def to_canonical_json(self) -> str:
        """Export full canonical artifact JSON."""
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        """Export full canonical artifact bytes."""
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class WorldAdapterSpec:
    """Specification binding an archive to its action alphabet."""

    archive_manifest_hash: str
    world_family: str
    adapter_mode: str
    action_alphabet: ActionAlphabet
    adapter_spec_hash: str

    def __post_init__(self) -> None:
        _validate_world_adapter_spec_integrity(self)

    def _hash_payload(self) -> dict[str, Any]:
        """Return the payload used for hash computation (excludes self-hash)."""
        return _world_adapter_spec_payload(
            self.archive_manifest_hash,
            self.world_family,
            self.adapter_mode,
            self.action_alphabet,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export full artifact dict including hash."""
        return {
            "archive_manifest_hash": self.archive_manifest_hash,
            "world_family": self.world_family,
            "adapter_mode": self.adapter_mode,
            "action_alphabet": self.action_alphabet.to_dict(),
            "adapter_spec_hash": self.adapter_spec_hash,
        }

    def to_canonical_json(self) -> str:
        """Export full canonical artifact JSON."""
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        """Export full canonical artifact bytes."""
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class WorldAdapterContractReceipt:
    """Receipt proving adapter contracts were created for all archives."""

    corpus_manifest_hash: str
    intake_receipt_hash: str
    adapter_specs: tuple[WorldAdapterSpec, ...]
    total_adapters: int
    adapter_contract_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_world_adapter_contract_receipt_integrity(self)

    def _hash_payload(self) -> dict[str, Any]:
        """Return the payload used for hash computation (excludes self-hash)."""
        return _world_adapter_contract_receipt_payload(
            self.corpus_manifest_hash,
            self.intake_receipt_hash,
            self.adapter_specs,
            self.total_adapters,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export full artifact dict including hash."""
        return {
            "corpus_manifest_hash": self.corpus_manifest_hash,
            "intake_receipt_hash": self.intake_receipt_hash,
            "adapter_specs": [s.to_dict() for s in self.adapter_specs],
            "total_adapters": self.total_adapters,
            "adapter_contract_receipt_hash": self.adapter_contract_receipt_hash,
        }

    def to_canonical_json(self) -> str:
        """Export full canonical artifact JSON."""
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        """Export full canonical artifact bytes."""
        return canonical_bytes(self.to_dict())


# ---------------------------------------------------------------------------
# Default action definitions by world family
# ---------------------------------------------------------------------------
_ACTIONS_BY_FAMILY = {
    "UNKNOWN": (("NO_OP", "META", ()),),
    "RAYCAST_FPS": (
        ("NO_OP", "META", ()),
        ("MOVE_FORWARD", "MOVEMENT", ()),
        ("MOVE_BACKWARD", "MOVEMENT", ()),
        ("TURN_LEFT", "MOVEMENT", ()),
        ("TURN_RIGHT", "MOVEMENT", ()),
        ("STRAFE_LEFT", "MOVEMENT", ()),
        ("STRAFE_RIGHT", "MOVEMENT", ()),
        ("FIRE", "COMBAT", ()),
        ("INTERACT", "MENU", ()),
    ),
    "DOOMLIKE_2_5D": (
        ("NO_OP", "META", ()),
        ("MOVE_FORWARD", "MOVEMENT", ()),
        ("MOVE_BACKWARD", "MOVEMENT", ()),
        ("TURN_LEFT", "MOVEMENT", ()),
        ("TURN_RIGHT", "MOVEMENT", ()),
        ("STRAFE_LEFT", "MOVEMENT", ()),
        ("STRAFE_RIGHT", "MOVEMENT", ()),
        ("FIRE", "COMBAT", ()),
        ("INTERACT", "MENU", ()),
    ),
    "ATARI_RL": (
        ("NO_OP", "META", ()),
        ("UP", "PIXEL_BUTTON", ()),
        ("DOWN", "PIXEL_BUTTON", ()),
        ("LEFT", "PIXEL_BUTTON", ()),
        ("RIGHT", "PIXEL_BUTTON", ()),
        ("FIRE", "PIXEL_BUTTON", ()),
        ("UP_FIRE", "PIXEL_BUTTON", ()),
        ("DOWN_FIRE", "PIXEL_BUTTON", ()),
        ("LEFT_FIRE", "PIXEL_BUTTON", ()),
        ("RIGHT_FIRE", "PIXEL_BUTTON", ()),
    ),
    "PIXEL_ACTION_INTERFACE": (
        ("NO_OP", "META", ()),
        ("KEY_PRESS", "PIXEL_BUTTON", ("key_code",)),
        ("MOUSE_CLICK", "PIXEL_BUTTON", ("button", "x", "y")),
        ("POINTER_MOVE", "PIXEL_BUTTON", ("x", "y")),
    ),
    "ABSTRACT_STRATEGY": (
        ("NO_OP", "META", ()),
        ("SELECT_ACTION_INDEX", "STRATEGY", ("action_index",)),
        ("PASS_TURN", "STRATEGY", ()),
        ("CONFIRM", "MENU", ()),
        ("CANCEL", "MENU", ()),
    ),
    "GAME_AI_FRAMEWORK": (
        ("NO_OP", "META", ()),
        ("TICK_AGENT", "FRAMEWORK_CONTROL", ()),
        ("SELECT_BEHAVIOR_INDEX", "FRAMEWORK_CONTROL", ("behavior_index",)),
        ("REQUEST_PATH_INDEX", "FRAMEWORK_CONTROL", ("path_index",)),
        ("CANCEL_ACTION", "FRAMEWORK_CONTROL", ()),
    ),
    "RHYTHM_GENERATIVE": (
        ("NO_OP", "META", ()),
        ("PLACE_NOTE", "RHYTHM_EDIT", ("lane_index", "time_index")),
        ("DELETE_NOTE", "RHYTHM_EDIT", ("note_index",)),
        ("SHIFT_NOTE_LEFT", "RHYTHM_EDIT", ("note_index",)),
        ("SHIFT_NOTE_RIGHT", "RHYTHM_EDIT", ("note_index",)),
    ),
    "BOARD_ECONOMIC_STRATEGY": (
        ("NO_OP", "META", ()),
        ("ROLL", "BOARD_ACTION", ()),
        ("BUY", "BOARD_ACTION", ()),
        ("SELL", "BOARD_ACTION", ("property_index",)),
        ("TRADE", "BOARD_ACTION", ("counterparty_index",)),
        ("END_TURN", "BOARD_ACTION", ()),
        ("SELECT_PROPERTY", "BOARD_ACTION", ("property_index",)),
    ),
}


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------
def build_action_atom(
    action_code: str,
    action_kind: str,
    parameter_slots: list[str] | tuple[str, ...] = (),
) -> ActionAtom:
    """Build an ActionAtom with computed hash."""
    if not isinstance(parameter_slots, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    if any(not isinstance(slot, str) for slot in parameter_slots):
        raise ValueError(_ERR_INVALID_INPUT)
    slots = tuple(sorted(parameter_slots))
    if len(set(slots)) != len(slots):
        raise ValueError(_ERR_DUPLICATE_PARAMETER_SLOT)
    payload = _action_atom_payload(action_code, action_kind, slots)
    return ActionAtom(
        action_code=action_code,
        action_kind=action_kind,
        parameter_slots=slots,
        action_atom_hash=sha256_hex(payload),
    )


def build_action_alphabet(world_family: str) -> ActionAlphabet:
    """Build an ActionAlphabet for a world family with computed hash."""
    _validate_world_family(world_family)
    actions = tuple(
        sorted(
            (build_action_atom(c, k, p) for c, k, p in _ACTIONS_BY_FAMILY[world_family]),
            key=lambda a: (a.action_code, a.action_atom_hash),
        )
    )
    payload = _action_alphabet_payload(world_family, actions, len(actions))
    return ActionAlphabet(
        world_family=world_family,
        actions=actions,
        action_count=len(actions),
        action_alphabet_hash=sha256_hex(payload),
    )


def build_world_adapter_spec(archive: GameWorldArchive) -> WorldAdapterSpec:
    """Build a WorldAdapterSpec for an archive with computed hash."""
    if not isinstance(archive, GameWorldArchive):
        raise ValueError(_ERR_INVALID_INPUT)
    validate_game_world_archive(archive)
    alphabet = build_action_alphabet(archive.world_family)
    payload = _world_adapter_spec_payload(
        archive.archive_manifest_hash,
        archive.world_family,
        "STATIC_ACTION_ALPHABET_ONLY",
        alphabet,
    )
    return WorldAdapterSpec(
        archive_manifest_hash=archive.archive_manifest_hash,
        world_family=archive.world_family,
        adapter_mode="STATIC_ACTION_ALPHABET_ONLY",
        action_alphabet=alphabet,
        adapter_spec_hash=sha256_hex(payload),
    )


def build_world_adapter_contract_receipt(
    corpus_manifest: GameWorldCorpusManifest,
    intake_receipt: GameWorldIntakeReceipt,
) -> WorldAdapterContractReceipt:
    """Build a WorldAdapterContractReceipt with computed hash."""
    if not isinstance(corpus_manifest, GameWorldCorpusManifest) or not isinstance(
        intake_receipt, GameWorldIntakeReceipt
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    validate_game_world_corpus_manifest(corpus_manifest)
    validate_game_world_intake_receipt(intake_receipt)
    if intake_receipt.corpus_manifest_hash != corpus_manifest.corpus_manifest_hash:
        raise ValueError(_ERR_CORPUS_RECEIPT_MISMATCH)
    specs = tuple(
        sorted(
            (build_world_adapter_spec(a) for a in corpus_manifest.archives),
            key=lambda s: (s.archive_manifest_hash, s.adapter_spec_hash),
        )
    )
    payload = _world_adapter_contract_receipt_payload(
        corpus_manifest.corpus_manifest_hash,
        intake_receipt.receipt_hash,
        specs,
        len(specs),
    )
    return WorldAdapterContractReceipt(
        corpus_manifest_hash=corpus_manifest.corpus_manifest_hash,
        intake_receipt_hash=intake_receipt.receipt_hash,
        adapter_specs=specs,
        total_adapters=len(specs),
        adapter_contract_receipt_hash=sha256_hex(payload),
    )


# ---------------------------------------------------------------------------
# Public validators
# ---------------------------------------------------------------------------
def validate_action_atom(atom: ActionAtom) -> bool:
    """Validate an ActionAtom's integrity."""
    if not isinstance(atom, ActionAtom):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_action_atom_integrity(atom)
    return True


def validate_action_alphabet(alphabet: ActionAlphabet) -> bool:
    """Validate an ActionAlphabet's integrity."""
    if not isinstance(alphabet, ActionAlphabet):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_action_alphabet_integrity(alphabet)
    return True


def validate_world_adapter_spec(spec: WorldAdapterSpec) -> bool:
    """Validate a WorldAdapterSpec's integrity."""
    if not isinstance(spec, WorldAdapterSpec):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_world_adapter_spec_integrity(spec)
    return True


def validate_world_adapter_contract_receipt(
    receipt: WorldAdapterContractReceipt,
) -> bool:
    """Validate a WorldAdapterContractReceipt's integrity."""
    if not isinstance(receipt, WorldAdapterContractReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_world_adapter_contract_receipt_integrity(receipt)
    return True
