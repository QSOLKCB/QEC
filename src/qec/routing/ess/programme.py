# SPDX-License-Identifier: MPL-2.0
"""v173.1–.4 source-bound immutable programmes; no caller-supplied code executes."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from qec.sonify.canonical import canonical_json, canonical_sha256, validate_sha256

CONTRACT_VERSION = "173.5"
MAX_CALLS = 8
MAX_TICK = 2**31 - 1
MAX_PAYLOAD_BYTES = 4096
MAX_NATIVE_COMMANDS = 32
MAX_SEARCH_EVALUATIONS = 65536
# Shared canonical UTF-8 artifact and actual CLI input-file byte ceiling.
MAX_JSON_BYTES = 256 * 1024 * 1024
SOURCE_UNITS = (
    "routing/ess/programme.py", "routing/ess/lifecycle.py",
    "routing/crossbar/core.py", "routing/crossbar/marker.py",
    "routing/crossbar/multistage.py", "routing/crossbar/contention.py",
    "routing/crossbar/continuity.py", "sonify/canonical.py",
)


def seal(name: str, **fields: object) -> dict:
    unsigned = {"schema": "qec.ess-" + name + ".v2", "contract_version": CONTRACT_VERSION, **fields}
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def exact(actual: object, expected: dict) -> None:
    try:
        if type(actual) is not dict or canonical_json(actual) != canonical_json(expected):
            raise ValueError("INVALID_INPUT")
    except (TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def text(value: object) -> None:
    if type(value) is not str or not 1 <= len(value) <= 128:
        raise ValueError("INVALID_INPUT")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("INVALID_INPUT") from exc


def integer(value: object, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("INVALID_INPUT")


def sequence(value: object, maximum: int, minimum: int = 0) -> tuple:
    if type(value) not in (tuple, list) or not minimum <= len(value) <= maximum:
        raise ValueError("INVALID_INPUT")
    return tuple(value)


def bounded_json(value: object) -> None:
    """Bound exact canonical UTF-8 bytes, nodes and depth without a full encoding.

    Count keys, quotes, escaped/control/non-ASCII strings, delimiters and scalar
    spellings exactly as canonical_json does. Serializing each bounded string
    avoids allocating a second copy of an entire large native receipt.
    """
    stack, count, characters, byte_count = [(value, 0)], 0, 0, 0

    def string_bytes(item: str) -> int:
        return len(json.dumps(item, ensure_ascii=False).encode("utf-8"))

    try:
        while stack:
            item, depth = stack.pop()
            count += 1
            if count > 2_000_000 or depth > 64:
                raise ValueError("INVALID_INPUT")
            if type(item) is dict:
                if len(item) > 128:
                    raise ValueError("INVALID_INPUT")
                # Braces, separating commas and one colon per key/value pair.
                byte_count += 2 + max(len(item) - 1, 0) + len(item)
                for key, child in item.items():
                    if type(key) is not str or len(key) > 128:
                        raise ValueError("INVALID_INPUT")
                    byte_count += string_bytes(key)
                    stack.append((child, depth + 1))
            elif type(item) is list:
                if len(item) > 8192:
                    raise ValueError("INVALID_INPUT")
                byte_count += 2 + max(len(item) - 1, 0)
                stack.extend((child, depth + 1) for child in item)
            elif type(item) is str:
                characters += len(item)
                if len(item) > 16384 or characters > 64_000_000:
                    raise ValueError("INVALID_INPUT")
                byte_count += string_bytes(item)
            elif type(item) is bool:
                byte_count += 4 if item else 5
            elif type(item) is int:
                byte_count += len(str(item))
            elif item is None:
                byte_count += 4
            else:
                raise ValueError("INVALID_INPUT")
            if byte_count > MAX_JSON_BYTES:
                raise ValueError("INVALID_INPUT")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def bindings(pairs: dict) -> dict:
    for expected, actual in pairs.values():
        if expected is not None and validate_sha256(expected) != actual:
            raise ValueError("INVALID_INPUT")
    return {key: expected is not None for key, (expected, _) in pairs.items()}


@dataclass(frozen=True)
class TranslationTable:
    """Canonical dialed-name → ordered logical destination rows (up to four)."""
    rows: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        rows = []
        for row in sequence(self.rows, 32):
            name, targets = sequence(row, 2, 2)
            text(name)
            targets = sequence(targets, 4, 1)
            for target in targets:
                text(target)
            if len(set(targets)) != len(targets):
                raise ValueError("INVALID_INPUT")
            rows.append((name, targets))
        names = tuple(row[0] for row in rows)
        if names != tuple(sorted(set(names))):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "rows", tuple(rows))

    def as_dict(self) -> dict:
        return seal("translation-table-manifest", rows=[{"dialed": name, "destinations": list(targets)}
                                                       for name, targets in self.rows],
                    policy="exact-lookup-preserve-declared-target-order.v1")

    @classmethod
    def from_dict(cls, value: object) -> TranslationTable:
        try:
            bounded_json(value)
            result = cls(tuple((row["dialed"], row["destinations"])
                               for row in sequence(value["rows"], 32)))
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class FeatureModules:
    """One-hop forwarding, at most two delayed retries, bounded independent fan-out."""
    forwarding: tuple[tuple[str, str], ...] = ()
    retry_delays: tuple[int, ...] = ()
    max_fanout: int = 1

    def __post_init__(self) -> None:
        forwarding = tuple(sequence(row, 2, 2) for row in sequence(self.forwarding, 32))
        for source, target in forwarding:
            text(source)
            text(target)
            if source == target:
                raise ValueError("INVALID_INPUT")
        names = tuple(row[0] for row in forwarding)
        if names != tuple(sorted(set(names))):
            raise ValueError("INVALID_INPUT")
        # No cycles/chains: forwarding is deliberately one exact transformation.
        if set(names) & {row[1] for row in forwarding}:
            raise ValueError("INVALID_INPUT")
        delays = sequence(self.retry_delays, 2)
        for delay in delays:
            integer(delay, 1_000_000, 1)
        integer(self.max_fanout, 4, 1)
        object.__setattr__(self, "forwarding", forwarding)
        object.__setattr__(self, "retry_delays", delays)

    def as_dict(self) -> dict:
        return seal("feature-module-manifest", forwarding=[{"from": a, "to": b} for a, b in self.forwarding],
                    retry_delays=list(self.retry_delays), max_fanout=self.max_fanout,
                    retry_policy="after-native-rejection-only-delay-from-previous-attempt.v1",
                    fanout_policy="independent-branches-no-atomic-group-claim.v1",
                    forwarding_policy="single-exact-lookup-no-recursion.v1",
                    force_accept_permitted=False, payload_mutation_permitted=False,
                    native_validation_required=True)

    @classmethod
    def from_dict(cls, value: object) -> FeatureModules:
        try:
            bounded_json(value)
            result = cls(tuple((row["from"], row["to"]) for row in sequence(value["forwarding"], 32)),
                         value["retry_delays"], value["max_fanout"])
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def routing_logic() -> dict:
    return seal("routing-logic", states=["intake", "analyse", "select", "connect", "verify", "release", "reject"],
                transitions=[["intake", "analyse"], ["analyse", "select"], ["analyse", "reject"],
                             ["select", "connect"], ["select", "reject"], ["connect", "verify"],
                             ["verify", "release"]],
                scheduler_order=["logical_tick", "fixed_priority", "call_id", "branch", "attempt"],
                fixed_priority={"release": 0, "retry": 1, "arrival": 2},
                retry_creates_new_attempt=True, hold_ticks_minimum=1,
                connect_semantics="atomic-native-owned-path-reservation",
                verify_semantics="independent-native-replay-and-forward-continuity",
                release_semantics="native-owner-bound-whole-path-release",
                native_tick_semantics="monotonic-fabric-command-ordinal-with-logical-schedule-binding",
                max_native_commands=MAX_NATIVE_COMMANDS)


def source_identity() -> tuple[tuple[str, int, str], ...]:
    root = Path(__file__).resolve().parents[2]
    rows = []
    for unit in SOURCE_UNITS:
        data = (root / unit).read_bytes()
        rows.append(("qec/" + unit, len(data), hashlib.sha256(data).hexdigest()))
    return tuple(rows)


@dataclass(frozen=True)
class ProgramManifest:
    """Immutable identity of installed source, fixed logic, version and separate data.

    Embedded source hashes must match these installed modules on replay. Labels
    never load a module or arbitrary code. Hash identity is not code provenance.
    """
    translations: TranslationTable
    features: FeatureModules = field(default_factory=FeatureModules)
    program_version: str = "1"
    marker_id: str = "ess-lifecycle-marker"
    max_search_evaluations: int = MAX_SEARCH_EVALUATIONS
    sources: tuple = field(default_factory=source_identity, init=False, repr=False)

    def __post_init__(self) -> None:
        for name, cls in (("translations", TranslationTable), ("features", FeatureModules)):
            value = getattr(self, name)
            if type(value) is not cls:
                raise ValueError("INVALID_INPUT")
            object.__setattr__(self, name, cls.from_dict(value.as_dict()))
        text(self.program_version)
        text(self.marker_id)
        integer(self.max_search_evaluations, MAX_SEARCH_EVALUATIONS, 1)
        if any(len(targets) > self.features.max_fanout for _, targets in self.translations.rows):
            raise ValueError("INVALID_INPUT")
        if any(target not in dict(self.translations.rows) for _, target in self.features.forwarding):
            raise ValueError("INVALID_INPUT")

    def as_dict(self) -> dict:
        source = seal("source-manifest", files=[{"path": path, "byte_count": count, "sha256": digest}
                                               for path, count, digest in self.sources],
                      identity="sha256-of-complete-installed-source-bytes", authenticated_provenance=False)
        logic = routing_logic()
        return seal("program-store-manifest", program_id="ess-native-lifecycle", program_version=self.program_version,
                    marker_id=self.marker_id, max_search_evaluations=self.max_search_evaluations,
                    source_manifest=source, source_sha256=source["sha256"], routing_logic=logic,
                    routing_logic_sha256=logic["sha256"], translations=self.translations.as_dict(),
                    translation_table_sha256=self.translations.as_dict()["sha256"],
                    features=self.features.as_dict(), feature_module_sha256=self.features.as_dict()["sha256"])

    @classmethod
    def from_dict(cls, value: object) -> ProgramManifest:
        try:
            bounded_json(value)
            result = cls(TranslationTable.from_dict(value["translations"]),
                         FeatureModules.from_dict(value["features"]), value["program_version"],
                         value["marker_id"], value["max_search_evaluations"])
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def validate_program_manifest(value: object, *, expected_manifest_sha256=None,
                              expected_source_sha256=None, expected_translation_table_sha256=None) -> dict:
    try:
        bounded_json(value)
        manifest = ProgramManifest.from_dict(value).as_dict()
        pins = bindings({"manifest": (expected_manifest_sha256, manifest["sha256"]),
                         "source": (expected_source_sha256, manifest["source_sha256"]),
                         "translations": (expected_translation_table_sha256, manifest["translation_table_sha256"])})
        return seal("program-store-validation", ess_program_store_manifest_hash=manifest["sha256"],
                    installed_source_bytes_verified=True, routing_logic_verified=True,
                    external_bindings_verified=pins, all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
