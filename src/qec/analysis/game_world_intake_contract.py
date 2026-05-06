from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import hashlib
import re
import zipfile

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_ARCHIVE_NOT_FOUND = "ARCHIVE_NOT_FOUND"
_ERR_INVALID_ARCHIVE_FORMAT = "INVALID_ARCHIVE_FORMAT"
_ERR_UNSAFE_ARCHIVE_PATH = "UNSAFE_ARCHIVE_PATH"
_ERR_DUPLICATE_ARCHIVE = "DUPLICATE_ARCHIVE"
_ERR_EXECUTION_NOT_ALLOWED = "EXECUTION_NOT_ALLOWED"

_ALLOWED_INTAKE_MODES = {"MANIFEST_ONLY", "STATIC_SCAN_ONLY"}
_ALLOWED_WORLD_FAMILIES = {
    "RAYCAST_FPS",
    "DOOMLIKE_2_5D",
    "ATARI_RL",
    "ABSTRACT_STRATEGY",
    "GAME_AI_FRAMEWORK",
    "PIXEL_ACTION_INTERFACE",
    "RHYTHM_GENERATIVE",
    "BOARD_ECONOMIC_STRATEGY",
    "UNKNOWN",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_LANGUAGE_BY_EXT = {".py": "Python", ".java": "Java", ".kt": "Kotlin", ".js": "JavaScript", ".ts": "TypeScript", ".c": "C", ".cpp": "C++", ".cs": "C#", ".ipynb": "Jupyter"}
_EXECUTABLE_ENTRYPOINT_NAMES = {"main.py", "app.py", "run.py", "train.py", "setup.py"}
_NOTABLE_PROJECT_FILES = {"build.gradle", "pom.xml", "package.json", "README.md"}
_ASSET_TYPE_BY_EXT = {".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".wav": "audio", ".mp3": "audio", ".ogg": "audio", ".wad": "doom_wad", ".json": "config", ".yaml": "config", ".yml": "config", ".pt": "model_weight", ".pth": "model_weight", ".ckpt": "model_weight", ".safetensors": "model_weight", ".txt": "text", ".md": "text"}
_BUILD_FILE_NAMES = {"makefile", "cmakelists.txt"}
_BUILD_FILE_SUFFIXES = ("build.gradle", "pom.xml")


def _validate_hash_string(value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)


def _validate_count(value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(_ERR_INVALID_INPUT)


def _validate_tuple_of_strings(value: object) -> None:
    if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
        raise ValueError(_ERR_INVALID_INPUT)


def _validate_sorted_tuple_of_strings(value: object) -> None:
    _validate_tuple_of_strings(value)
    if tuple(sorted(value)) != value:
        raise ValueError(_ERR_INVALID_INPUT)


def _validate_world_family(value: object) -> None:
    if not isinstance(value, str) or value not in _ALLOWED_WORLD_FAMILIES:
        raise ValueError(_ERR_INVALID_INPUT)


def _validate_archive_entry_path(name: str) -> None:
    if "\x00" in name:
        raise ValueError(_ERR_UNSAFE_ARCHIVE_PATH)
    normalized = name.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    for part in parts:
        if part == "..":
            raise ValueError(_ERR_UNSAFE_ARCHIVE_PATH)
        if re.match(r"^[a-zA-Z]:$", part):
            raise ValueError(_ERR_UNSAFE_ARCHIVE_PATH)
    if normalized.startswith("/"):
        raise ValueError(_ERR_UNSAFE_ARCHIVE_PATH)


def _detect_world_family(archive_name: str) -> str:
    n = archive_name.lower()
    if "wolfenstein" in n or "raycast" in n:
        return "RAYCAST_FPS"
    if "doom" in n:
        return "DOOMLIKE_2_5D"
    if "atari" in n:
        return "ATARI_RL"
    if "easyai" in n:
        return "ABSTRACT_STRATEGY"
    if "gdx-ai" in n or "gdx" in n:
        return "GAME_AI_FRAMEWORK"
    if "universe" in n:
        return "PIXEL_ACTION_INTERFACE"
    if "mug" in n or "diffusion" in n:
        return "RHYTHM_GENERATIVE"
    if "monopoly" in n:
        return "BOARD_ECONOMIC_STRATEGY"
    return "UNKNOWN"


def _has_native_build_files(entries: list[str]) -> bool:
    for name in entries:
        lower = name.lower()
        filename = lower.rsplit("/", 1)[-1]
        if lower.endswith(_BUILD_FILE_SUFFIXES) or filename in _BUILD_FILE_NAMES:
            return True
    return False


def _compute_intake_warnings(
    entries: list[str],
    entrypoints: set[str],
    asset_types: set[str],
    languages: set[str],
    world_family: str,
) -> set[str]:
    warnings: set[str] = set()
    if entrypoints:
        warnings.add("CONTAINS_EXECUTABLE_ENTRYPOINT")
    if "model_weight" in asset_types:
        warnings.add("CONTAINS_MODEL_WEIGHTS")
    if "doom_wad" in asset_types:
        warnings.add("CONTAINS_DOOM_WAD")
    if _has_native_build_files(entries):
        warnings.add("CONTAINS_NATIVE_BUILD_FILES")
    if "Jupyter" in languages:
        warnings.add("CONTAINS_JUPYTER_NOTEBOOK")
    if world_family == "UNKNOWN":
        warnings.add("UNKNOWN_WORLD_FAMILY")
    return warnings


@dataclass(frozen=True)
class GameWorldArchive:
    archive_name: str
    archive_path: str
    archive_sha256: str
    file_count: int
    total_uncompressed_bytes: int
    top_level_entries: tuple[str, ...]
    detected_languages: tuple[str, ...]
    detected_entrypoints: tuple[str, ...]
    detected_asset_types: tuple[str, ...]
    world_family: str
    intake_warnings: tuple[str, ...]
    archive_manifest_hash: str

    def __post_init__(self) -> None:
        _validate_game_world_archive_integrity(self)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "archive_name": self.archive_name,
            "archive_sha256": self.archive_sha256,
            "file_count": self.file_count,
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
            "top_level_entries": list(self.top_level_entries),
            "detected_languages": list(self.detected_languages),
            "detected_entrypoints": list(self.detected_entrypoints),
            "detected_asset_types": list(self.detected_asset_types),
            "world_family": self.world_family,
            "intake_warnings": list(self.intake_warnings),
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._hash_payload().copy()
        payload["archive_path"] = self.archive_path
        payload["archive_manifest_hash"] = self.archive_manifest_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class GameWorldCorpusManifest:
    archives: tuple[GameWorldArchive, ...]
    total_archives: int
    total_files: int
    total_uncompressed_bytes: int
    corpus_manifest_hash: str

    def __post_init__(self) -> None:
        _validate_game_world_corpus_manifest_integrity(self)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "archives": [a.to_dict() for a in self.archives],
            "total_archives": self.total_archives,
            "total_files": self.total_files,
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._hash_payload().copy()
        payload["corpus_manifest_hash"] = self.corpus_manifest_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class GameWorldIntakeReceipt:
    corpus_manifest_hash: str
    source_policy_hash: str
    execution_allowed: bool
    intake_mode: str
    receipt_hash: str

    def __post_init__(self) -> None:
        _validate_game_world_intake_receipt_integrity(self)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "corpus_manifest_hash": self.corpus_manifest_hash,
            "source_policy_hash": self.source_policy_hash,
            "execution_allowed": self.execution_allowed,
            "intake_mode": self.intake_mode,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._hash_payload().copy()
        payload["receipt_hash"] = self.receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _validate_game_world_archive_integrity(archive: GameWorldArchive) -> None:
    if not isinstance(archive.archive_name, str) or not isinstance(archive.archive_path, str) or archive.archive_path == "":
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(archive.archive_sha256)
    _validate_count(archive.file_count)
    _validate_count(archive.total_uncompressed_bytes)
    _validate_sorted_tuple_of_strings(archive.top_level_entries)
    _validate_sorted_tuple_of_strings(archive.detected_languages)
    _validate_sorted_tuple_of_strings(archive.detected_entrypoints)
    _validate_sorted_tuple_of_strings(archive.detected_asset_types)
    _validate_sorted_tuple_of_strings(archive.intake_warnings)
    _validate_world_family(archive.world_family)
    _validate_hash_string(archive.archive_manifest_hash)
    if archive.archive_manifest_hash != sha256_hex(archive._hash_payload()):
        raise ValueError(_ERR_HASH_MISMATCH)


def _validate_game_world_corpus_manifest_integrity(manifest: GameWorldCorpusManifest) -> None:
    if not isinstance(manifest.archives, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    for archive in manifest.archives:
        if not isinstance(archive, GameWorldArchive):
            raise ValueError(_ERR_INVALID_INPUT)
        _validate_game_world_archive_integrity(archive)
    _validate_count(manifest.total_archives)
    _validate_count(manifest.total_files)
    _validate_count(manifest.total_uncompressed_bytes)
    sorted_archives = tuple(sorted(manifest.archives, key=lambda a: (a.archive_name, a.archive_sha256)))
    if manifest.archives != sorted_archives:
        raise ValueError(_ERR_INVALID_INPUT)
    names = [a.archive_name for a in manifest.archives]
    if len(set(names)) != len(names):
        raise ValueError(_ERR_DUPLICATE_ARCHIVE)
    if manifest.total_archives != len(manifest.archives):
        raise ValueError(_ERR_INVALID_INPUT)
    if manifest.total_files != sum(a.file_count for a in manifest.archives):
        raise ValueError(_ERR_INVALID_INPUT)
    if manifest.total_uncompressed_bytes != sum(a.total_uncompressed_bytes for a in manifest.archives):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(manifest.corpus_manifest_hash)
    if manifest.corpus_manifest_hash != sha256_hex(manifest._hash_payload()):
        raise ValueError(_ERR_HASH_MISMATCH)


def _validate_game_world_intake_receipt_integrity(receipt: GameWorldIntakeReceipt) -> None:
    _validate_hash_string(receipt.corpus_manifest_hash)
    _validate_hash_string(receipt.source_policy_hash)
    if not isinstance(receipt.execution_allowed, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    if receipt.execution_allowed is True:
        raise ValueError(_ERR_EXECUTION_NOT_ALLOWED)
    if receipt.intake_mode not in _ALLOWED_INTAKE_MODES:
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(receipt.receipt_hash)
    if receipt.receipt_hash != sha256_hex(receipt._hash_payload()):
        raise ValueError(_ERR_HASH_MISMATCH)


def build_game_world_archive(archive_path: str) -> GameWorldArchive:
    if not isinstance(archive_path, str) or archive_path == "":
        raise ValueError(_ERR_INVALID_INPUT)
    path = Path(archive_path)
    if not path.exists():
        raise ValueError(_ERR_ARCHIVE_NOT_FOUND)
    if path.suffix.lower() != ".zip":
        raise ValueError(_ERR_INVALID_ARCHIVE_FORMAT)
    try:
        archive_bytes = path.read_bytes()
        with zipfile.ZipFile(path, "r") as zf:
            infos = sorted(zf.infolist(), key=lambda zi: zi.filename)
    except (OSError, zipfile.BadZipFile):
        raise ValueError(_ERR_INVALID_ARCHIVE_FORMAT)

    entries: list[str] = []
    top_levels: set[str] = set()
    languages: set[str] = set()
    entrypoints: set[str] = set()
    asset_types: set[str] = set()
    total_bytes = 0
    for zi in infos:
        name = zi.filename
        _validate_archive_entry_path(name)
        entries.append(name)
        top_levels.add(name.split("/", 1)[0])
        if zi.is_dir():
            continue
        total_bytes += zi.file_size
        file_name = name.rsplit("/", 1)[-1]
        lower = name.lower()
        for ext, lang in _LANGUAGE_BY_EXT.items():
            if lower.endswith(ext):
                languages.add(lang)
        if file_name in _EXECUTABLE_ENTRYPOINT_NAMES:
            entrypoints.add(name)
        for ext, label in _ASSET_TYPE_BY_EXT.items():
            if lower.endswith(ext):
                asset_types.add(label)

    world_family = _detect_world_family(path.name)
    warnings = _compute_intake_warnings(entries, entrypoints, asset_types, languages, world_family)

    hash_payload = {
        "archive_name": path.name,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "file_count": sum(1 for zi in infos if not zi.is_dir()),
        "total_uncompressed_bytes": total_bytes,
        "top_level_entries": list(sorted(top_levels)),
        "detected_languages": list(sorted(languages)),
        "detected_entrypoints": list(sorted(entrypoints)),
        "detected_asset_types": list(sorted(asset_types)),
        "world_family": world_family,
        "intake_warnings": list(sorted(warnings)),
    }
    return GameWorldArchive(
        archive_name=path.name,
        archive_path=archive_path,
        archive_sha256=hash_payload["archive_sha256"],
        file_count=hash_payload["file_count"],
        total_uncompressed_bytes=total_bytes,
        top_level_entries=tuple(sorted(top_levels)),
        detected_languages=tuple(sorted(languages)),
        detected_entrypoints=tuple(sorted(entrypoints)),
        detected_asset_types=tuple(sorted(asset_types)),
        world_family=world_family,
        intake_warnings=tuple(sorted(warnings)),
        archive_manifest_hash=sha256_hex(hash_payload),
    )


def build_game_world_corpus_manifest(archive_paths: list[str] | tuple[str, ...]) -> GameWorldCorpusManifest:
    if not isinstance(archive_paths, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    archives = tuple(sorted((build_game_world_archive(p) for p in archive_paths), key=lambda a: (a.archive_name, a.archive_sha256)))
    names = [a.archive_name for a in archives]
    if len(names) != len(set(names)):
        raise ValueError(_ERR_DUPLICATE_ARCHIVE)
    payload = {"archives": [a.to_dict() for a in archives], "total_archives": len(archives), "total_files": sum(a.file_count for a in archives), "total_uncompressed_bytes": sum(a.total_uncompressed_bytes for a in archives)}
    return GameWorldCorpusManifest(archives=archives, total_archives=payload["total_archives"], total_files=payload["total_files"], total_uncompressed_bytes=payload["total_uncompressed_bytes"], corpus_manifest_hash=sha256_hex(payload))


def build_game_world_intake_receipt(corpus_manifest: GameWorldCorpusManifest, source_policy_hash: str, execution_allowed: bool = False, intake_mode: str = "MANIFEST_ONLY") -> GameWorldIntakeReceipt:
    if not isinstance(corpus_manifest, GameWorldCorpusManifest):
        raise ValueError(_ERR_INVALID_INPUT)
    validate_game_world_corpus_manifest(corpus_manifest)
    _validate_hash_string(source_policy_hash)
    if not isinstance(execution_allowed, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    if execution_allowed is True:
        raise ValueError(_ERR_EXECUTION_NOT_ALLOWED)
    if intake_mode not in _ALLOWED_INTAKE_MODES:
        raise ValueError(_ERR_INVALID_INPUT)
    payload = {"corpus_manifest_hash": corpus_manifest.corpus_manifest_hash, "source_policy_hash": source_policy_hash, "execution_allowed": execution_allowed, "intake_mode": intake_mode}
    return GameWorldIntakeReceipt(**payload, receipt_hash=sha256_hex(payload))


def validate_game_world_archive(archive: GameWorldArchive) -> bool:
    if not isinstance(archive, GameWorldArchive):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_game_world_archive_integrity(archive)
    return True


def validate_game_world_corpus_manifest(manifest: GameWorldCorpusManifest) -> bool:
    if not isinstance(manifest, GameWorldCorpusManifest):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_game_world_corpus_manifest_integrity(manifest)
    return True


def validate_game_world_intake_receipt(receipt: GameWorldIntakeReceipt) -> bool:
    if not isinstance(receipt, GameWorldIntakeReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_game_world_intake_receipt_integrity(receipt)
    return True
