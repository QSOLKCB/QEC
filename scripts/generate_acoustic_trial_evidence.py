# SPDX-License-Identifier: MPL-2.0
"""Export bounded replayable comparisons, actual WAV/JSON, and a native byte demo.

The canonical battery excludes runtime, memory, git state and dependency versions.
Those are separately measured/recorded in environment.json and a file manifest.
No installation, hardware testing, SPECTRAL UI integration or publishing occurs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import sys
import types
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import stat
import struct
import subprocess
import time
import tracemalloc

from qec.experiments.acoustic.channel import PCM, read_wav, wav_bytes
from qec.experiments.acoustic.export_fs import SecureExportWriter
from qec.experiments.acoustic.evidence import (
    CASE_NAMES, build_battery_source, build_demonstration, build_trial_source,
    run_comparative_battery, validate_comparative_battery, validate_demonstration,
)
from qec.experiments.acoustic.transport import (
    ARTIFACT_ROLES, TRANSPORTS, AcousticTransportArtifact,
    execute_acoustic_transport, validate_acoustic_transport_receipt,
)
from qec.sonify.canonical import canonical_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
MAX_EXPORT_FILE_BYTES = 32 * 1024 * 1024
MAX_EXPORT_TOTAL_BYTES = 32 * 1024 * 1024


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not_installed"


def _portable_relative(path, root):
    """Canonical manifest paths always use forward slashes, including Windows."""
    return path.relative_to(root).as_posix()


def _code_manifest():
    # Hash actual source bytes, including imported decoder and native transport
    # dependency trees. This is deliberately broader than just entrypoints.
    files = [ROOT / "scripts/generate_acoustic_trial_evidence.py"]
    files.extend(ROOT / name for name in ("src/qec/__init__.py", "src/qec/experiments/__init__.py",
                                          "src/qec/routing/__init__.py"))
    for relative in ("src/qec/experiments/acoustic", "src/qec/decoder", "src/qec/sonify",
                     "src/qec/routing/digital", "src/qec/routing/packet"):
        files.extend(sorted((ROOT / relative).rglob("*.py")))
    for name in ("pyproject.toml", "requirements.txt", "requirements-test.txt", "dev-requirements.txt"):
        if (ROOT / name).is_file():
            files.append(ROOT / name)
    return [{"path": _portable_relative(path, ROOT), "sha256": _sha(path.read_bytes()), "bytes": path.stat().st_size}
            for path in sorted(set(files))]


def _write_json(path, value):
    path.write_bytes(canonical_bytes(value) + b"\n")


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _same_file_identity(left, right, *, path_descriptor=False):
    left_identity, right_identity = _file_identity(left), _file_identity(right)
    if path_descriptor and os.name == "nt":
        # Windows path stat reports creation time as ctime; fstat reports change time.
        left_identity, right_identity = left_identity[:-1], right_identity[:-1]
    return left_identity == right_identity


def _regular_stat(path, *, dir_fd=None):
    info = path.lstat() if dir_fd is None else os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
    if (not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= MAX_EXPORT_FILE_BYTES
            or getattr(info, "st_file_attributes", 0) & 0x400):
        raise ValueError("INVALID_INPUT")
    return info


def _read_regular_bytes(path, *, expected_stat=None, dir_fd=None):
    """Read one bounded regular-file snapshot without blocking on a FIFO race.

    POSIX nonblocking/no-follow flags are used when available. Every platform
    checks both lstat and descriptor fstat before reading; identity, size and
    timestamps must remain stable through the bounded read. No special file or
    symlink is accepted. The extra byte detects growth beyond the declared size.
    """
    descriptor = None
    try:
        before = _regular_stat(path, dir_fd=dir_fd)
        if expected_stat is not None and _file_identity(before) != _file_identity(expected_stat):
            raise ValueError("INVALID_INPUT")
        flags = os.O_RDONLY
        for name in ("O_NONBLOCK", "O_NOFOLLOW", "O_CLOEXEC", "O_BINARY", "O_NOINHERIT"):
            flags |= getattr(os, name, 0)
        descriptor = os.open(path, flags) if dir_fd is None else os.open(path, flags, dir_fd=dir_fd)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(before, opened, path_descriptor=True):
            raise ValueError("INVALID_INPUT")
        if _file_identity(_regular_stat(path, dir_fd=dir_fd)) != _file_identity(before):
            raise ValueError("INVALID_INPUT")
        chunks, count = [], 0
        while count <= opened.st_size:
            chunk = os.read(descriptor, min(65536, opened.st_size + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
        if (count != opened.st_size or _file_identity(os.fstat(descriptor)) != _file_identity(opened)
                or _file_identity(_regular_stat(path, dir_fd=dir_fd)) != _file_identity(before)):
            raise ValueError("INVALID_INPUT")
        return b"".join(chunks)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _parse_json(data):
    try:
        value = json.loads(data)
        if type(value) is not dict or canonical_bytes(value) + b"\n" != data:
            raise ValueError("INVALID_INPUT")
        return value
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _read_json(path):
    return _parse_json(_read_regular_bytes(path))


class _InventoryAccess:
    """File access restricted to names discovered under retained directory guards."""
    def __init__(self, names, locations, *, stat_file=None, read_file=None):
        self.names, self._locations = sorted(names), locations
        self._stat_file, self._read_file = stat_file, read_file

    def stat(self, name):
        if self._stat_file is not None:
            return self._stat_file(name)
        path, parent, discovered = self._locations[name]
        observed = _regular_stat(path, dir_fd=parent)
        if _file_identity(observed) != _file_identity(discovered):
            raise ValueError("INVALID_INPUT")
        return observed

    def read(self, name, *, expected_stat=None):
        if self._read_file is not None:
            return self._read_file(name, expected_stat=expected_stat)
        path, parent, discovered = self._locations[name]
        if expected_stat is not None and _file_identity(expected_stat) != _file_identity(discovered):
            raise ValueError("INVALID_INPUT")
        return _read_regular_bytes(path, expected_stat=discovered, dir_fd=parent)


@contextmanager
def _inventory_posix(output, *, file_access=False):
    """Walk retained no-follow directory descriptors, never queued pathnames."""
    if (os.scandir not in os.supports_fd or os.open not in os.supports_dir_fd
            or os.stat not in os.supports_dir_fd or os.stat not in os.supports_follow_symlinks
            or not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))):
        raise ValueError("INVALID_INPUT")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    retained, snapshots, names, locations, directories = [], [], [], {}, 0
    def open_directory(name, expected, parent=None):
        descriptor = os.open(name, flags, dir_fd=parent)
        retained.append(descriptor)
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode) or _file_identity(observed) != _file_identity(expected):
            raise ValueError("INVALID_INPUT")
        snapshots.append((descriptor, observed))
        return descriptor, observed
    try:
        root, root_info = open_directory(output, output.lstat())
        pending = [(root, "", 0, root_info)]
        while pending:
            descriptor, prefix, depth, initial = pending.pop()
            if _file_identity(os.fstat(descriptor)) != _file_identity(initial):
                raise ValueError("INVALID_INPUT")
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    relative = (Path(prefix) / entry.name).as_posix()
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        directories += 1
                        if directories > 16 or depth >= 4:
                            raise ValueError("INVALID_INPUT")
                        child, child_info = open_directory(entry.name, info, descriptor)
                        pending.append((child, relative, depth + 1, child_info))
                    elif not stat.S_ISREG(info.st_mode):
                        raise ValueError("INVALID_INPUT")
                    else:
                        locations[relative] = (Path(entry.name), descriptor, info)
                        if relative != "manifest.json":
                            names.append(relative)
                            if len(names) > 128:
                                raise ValueError("INVALID_INPUT")
            if _file_identity(os.fstat(descriptor)) != _file_identity(initial):
                raise ValueError("INVALID_INPUT")
        yield _InventoryAccess(names, locations) if file_access else sorted(names)
        for descriptor, initial in snapshots:
            if _file_identity(os.fstat(descriptor)) != _file_identity(initial):
                raise ValueError("INVALID_INPUT")
    finally:
        for descriptor in reversed(retained):
            os.close(descriptor)


def _parse_windows_directory_buffer(data):
    """Parse bounded FILE_ID_EXTD_DIR_INFO records; no pathname traversal here."""
    header = struct.Struct("<II6q4I16s")
    if type(data) is not bytes or not header.size <= len(data) <= 65536:
        raise ValueError("INVALID_INPUT")
    records, offset = [], 0
    while True:
        if offset + header.size > len(data) or len(records) >= 160:
            raise ValueError("INVALID_INPUT")
        fields = header.unpack_from(data, offset)
        following, attributes, length, reparse_tag, file_id = fields[0], fields[8], fields[9], fields[11], fields[12]
        end = offset + header.size + length
        if not length or length % 2 or end > len(data):
            raise ValueError("INVALID_INPUT")
        name = data[offset + header.size:end].decode("utf-16-le")
        if any(character in name for character in ("/", "\\", ":", "\0")):
            raise ValueError("INVALID_INPUT")
        records.append((name, attributes, reparse_tag, file_id))
        if not following:
            return records
        if following < header.size + length or following % 8:
            raise ValueError("INVALID_INPUT")
        offset += following


@contextmanager
def _inventory_windows(output, *, file_access=False):
    """Enumerate guarded Windows handles, with native 128-bit child identity.

    GetFileInformationByHandleEx(FileIdExtdDirectoryInfo) enumerates the retained
    handle itself. Only the root uses CreateFileW; descendants and payloads
    use inventoried file IDs with OpenFileById and OPEN_REPARSE_POINT. Unsupported
    APIs/filesystems fail closed, without pathname fallbacks. All acquired handles
    remain open through the caller's byte snapshot.
    Microsoft API contracts: https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-getfileinformationbyhandleex
    and https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-openfilebyid
    """
    import ctypes
    from ctypes import wintypes
    from types import SimpleNamespace
    if not hasattr(ctypes, "WinDLL"):
        raise ValueError("INVALID_INPUT")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    query.restype = wintypes.BOOL
    close = kernel.CloseHandle
    close.argtypes, close.restype = (wintypes.HANDLE,), wintypes.BOOL
    open_id = kernel.OpenFileById
    open_id.argtypes = (wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                        wintypes.LPVOID, wintypes.DWORD)
    open_id.restype = wintypes.HANDLE
    file_type = kernel.GetFileType
    file_type.argtypes, file_type.restype = (wintypes.HANDLE,), wintypes.DWORD
    read = kernel.ReadFile
    read.argtypes = (wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                     ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID)
    read.restype = wintypes.BOOL
    seek = kernel.SetFilePointerEx
    seek.argtypes = (wintypes.HANDLE, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), wintypes.DWORD)
    seek.restype = wintypes.BOOL
    file_handles = {}
    retained, snapshots, names, locations, directories, calls = [], [], [], {}, 0, 0
    def native_failure(operation):
        # Keep API/error diagnostics in the chain, without file paths or bytes.
        raise ValueError("INVALID_INPUT") from OSError(ctypes.get_last_error(), operation)
    def info(handle, kind, size):
        buffer = ctypes.create_string_buffer(size)
        if not query(handle, kind, buffer, size):
            native_failure(f"GetFileInformationByHandleEx({kind})")
        return buffer.raw
    def open_by_id(expected, access):
        # FILE_ID_DESCRIPTOR is two DWORDs followed by the 16-byte union.
        # The retained root is a volume hint, not a pathname resolution base.
        descriptor = ctypes.create_string_buffer(struct.pack("<II16s", 24, 2, expected[1]))
        handle = open_id(root, descriptor, access, 0x1, None, 0x02200000)
        if (handle in (None, ctypes.c_void_p(-1).value) and ctypes.get_last_error() == 87
                and expected[1][8:] == b"\0" * 8):
            # Some filesystems accept only FileIdType (64 bits). This capability
            # retry loses no ID bits, changes no access flags, and still requires
            # exact full 128-bit/volume verification before enumeration or reads.
            # ACCESS_DENIED and sharing errors are never retried.
            descriptor = ctypes.create_string_buffer(struct.pack("<II16s", 24, 0, expected[1]))
            handle = open_id(root, descriptor, access, 0x1, None, 0x02200000)
        if handle in (None, ctypes.c_void_p(-1).value):
            native_failure("OpenFileById")
        return handle
    def open_directory(path, expected=None):
        # FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES; FILE_SHARE_READ only;
        # OPEN_EXISTING; BACKUP_SEMANTICS | OPEN_REPARSE_POINT.
        handle = (create(str(path), 0x81, 0x1, None, 3, 0x02200000, None)
                  if expected is None else open_by_id(expected, 0x81))
        if handle in (None, ctypes.c_void_p(-1).value):
            native_failure("CreateFileW")
        retained.append(handle)
        snapshot = directory_snapshot(handle)
        identity = snapshot[0]
        if expected is not None and identity != expected:
            raise ValueError("INVALID_INPUT")
        snapshots.append((handle, snapshot))
        return handle, identity
    def directory_snapshot(handle):
        attributes, reparse_tag = struct.unpack("<II", info(handle, 9, 8))
        identity = struct.unpack("<Q16s", info(handle, 18, 24))
        basic = struct.unpack_from("<4qI", info(handle, 0, 40))
        if (not attributes & 0x10 or attributes & (0x400 | 0x40)
                or not any(identity[1]) or basic[4] != attributes):
            raise ValueError("INVALID_INPUT")
        # Access time can change due to reading. Creation/write/change time and
        # reparse attributes must stay exact, including after caller snapshot reads.
        return identity, attributes, basic[0], basic[2], basic[3]
    def file_snapshot(handle, expected):
        if file_type(handle) != 1:  # FILE_TYPE_DISK, never a pipe/device.
            raise ValueError("INVALID_INPUT")
        attributes, _ = struct.unpack("<II", info(handle, 9, 8))
        identity = struct.unpack("<Q16s", info(handle, 18, 24))
        basic = struct.unpack_from("<4qI", info(handle, 0, 40))
        standard = struct.unpack("<qqIBB2x", info(handle, 1, 24))
        if (identity != expected or not any(identity[1]) or attributes & (0x10 | 0x400 | 0x40)
                or basic[4] != attributes or standard[3] or standard[4]
                or not 0 <= standard[1] <= MAX_EXPORT_FILE_BYTES):
            raise ValueError("INVALID_INPUT")
        return identity, standard[1], attributes, basic[0], basic[2], basic[3], standard
    def stat_file(name):
        expected = locations[name]
        if name not in file_handles:
            handle = open_by_id(expected, 0x80000000)  # GENERIC_READ
            retained.append(handle)
            initial = file_snapshot(handle, expected)
            file_handles[name] = handle, initial
        handle, initial = file_handles[name]
        if file_snapshot(handle, expected) != initial:
            raise ValueError("INVALID_INPUT")
        return SimpleNamespace(st_size=initial[1], native_snapshot=initial)
    def read_file(name, *, expected_stat=None):
        observed = stat_file(name)
        if expected_stat is not None and expected_stat.native_snapshot != observed.native_snapshot:
            raise ValueError("INVALID_INPUT")
        handle, initial = file_handles[name]
        if not seek(handle, 0, None, 0):  # FILE_BEGIN
            native_failure("SetFilePointerEx")
        chunks, count, calls = [], 0, 0
        while count <= observed.st_size:
            calls += 1
            if calls > MAX_EXPORT_FILE_BYTES // 65536 + 2:
                raise ValueError("INVALID_INPUT")
            requested = min(65536, observed.st_size + 1 - count)
            buffer, received = ctypes.create_string_buffer(requested), wintypes.DWORD()
            if not read(handle, buffer, requested, ctypes.byref(received), None):
                native_failure("ReadFile")
            if received.value > requested:
                raise ValueError("INVALID_INPUT")
            if not received.value:
                break
            chunks.append(buffer.raw[:received.value])
            count += received.value
        if count != observed.st_size or file_snapshot(handle, locations[name]) != initial:
            raise ValueError("INVALID_INPUT")
        return b"".join(chunks)
    try:
        output = output.absolute()
        root, root_id = open_directory(output)
        pending = [(root, "", 0, root_id)]
        while pending:
            handle, prefix, depth, directory_id = pending.pop()
            while True:
                calls += 1
                if calls > 256:
                    raise ValueError("INVALID_INPUT")
                buffer = ctypes.create_string_buffer(65536)
                if not query(handle, 0x13, buffer, len(buffer)):
                    if ctypes.get_last_error() == 18:  # ERROR_NO_MORE_FILES
                        break
                    native_failure("GetFileInformationByHandleEx(19)")
                for name, attributes, reparse_tag, file_id in _parse_windows_directory_buffer(buffer.raw):
                    if name in (".", ".."):
                        continue
                    if attributes & (0x400 | 0x40):
                        raise ValueError("INVALID_INPUT")
                    relative = (Path(prefix) / name).as_posix()
                    if attributes & 0x10:
                        directories += 1
                        if directories > 16 or depth >= 4:
                            raise ValueError("INVALID_INPUT")
                        child, child_id = open_directory(output / relative, (directory_id[0], file_id))
                        pending.append((child, relative, depth + 1, child_id))
                    else:
                        locations[relative] = (directory_id[0], file_id)
                        if relative != "manifest.json":
                            names.append(relative)
                            if len(names) > 128:
                                raise ValueError("INVALID_INPUT")
        for handle, initial in snapshots:
            if directory_snapshot(handle) != initial:
                raise ValueError("INVALID_INPUT")
        yield (_InventoryAccess(names, locations, stat_file=stat_file, read_file=read_file)
               if file_access else sorted(names))
        for name, (handle, initial) in file_handles.items():
            if file_snapshot(handle, locations[name]) != initial:
                raise ValueError("INVALID_INPUT")
        for handle, initial in snapshots:
            if directory_snapshot(handle) != initial:
                raise ValueError("INVALID_INPUT")
    finally:
        for handle in reversed(retained):
            close(handle)


@contextmanager
def _open_inventory(output, *, file_access=False):
    """Retain no-follow directory guards until snapshot reads finish."""
    try:
        implementation = _inventory_posix if os.name == "posix" else _inventory_windows if os.name == "nt" else None
        if implementation is None:
            raise ValueError("INVALID_INPUT")
        with implementation(output, file_access=file_access) as names:
            yield names
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _inventory_files(output):
    with _open_inventory(output) as names:
        return names


_MANIFEST_FIXED = {
    "schema": "qec.acoustic-export-manifest.v1", "contract_version": "177.5",
    "transport_scope": "six_actual_role_payloads_in_separate_demo; no_per_battery_transport_claim",
    "spectral_compatibility": "ordinary_mono_PCM16_WAV_and_canonical_JSON_export_only",
    "spectral_repository_or_ui_integration": False,
    "environment_metadata_is_canonical_scientific_proof": False,
}
_MANIFEST_DYNAMIC = ("sha256", "canonical_battery_sha256", "battery_source_sha256",
    "demonstration_source_sha256", "measured_reference_sha256", "source_commit_sha", "source_files",
    "clean_working_tree_at_start", "dependencies", "files", "transport_demonstration",
    "artifact_total_bytes_excluding_manifest", "digital_binding_specimen")
_ENVIRONMENT_FIXED = {
    "schema": "qec.acoustic-environment.v1",
    "measurement_scope": "battery_build_and_replay_plus_export_and_native_demo; not_process_RSS",
    "performance_is_environment_dependent": True, "performance_is_canonical_proof": False,
    "physical_latency_or_hardware_QA": False, "full_suite_gate": "separate_exact_commit_test_run",
}
_ENVIRONMENT_DYNAMIC = ("source_commit_sha", "clean_working_tree_at_start", "git_status_porcelain_at_start",
    "source_files", "python", "platform", "dependencies", "runtime_ns", "peak_python_tracemalloc_bytes")


def _strict_metadata(value, fixed, dynamic):
    from qec.experiments.acoustic._common import exact
    # Reconstruct the exact permitted schema; external hash pins never authorize
    # altered scientific claim flags or unknown fields.
    exact(value, {**fixed, **{key: value[key] for key in dynamic}})


def _validate_environment(value):
    _strict_metadata(value, _ENVIRONMENT_FIXED, _ENVIRONMENT_DYNAMIC)
    if (type(value["source_commit_sha"]) is not str or
            re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value["source_commit_sha"]) is None):
        raise ValueError("INVALID_INPUT")
    for key, limit in (("python", 128), ("platform", 2048), ("git_status_porcelain_at_start", 1_000_000)):
        if type(value[key]) is not str or len(value[key]) > limit:
            raise ValueError("INVALID_INPUT")
    if (type(value["clean_working_tree_at_start"]) is not bool or
            value["clean_working_tree_at_start"] != (not bool(value["git_status_porcelain_at_start"]))):
        raise ValueError("INVALID_INPUT")
    for key in ("runtime_ns", "peak_python_tracemalloc_bytes"):
        if type(value[key]) is not int or not 0 <= value[key] < 2**63:
            raise ValueError("INVALID_INPUT")
    dependencies = value["dependencies"]
    if (type(dependencies) is not dict or set(dependencies) != {"qec", "numpy", "scipy", "pytest"}
            or any(type(v) is not str or not 1 <= len(v) <= 128 for v in dependencies.values())):
        raise ValueError("INVALID_INPUT")
    if value["source_files"] != _code_manifest():
        raise ValueError("INVALID_INPUT")



def _source_code_objects(code):
    """Index compiled functions without executing source."""
    result = {}
    def walk(node, prefix=""):
        for item in node.co_consts:
            if isinstance(item, types.CodeType):
                # co_qualname was introduced in CPython 3.11; 3.10 exposes
                # only co_name, so derive lexical class/function nesting.
                qualified = getattr(item, "co_qualname", None) or (
                    prefix + "." + item.co_name if prefix else item.co_name)
                result[qualified] = item
                walk(item, qualified)
    walk(code)
    return result


def _loaded_code_matches_disk():
    """Compare imported functions with freshly compiled tracked source bytes."""
    tracked = {row["path"] for row in _code_manifest()}
    for module in tuple(sys.modules.values()):
        if module is None or not str(getattr(module, "__name__", "")).startswith("qec."):
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        path = Path(origin).resolve()
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            continue
        if relative not in tracked:
            continue
        compiled = _source_code_objects(compile(path.read_bytes(), origin, "exec", dont_inherit=True))
        def check(candidate):
            if isinstance(candidate, (staticmethod, classmethod)):
                candidate = candidate.__func__
            if isinstance(candidate, property):
                for accessor in (candidate.fget, candidate.fset, candidate.fdel):
                    if accessor:
                        check(accessor)
                return
            if inspect.isfunction(candidate) and candidate.__module__ == module.__name__:
                unwrapped = inspect.unwrap(candidate)
                runtime = unwrapped.__code__
                # CPython 3.10 code objects lack co_qualname.
                source = compiled.get(getattr(runtime, "co_qualname", unwrapped.__qualname__))
                if source is not None and runtime != source:
                    raise ValueError("imported source changed; restart against a stable checkout")
        for item in vars(module).values():
            check(item)
            if inspect.isclass(item) and item.__module__ == module.__name__:
                for attribute in vars(item).values():
                    check(attribute)


def _environment_snapshot(commit, dirty, code_files):
    return {"schema": "qec.acoustic-environment.v1", "source_commit_sha": commit,
                   "clean_working_tree_at_start": not bool(dirty), "git_status_porcelain_at_start": dirty,
                   "source_files": code_files, "python": platform.python_version(), "platform": platform.platform(),
                   "dependencies": {name: _version(name) for name in ("qec", "numpy", "scipy", "pytest")},
                   "runtime_ns": 0, "peak_python_tracemalloc_bytes": 0,
                   "measurement_scope": "battery_build_and_replay_plus_export_and_native_demo; not_process_RSS",
                   "performance_is_environment_dependent": True, "performance_is_canonical_proof": False,
                   "physical_latency_or_hardware_QA": False, "full_suite_gate": "separate_exact_commit_test_run"}


def _ensure_source_snapshot(commit, code_files):
    if _git("rev-parse", "HEAD") != commit or _code_manifest() != code_files:
        raise ValueError("source changed during export; rerun against a stable checkout")


def generate(output: Path, *, cases=CASE_NAMES, alphabets=(3, 4), seed=1775, transports=("packet",)):
    """Create an empty-directory export; return canonical identity and source pins."""
    if type(output) is not Path:
        output = Path(output)
    if (not transports or type(transports) not in (tuple, list) or len(transports) > 2
            or any(type(t) is not str or t not in TRANSPORTS for t in transports)
            or len(set(transports)) != len(transports)):
        raise ValueError("INVALID_INPUT")
    # The writer performs its empty-directory check under an exclusive claim.
    # Do not traverse the destination via mutable pathnames before claiming it.
    # Pin and validate the environment before any fixture computation. Inputs
    # built under an earlier source state may never be labelled with a later one.
    commit, dirty, code_files = _git("rev-parse", "HEAD"), _git("status", "--porcelain"), _code_manifest()
    environment = _environment_snapshot(commit, dirty, code_files)
    _validate_environment(environment)
    _loaded_code_matches_disk()
    source = build_battery_source(cases=cases, alphabets=alphabets, seed=seed)
    demo_source = build_trial_source("broadband", alphabet_size=3, seed=seed)
    _ensure_source_snapshot(commit, code_files)
    with SecureExportWriter(output, MAX_EXPORT_FILE_BYTES, MAX_EXPORT_TOTAL_BYTES) as writer:
        def save(path, value):
            writer.save(_portable_relative(path, output), canonical_bytes(value) + b"\n")
        tracemalloc.start()
        start = time.perf_counter_ns()
        try:
            battery = run_comparative_battery(source)
            validation = validate_comparative_battery(battery, expected_source_sha256=source["sha256"])
            demo = build_demonstration(demo_source)
            demo_validation = validate_demonstration(demo, expected_source_sha256=demo_source["sha256"],
                                                    expected_reference_sha256=demo_source["measured_reference"]["sha256"])
            save(output / "battery.json", battery)
            save(output / "battery_source.json", source)
            save(output / "battery_validation.json", validation)
            save(output / "demonstration.json", demo)
            save(output / "demonstration_source.json", demo_source)
            save(output / "demonstration_validation.json", demo_validation)
            writer.mkdir("trials")
            for index, trial in enumerate(battery["trials"]):
                stem = f"{index:02d}_q{trial['source']['alphabet_size']}_{trial['source']['case_name']}"
                save(output / "trials" / (stem + ".json"), trial)
                save(output / "trials" / (stem + "_source.json"), trial["source"])
            writer.mkdir("lab")
            writer.mkdir("transport")
            transport_index = []
            for role in ARTIFACT_ROLES:
                value = demo["roles"][role]
                save(output / "lab" / (role + ".json"), value)
                if role in ("original", "observed", "filtered"):
                    writer.save("lab/" + role + ".wav", wav_bytes(PCM.from_dict(value)))
                artifact = AcousticTransportArtifact.from_artifact(value, role=role, source_sha256=demo_source["sha256"])
                for transport in transports:
                    receipt = execute_acoustic_transport(artifact, transport=transport)
                    result = validate_acoustic_transport_receipt(
                        receipt, expected_artifact_sha256=artifact.as_dict()["sha256"],
                        expected_source_sha256=demo_source["sha256"], expected_input_sha256=receipt["input"]["sha256"])
                    name = f"transport/{role}_{transport}.json"
                    save(output / name, receipt)
                    transport_index.append({"role": role, "transport": transport, "path": name,
                                            "artifact_sha256": artifact.as_dict()["sha256"],
                                            "input_sha256": receipt["input"]["sha256"],
                                            "receipt_sha256": receipt["sha256"], "native_replay": result["native_replay"],
                                            "payload_bytes": len(artifact.payload)})
            # A short Digital binding record is intentionally distinct from the six
            # full Packet stage payloads. It does not claim to transport PCM.
            digital_specimen = {"schema": "qec.acoustic-digital-binding-specimen.v1",
                                "source_sha256": demo_source["sha256"],
                                "candidate_observation_sha256": demo["roles"]["candidate"]["sha256"],
                                "scope": "binding_record_only; not_full_observation_or_PCM"}
            digital_artifact = AcousticTransportArtifact.from_artifact(
                digital_specimen, role="candidate", source_sha256=demo_source["sha256"])
            digital_receipt = execute_acoustic_transport(digital_artifact, transport="digital")
            validate_acoustic_transport_receipt(digital_receipt,
                expected_artifact_sha256=digital_artifact.as_dict()["sha256"], expected_source_sha256=demo_source["sha256"],
                expected_input_sha256=digital_receipt["input"]["sha256"])
            save(output / "lab/digital_binding_specimen.json", digital_specimen)
            save(output / "transport/digital_binding_specimen.json", digital_receipt)
            elapsed_ns = time.perf_counter_ns() - start
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        _ensure_source_snapshot(commit, code_files)
        environment.update(runtime_ns=elapsed_ns, peak_python_tracemalloc_bytes=peak_bytes)
        _validate_environment(environment)
        save(output / "environment.json", environment)
        files = list(writer.files)
        manifest = {"schema": "qec.acoustic-export-manifest.v1", "contract_version": "177.5",
                    "canonical_battery_sha256": battery["sha256"], "battery_source_sha256": source["sha256"],
                    "demonstration_source_sha256": demo_source["sha256"],
                    "measured_reference_sha256": demo_source["measured_reference"]["sha256"],
                    "source_commit_sha": commit, "source_files": code_files,
                    "clean_working_tree_at_start": not bool(dirty), "dependencies": environment["dependencies"],
                    "files": files, "transport_demonstration": transport_index,
                    "artifact_total_bytes_excluding_manifest": sum(f["bytes"] for f in files),
                    "digital_binding_specimen": {"path": "transport/digital_binding_specimen.json",
                        "input_sha256": digital_receipt["input"]["sha256"],
                        "artifact_sha256": digital_artifact.as_dict()["sha256"], "payload_bytes": len(digital_artifact.payload),
                        "scope": "binding_record_only; not_one_of_six_full_stage_payloads"},
                    "transport_scope": "six_actual_role_payloads_in_separate_demo; no_per_battery_transport_claim",
                    "spectral_compatibility": "ordinary_mono_PCM16_WAV_and_canonical_JSON_export_only",
                    "spectral_repository_or_ui_integration": False, "environment_metadata_is_canonical_scientific_proof": False}
        manifest["sha256"] = canonical_sha256(manifest)
        save(output / "manifest.json", manifest)
        _ensure_source_snapshot(commit, code_files)
        result = {"battery_sha256": battery["sha256"], "battery_source_sha256": source["sha256"],
                "demonstration_source_sha256": demo_source["sha256"],
                "measured_reference_sha256": demo_source["measured_reference"]["sha256"],
                "manifest_sha256": manifest["sha256"], "trial_count": len(battery["trials"])}

        writer.check_root()
    # Recheck every artifact through the retained validation inventory
    # after releasing the writer's exclusive claim.
    with _open_inventory(output, file_access=True) as access:
        if access.names != sorted(row["path"] for row in files):
            raise ValueError("INVALID_INPUT")
        for row in files:
            observed = access.stat(row["path"])
            if observed.st_size != row["bytes"] or _sha(access.read(row["path"], expected_stat=observed)) != row["sha256"]:
                raise ValueError("INVALID_INPUT")
        if _parse_json(access.read("manifest.json")) != manifest:
            raise ValueError("INVALID_INPUT")
    return result

def validate_export(output: Path, *, expected_manifest_sha256: str, expected_battery_source_sha256: str,
                    expected_demonstration_source_sha256: str, expected_reference_sha256: str):
    """Replay disk evidence against externally retained source AND manifest pins."""
    from qec.experiments.acoustic._common import digest, exact
    try:
        output = Path(output)
        for pin in (expected_manifest_sha256, expected_battery_source_sha256,
                    expected_demonstration_source_sha256, expected_reference_sha256):
            digest(pin)
        # Discover/retain the complete tree before opening even the root manifest.
        # POSIX file access uses only a discovered basename and its retained
        # parent descriptor, so ancestor replacement cannot redirect stat/open.
        with _open_inventory(output, file_access=True) as access:
            manifest = _parse_json(access.read("manifest.json"))
            _strict_metadata(manifest, _MANIFEST_FIXED, _MANIFEST_DYNAMIC)
            unsigned = {k: v for k, v in manifest.items() if k != "sha256"}
            if (manifest["sha256"] != expected_manifest_sha256 or canonical_sha256(unsigned) != expected_manifest_sha256
                    or manifest["battery_source_sha256"] != expected_battery_source_sha256
                    or manifest["demonstration_source_sha256"] != expected_demonstration_source_sha256
                    or manifest["measured_reference_sha256"] != expected_reference_sha256):
                raise ValueError("INVALID_INPUT")
            if type(manifest["files"]) is not list or len(manifest["files"]) > 128:
                raise ValueError("INVALID_INPUT")
            requested = []
            for row in manifest["files"]:
                name = row["path"]
                if (type(name) is not str or not 1 <= len(name) <= 4096
                        or any(character in name for character in ("\\", ":", "\0"))
                        or any(part in ("", ".", "..") for part in name.split("/"))):
                    raise ValueError("INVALID_INPUT")
                requested.append(name)
            # Unknown/aliased paths must fail before any payload lookup or stat.
            if access.names != sorted(requested):
                raise ValueError("INVALID_INPUT")
            inventory, total_bytes = [], 0
            for name in requested:
                info = access.stat(name)
                total_bytes += info.st_size
                if total_bytes > MAX_EXPORT_TOTAL_BYTES:
                    raise ValueError("INVALID_INPUT")
                inventory.append((name, info))
            contents = {name: access.read(name, expected_stat=info) for name, info in inventory}
        expected_files = [{"path": name, "sha256": _sha(contents[name]), "bytes": len(contents[name])}
                          for name, _ in inventory]
        exact({"files": manifest["files"]}, {"files": expected_files})
        def read_json(path):
            return _parse_json(contents[_portable_relative(path, output)])
        battery = read_json(output / "battery.json")
        exact(read_json(output / "battery_source.json"), battery["source"])
        validation = validate_comparative_battery(battery, expected_source_sha256=expected_battery_source_sha256)
        exact(read_json(output / "battery_validation.json"), validation)
        if manifest["canonical_battery_sha256"] != battery["sha256"]:
            raise ValueError("INVALID_INPUT")
        for index, trial in enumerate(battery["trials"]):
            stem = f"{index:02d}_q{trial['source']['alphabet_size']}_{trial['source']['case_name']}"
            exact(read_json(output / "trials" / (stem + ".json")), trial)
            exact(read_json(output / "trials" / (stem + "_source.json")), trial["source"])
        demo = read_json(output / "demonstration.json")
        exact(read_json(output / "demonstration_source.json"), demo["source"])
        demo_validation = validate_demonstration(demo, expected_source_sha256=expected_demonstration_source_sha256,
                                                expected_reference_sha256=expected_reference_sha256)
        exact(read_json(output / "demonstration_validation.json"), demo_validation)
        for role in ARTIFACT_ROLES:
            value = read_json(output / "lab" / (role + ".json"))
            exact(value, demo["roles"][role])
            if role in ("original", "observed", "filtered"):
                exact(read_wav(contents["lab/" + role + ".wav"]).as_dict(), value)
        if type(manifest["transport_demonstration"]) is not list or not 6 <= len(manifest["transport_demonstration"]) <= 12:
            raise ValueError("INVALID_INPUT")
        seen = set()
        for row in manifest["transport_demonstration"]:
            role, transport = row["role"], row["transport"]
            if role not in ARTIFACT_ROLES or transport not in TRANSPORTS or (role, transport) in seen:
                raise ValueError("INVALID_INPUT")
            seen.add((role, transport))
            if row["path"] != f"transport/{role}_{transport}.json":
                raise ValueError("INVALID_INPUT")
            artifact = AcousticTransportArtifact.from_artifact(demo["roles"][role], role=role,
                                                               source_sha256=expected_demonstration_source_sha256)
            receipt = read_json(output / row["path"])
            result = validate_acoustic_transport_receipt(receipt,
                expected_artifact_sha256=artifact.as_dict()["sha256"], expected_source_sha256=expected_demonstration_source_sha256,
                expected_input_sha256=row["input_sha256"])
            exact(row, {"role": role, "transport": transport, "path": f"transport/{role}_{transport}.json",
                        "artifact_sha256": artifact.as_dict()["sha256"], "input_sha256": receipt["input"]["sha256"],
                        "receipt_sha256": receipt["sha256"], "native_replay": "FULL", "payload_bytes": len(artifact.payload)})
            if (not result["accepted"] or receipt["input"]["transport"] != transport or
                    receipt["sha256"] != row["receipt_sha256"] or artifact.as_dict()["sha256"] != row["artifact_sha256"]
                    or len(artifact.payload) != row["payload_bytes"] or row["native_replay"] != "FULL"):
                raise ValueError("INVALID_INPUT")
        if {r for r, _ in seen} != set(ARTIFACT_ROLES):
            raise ValueError("INVALID_INPUT")
        expected_specimen = {"schema": "qec.acoustic-digital-binding-specimen.v1",
                             "source_sha256": expected_demonstration_source_sha256,
                             "candidate_observation_sha256": demo["roles"]["candidate"]["sha256"],
                             "scope": "binding_record_only; not_full_observation_or_PCM"}
        exact(read_json(output / "lab/digital_binding_specimen.json"), expected_specimen)
        artifact = AcousticTransportArtifact.from_artifact(expected_specimen, role="candidate",
                                                           source_sha256=expected_demonstration_source_sha256)
        entry = manifest["digital_binding_specimen"]
        if (entry["path"] != "transport/digital_binding_specimen.json" or
                entry["artifact_sha256"] != artifact.as_dict()["sha256"] or entry["payload_bytes"] != len(artifact.payload)):
            raise ValueError("INVALID_INPUT")
        digital_receipt = read_json(output / entry["path"])
        result = validate_acoustic_transport_receipt(digital_receipt,
            expected_artifact_sha256=artifact.as_dict()["sha256"], expected_source_sha256=expected_demonstration_source_sha256,
            expected_input_sha256=entry["input_sha256"])
        exact(entry, {"path": "transport/digital_binding_specimen.json",
                      "input_sha256": digital_receipt["input"]["sha256"],
                      "artifact_sha256": artifact.as_dict()["sha256"], "payload_bytes": len(artifact.payload),
                      "scope": "binding_record_only; not_one_of_six_full_stage_payloads"})
        if not result["accepted"] or digital_receipt["input"]["transport"] != "digital":
            raise ValueError("INVALID_INPUT")
        if manifest["artifact_total_bytes_excluding_manifest"] != sum(f["bytes"] for f in expected_files):
            raise ValueError("INVALID_INPUT")
        environment = read_json(output / "environment.json")
        _validate_environment(environment)
        for key in ("source_commit_sha", "source_files", "clean_working_tree_at_start", "dependencies"):
            exact({key: manifest[key]}, {key: environment[key]})
        if environment["source_files"] != _code_manifest():
            raise ValueError("INVALID_INPUT")
        return {"all_passed": True, "full_comparative_replay": True, "full_native_replay": True,
                "manifest_sha256": expected_manifest_sha256, "physical_provenance_authenticated": False}
    except (OSError, KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1775)
    parser.add_argument("--case", action="append", choices=CASE_NAMES)
    parser.add_argument("--alphabet", action="append", type=int, choices=(3, 4))
    parser.add_argument("--transport", action="append", choices=TRANSPORTS)
    args = parser.parse_args(argv)
    result = generate(args.output, cases=tuple(args.case or CASE_NAMES), alphabets=tuple(args.alphabet or (3, 4)),
                      seed=args.seed, transports=tuple(args.transport or ("packet",)))
    print(canonical_bytes(result).decode("utf-8"))


if __name__ == "__main__":
    main()
