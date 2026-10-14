# SPDX-License-Identifier: MPL-2.0
"""Retained-handle, exclusively claimed acoustic evidence creation.

Artifact names are internal single components under retained directory handles.
No payload is written through a mutable descendant pathname.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat


class SecureExportWriter:
    _CLAIM = ".qec-acoustic-export-claim"

    def __init__(self, output, max_file_bytes, max_total_bytes):
        self.output = Path(output).absolute()
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.files = []
        self.total = 0
        self.directories = {}
        self.handles = []
        self.claim = None
        self.root_identity = None

    def __enter__(self):
        try:
            if os.name == "posix":
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                ancestor = os.open(self.output.anchor, flags)
                self.handles.append(ancestor)
                for part in self.output.parts[1:]:
                    try:
                        os.mkdir(part, 0o700, dir_fd=ancestor)
                    except FileExistsError:
                        pass
                    ancestor = os.open(part, flags, dir_fd=ancestor)
                    self.handles.append(ancestor)
                root = ancestor
                self.directories[""] = root
                self.root_identity = self._identity(os.fstat(root))
                self.claim = os.open(self._CLAIM, os.O_CREAT | os.O_EXCL | os.O_WRONLY |
                                     os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=root)
                self.handles.append(self.claim)
                with os.scandir(root) as entries:
                    if any(entry.name != self._CLAIM for entry in entries):
                        raise ValueError("output must be an empty directory")
            elif os.name == "nt":
                self.output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self.output.mkdir()
                except FileExistsError:
                    pass
                self._init_windows()
            else:
                raise ValueError("unsupported filesystem")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    @staticmethod
    def _identity(info):
        return (info.st_dev, info.st_ino)

    def _split(self, name):
        if (type(name) is not str or not name or "\\" in name or ":" in name or "\0" in name
                or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError("INVALID_INPUT")
        components = name.split("/")
        parent = "/".join(components[:-1])
        if parent not in self.directories:
            raise ValueError("INVALID_INPUT")
        return self.directories[parent], components[-1]

    def mkdir(self, name):
        parent, basename = self._split(name)
        if name in self.directories:
            raise ValueError("INVALID_INPUT")
        if os.name == "posix":
            os.mkdir(basename, 0o700, dir_fd=parent)
            descriptor = os.open(basename, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW |
                                 getattr(os, "O_CLOEXEC", 0), dir_fd=parent)
            self.handles.append(descriptor)
            self.directories[name] = descriptor
        else:
            self.directories[name] = self._nt_create(parent, basename, directory=True)

    def save(self, name, data):
        if type(data) is not bytes:
            raise ValueError("INVALID_INPUT")
        parent, basename = self._split(name)
        if (basename == self._CLAIM or name in (row["path"] for row in self.files)
                or len(data) > self.max_file_bytes or self.total + len(data) > self.max_total_bytes):
            raise ValueError("export exceeds 32 MiB bound; choose fewer full-stage transports")
        if os.name == "posix":
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(basename, flags, 0o600, dir_fd=parent)
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or opened.st_size:
                    raise ValueError("INVALID_INPUT")
                view = memoryview(data)
                while view:
                    count = os.write(descriptor, view)
                    if count <= 0:
                        raise ValueError("INVALID_INPUT")
                    view = view[count:]
                if os.fstat(descriptor).st_size != len(data):
                    raise ValueError("INVALID_INPUT")
            finally:
                os.close(descriptor)
        else:
            descriptor = self._nt_create(parent, basename)
            self._nt_write(descriptor, data)
        self.files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        self.total += len(data)

    def check_root(self):
        if os.name == "posix":
            original = self.root_identity
            if (self._identity(os.fstat(self.directories[""])) != original or
                    self._identity(self.output.lstat()) != original):
                raise ValueError("INVALID_INPUT")
        else:
            if self._win_identity(self.directories[""]) != self.root_identity:
                raise ValueError("INVALID_INPUT")
            h = self._win_open_root()
            try:
                if self._win_identity(h) != self.root_identity:
                    raise ValueError("INVALID_INPUT")
            finally:
                self._win_close(h)

    def __exit__(self, typ, value, tb):
        if os.name == "posix" and self.claim is not None:
            try:
                os.unlink(self._CLAIM, dir_fd=self.directories[""])
            except OSError:
                if typ is None:
                    raise
        for handle in reversed(self.handles):
            if os.name == "nt":
                self._win_close(handle)
            else:
                os.close(handle)
        self.handles.clear()
        self.directories.clear()

    def _init_windows(self):
        import ctypes
        from ctypes import wintypes
        self._ctypes, self._wintypes = ctypes, wintypes
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        class UnicodeString(ctypes.Structure):
            _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                        ("Buffer", wintypes.LPWSTR)]
        class ObjectAttributes(ctypes.Structure):
            _fields_ = [("Length", wintypes.ULONG), ("RootDirectory", wintypes.HANDLE),
                        ("ObjectName", ctypes.POINTER(UnicodeString)), ("Attributes", wintypes.ULONG),
                        ("SecurityDescriptor", wintypes.LPVOID), ("SecurityQualityOfService", wintypes.LPVOID)]
        class IoStatusBlock(ctypes.Structure):
            _fields_ = [("Status", ctypes.c_size_t), ("Information", ctypes.c_size_t)]
        self._UnicodeString, self._ObjectAttributes, self._IoStatusBlock = UnicodeString, ObjectAttributes, IoStatusBlock
        self._create = self._kernel.CreateFileW
        self._create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        self._create.restype = wintypes.HANDLE
        self._query = self._kernel.GetFileInformationByHandleEx
        self._query.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
        self._query.restype = wintypes.BOOL
        self._win_close = self._kernel.CloseHandle
        self._win_close.argtypes = (wintypes.HANDLE,)
        self._win_close.restype = wintypes.BOOL
        self._ntcreate = self._ntdll.NtCreateFile
        self._ntcreate.argtypes = (ctypes.POINTER(wintypes.HANDLE), wintypes.ULONG,
                                    ctypes.POINTER(ObjectAttributes), ctypes.POINTER(IoStatusBlock),
                                    wintypes.LPVOID, wintypes.ULONG, wintypes.ULONG, wintypes.ULONG,
                                    wintypes.ULONG, wintypes.LPVOID, wintypes.ULONG)
        self._ntcreate.restype = ctypes.c_long
        self._write = self._kernel.WriteFile
        self._write.argtypes = (wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                                ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID)
        self._write.restype = wintypes.BOOL
        root = self._win_open_root()
        self.handles.append(root)
        self.directories[""] = root
        self.root_identity = self._win_identity(root)
        self.claim = self._nt_create(root, self._CLAIM, claim=True)
        if any(entry.name != self._CLAIM for entry in os.scandir(self.output)):
            raise ValueError("output must be an empty directory")

    def _win_open_root(self):
        ctypes = self._ctypes
        handle = self._create(str(self.output), 0x81, 0x1, None, 3, 0x02200000, None)
        if handle in (None, ctypes.c_void_p(-1).value):
            raise ValueError("INVALID_INPUT")
        try:
            self._win_identity(handle)
        except BaseException:
            self._win_close(handle)
            raise
        return handle

    def _win_identity(self, handle):
        import struct
        ctypes = self._ctypes
        basic = ctypes.create_string_buffer(8)
        identity = ctypes.create_string_buffer(24)
        if not self._query(handle, 9, basic, 8) or not self._query(handle, 18, identity, 24):
            raise ValueError("INVALID_INPUT")
        attributes = struct.unpack_from("<I", basic.raw)[0]
        if attributes & (0x400 | 0x40) or not attributes & 0x10:
            raise ValueError("INVALID_INPUT")
        return identity.raw

    def _nt_create(self, parent, basename, *, directory=False, claim=False):
        ctypes, wt = self._ctypes, self._wintypes
        native_string = ctypes.create_unicode_buffer(basename)
        name = self._UnicodeString(len(basename) * 2, (len(basename) + 1) * 2,
                                   ctypes.cast(native_string, wt.LPWSTR))
        attrs = self._ObjectAttributes(ctypes.sizeof(self._ObjectAttributes), parent,
                                       ctypes.pointer(name), 0x40, None, None)
        status = self._IoStatusBlock()
        result_handle = wt.HANDLE()
        access = 0x00100000 | 0x80 | (0x1 if directory else 0x2) | (0x00010000 if claim else 0)
        options = 0x20 | 0x200000 | (0x1 if directory else 0x40) | (0x1000 if claim else 0)
        status_code = self._ntcreate(ctypes.byref(result_handle), access, ctypes.byref(attrs),
                                    ctypes.byref(status), None, 0x10 if directory else 0x80,
                                    0x1, 2, options, None, 0)
        if status_code < 0 or result_handle.value in (None, ctypes.c_void_p(-1).value):
            raise ValueError("INVALID_INPUT")
        self.handles.append(result_handle)
        return result_handle

    def _nt_write(self, handle, data):
        ctypes, wt = self._ctypes, self._wintypes
        for offset in range(0, len(data), 65536):
            payload = data[offset:offset + 65536]
            buffer = ctypes.create_string_buffer(payload)
            written = wt.DWORD()
            if not self._write(handle, buffer, len(payload), ctypes.byref(written), None) or written.value != len(payload):
                raise ValueError("INVALID_INPUT")
