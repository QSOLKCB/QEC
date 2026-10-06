"""Execute the POSIX installer offline with real archives and fake transports."""

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO = "QSOLKCB/QEC"
TAG = "v173.0"
ASSET = "qec-tui-linux-x86_64.tar.gz"
ASSET_URL = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
# If installation accidentally runs this old interactive binary, fail the test.
BINARY = b"#!/bin/sh\nprintf launched > \"$TEST_ROOT/launched\"\nexit 99\n"


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.dest = self.root / "install space"
        # Isolate the command path so cargo/cc cannot leak in from the host.
        for tool in ("python3", "tar", "gzip", "mktemp", "install", "sed", "mkdir", "rm", "chmod", "cp"):
            (self.bin / tool).symlink_to(shutil.which(tool))
        self.env = dict(os.environ, PATH=str(self.bin), TEST_ROOT=str(self.root),
                        QEC_INSTALL_DIR=str(self.dest), TMPDIR=str(self.root),
                        TEST_ARCH="x86_64")
        self.stub("uname", '#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo "$TEST_ARCH";; esac\n')
        self.stub("cc", "#!/bin/sh\nexit 0\n")
        self.stub("cargo", """#!/bin/sh
printf '%s\n' "$*" > "$TEST_ROOT/cargo.args"
[ "${TEST_BUILD_FAIL:-0}" != 1 ] || exit 1
[ -f "$5" ] || exit 2
mkdir -p "$CARGO_TARGET_DIR/release"
cp "$TEST_ROOT/binary" "$CARGO_TARGET_DIR/release/qec-tui"
""")
        self.stub("curl", """#!/bin/sh
output=
while [ "$#" -gt 0 ]; do
    case "$1" in -o) output=$2; shift 2;; -*) shift;; *) url=$1; shift;; esac
done
printf '%s\n' "$url" >> "$TEST_ROOT/urls"
case "$url" in
    */releases/latest) cp "$TEST_ROOT/release.json" /dev/stdout;;
    */releases/download/*)
        [ "${TEST_DOWNLOAD_FAIL:-0}" != 1 ] || exit 22
        cp "$TEST_ROOT/asset.tar.gz" "$output";;
    */archive/refs/tags/*.tar.gz) cp "$TEST_ROOT/source.tar.gz" "$output";;
    *) exit 22;;
esac
""")
        (self.root / "binary").write_bytes(BINARY)
        self.archive("asset.tar.gz", {"qec-tui": BINARY})
        self.archive("source.tar.gz", {
            "QEC-173.0/tui/Cargo.toml": b'[package]\nname = "qec-tui"\nversion = "106.0.0"\n',
            "QEC-173.0/tui/Cargo.lock": b"# lock fixture\n",
        })
        self.release([])

    def stub(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)

    def archive(self, name, members):
        with tarfile.open(self.root / name, "w:gz") as archive:
            for path, data in members.items():
                info = tarfile.TarInfo(path)
                info.size = len(data)
                info.mode = 0o755
                archive.addfile(info, io.BytesIO(data))

    def release(self, assets):
        (self.root / "release.json").write_text(json.dumps({"tag_name": TAG, "assets": assets}))

    def run_installer(self):
        # Match curl | sh: the script arrives on stdin, rather than by filename.
        result = subprocess.run(["/bin/sh"], input=(ROOT / "tui/install.sh").read_text(),
                                env=self.env, capture_output=True, text=True, timeout=10)
        self.assertFalse((self.root / "launched").exists(), "installer launched the TUI")
        self.assertFalse(list(self.root.glob("tmp.*")), "temporary directory leaked")
        return result

    def assert_installed(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        binary = self.dest / "qec-tui"
        self.assertEqual(binary.read_bytes(), BINARY)
        self.assertEqual(binary.stat().st_mode & 0o777, 0o755)
        self.assertIn("installed successfully from QEC v173.0", result.stdout)

    def test_assetless_latest_release_builds_source_and_reinstalls(self):
        self.assert_installed(self.run_installer())
        self.assertIn("--locked --release --manifest-path", (self.root / "cargo.args").read_text())
        urls = (self.root / "urls").read_text()
        self.assertIn("/archive/refs/tags/v173.0.tar.gz", urls)
        self.assertNotIn("/releases/download/", urls)
        self.assert_installed(self.run_installer())

    def test_exact_asset_download_skips_build_and_preserves_tmpdir(self):
        self.release([{"name": ASSET, "browser_download_url": ASSET_URL}])
        self.assert_installed(self.run_installer())
        self.assertIn(ASSET_URL, (self.root / "urls").read_text())
        self.assertFalse((self.root / "cargo.args").exists())
        self.assertTrue(self.root.exists())

    def test_next_source_only_release_needs_no_installer_change(self):
        (self.root / "release.json").write_text(json.dumps({"tag_name": "v174.0", "assets": []}))
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("installed successfully from QEC v174.0", result.stdout)
        self.assertIn("/archive/refs/tags/v174.0.tar.gz", (self.root / "urls").read_text())
        self.assertEqual((self.dest / "qec-tui").read_bytes(), BINARY)

    def test_unrelated_assets_do_not_prevent_source_build(self):
        self.release([{"name": "notes.txt", "browser_download_url": "https://example.invalid/notes"}])
        self.assert_installed(self.run_installer())
        self.assertTrue((self.root / "cargo.args").exists())

    def test_arm_host_builds_natively_instead_of_downloading_x86(self):
        self.env["TEST_ARCH"] = "aarch64"
        self.release([{"name": ASSET, "browser_download_url": ASSET_URL}])
        self.assert_installed(self.run_installer())
        self.assertNotIn(ASSET_URL, (self.root / "urls").read_text())

    def test_missing_cargo_explains_prerequisites(self):
        (self.bin / "cargo").unlink()
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires Rust/Cargo", result.stderr)
        self.assertFalse(self.dest.exists())

    def test_missing_linker_explains_prerequisites(self):
        (self.bin / "cc").unlink()
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires a C compiler/linker", result.stderr)

    def test_build_failure_does_not_overwrite_existing_binary(self):
        self.dest.mkdir()
        (self.dest / "qec-tui").write_bytes(b"previous install")
        self.env["TEST_BUILD_FAIL"] = "1"
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Rust TUI build failed", result.stderr)
        self.assertEqual((self.dest / "qec-tui").read_bytes(), b"previous install")

    def test_download_failure_is_not_treated_as_missing_asset(self):
        self.release([{"name": ASSET, "browser_download_url": ASSET_URL}])
        self.env["TEST_DOWNLOAD_FAIL"] = "1"
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "cargo.args").exists())
        self.assertFalse(self.dest.exists())

    def test_malformed_metadata_fails_before_install(self):
        for metadata in ("not json", '{"tag_name":"v173.0"}',
                         '{"tag_name":"../bad","assets":[]}',
                         '{"tag_name":"v173.0","assets":{}}'):
            with self.subTest(metadata=metadata):
                (self.root / "release.json").write_text(metadata)
                self.assertNotEqual(self.run_installer().returncode, 0)
                self.assertFalse(self.dest.exists())

    def test_wrong_asset_url_is_rejected(self):
        self.release([{"name": ASSET, "browser_download_url": "https://example.invalid/binary"}])
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertFalse(self.dest.exists())

    def test_archive_without_binary_fails(self):
        self.release([{"name": ASSET, "browser_download_url": ASSET_URL}])
        self.archive("asset.tar.gz", {"README.md": b"no binary"})
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main()
