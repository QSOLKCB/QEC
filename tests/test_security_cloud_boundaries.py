"""Bounded regression fixtures for the provisional QEC Security Cloud IRC/CI findings.

No internet access, external credentials, or real remote services are required.
"""
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irc_bot import IRCBot, _safe_terminal_text
from qec.operator import irc_server
from qec.operator.irc_server import IRCServer


class FakeSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False
    def recv(self, n):
        return self.chunks.pop(0) if self.chunks else b""
    def close(self):
        self.closed = True
    def sendall(self, bytes_):
        pass
    def settimeout(self, value):
        self.timeout = value


class LegacyIrcBoundaryTests(unittest.TestCase):
    def test_tls_default_and_private_optout(self):
        self.assertTrue(IRCBot("irc.example").use_tls)
        self.assertEqual(IRCBot("irc.example").port, 6697)
        self.assertFalse(IRCBot("localhost", use_tls=False).use_tls)

    def test_eof_does_not_spin(self):
        bot = IRCBot("localhost", use_tls=False)
        bot.connected = True
        fake = FakeSocket([b""])
        bot.sock = fake
        bot.run()
        self.assertFalse(bot.running)
        self.assertFalse(bot.connected)
        self.assertTrue(fake.closed)

    def test_unterminated_frame_is_bounded(self):
        bot = IRCBot("localhost", use_tls=False)
        bot.connected = True
        fake = FakeSocket([b"A" * 600, b"PING :x\r\n"])
        bot.sock = fake
        bot.run()
        self.assertEqual(len(fake.chunks), 1)  # does not process next frame
        self.assertTrue(fake.closed)

    def test_terminal_escape_is_not_executed(self):
        raw = "\x1b]52;c;secret\x07\r\n\x1b[31m\u202e"
        escaped = _safe_terminal_text(raw)
        self.assertNotIn("\x1b", escaped)
        self.assertNotIn("\u202e", escaped)
        self.assertIn("\\x1b", escaped)
        self.assertIn("\\u202e", escaped)

    def test_valid_frames_survive_arbitrary_tcp_segmentation(self):
        bot = IRCBot("localhost", use_tls=False)
        bot.connected = True
        frames = b"".join(b":server NOTICE #q :" + b"A" * 491 + b"\r\n" for _ in range(5))
        self.assertEqual(len(frames), 512 * 5)
        fake = FakeSocket([frames[:100], frames[100:2148], frames[2148:]])
        bot.sock = fake
        bot.run()
        self.assertEqual(fake.chunks, [])
        self.assertTrue(fake.closed)

    def test_connection_timeout_does_not_become_receive_idle_timeout(self):
        bot = IRCBot("irc.example", use_tls=True)
        plain = FakeSocket([])
        wrapped = FakeSocket([])
        class Context:
            def wrap_socket(self, sock, server_hostname):
                self.assert_hostname = server_hostname
                return wrapped
        ctx = Context()
        with patch("irc_bot.socket.create_connection", return_value=plain), \
             patch("irc_bot.ssl.create_default_context", return_value=ctx):
            self.assertTrue(bot.connect())
        self.assertEqual(wrapped.timeout, None)
        self.assertEqual(ctx.assert_hostname, "irc.example")
        bot.disconnect()

    def test_invalid_tls_certificate_does_not_fall_back(self):
        bot = IRCBot("irc.example")
        fake = FakeSocket([])
        context = type("Context", (), {
            "wrap_socket": lambda self, sock, server_hostname: (_ for _ in ()).throw(
                OSError("certificate invalid"))
        })()
        with patch("irc_bot.socket.create_connection", return_value=fake), \
             patch("irc_bot.ssl.create_default_context", return_value=context):
            self.assertFalse(bot.connect())
        self.assertTrue(fake.closed)
        self.assertFalse(bot.connected)


class OperatorIrcTransportTests(unittest.IsolatedAsyncioTestCase):
    async def _start(self, max_clients=1):
        s = IRCServer(host="127.0.0.1", port=0, max_clients=max_clients)
        await s.start()
        port = s._server.sockets[0].getsockname()[1]
        return s, port

    async def test_quit_closes_transport_and_releases_slot(self):
        s, port = await self._start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"NICK first\r\nUSER one 0 * :One\r\n")
            await writer.drain()
            await asyncio.wait_for(reader.readline(), 2)
            writer.write(b"QUIT :bye\r\n")
            await writer.drain()
            self.assertEqual(await asyncio.wait_for(reader.read(), 2), b"")
            for _ in range(10):
                if not s._socket_clients:
                    break
                await asyncio.sleep(.01)
            self.assertEqual(len(s._socket_clients), 0)
            second_reader, second_writer = await asyncio.open_connection("127.0.0.1", port)
            second_writer.write(b"NICK second\r\nUSER two 0 * :Two\r\n")
            await second_writer.drain()
            self.assertIn(b" 001 second ", await asyncio.wait_for(second_reader.readline(), 2))
            second_writer.close()
            await second_writer.wait_closed()
            writer.close()
            await writer.wait_closed()
        finally:
            await s.stop()

    async def test_idle_and_overlong_frame_close(self):
        with patch.object(irc_server, "_SOCKET_IDLE_TIMEOUT_SECONDS", 0.07):
            s, port = await self._start()
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                self.assertEqual(await asyncio.wait_for(reader.read(), 2), b"")
                writer.close()
                await writer.wait_closed()
                reader2, writer2 = await asyncio.open_connection("127.0.0.1", port)
                writer2.write(b"A" * 750 + b"\r\n")
                await writer2.drain()
                self.assertEqual(await asyncio.wait_for(reader2.read(), 2), b"")
                writer2.close()
                await writer2.wait_closed()
            finally:
                await s.stop()


class GitHubAndMaintenancePolicyTests(unittest.TestCase):
    def test_pages_permissions_only_in_trusted_deploy_job(self):
        text = (ROOT / ".github/workflows/ququart-fer-pages.yml").read_text()
        global_scope = text.split("\npermissions:", 1)[1].split("\nconcurrency:", 1)[0]
        build_scope = text.split("\n  build:", 1)[1].split("\n  deploy:", 1)[0]
        deploy_scope = text.split("\n  deploy:", 1)[1]
        self.assertNotIn("pages: write", global_scope)
        self.assertNotIn("id-token: write", global_scope)
        self.assertNotIn("pages: write", build_scope)
        self.assertNotIn("id-token: write", build_scope)
        self.assertIn("persist-credentials: false", build_scope)
        self.assertIn("github.ref == 'refs/heads/main'", deploy_scope)
        self.assertIn("github.repository == 'QSOLKCB/QEC'", deploy_scope)
        self.assertIn("pages: write", deploy_scope)
        self.assertIn("id-token: write", deploy_scope)

    def test_dry_run_makes_no_mutating_gh_or_git_calls(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            logfile = root / "calls"
            shared = root / "untrusted"
            shared.mkdir()
            gh = bin_dir / "gh"
            gh.write_text(
                '#!/bin/sh\n'
                'printf "gh %s\\n" "$*" >> "$QEC_TEST_CALLS"\n'
                'if [ "$1" = repo ] && [ "$2" = list ]; then printf "fixture\\n"; fi\n'
            )
            gh.chmod(0o755)
            git = bin_dir / "git"
            git.write_text(
                '#!/bin/sh\n'
                'printf "git %s\\n" "$*" >> "$QEC_TEST_CALLS"\n'
                'exit 10\n'
            )
            git.chmod(0o755)
            env = os.environ.copy()
            env.update({"PATH": str(bin_dir) + os.pathsep + env.get("PATH",""),
                        "DRY_RUN": "1", "HOME": str(root),
                        "TMPDIR": str(shared), "QEC_TEST_CALLS": str(logfile)})
            result = subprocess.run(
                ["bash", str(ROOT/"scripts/maintenance/mirror_repos.sh")],
                cwd=root, env=env, capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PREVIEW", result.stdout)
            calls = logfile.read_text().splitlines()
            self.assertTrue(calls and all(x.startswith("gh repo list ") for x in calls))
            self.assertFalse(list(shared.iterdir()))
            self.assertFalse(list(root.glob(".qec-mirror.*")))


if __name__ == "__main__":
    unittest.main()
