import asyncio

import pytest

from qec.operator.irc_server import IRCServer


def _register(server: IRCServer, client: str, nick: str = "alice"):
    server.handle_message(client, f"NICK {nick}")
    return server.handle_message(client, "USER user 0 * :Real")


def test_registration_and_ping():
    s = IRCServer()
    replies = _register(s, "c1")
    assert any(" 001 " in r for r in replies)
    pong = s.handle_message("c1", "PING :tok")
    assert pong == ("PONG tok",)


def test_join_part_and_ordering_and_unknown_and_malformed_and_dup_nick():
    s = IRCServer()
    _register(s, "c1", "alice")
    _register(s, "c2", "bob")
    assert "JOIN #qec" in s.handle_message("c1", "JOIN #qec")[0]
    assert tuple(s.state.channels["#qec"].members) == ("c1",)
    s.handle_message("c2", "JOIN #qec")
    assert tuple(s.state.channels["#qec"].members) == ("c1", "c2")
    part = s.handle_message("c1", "PART #qec :bye")
    assert "PART #qec" in part[0]
    assert "421 WHAT" in s.handle_message("c1", "WHAT")[0]
    assert "461" in s.handle_message("c1", "")[0]
    dup = s.handle_message("c3", "NICK alice")
    assert "433" in dup[0]


def test_privmsg_and_unregistered_policy_and_no_module_cache_scan():
    s = IRCServer()
    _register(s, "c1", "alice")
    _register(s, "c2", "bob")
    s.handle_message("c1", "JOIN #qec")
    s.handle_message("c2", "JOIN #qec")
    out = s.handle_message("c1", "PRIVMSG #qec :hi")
    assert out == (":alice PRIVMSG #qec :hi",)
    out2 = s.handle_message("cx", "PRIVMSG #qec :hi")
    assert "451" in out2[0]
    banned = ["eval(", "exec(", "subprocess", "os.system", "shell=True", "importlib", "__import__(", "requests", "urllib.request", "openai", "anthropic", "random.", "time.time", "datetime.now"]
    for file in ("src/qec/operator/irc_protocol.py", "src/qec/operator/irc_server.py"):
        text = open(file, encoding="utf-8").read()
        for token in banned:
            assert token not in text


def test_local_defaults_and_nonlocal_bind_requirement():
    s = IRCServer()
    assert s.host == "127.0.0.1"
    from scripts.qec_irc_server import build_parser

    parser = build_parser()
    args = parser.parse_args(["--host", "0.0.0.0"])
    assert args.host == "0.0.0.0"


def test_socket_smoke_and_shutdown_cleanly():
    async def _run():
        s = IRCServer(host="127.0.0.1", port=0)
        await s.start()
        sock = s._server.sockets[0]
        host, port = sock.getsockname()[0], sock.getsockname()[1]
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(b"NICK a\r\n")
        writer.write(b"USER u 0 * :r\r\n")
        await writer.drain()
        data = await reader.readline()
        assert b"001" in data
        writer.close()
        await writer.wait_closed()
        await s.stop()
    asyncio.run(_run())
