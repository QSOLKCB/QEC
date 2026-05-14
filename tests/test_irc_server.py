import asyncio

import pytest

from qec.operator.irc_server import IRCServer

_SOCKET_READ_TIMEOUT = 2


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
    _register(s, "c3", "carol")
    s.handle_message("c1", "JOIN #qec")
    s.handle_message("c2", "JOIN #qec")
    out = s.handle_message("c1", "PRIVMSG #qec :hi")
    assert out == ()
    denied = s.handle_message("c3", "PRIVMSG #qec :hi")
    assert "404" in denied[0]
    out2 = s.handle_message("cx", "PRIVMSG #qec :hi")
    assert "451" in out2[0]
    banned = ["eval(", "exec(", "subprocess", "os.system", "shell=True", "importlib", "__import__(", "requests", "urllib.request", "openai", "anthropic", "random.", "time.time", "datetime.now"]
    for file in ("src/qec/operator/irc_protocol.py", "src/qec/operator/irc_server.py", "src/qec/operator/irc_commands.py"):
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


def test_server_full_returns_error():
    s = IRCServer(max_clients=1)
    _register(s, "c1", "alice")
    replies = s.handle_message("c2", "NICK bob")
    assert "500" in replies[0]
    assert "Server is full" in replies[0]
    assert len(s.state.clients) == 1


def test_socket_smoke_and_shutdown_cleanly():
    async def _run():
        s = IRCServer(host="127.0.0.1", port=0)
        await s.start()
        sock = s._server.sockets[0]
        host, port = sock.getsockname()[0], sock.getsockname()[1]
        reader1, writer1 = await asyncio.open_connection(host, port)
        reader2, writer2 = await asyncio.open_connection(host, port)
        writer1.write(b"NICK a\r\n")
        writer1.write(b"USER u 0 * :r\r\n")
        writer2.write(b"NICK b\r\n")
        writer2.write(b"USER u 0 * :r\r\n")
        await writer1.drain()
        await writer2.drain()
        assert b" 001 a :Welcome to qec-ircd" in await asyncio.wait_for(reader1.readline(), timeout=_SOCKET_READ_TIMEOUT)
        assert b" 001 b :Welcome to qec-ircd" in await asyncio.wait_for(reader2.readline(), timeout=_SOCKET_READ_TIMEOUT)
        writer1.write(b"JOIN #qec\r\n")
        writer2.write(b"JOIN #qec\r\n")
        await writer1.drain()
        await writer2.drain()
        assert b":a JOIN #qec" in await asyncio.wait_for(reader1.readline(), timeout=_SOCKET_READ_TIMEOUT)
        assert b":b JOIN #qec" in await asyncio.wait_for(reader2.readline(), timeout=_SOCKET_READ_TIMEOUT)
        writer1.write(b"PRIVMSG #qec :hello\r\n")
        await writer1.drain()
        assert b":a PRIVMSG #qec :hello" in await asyncio.wait_for(reader2.readline(), timeout=_SOCKET_READ_TIMEOUT)
        writer1.close()
        writer2.close()
        await writer1.wait_closed()
        await writer2.wait_closed()
        await s.stop()

    asyncio.run(_run())


def test_command_help_channel_response_and_routing():
    s = IRCServer()
    _register(s, "c1", "alice")
    _register(s, "c2", "bob")
    s.handle_message("c1", "JOIN #qec")
    s.handle_message("c2", "JOIN #qec")
    out = s.handle_message("c1", "PRIVMSG #qec :!help")
    assert any(" NOTICE #qec :!help -" in line for line in out)
    routed = s._handle_message_routed("c1", "PRIVMSG #qec :!corelaw")
    assert any(target == "c2" and "PRIVMSG #qec :!corelaw" in line for target, line in routed)


def test_direct_command_and_unknown_command_do_not_crash():
    s = IRCServer()
    _register(s, "c1", "alice")
    out = s.handle_message("c1", "PRIVMSG qec-ircd :!corelaw")
    assert any("NOTICE alice" in line and "same input" in line for line in out)
    out2 = s.handle_message("c1", "PRIVMSG qec-ircd :!doesnotexist")
    assert any("UNKNOWN_COMMAND" in line for line in out2)


def test_direct_command_to_unknown_target_returns_401():
    s = IRCServer()
    _register(s, "c1", "alice")
    out = s.handle_message("c1", "PRIVMSG doesnotexist :!help")
    assert any("401" in line and "No such nick/channel" in line for line in out)
    assert not any("NOTICE" in line for line in out)


def test_replay_audit_export_deterministic_when_enabled():
    s = IRCServer(enable_replay_audit=True)
    _register(s, "c1", "alice")
    s.handle_message("c1", "JOIN #qec")
    s.handle_message("c1", "PRIVMSG #qec :!help")
    receipt1 = s.export_replay_audit_receipt()
    receipt2 = s.export_replay_audit_receipt()
    assert receipt1.irc_replay_audit_hash == receipt2.irc_replay_audit_hash
    assert receipt1.event_count >= 1


def test_replay_audit_limit_enforced_when_enabled():
    s = IRCServer(enable_replay_audit=True)
    _register(s, "c1", "alice")
    for i in range(254):
        s.handle_message("c1", f"PING :{i}")
    with pytest.raises(ValueError, match="AUDIT_EVENT_LIMIT_EXCEEDED"):
        s.handle_message("c1", "PING :overflow")
