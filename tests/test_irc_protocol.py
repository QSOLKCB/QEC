import pytest

from qec.operator.irc_protocol import (
    IRCMessage,
    IRCParseError,
    format_irc_message,
    is_valid_channel,
    is_valid_nick,
    parse_irc_line,
)


def test_parse_nick():
    m = parse_irc_line("NICK alice")
    assert m.command == "NICK"
    assert m.params == ("alice",)


def test_parse_user():
    m = parse_irc_line("USER alice 0 * :Alice")
    assert m.command == "USER"
    assert m.trailing == "Alice"


def test_parse_ping_join_part_privmsg_quit_and_upper_tuple():
    assert parse_irc_line("ping :x").command == "PING"
    assert parse_irc_line("JOIN #qec").command == "JOIN"
    assert parse_irc_line("PART #qec :bye").command == "PART"
    p = parse_irc_line("PRIVMSG #qec :hello")
    assert p.trailing == "hello"
    assert isinstance(p.params, tuple)
    assert parse_irc_line("QUIT :later").command == "QUIT"


def test_invalid_newline_nul_length_rejected():
    with pytest.raises(IRCParseError):
        parse_irc_line("NICK a\n")
    with pytest.raises(IRCParseError):
        parse_irc_line("NICK a\0")
    with pytest.raises(IRCParseError):
        parse_irc_line("A" * 513)


def test_invalid_nick_channel_rejected():
    assert not is_valid_nick("1bad")
    assert not is_valid_channel("qec")


def test_format_deterministic():
    msg = IRCMessage(prefix="x", command="PRIVMSG", params=("#qec",), trailing="hello")
    assert format_irc_message(msg) == ":x PRIVMSG #qec :hello\r\n"
