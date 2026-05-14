from __future__ import annotations

from dataclasses import dataclass

_MAX_LINE_BYTES = 512
_MAX_NICK_LENGTH = 32
_MAX_USERNAME_LENGTH = 64
_MAX_CHANNEL_LENGTH = 64
_MAX_MESSAGE_LENGTH = 400

_MAX_CHANNELS_PER_CLIENT = 16
_MAX_CLIENTS = 32
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 6667


class IRCParseError(ValueError):
    """Deterministic IRC parse error."""


@dataclass(frozen=True)
class IRCMessage:
    prefix: str | None
    command: str
    params: tuple[str, ...]
    trailing: str | None


@dataclass(frozen=True)
class IRCReply:
    command: str
    params: tuple[str, ...] = ()
    trailing: str | None = None
    prefix: str | None = None


def is_valid_nick(nick: str) -> bool:
    if not nick or len(nick) > _MAX_NICK_LENGTH:
        return False
    if not (nick[0].isalpha() or nick[0] in "[]\\`_^{|}"):
        return False
    for c in nick:
        if not (c.isalnum() or c in "-[]\\`_^{|}"):
            return False
    return True


def normalize_nick(nick: str) -> str:
    n = nick.strip()
    if not is_valid_nick(n):
        raise IRCParseError("invalid nickname")
    return n


def is_valid_channel(channel: str) -> bool:
    if not channel.startswith("#"):
        return False
    if len(channel) < 2 or len(channel) > _MAX_CHANNEL_LENGTH:
        return False
    for c in channel:
        if c in " \r\n\0,:":
            return False
    return True


def normalize_channel(channel: str) -> str:
    c = channel.strip()
    if not is_valid_channel(c):
        raise IRCParseError("invalid channel")
    return c


def parse_irc_line(raw_line: str) -> IRCMessage:
    if "\0" in raw_line:
        raise IRCParseError("NUL byte not allowed")
    if "\n" in raw_line or "\r" in raw_line:
        raise IRCParseError("newline not allowed")
    if len(raw_line.encode("utf-8")) > _MAX_LINE_BYTES:
        raise IRCParseError("line too long")

    line = raw_line.strip(" ")
    if not line:
        raise IRCParseError("empty line")

    prefix = None
    if line.startswith(":"):
        if " " not in line:
            raise IRCParseError("missing command")
        prefix, line = line[1:].split(" ", 1)

    trailing = None
    if " :" in line:
        line, trailing = line.split(" :", 1)

    parts = [p for p in line.split(" ") if p]
    if not parts:
        raise IRCParseError("missing command")

    command = parts[0].upper()
    params = tuple(parts[1:])
    return IRCMessage(prefix=prefix, command=command, params=params, trailing=trailing)


def format_irc_message(message: IRCMessage | IRCReply) -> str:
    prefix = f":{message.prefix} " if message.prefix else ""
    base = " ".join((message.command, *message.params)).strip()
    if message.trailing is not None:
        return f"{prefix}{base} :{message.trailing}\r\n"
    return f"{prefix}{base}\r\n"
