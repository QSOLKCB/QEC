from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .irc_protocol import (
    _DEFAULT_HOST,
    _DEFAULT_PORT,
    _MAX_CHANNELS_PER_CLIENT,
    _MAX_CLIENTS,
    _MAX_MESSAGE_LENGTH,
    _MAX_USERNAME_LENGTH,
    IRCMessage,
    IRCParseError,
    IRCReply,
    format_irc_message,
    normalize_channel,
    normalize_nick,
    parse_irc_line,
)

_SERVER_NAME = "qec-ircd"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 6667
_MAX_CHANNELS_PER_CLIENT = 16
_MAX_CLIENTS = 32


@dataclass
class IRCClientState:
    nick: str | None = None
    username: str | None = None
    realname: str | None = None
    registered: bool = False
    channels: tuple[str, ...] = ()


@dataclass
class IRCChannelState:
    name: str
    members: tuple[str, ...] = ()


@dataclass
class IRCServerState:
    clients: dict[str, IRCClientState] = field(default_factory=dict)
    channels: dict[str, IRCChannelState] = field(default_factory=dict)


class IRCServer:
    def __init__(self, host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT, max_clients: int = _MAX_CLIENTS) -> None:
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.state = IRCServerState()
        self._writer_by_client: dict[str, asyncio.StreamWriter] = {}
        self._server: asyncio.base_events.Server | None = None
        self._client_seq = 0

    def _sorted_clients(self) -> list[str]:
        return sorted(self.state.clients)

    def _make_reply(self, command: str, *params: str, trailing: str | None = None) -> str:
        return format_irc_message(IRCReply(prefix=_SERVER_NAME, command=command, params=tuple(params), trailing=trailing)).rstrip("\r\n")

    def _register_if_ready(self, client_id: str) -> list[str]:
        c = self.state.clients[client_id]
        if c.registered or not c.nick or not c.username:
            return []
        c.registered = True
        return [self._make_reply("001", c.nick, trailing="Welcome to qec-ircd")]

    def add_client(self, client_id: str) -> None:
        if len(self.state.clients) >= self.max_clients:
            raise ValueError("max clients reached")
        self.state.clients[client_id] = IRCClientState()

    def remove_client(self, client_id: str) -> None:
        c = self.state.clients.get(client_id)
        if not c:
            return
        for ch in list(c.channels):
            self._part_channel(client_id, ch)
        self.state.clients.pop(client_id, None)
        self._writer_by_client.pop(client_id, None)

    def _part_channel(self, client_id: str, channel: str) -> None:
        ch = self.state.channels.get(channel)
        c = self.state.clients[client_id]
        if not ch:
            return
        ch.members = tuple(m for m in ch.members if m != client_id)
        c.channels = tuple(x for x in c.channels if x != channel)
        if not ch.members:
            self.state.channels.pop(channel, None)

    def handle_message(self, client_id: str, raw_line: str) -> tuple[str, ...]:
        if client_id not in self.state.clients:
            self.add_client(client_id)
        try:
            msg = parse_irc_line(raw_line)
        except IRCParseError as exc:
            return (self._make_reply("461", "*", trailing=str(exc)),)
        return tuple(self._dispatch(client_id, msg))

    def _dispatch(self, client_id: str, msg: IRCMessage) -> list[str]:
        c = self.state.clients[client_id]
        cmd = msg.command
        if cmd == "NICK":
            if not msg.params:
                return [self._make_reply("431", "*", trailing="No nickname given")]
            try:
                nick = normalize_nick(msg.params[0])
            except IRCParseError:
                return [self._make_reply("432", "*", trailing="Erroneous nickname")]
            if any(client.nick == nick and cid != client_id for cid, client in self.state.clients.items()):
                return [self._make_reply("433", "*", trailing="Nickname is already in use")]
            c.nick = nick
            return self._register_if_ready(client_id)
        if cmd == "USER":
            if len(msg.params) < 3 or msg.trailing is None:
                return [self._make_reply("461", "USER", trailing="Not enough parameters")]
            if len(msg.params[0]) > _MAX_USERNAME_LENGTH:
                return [self._make_reply("461", "USER", trailing="Username too long")]
            c.username = msg.params[0]
            c.realname = msg.trailing
            return self._register_if_ready(client_id)
        if cmd == "PING":
            token = msg.trailing if msg.trailing is not None else (msg.params[0] if msg.params else "")
            return [format_irc_message(IRCReply(command="PONG", params=(token,) if token else ())).rstrip("\r\n")]
        if cmd == "PONG":
            return []
        if cmd == "JOIN":
            if not c.registered:
                return [self._make_reply("451", "*", trailing="You have not registered")]
            if not msg.params:
                return [self._make_reply("461", "JOIN", trailing="Not enough parameters")]
            try:
                channel = normalize_channel(msg.params[0])
            except IRCParseError:
                return [self._make_reply("401", c.nick or "*", trailing="No such nick/channel")]
            if len(c.channels) >= _MAX_CHANNELS_PER_CLIENT:
                return [self._make_reply("461", "JOIN", trailing="Too many channels")]
            ch = self.state.channels.setdefault(channel, IRCChannelState(name=channel))
            if client_id not in ch.members:
                ch.members = tuple(sorted((*ch.members, client_id)))
            if channel not in c.channels:
                c.channels = tuple(sorted((*c.channels, channel)))
            return [format_irc_message(IRCReply(prefix=c.nick, command="JOIN", params=(channel,))).rstrip("\r\n")]
        if cmd == "PART":
            if not msg.params:
                return [self._make_reply("461", "PART", trailing="Not enough parameters")]
            channel = msg.params[0]
            if channel not in c.channels:
                return [self._make_reply("401", c.nick or "*", trailing="No such nick/channel")]
            self._part_channel(client_id, channel)
            return [format_irc_message(IRCReply(prefix=c.nick, command="PART", params=(channel,), trailing=msg.trailing)).rstrip("\r\n")]
        if cmd == "PRIVMSG":
            if not c.registered:
                return [self._make_reply("451", "*", trailing="You have not registered")]
            if len(msg.params) < 1 or msg.trailing is None:
                return [self._make_reply("461", "PRIVMSG", trailing="Not enough parameters")]
            target = msg.params[0]
            if len(msg.trailing) > _MAX_MESSAGE_LENGTH:
                return [self._make_reply("461", "PRIVMSG", trailing="Message too long")]
            if target.startswith("#"):
                ch = self.state.channels.get(target)
                if not ch:
                    return [self._make_reply("401", c.nick or "*", trailing="No such nick/channel")]
                lines = []
                for member_id in ch.members:
                    if member_id == client_id:
                        continue
                    lines.append(format_irc_message(IRCReply(prefix=c.nick, command="PRIVMSG", params=(target,), trailing=msg.trailing)).rstrip("\r\n"))
                return lines
            return [self._make_reply("401", c.nick or "*", trailing="No such nick/channel")]
        if cmd == "QUIT":
            self.remove_client(client_id)
            return []
        return [self._make_reply("421", cmd, trailing="Unknown command")]

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_socket_client, self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_socket_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._client_seq += 1
        client_id = f"client-{self._client_seq:04d}"
        self.add_client(client_id)
        self._writer_by_client[client_id] = writer
        try:
            while not reader.at_eof():
                data = await reader.readline()
                if not data:
                    break
                line = data.decode("utf-8", errors="strict").rstrip("\r\n")
                replies = self.handle_message(client_id, line)
                for reply in replies:
                    writer.write((reply + "\r\n").encode("utf-8"))
                await writer.drain()
        finally:
            self.remove_client(client_id)
            writer.close()
            await writer.wait_closed()
