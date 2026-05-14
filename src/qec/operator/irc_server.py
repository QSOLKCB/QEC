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

RoutedReply = tuple[str, str]


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

    def _register_if_ready(self, client_id: str) -> list[RoutedReply]:
        c = self.state.clients[client_id]
        if c.registered or not c.nick or not c.username:
            return []
        c.registered = True
        return [(client_id, self._make_reply("001", c.nick, trailing="Welcome to qec-ircd"))]

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

    def _handle_message_routed(self, client_id: str, raw_line: str) -> tuple[RoutedReply, ...]:
        if client_id not in self.state.clients:
            try:
                self.add_client(client_id)
            except ValueError:
                return ((client_id, self._make_reply("461", "*", trailing="Server is full")),)
        try:
            msg = parse_irc_line(raw_line)
        except IRCParseError as exc:
            return ((client_id, self._make_reply("461", "*", trailing=str(exc))),)
        return tuple(self._dispatch(client_id, msg))

    def handle_message(self, client_id: str, raw_line: str) -> tuple[str, ...]:
        routed = self._handle_message_routed(client_id, raw_line)
        return tuple(line for target_client, line in routed if target_client == client_id)

    def _dispatch(self, client_id: str, msg: IRCMessage) -> list[RoutedReply]:
        c = self.state.clients[client_id]
        cmd = msg.command
        if cmd == "NICK":
            if not msg.params:
                return [(client_id, self._make_reply("431", "*", trailing="No nickname given"))]
            try:
                nick = normalize_nick(msg.params[0])
            except IRCParseError:
                return [(client_id, self._make_reply("432", "*", trailing="Erroneous nickname"))]
            if any(client.nick == nick and cid != client_id for cid, client in self.state.clients.items()):
                return [(client_id, self._make_reply("433", "*", trailing="Nickname is already in use"))]
            c.nick = nick
            return self._register_if_ready(client_id)
        if cmd == "USER":
            if len(msg.params) < 3 or msg.trailing is None:
                return [(client_id, self._make_reply("461", "USER", trailing="Not enough parameters"))]
            if len(msg.params[0]) > _MAX_USERNAME_LENGTH:
                return [(client_id, self._make_reply("461", "USER", trailing="Username too long"))]
            c.username = msg.params[0]
            c.realname = msg.trailing
            return self._register_if_ready(client_id)
        if cmd == "PING":
            token = msg.trailing if msg.trailing is not None else (msg.params[0] if msg.params else "")
            return [(client_id, format_irc_message(IRCReply(command="PONG", params=(token,) if token else ())).rstrip("\r\n"))]
        if cmd == "PONG":
            return []
        if cmd == "JOIN":
            if not c.registered:
                return [(client_id, self._make_reply("451", "*", trailing="You have not registered"))]
            if not msg.params:
                return [(client_id, self._make_reply("461", "JOIN", trailing="Not enough parameters"))]
            try:
                channel = normalize_channel(msg.params[0])
            except IRCParseError:
                return [(client_id, self._make_reply("401", c.nick or "*", trailing="No such nick/channel"))]
            if len(c.channels) >= _MAX_CHANNELS_PER_CLIENT:
                return [(client_id, self._make_reply("461", "JOIN", trailing="Too many channels"))]
            ch = self.state.channels.setdefault(channel, IRCChannelState(name=channel))
            if client_id not in ch.members:
                ch.members = tuple(sorted((*ch.members, client_id)))
            if channel not in c.channels:
                c.channels = tuple(sorted((*c.channels, channel)))
            return [(client_id, format_irc_message(IRCReply(prefix=c.nick, command="JOIN", params=(channel,))).rstrip("\r\n"))]
        if cmd == "PART":
            if not msg.params:
                return [(client_id, self._make_reply("461", "PART", trailing="Not enough parameters"))]
            channel = msg.params[0]
            if channel not in c.channels:
                return [(client_id, self._make_reply("401", c.nick or "*", trailing="No such nick/channel"))]
            self._part_channel(client_id, channel)
            return [(client_id, format_irc_message(IRCReply(prefix=c.nick, command="PART", params=(channel,), trailing=msg.trailing)).rstrip("\r\n"))]
        if cmd == "PRIVMSG":
            if not c.registered:
                return [(client_id, self._make_reply("451", "*", trailing="You have not registered"))]
            if len(msg.params) < 1 or msg.trailing is None:
                return [(client_id, self._make_reply("461", "PRIVMSG", trailing="Not enough parameters"))]
            target = msg.params[0]
            if len(msg.trailing) > _MAX_MESSAGE_LENGTH:
                return [(client_id, self._make_reply("461", "PRIVMSG", trailing="Message too long"))]
            if target.startswith("#"):
                ch = self.state.channels.get(target)
                if not ch:
                    return [(client_id, self._make_reply("401", c.nick or "*", trailing="No such nick/channel"))]
                if client_id not in ch.members:
                    return [(client_id, self._make_reply("404", c.nick or "*", target, trailing="Cannot send to channel"))]
                lines: list[RoutedReply] = []
                for member_id in ch.members:
                    if member_id == client_id:
                        continue
                    lines.append((member_id, format_irc_message(IRCReply(prefix=c.nick, command="PRIVMSG", params=(target,), trailing=msg.trailing)).rstrip("\r\n")))
                return lines
            return [(client_id, self._make_reply("401", c.nick or "*", trailing="No such nick/channel"))]
        if cmd == "QUIT":
            self.remove_client(client_id)
            return []
        return [(client_id, self._make_reply("421", cmd, trailing="Unknown command"))]

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
        try:
            self.add_client(client_id)
        except ValueError:
            writer.write((self._make_reply("461", "*", trailing="Server is full") + "\r\n").encode("utf-8"))
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        self._writer_by_client[client_id] = writer
        try:
            while not reader.at_eof():
                data = await reader.readline()
                if not data:
                    break
                line = data.decode("utf-8", errors="strict").rstrip("\r\n")
                replies = self._handle_message_routed(client_id, line)
                touched_writers: set[asyncio.StreamWriter] = set()
                for target_client, reply in replies:
                    target_writer = self._writer_by_client.get(target_client)
                    if target_writer is None:
                        continue
                    target_writer.write((reply + "\r\n").encode("utf-8"))
                    touched_writers.add(target_writer)
                for target_writer in touched_writers:
                    await target_writer.drain()
        finally:
            self.remove_client(client_id)
            writer.close()
            await writer.wait_closed()
