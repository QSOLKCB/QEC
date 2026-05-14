"""Local IRC operator surface for deterministic control-plane input."""

from .irc_protocol import IRCMessage, IRCParseError, IRCReply, format_irc_message, parse_irc_line
from .irc_commands import (
    IRCCommandRequest,
    IRCCommandResult,
    IRCCommandSpec,
    command_manifest_to_dict,
    format_command_response,
    get_command_specs,
    parse_operator_command,
    route_operator_command,
)
from .irc_server import IRCServer
from .irc_replay_audit import (
    IRCReplayAuditEvent,
    IRCReplayAuditReceipt,
    build_irc_replay_audit_event,
    build_irc_replay_audit_receipt,
    get_allowed_irc_audit_event_statuses,
    normalize_audit_line,
    replay_irc_audit_from_interactions,
    validate_irc_replay_audit_event,
    validate_irc_replay_audit_receipt,
)

__all__ = [
    "IRCMessage",
    "IRCParseError",
    "IRCReply",
    "IRCServer",
    "IRCCommandRequest",
    "IRCCommandResult",
    "IRCCommandSpec",
    "command_manifest_to_dict",
    "format_command_response",
    "get_command_specs",
    "parse_operator_command",
    "route_operator_command",
    "format_irc_message",
    "parse_irc_line",
    "IRCReplayAuditEvent",
    "IRCReplayAuditReceipt",
    "build_irc_replay_audit_event",
    "build_irc_replay_audit_receipt",
    "get_allowed_irc_audit_event_statuses",
    "normalize_audit_line",
    "replay_irc_audit_from_interactions",
    "validate_irc_replay_audit_event",
    "validate_irc_replay_audit_receipt",
]
