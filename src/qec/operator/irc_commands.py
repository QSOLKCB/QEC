from __future__ import annotations

from dataclasses import dataclass

_MAX_COMMAND_LENGTH = 64
_MAX_ARGUMENT_LENGTH = 256
_MAX_RESPONSE_LENGTH = 400
_COMMAND_PREFIX = "!"
_LATEST_RELEASE = "v162.1"


@dataclass(frozen=True)
class IRCCommandSpec:
    name: str
    summary: str
    usage: str
    category: str


@dataclass(frozen=True)
class IRCCommandRequest:
    raw_text: str
    command: str
    args: tuple[str, ...]
    target: str
    nick: str | None


@dataclass(frozen=True)
class IRCCommandResult:
    command: str
    ok: bool
    response_lines: tuple[str, ...]
    error: str | None = None


_COMMAND_SPECS: tuple[IRCCommandSpec, ...] = tuple(
    sorted(
        (
            IRCCommandSpec("help", "List available deterministic operator commands.", "!help", "operator"),
            IRCCommandSpec("status", "Show local read-only operator router status.", "!status", "operator"),
            IRCCommandSpec("corelaw", "Show deterministic core-law statement.", "!corelaw", "determinism"),
            IRCCommandSpec("hashchain", "Show terminal proof-hash chain labels.", "!hashchain", "proof"),
            IRCCommandSpec("release", "Show current release and arc context.", "!release", "release"),
            IRCCommandSpec("sphaera", "Show SPHAERA demo command (instruction only).", "!sphaera", "instructions"),
            IRCCommandSpec("test", "Show pytest commands (instruction only).", "!test", "instructions"),
            IRCCommandSpec("midi", "Explain deterministic MIDI telemetry perspective.", "!midi", "telemetry"),
            IRCCommandSpec("steane", "Explain Steane [[7,1,3]] code role.", "!steane", "qec"),
            IRCCommandSpec("qldpc", "Explain QLDPC/CSS role in deterministic QEC.", "!qldpc", "qec"),
            IRCCommandSpec("qudit", "Explain high-dimensional stabilizer readiness.", "!qudit", "qec"),
            IRCCommandSpec("about", "One-line QEC description.", "!about", "operator"),
        ),
        key=lambda s: s.name,
    )
)
_COMMAND_SPEC_BY_NAME = {spec.name: spec for spec in _COMMAND_SPECS}


def get_command_specs() -> tuple[IRCCommandSpec, ...]:
    return _COMMAND_SPECS


def parse_operator_command(text: str, *, target: str, nick: str | None = None) -> IRCCommandRequest | None:
    if not text.startswith(_COMMAND_PREFIX):
        return None
    if "\x00" in text or "\n" in text or "\r" in text:
        return IRCCommandRequest(raw_text=text, command="", args=(), target=target, nick=nick)
    body = text[len(_COMMAND_PREFIX) :].strip()
    if not body:
        return IRCCommandRequest(raw_text=text, command="", args=(), target=target, nick=nick)
    parts = body.split()
    command = parts[0].lower()
    if len(command) > _MAX_COMMAND_LENGTH:
        return IRCCommandRequest(raw_text=text, command=command, args=(), target=target, nick=nick)
    if not all(ch.isalnum() or ch in ("-", "_") for ch in command):
        return IRCCommandRequest(raw_text=text, command="", args=(), target=target, nick=nick)
    args = tuple(parts[1:])
    if any(len(arg) > _MAX_ARGUMENT_LENGTH for arg in args):
        return IRCCommandRequest(raw_text=text, command=command, args=args, target=target, nick=nick)
    return IRCCommandRequest(raw_text=text, command=command, args=args, target=target, nick=nick)


def route_operator_command(request: IRCCommandRequest) -> IRCCommandResult:
    if "\x00" in request.raw_text or "\n" in request.raw_text or "\r" in request.raw_text:
        return IRCCommandResult(request.command, False, ("INVALID_COMMAND_TEXT",), "INVALID_COMMAND_TEXT")
    if not request.command:
        return IRCCommandResult(request.command, False, ("INVALID_COMMAND",), "INVALID_COMMAND")
    if len(request.command) > _MAX_COMMAND_LENGTH:
        return IRCCommandResult(request.command, False, ("COMMAND_TOO_LONG",), "COMMAND_TOO_LONG")
    if any(len(arg) > _MAX_ARGUMENT_LENGTH for arg in request.args):
        return IRCCommandResult(request.command, False, ("ARGUMENT_TOO_LONG",), "ARGUMENT_TOO_LONG")

    if request.command == "help":
        lines = tuple(f"!{spec.name} - {spec.summary}" for spec in _COMMAND_SPECS)
        return IRCCommandResult(request.command, True, lines)
    if request.command == "status":
        return IRCCommandResult(request.command, True, ("QEC IRC operator surface: local-only, read-only, deterministic command router active.",))
    if request.command == "corelaw":
        return IRCCommandResult(request.command, True, ("same input → same ordering → same canonical JSON → same stable SHA-256 hash → same bytes → same proof artifact → same outcome",))
    if request.command == "hashchain":
        return IRCCommandResult(request.command, True, (
            "reality_loop_proof_receipt_hash → global_validation_entry_hash → global_validation_index_hash",
            "global_threshold_contract_hash → global_truth_receipt_hash → replay_record_hash → global_replay_proof_hash",
        ))
    if request.command == "release":
        return IRCCommandResult(request.command, True, (
            f"latest implemented release: {_LATEST_RELEASE}",
            "current arc: v162.x IRC Operator Control Surface",
            "current implementation: v162.1 QEC IRC Command Router",
        ))
    if request.command == "sphaera":
        return IRCCommandResult(request.command, True, ("python scripts/sphaera_proof_demo.py",))
    if request.command == "test":
        return IRCCommandResult(request.command, True, ("pytest -q", "pytest -q -ra"))
    if request.command == "midi":
        return IRCCommandResult(request.command, True, ("QEC telemetry / MIDI export is a deterministic rendering target; sound is a view, not the proof.",))
    if request.command == "steane":
        return IRCCommandResult(request.command, True, ("Steane [[7,1,3]] is a CSS stabilizer code encoding 1 logical qubit into 7 physical qubits with distance 3.",))
    if request.command == "qldpc":
        return IRCCommandResult(request.command, True, ("QLDPC codes use sparse parity structure; CSS constructions support scalable deterministic QEC modeling.",))
    if request.command == "qudit":
        return IRCCommandResult(request.command, True, ("Qudit/ququart paths extend stabilizer reasoning to higher-dimensional alphabets while preserving deterministic interfaces.",))
    if request.command == "about":
        return IRCCommandResult(request.command, True, ("QEC is a deterministic quantum error-correction research and proof system.",))
    return IRCCommandResult(request.command, False, ("UNKNOWN_COMMAND",), "UNKNOWN_COMMAND")


def format_command_response(result: IRCCommandResult) -> tuple[str, ...]:
    out: list[str] = []
    for line in result.response_lines:
        normalized = " ".join(line.rstrip().split())
        out.append(normalized[:_MAX_RESPONSE_LENGTH])
    return tuple(out)


def command_manifest_to_dict() -> dict[str, object]:
    return {
        "prefix": _COMMAND_PREFIX,
        "max_command_length": _MAX_COMMAND_LENGTH,
        "max_argument_length": _MAX_ARGUMENT_LENGTH,
        "max_response_length": _MAX_RESPONSE_LENGTH,
        "commands": [
            {"name": spec.name, "summary": spec.summary, "usage": spec.usage, "category": spec.category}
            for spec in _COMMAND_SPECS
        ],
    }
