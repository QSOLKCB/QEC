import json

from qec.operator.irc_commands import (
    _MAX_ARGUMENT_LENGTH,
    _MAX_COMMAND_LENGTH,
    _MAX_RESPONSE_LENGTH,
    command_manifest_to_dict,
    format_command_response,
    get_command_specs,
    parse_operator_command,
    route_operator_command,
)


def _parse(text: str):
    req = parse_operator_command(text, target="#qec", nick="alice")
    assert req is not None
    return req


def test_specs_sorted_unique_and_no_duplicates():
    specs = get_command_specs()
    names = [s.name for s in specs]
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_help_deterministic_and_contains_commands():
    out = format_command_response(route_operator_command(_parse("!help")))
    assert out == format_command_response(route_operator_command(_parse("!help")))
    assert any(line.startswith("!corelaw") for line in out)


def test_parse_non_command_none_and_lowercase():
    assert parse_operator_command("hello", target="#qec", nick="alice") is None
    req = _parse("!HeLp")
    assert req.command == "help"


def test_parse_rejects_invalid_text_and_lengths():
    bad = _parse("!help\n")
    assert route_operator_command(bad).error == "INVALID_COMMAND_TEXT"
    too_long = _parse("!" + ("a" * (_MAX_COMMAND_LENGTH + 1)))
    assert route_operator_command(too_long).error == "COMMAND_TOO_LONG"
    arg_long = _parse("!help " + ("a" * (_MAX_ARGUMENT_LENGTH + 1)))
    assert route_operator_command(arg_long).error == "ARGUMENT_TOO_LONG"


def test_routes_expected_commands_and_unknown():
    assert "same input" in format_command_response(route_operator_command(_parse("!corelaw")))[0]
    assert "global_replay_proof_hash" in " ".join(format_command_response(route_operator_command(_parse("!hashchain"))))
    assert "v162.1" in " ".join(format_command_response(route_operator_command(_parse("!release"))))
    assert "scripts/sphaera_proof_demo.py" in format_command_response(route_operator_command(_parse("!sphaera")))[0]
    test_lines = format_command_response(route_operator_command(_parse("!test")))
    assert "pytest -q" in test_lines[0]
    assert "pytest -q -ra" in test_lines[1]
    assert "deterministic rendering target" in format_command_response(route_operator_command(_parse("!midi")))[0]
    assert "[[7,1,3]]" in format_command_response(route_operator_command(_parse("!steane")))[0]
    assert "QLDPC" in format_command_response(route_operator_command(_parse("!qldpc")))[0]
    assert "Qudit" in format_command_response(route_operator_command(_parse("!qudit")))[0]
    unknown = route_operator_command(_parse("!unknown"))
    assert unknown.error == "UNKNOWN_COMMAND"


def test_response_bound_and_manifest_json_safe():
    for line in format_command_response(route_operator_command(_parse("!help"))):
        assert len(line) <= _MAX_RESPONSE_LENGTH
    manifest = command_manifest_to_dict()
    encoded = json.dumps(manifest, sort_keys=True)
    assert isinstance(encoded, str)
