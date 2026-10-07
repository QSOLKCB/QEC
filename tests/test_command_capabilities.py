"""Public metadata/CLI parity; no private argparse action inspection."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from qec.capabilities import descriptor
from qec.cli_contract import ScalarOption
from qec.command_specs import COMMANDS, QUQUART_BATTERY, QUQUART_VALIDATION, QUTRIT_BATTERY


def test_export_does_not_import_scientific_modules():
    result = subprocess.run([sys.executable, "-c",
        "import json,sys; from qec.capabilities import descriptor; "
        "print(json.dumps({'descriptor':descriptor(), 'scientific':[name for name in sys.modules "
        "if name.startswith(('qec.benchmark','qec.decoder','qec.sonify'))]}))"],
        check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["scientific"] == []
    assert payload["descriptor"] == descriptor()


def test_installed_module_export_matches_public_function():
    result = subprocess.run([sys.executable, "-m", "qec.capabilities"],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == descriptor()
    assert len(descriptor()["actions"]) == 3


@pytest.mark.parametrize("command", COMMANDS, ids=lambda command: command.id)
def test_defaults_and_required_paths_match_direct_parser(command):
    argv = ["--claims=claims.json", "--evidence=results"] if command == QUQUART_VALIDATION else []
    arguments = vars(command.parser().parse_args(argv))
    exported = command.descriptor()
    for field in exported["fields"]:
        if "default" in field:
            value = arguments[field["name"]]
            assert (str(value) if isinstance(value, Path) else value) == field["default"]
        if field.get("path_role") and arguments[field["name"]] is not None:
            assert isinstance(arguments[field["name"]], Path)
    if command == QUQUART_VALIDATION:
        assert {field["name"] for field in exported["fields"] if field["required"]} == {"claims", "evidence"}
        with pytest.raises(SystemExit):
            command.parser().parse_args([])


def test_new_supported_option_updates_parser_and_descriptor_together():
    command = replace(QUQUART_BATTERY, options=QUQUART_BATTERY.options +
                      (ScalarOption("label", choices=("a", "b"), default="a"),))
    assert command.parser().parse_args(["--label=b"]).label == "b"
    field = command.descriptor()["fields"][-1]
    assert field["choices"] == ["a", "b"] and field["default"] == "a"


def test_string_defaults_path_choices_booleans_and_omission_match_argparse():
    options = (ScalarOption("count", "integer", "10"),
               ScalarOption("path", default="a", choices=(Path("a"), Path("b")), path_role="read-file"),
               ScalarOption("enabled", "boolean", "false", choices=(True, False)),
               ScalarOption("optional", "integer", argparse.SUPPRESS))
    parser = argparse.ArgumentParser()
    for option in options:
        option.add_to(parser)
    args = vars(parser.parse_args(["--path=b", "--enabled=true"]))
    assert args == {"count": 10, "path": Path("b"), "enabled": True}
    assert options[0].descriptor()["default"] == 10
    assert options[1].descriptor()["choices"] == ["a", "b"]
    assert options[2].descriptor()["default"] is False
    assert "default" not in options[3].descriptor()


def test_custom_input_type_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="Unsupported scalar input type"):
        ScalarOption("matrix", "array").descriptor()


def test_existing_backend_bounds_are_exported_without_scientific_reinterpretation():
    assert {field["name"]: field["minimum"] for field in QUQUART_BATTERY.descriptor()["fields"]
            if "minimum" in field} == {"trials": 1, "harmonic_trials": 1}
    assert QUTRIT_BATTERY.descriptor()["fields"][-1]["minimum"] == 1
    assert QUQUART_VALIDATION.descriptor()["output"]["success_field"] == "passed"
