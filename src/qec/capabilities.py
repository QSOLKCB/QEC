"""Export installed QEC command metadata without importing scientific CLIs.

Usage: python -m qec.capabilities
"""
import json

from .command_specs import COMMANDS

PROTOCOL = "qec-capabilities/1"


def descriptor() -> dict:
    return {"protocol": PROTOCOL,
            "implementation_modules": ["qec.capabilities", "qec.command_specs", "qec.cli_contract"],
            "actions": [command.descriptor() for command in COMMANDS]}


def main() -> None:
    print(json.dumps(descriptor(), ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
