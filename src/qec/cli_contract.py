"""Public scalar command declarations shared by CLI parsers and descriptors."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


def boolean(text: str) -> bool:
    if text not in ("true", "false"):
        raise argparse.ArgumentTypeError("boolean values must be true or false")
    return text == "true"


@dataclass(frozen=True)
class ScalarOption:
    name: str
    kind: str = "string"
    default: object = None
    required: bool = False
    choices: tuple | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    help: str = ""
    path_role: str | None = None

    @property
    def flag(self) -> str:
        return "--" + self.name.replace("_", "-")

    @property
    def converter(self):
        converters = {"string": str, "integer": int, "number": float, "boolean": boolean}
        if self.kind not in converters:
            raise ValueError(f"Unsupported scalar input type: {self.name}: {self.kind}")
        if self.path_role is not None:
            if self.kind != "string" or self.path_role not in ("read-file", "write-file", "read-directory", "write-directory"):
                raise ValueError(f"Unsupported path contract: {self.name}")
            return Path
        return converters[self.kind]

    def add_to(self, parser: argparse.ArgumentParser) -> None:
        converter = self.converter
        choices = None if self.choices is None else tuple(
            Path(value) if self.path_role else value for value in self.choices)
        parser.add_argument(self.flag, type=converter, default=self.default,
                            required=self.required, choices=choices, help=self.help)

    def descriptor(self) -> dict:
        converter = self.converter
        result = {"name": self.name, "flag": self.flag, "type": self.kind,
                  "label": self.name.replace("_", " ").title(), "required": self.required,
                  "help": self.help}
        if self.default is not None and self.default != argparse.SUPPRESS:
            default = converter(self.default) if isinstance(self.default, str) else self.default
            result["default"] = str(default) if isinstance(default, Path) else default
        if self.choices is not None:
            result["choices"] = [str(value) if isinstance(value, Path) else value for value in self.choices]
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.path_role:
            result["path_role"] = self.path_role
            result["path_base"] = "cwd"
        return result


@dataclass(frozen=True)
class CommandSpec:
    id: str
    module: str
    title: str
    description: str
    options: tuple[ScalarOption, ...]
    effect: str
    output: dict

    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.description)
        for option in self.options:
            option.add_to(parser)
        return parser

    def descriptor(self) -> dict:
        return {"id": self.id, "module": self.module, "title": self.title,
                "description": self.description, "fields": [option.descriptor() for option in self.options],
                "effect": self.effect, "output": dict(self.output)}
