"""Optional CLI utility for spectral artifact sonification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.qec.analysis.spectral_sonifier import sonify_experiment_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qec-exp spectral-sonify")
    parser.add_argument("artifact", help="Path to experiment artifact JSON")
    parser.add_argument("output", help="Output WAV file path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    artifact_path = Path(args.artifact)
    output_path = Path(args.output)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    sonify_experiment_artifact(artifact, str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
