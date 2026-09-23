# SPDX-License-Identifier: MPL-2.0
"""CLI for immutable Crossbar contracts and bounded owned contention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qec.sonify.canonical import canonical_json

from ..equivalence import (
    EquivalenceCorpus, demo_equivalence_corpus, equivalence_adapter_manifest,
    run_equivalence_battery, validate_equivalence_matrix,
)
from .contention import (
    MAX_BATCH_SEARCH_EVALUATIONS, ContentionBatch, compile_contention_program,
    execute_contention_program, validate_contention_receipt, demo_contention_batch,
)
from .continuity import create_continuity_receipt, validate_continuity_receipt
from .core import LINK_STATES, demo_matrix, validate_matrix_manifest
from .marker import (
    MAX_PAYLOAD_BYTES,
    CrossbarRequest,
    compile_marker_program,
    execute_marker_program,
    validate_common_control_receipt,
)

from .multistage import (
    MAX_SEARCH_EVALUATIONS,
    MultiStageRequest,
    compile_path_program,
    demo_fabric,
    execute_path_program,
    validate_path_search_receipt,
)


def _read_json(path: Path) -> object:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)



def _link_state(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("link state must use LINK_ID=STATE")
    link_id, state = (part.strip() for part in text.split("=", 1))
    if not link_id:
        raise argparse.ArgumentTypeError("link id must be non-empty")
    if state not in LINK_STATES:
        raise argparse.ArgumentTypeError(f"state must be one of {LINK_STATES}")
    return link_id, state


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="qec-crossbar")
    sub = command.add_subparsers(dest="command", required=True)

    matrix = sub.add_parser("matrix", help="emit a canonical v172.0 matrix manifest")
    matrix.add_argument("--matrix-id", default="crossbar-demo")
    matrix.add_argument("--horizontal-count", type=int, default=4)
    matrix.add_argument("--vertical-count", type=int, default=4)
    matrix.add_argument(
        "--link-state",
        action="append",
        type=_link_state,
        default=[],
        metavar="LINK_ID=STATE",
        help="override a deterministic demo link state; may be repeated",
    )
    matrix.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar"))

    validate = sub.add_parser("validate", help="replay-validate a matrix manifest")
    validate.add_argument("--manifest", required=True, type=Path)
    marker = sub.add_parser("marker", help="compute a sealed exact-coordinate plan")
    marker.add_argument("--manifest", required=True, type=Path)
    marker.add_argument("--request-id", required=True)
    marker.add_argument("--horizontal-link-id", required=True)
    marker.add_argument("--vertical-link-id", required=True)
    marker.add_argument("--payload-file", required=True, type=Path)
    marker.add_argument("--decoder-output-sha256", required=True)
    marker.add_argument("--marker-id", default="marker-0")
    marker.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar-marker"))

    marker_validate = sub.add_parser("marker-validate", help="replay a common-control receipt")
    marker_validate.add_argument("--receipt", required=True, type=Path)
    marker_validate.add_argument("--expected-matrix-sha256")
    marker_validate.add_argument("--expected-request-sha256")
    marker_validate.add_argument("--expected-program-sha256")
    fabric = sub.add_parser("fabric-demo", help="emit a three-stage look-ahead fixture")
    fabric.add_argument("--all-idle", action="store_true")
    fabric.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar-fabric"))

    path = sub.add_parser("path-search", help="select the first complete admissible path")
    path.add_argument("--fabric", required=True, type=Path)
    path.add_argument("--source-matrix-id", required=True)
    path.add_argument("--destination-matrix-id", required=True)
    path.add_argument("--request-id", required=True)
    path.add_argument("--horizontal-link-id", required=True)
    path.add_argument("--vertical-link-id", required=True)
    path.add_argument("--payload-file", required=True, type=Path)
    path.add_argument("--decoder-output-sha256", required=True)
    path.add_argument("--marker-id", default="marker-0")
    path.add_argument("--max-search-evaluations", type=int, default=MAX_SEARCH_EVALUATIONS)
    path.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar-path"))

    path_validate = sub.add_parser("path-validate", help="replay a multi-stage search receipt")
    path_validate.add_argument("--receipt", required=True, type=Path)
    path_validate.add_argument("--expected-fabric-sha256")
    path_validate.add_argument("--expected-request-sha256")
    path_validate.add_argument("--expected-program-sha256")
    contention_demo = sub.add_parser("contention-demo", help="emit a bounded contention input fixture")
    contention_demo.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar-contention"))
    contention = sub.add_parser("contend", help="execute a canonical reservation/release/quarantine batch")
    contention.add_argument("--input", required=True, type=Path)
    contention.add_argument("--marker-id", default="marker-0")
    contention.add_argument("--max-search-evaluations", type=int, default=MAX_BATCH_SEARCH_EVALUATIONS)
    contention.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar-contention"))
    contention_validate = sub.add_parser("contention-validate", help="replay all contention and resource transitions")
    contention_validate.add_argument("--receipt", required=True, type=Path)
    contention_validate.add_argument("--expected-input-sha256")
    contention_validate.add_argument("--expected-program-sha256")
    contention_validate.add_argument("--expected-fabric-sha256")
    for name, help_text in (
        ("continuity", "verify all selected routes in a path-search or contention receipt"),
        ("continuity-validate", "replay a source-bound continuity receipt"),
    ):
        continuity = sub.add_parser(name, help=help_text)
        continuity.add_argument("--receipt", required=True, type=Path)
        for identity in ("source", "fabric", "input", "program"):
            continuity.add_argument("--expected-" + identity + "-sha256")
        if name == "continuity":
            continuity.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar-continuity"))
    equivalence_demo = sub.add_parser("equivalence-demo", help="emit the shared 41-case equivalence corpus")
    equivalence_demo.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar-equivalence"))
    equivalence = sub.add_parser("equivalence", help="run the Strowger/Panel/Crossbar comparison battery")
    equivalence.add_argument("--corpus", required=True, type=Path)
    equivalence.add_argument("--output-dir", type=Path, default=Path("artifacts/crossbar-equivalence"))
    equivalence_validate = sub.add_parser("equivalence-validate", help="replay every native source and comparison")
    equivalence_validate.add_argument("--matrix", required=True, type=Path)
    equivalence_validate.add_argument("--expected-corpus-sha256")
    equivalence_validate.add_argument("--expected-adapter-sha256")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "equivalence-demo":
            corpus = demo_equivalence_corpus().as_dict()
            _write(args.output_dir / "switch_equivalence_corpus.json", corpus)
            _write(args.output_dir / "switch_equivalence_adapter_manifest.json", equivalence_adapter_manifest())
            print(canonical_json({"corpus_sha256": corpus["sha256"]}))
            return 0
        if args.command == "equivalence-validate":
            validation = validate_equivalence_matrix(_read_json(args.matrix),
                expected_corpus_sha256=args.expected_corpus_sha256,
                expected_adapter_sha256=args.expected_adapter_sha256)
            print(canonical_json(validation))
            return 0 if validation["all_passed"] else 2
        if args.command == "equivalence":
            corpus = EquivalenceCorpus.from_dict(_read_json(args.corpus))
            matrix = run_equivalence_battery(corpus)
            validation = validate_equivalence_matrix(matrix,
                expected_corpus_sha256=corpus.as_dict()["sha256"],
                expected_adapter_sha256=matrix["adapter_manifest"]["sha256"])
            for name, value in (
                ("switch_equivalence_corpus.json", corpus.as_dict()),
                ("switch_equivalence_adapter_manifest.json", matrix["adapter_manifest"]),
                ("switch_equivalence_matrix.json", matrix),
                ("switch_equivalence_validation.json", validation),
            ):
                _write(args.output_dir / name, value)
            print(canonical_json(validation))
            return 0 if validation["all_passed"] else 2
        if args.command in ("continuity", "continuity-validate"):
            bindings = {"expected_" + name + "_sha256": getattr(args, "expected_" + name + "_sha256")
                        for name in ("source", "fabric", "input", "program")}
            value = _read_json(args.receipt)
            if args.command == "continuity-validate":
                print(canonical_json(validate_continuity_receipt(value, **bindings)))
                return 0
            receipt = create_continuity_receipt(value, **bindings)
            validation = validate_continuity_receipt(receipt, **bindings)
            _write(args.output_dir / "crossbar_continuity_receipt.json", receipt)
            _write(args.output_dir / "crossbar_continuity_validation.json", validation)
            print(canonical_json(validation))
            return 0 if validation["continuity_verified"] else 2
        if args.command == "contention-demo":
            batch = demo_contention_batch().as_dict()
            _write(args.output_dir / "crossbar_contention_input.json", batch)
            print(canonical_json({"input_sha256": batch["sha256"]}))
            return 0
        if args.command == "contention-validate":
            validation = validate_contention_receipt(
                _read_json(args.receipt), expected_input_sha256=args.expected_input_sha256,
                expected_program_sha256=args.expected_program_sha256,
                expected_fabric_sha256=args.expected_fabric_sha256)
            print(canonical_json(validation))
            return 0
        if args.command == "contend":
            batch = ContentionBatch.from_dict(_read_json(args.input))
            program = compile_contention_program(batch, marker_id=args.marker_id,
                max_search_evaluations=args.max_search_evaluations)
            receipt = execute_contention_program(batch, program)
            validation = validate_contention_receipt(receipt,
                expected_input_sha256=batch.as_dict()["sha256"],
                expected_program_sha256=program["sha256"],
                expected_fabric_sha256=batch.fabric.as_dict()["sha256"])
            for name, value in (
                ("crossbar_contention_input.json", batch.as_dict()),
                ("crossbar_contention_program.json", program),
                ("crossbar_contention_receipt.json", receipt),
                ("crossbar_contention_validation.json", validation),
            ):
                _write(args.output_dir / name, value)
            print(canonical_json(validation))
            return 0
        if args.command == "fabric-demo":
            fabric = demo_fabric(dead_end_first=not args.all_idle).as_dict()
            _write(args.output_dir / "crossbar_fabric_manifest.json", fabric)
            print(canonical_json({"fabric_sha256": fabric["sha256"]}))
            return 0
        if args.command == "path-validate":
            validation = validate_path_search_receipt(
                _read_json(args.receipt),
                expected_fabric_sha256=args.expected_fabric_sha256,
                expected_request_sha256=args.expected_request_sha256,
                expected_program_sha256=args.expected_program_sha256,
            )
            print(canonical_json(validation))
            return 0
        if args.command == "path-search":
            fabric = _read_json(args.fabric)
            with args.payload_file.open("rb") as stream:
                payload = stream.read(MAX_PAYLOAD_BYTES + 1)
            request = MultiStageRequest(
                args.source_matrix_id, args.destination_matrix_id,
                CrossbarRequest(args.request_id, args.horizontal_link_id,
                                args.vertical_link_id, payload, args.decoder_output_sha256),
            )
            program = compile_path_program(
                fabric, request, marker_id=args.marker_id,
                max_search_evaluations=args.max_search_evaluations,
            )
            receipt = execute_path_program(fabric, request, program)
            validation = validate_path_search_receipt(
                receipt, expected_fabric_sha256=fabric["sha256"],
                expected_request_sha256=request.as_dict()["sha256"],
                expected_program_sha256=program["sha256"],
            )
            for name, value in (
                ("crossbar_path_input_register.json", request.as_dict()),
                ("crossbar_path_search_program.json", program),
                ("crossbar_path_search_receipt.json", receipt),
                ("crossbar_path_search_validation.json", validation),
            ):
                _write(args.output_dir / name, value)
            print(canonical_json(validation))
            return 0 if receipt["outcome"] == "plan_selected" else 2
        if args.command == "marker-validate":
            validation = validate_common_control_receipt(
                _read_json(args.receipt),
                expected_matrix_sha256=args.expected_matrix_sha256,
                expected_request_sha256=args.expected_request_sha256,
                expected_program_sha256=args.expected_program_sha256,
            )
            print(canonical_json(validation))
            return 0
        if args.command == "marker":
            manifest = _read_json(args.manifest)
            with args.payload_file.open("rb") as stream:
                payload = stream.read(MAX_PAYLOAD_BYTES + 1)
            request = CrossbarRequest(
                args.request_id, args.horizontal_link_id, args.vertical_link_id,
                payload, args.decoder_output_sha256,
            )
            program = compile_marker_program(manifest, request, marker_id=args.marker_id)
            receipt = execute_marker_program(manifest, request, program)
            validation = validate_common_control_receipt(
                receipt,
                expected_matrix_sha256=manifest["sha256"],
                expected_request_sha256=request.as_dict()["sha256"],
                expected_program_sha256=program["sha256"],
            )
            for name, value in (
                ("crossbar_marker_input_register.json", request.as_dict()),
                ("crossbar_marker_program.json", program),
                ("crossbar_common_control_receipt.json", receipt),
                ("crossbar_common_control_validation.json", validation),
            ):
                _write(args.output_dir / name, value)
            print(canonical_json(validation))
            # A valid rejection receipt is evidence, not successful selection.
            return 0 if receipt["outcome"] == "plan_selected" else 2
        if args.command == "validate":
            value = json.loads(args.manifest.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("manifest must be a JSON object")
            print(canonical_json(validate_matrix_manifest(value)))
            return 0

        overrides: dict[str, str] = {}
        for link_id, state in args.link_state:
            if link_id in overrides:
                raise ValueError(f"duplicate link-state override for {link_id}")
            overrides[link_id] = state

        matrix = demo_matrix(
            args.matrix_id,
            horizontal_count=args.horizontal_count,
            vertical_count=args.vertical_count,
            state_overrides=overrides,
        )
        manifest = matrix.as_dict()
        validation = validate_matrix_manifest(manifest)
        _write(args.output_dir / "crossbar_matrix_manifest.json", manifest)
        _write(args.output_dir / "crossbar_matrix_validation.json", validation)
        print(
            canonical_json(
                {
                    "matrix_id": matrix.matrix_id,
                    "crossbar_matrix_receipt_hash": manifest["sha256"],
                    "intersection_count": len(matrix.intersections),
                    "validation_passed": validation["all_passed"],
                }
            )
        )
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"qec-crossbar: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
