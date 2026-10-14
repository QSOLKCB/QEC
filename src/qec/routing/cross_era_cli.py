# SPDX-License-Identifier: MPL-2.0
"""Offline corpus, migration, replay, equivalence, selection and report commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from qec.sonify.canonical import canonical_json
from .cross_era import (
    ArchitectureSelectionPolicy, CompatibilityCorpus, create_cross_era_equivalence_receipt,
    create_switching_evolution_report, default_compatibility_corpus, execute_cross_era_replay,
    select_architecture, validate_architecture_selection, validate_cross_era_equivalence,
    validate_cross_era_replay, validate_migration_contract, validate_switching_evolution_report,
)

MAX_FILE_BYTES = 64_000_000


def _read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('INVALID_INPUT')
            value[key] = item
        return value
    def invalid_constant(value):
        raise ValueError('INVALID_INPUT')
    with path.open('rb') as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError('INVALID_INPUT')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=invalid_constant)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m qec.routing.cross_era_cli')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('demo', 'run'):
        command = sub.add_parser(name)
        command.add_argument('--output-dir', type=Path, default=Path('artifacts/cross-era'))
        if name == 'run':
            command.add_argument('--input', type=Path, required=True)
    select = sub.add_parser('select')
    select.add_argument('--required', action='append', default=[])
    select.add_argument('--output-dir', type=Path, default=Path('artifacts/cross-era'))
    validate = sub.add_parser('validate')
    validate.add_argument('--kind', choices=('migration','replay','equivalence','selection','report'), required=True)
    validate.add_argument('--receipt', type=Path, required=True)
    validate.add_argument('--expected-sha256')
    args = parser.parse_args(argv)
    try:
        if args.command == 'demo':
            source = default_compatibility_corpus().as_dict()
            _write(args.output_dir / 'switch_compatibility_corpus_manifest.json', source)
            result = {'corpus_sha256': source['sha256'], 'case_count': len(source['cases'])}
        elif args.command == 'run':
            corpus = CompatibilityCorpus.from_dict(_read(args.input))
            replay = execute_cross_era_replay(corpus)
            equivalence = create_cross_era_equivalence_receipt(replay)
            report = create_switching_evolution_report(equivalence)
            result = validate_switching_evolution_report(report, expected_equivalence_sha256=equivalence['sha256'])
            artifacts = {
                'switch_compatibility_corpus_manifest': corpus.as_dict(),
                'switch_migration_contract': replay['migration_contract'],
                'cross_era_replay_receipt': replay, 'cross_era_equivalence_receipt': equivalence,
                'architecture_selection_policy': report['architecture_selection']['policy'],
                'architecture_selection_receipt': report['architecture_selection'],
                'switching_evolution_report': report, 'switching_evolution_validation': result,
            }
            for name, value in artifacts.items():
                _write(args.output_dir / (name + '.json'), value)
        elif args.command == 'select':
            receipt = select_architecture(ArchitectureSelectionPolicy(tuple(sorted(args.required))))
            _write(args.output_dir / 'architecture_selection_receipt.json', receipt)
            result = validate_architecture_selection(receipt)
        else:
            receipt = _read(args.receipt)
            if args.expected_sha256 is not None and receipt.get('sha256') != args.expected_sha256:
                raise ValueError('INVALID_INPUT')
            validators = {'migration': validate_migration_contract, 'replay': validate_cross_era_replay,
                          'equivalence': validate_cross_era_equivalence, 'selection': validate_architecture_selection,
                          'report': validate_switching_evolution_report}
            result = validators[args.kind](receipt)
        print(canonical_json(result))
        return 0 if result.get('all_passed', True) else 1
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
