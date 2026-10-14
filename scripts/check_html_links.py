# SPDX-License-Identifier: MPL-2.0
"""Offline local-link/fragment gate; external HTTP liveness is not asserted."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlsplit

# Produced by the existing Ququart FER Pages workflow, not committed source.
GENERATED_FER = frozenset(
    'viz/ququart-fer/results/' + name
    for name in (
        'exact_weight_enumerator.csv', 'exact_fer_curve.csv',
        'exact_channel_weight_enumerator.csv', 'exact_channel_fer.csv',
        'monte_carlo_fer.csv', 'harmonic_fault_matrix.csv',
        'harmonic_end_to_end.csv', 'receiver_operating_curve.csv',
        'lane_symmetry_certificate.json', 'report_claims.json',
        'claim_validation.json', 'qbraid_replication_receipt.json',
        'methodology.json', 'benchmark_manifest.json', 'report.js',
    )
)


class Document(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.duplicate_ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if not value:
                continue
            if key == 'id' or (tag == 'a' and key == 'name'):
                if value in self.ids:
                    self.duplicate_ids.add(value)
                self.ids.add(value)
            if key in ('href', 'src'):
                self.links.append(value)

    handle_startendtag = handle_starttag


def check_links(root: Path, paths: list[str], *, require_generated=False):
    root = root.resolve()
    documents = {}
    errors = []
    deferred = set()
    external = set()
    for name in paths:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            errors.append(f'{name}: document escapes root')
            continue
        try:
            document = Document()
            document.feed(path.read_text(encoding='utf-8'))
            document.close()
            documents[path] = document
            for duplicate in sorted(document.duplicate_ids):
                errors.append(f'{name}: duplicate id {duplicate!r}')
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f'{name}: unreadable HTML ({type(exc).__name__})')
    for path, document in sorted(documents.items()):
        for link in document.links:
            try:
                parsed = urlsplit(link)
            except ValueError:
                errors.append(f'{path.relative_to(root)}: malformed URL {link!r}')
                continue
            if parsed.scheme or parsed.netloc:
                external.add(link)
                continue
            target = ((root / unquote(parsed.path).lstrip('/')) if parsed.path.startswith('/')
                      else path.parent / unquote(parsed.path)) if parsed.path else path
            target = target.resolve()
            if not target.is_relative_to(root):
                errors.append(f'{path.relative_to(root)}: link escapes root {link!r}')
                continue
            relative = target.relative_to(root).as_posix()
            if target.is_dir():
                target = (target / 'index.html').resolve()
                if not target.is_relative_to(root):
                    errors.append(f'{path.relative_to(root)}: link escapes root {link!r}')
                    continue
            if not target.is_file():
                if relative in GENERATED_FER and not require_generated:
                    deferred.add(relative)
                else:
                    errors.append(f'{path.relative_to(root)}: missing target {link!r}')
                continue
            fragment = unquote(parsed.fragment)
            if fragment and target.suffix.lower() == '.html':
                if target not in documents:
                    try:
                        parsed_target = Document()
                        parsed_target.feed(target.read_text(encoding='utf-8'))
                        parsed_target.close()
                        documents_target = parsed_target
                    except (OSError, UnicodeError, ValueError):
                        errors.append(f'{path.relative_to(root)}: unreadable target {link!r}')
                        continue
                else:
                    documents_target = documents[target]
                if fragment not in documents_target.ids:
                    errors.append(f'{path.relative_to(root)}: missing fragment {link!r}')
    return {'errors': sorted(errors), 'deferred_generated': sorted(deferred),
            'external_not_checked': sorted(external), 'documents_checked': len(documents)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--require-generated', action='store_true')
    args = parser.parse_args(argv)
    paths = subprocess.check_output(['git', '-C', str(args.root), 'ls-files', '-z', '*.html']).decode().split('\0')
    result = check_links(args.root, [name for name in paths if name], require_generated=args.require_generated)
    for key, value in result.items():
        print(f'{key}: {value}')
    return int(bool(result['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
