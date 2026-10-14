# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from scripts.check_html_links import check_links


def test_local_links_fragments_and_external_boundary(tmp_path):
    (tmp_path / 'index.html').write_text('<a href="other.html#ok">ok</a><img src="pixel.svg"><a href="https://example.invalid">external</a>')
    (tmp_path / 'other.html').write_text('<div id="ok"></div>')
    (tmp_path / 'pixel.svg').write_text('<svg/>')
    result = check_links(tmp_path, ['index.html', 'other.html'])
    assert result['errors'] == []
    assert result['documents_checked'] == 2
    assert result['external_not_checked'] == ['https://example.invalid']


def test_missing_duplicate_and_escape_are_failures(tmp_path):
    (tmp_path / 'index.html').write_text('<p id="x"></p><p id="x"></p><a href="#no">bad</a><img src="no.svg"><a href="../outside.html">escape</a>')
    result = check_links(tmp_path, ['index.html'])
    assert len(result['errors']) == 4
    assert any('missing fragment' in x for x in result['errors'])
    assert any('missing target' in x for x in result['errors'])
    assert any('duplicate id' in x for x in result['errors'])
    assert any('escapes root' in x for x in result['errors'])


def test_generated_allowance_is_exact_and_can_be_required(tmp_path):
    folder = tmp_path / 'viz/ququart-fer'
    folder.mkdir(parents=True)
    (folder / 'index.html').write_text('<a href="results/report.js">generated</a>')
    paths = ['viz/ququart-fer/index.html']
    assert check_links(tmp_path, paths)['deferred_generated'] == ['viz/ququart-fer/results/report.js']
    assert check_links(tmp_path, paths)['errors'] == []
    assert len(check_links(tmp_path, paths, require_generated=True)['errors']) == 1
    (folder / 'results').mkdir()
    (folder / 'results/report.js').write_text('// generated')
    assert check_links(tmp_path, paths, require_generated=True)['errors'] == []


def test_declared_root_relative_and_encoded_fragment(tmp_path):
    (tmp_path / 'index.html').write_text('<a href="/other.html#a%20b">ok</a>')
    (tmp_path / 'other.html').write_text('<p id="a b">text</p>')
    assert check_links(tmp_path, ['index.html'])['errors'] == []


def test_directory_index_symlink_cannot_escape_root(tmp_path):
    root = tmp_path / 'root'
    (root / 'sub').mkdir(parents=True)
    (root / 'index.html').write_text('<a href="sub/#outside">escape</a>')
    outside = tmp_path / 'outside.html'
    outside.write_text('<p id="outside">must not be read</p>')
    (root / 'sub/index.html').symlink_to(outside)
    result = check_links(root, ['index.html'])
    assert len(result['errors']) == 1
    assert 'escapes root' in result['errors'][0]
