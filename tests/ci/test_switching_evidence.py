# SPDX-License-Identifier: MPL-2.0
import hashlib
import json

import pytest

from scripts.generate_switching_evidence import generate
from qec.sonify.canonical import canonical_sha256


def test_candidate_manifest_binds_real_generated_bytes_and_environment(tmp_path):
    output=tmp_path/'evidence'
    manifest=generate(output,('ess',))
    assert manifest['implementation_status']=='unreleased_candidate'
    assert len(manifest['source_commit_sha'])==40
    assert type(manifest['clean_working_tree']) is bool
    assert manifest['environment']['package_version']==manifest['source_package_version']
    assert all(row['validation']['all_passed'] for row in manifest['native_replay_validations'])
    assert len(manifest['artifacts'])==9
    for row in manifest['artifacts']:
        data=(output/row['path']).read_bytes()
        assert len(data)==row['bytes']
        assert hashlib.sha256(data).hexdigest()==row['file_sha256']
    assert canonical_sha256({k:v for k,v in manifest.items() if k!='sha256'})==manifest['sha256']
    assert json.loads((output/'candidate_evidence_manifest.json').read_text())==manifest
    with pytest.raises(ValueError,match='must be empty'): generate(output,('ess',))


@pytest.mark.parametrize('eras',[(),('unknown',),('ess','ess')])
def test_invalid_era_inventory_fails_before_writes(tmp_path,eras):
    output=tmp_path/'evidence'
    with pytest.raises(ValueError): generate(output,eras)
    assert not output.exists()
