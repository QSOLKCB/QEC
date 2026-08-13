# SPDX-License-Identifier: MPL-2.0
from copy import deepcopy
import pytest
from qec.routing.crossbar import CrossbarLink, CrossbarMatrix, demo_matrix, validate_matrix_manifest
from qec.sonify.canonical import canonical_sha256


def test_matrix_copies_input_sequences():
    horizontal=[CrossbarLink("horizontal","H000",0)]
    vertical=[CrossbarLink("vertical","V000",0)]
    matrix=CrossbarMatrix("copy-inputs",horizontal,vertical)
    expected=matrix.as_dict()
    horizontal.append(CrossbarLink("horizontal","H001",1))
    vertical.clear()
    assert matrix.as_dict()==expected


def test_claim_boundary_requires_boolean_types():
    manifest=deepcopy(demo_matrix("typed-boundary",horizontal_count=2,vertical_count=2).as_dict())
    manifest["claim_boundary"]["marker_authority_present"]=0
    unsigned=dict(manifest); unsigned.pop("sha256",None)
    manifest["sha256"]=canonical_sha256(unsigned)
    with pytest.raises(ValueError,match="claim boundary mismatch"):
        validate_matrix_manifest(manifest)
