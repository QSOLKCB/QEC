# SPDX-License-Identifier: MPL-2.0
"""Crossbar coordinate-matrix routing API."""

from .core import (
    CLAIM_BOUNDARY,
    CONTRACT_VERSION,
    INTERSECTION_ID_SCHEMA,
    LINK_STATES,
    MATRIX_SCHEMA,
    QEC_VERSION,
    VALIDATION_SCHEMA,
    CrossbarIntersection,
    CrossbarLink,
    CrossbarMatrix,
    demo_matrix,
    validate_matrix_manifest,
)

__all__ = [
    "CLAIM_BOUNDARY",
    "CONTRACT_VERSION",
    "INTERSECTION_ID_SCHEMA",
    "LINK_STATES",
    "MATRIX_SCHEMA",
    "QEC_VERSION",
    "VALIDATION_SCHEMA",
    "CrossbarIntersection",
    "CrossbarLink",
    "CrossbarMatrix",
    "demo_matrix",
    "validate_matrix_manifest",
]

# The v172.0 names above retain their published schemas and version constants.
from .marker import (
    MARKER_CLAIM_BOUNDARY,
    MARKER_CONTRACT_VERSION,
    CrossbarRequest,
    compile_marker_program,
    execute_marker_program,
    validate_common_control_receipt,
)

__all__ += [
    "MARKER_CLAIM_BOUNDARY",
    "MARKER_CONTRACT_VERSION",
    "CrossbarRequest",
    "compile_marker_program",
    "execute_marker_program",
    "validate_common_control_receipt",
]

# Additive v172.2 contracts; published matrix and marker names stay unchanged.
from .multistage import (
    PATH_CONTRACT_VERSION,
    PATH_CLAIM_BOUNDARY,
    CrossbarFabric,
    InterstageLink,
    MultiStageRequest,
    compile_path_program,
    execute_path_program,
    validate_path_search_receipt,
    demo_fabric,
)

__all__ += [
    "PATH_CONTRACT_VERSION",
    "PATH_CLAIM_BOUNDARY",
    "CrossbarFabric",
    "InterstageLink",
    "MultiStageRequest",
    "compile_path_program",
    "execute_path_program",
    "validate_path_search_receipt",
    "demo_fabric",
]
