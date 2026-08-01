"""QEC adapter boundary for pinned NEXUS geometry and execution evidence."""
from .attestation import (
    NexusAttestationError,
    build_attestation,
    validate_build_attestation,
)
from .contract import NexusConfig, NexusInvocation
from .evidence import (
    NexusEvidenceError,
    build_execution_receipt,
    validate_execution_receipt,
)
from .runner import (
    NexusExecutionError,
    run_nexus,
    validate_receipt_file,
)
from .source import (
    NEXUS_V3,
    NEXUS_V4,
    PROFILES,
    NexusSourceIdentity,
    source_profile,
)

__all__ = [
    "NEXUS_V3",
    "NEXUS_V4",
    "PROFILES",
    "NexusAttestationError",
    "NexusConfig",
    "NexusEvidenceError",
    "NexusExecutionError",
    "NexusInvocation",
    "NexusSourceIdentity",
    "build_attestation",
    "build_execution_receipt",
    "run_nexus",
    "source_profile",
    "validate_build_attestation",
    "validate_execution_receipt",
    "validate_receipt_file",
]
