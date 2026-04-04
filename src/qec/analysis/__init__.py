"""Stable analysis package exports.

Lazy-loading: heavy submodule imports are deferred to attribute access so
that importing lightweight analysis modules (e.g. attractor_analysis,
field_metrics) does not trigger optional or heavy transitive dependencies.
"""

import importlib as _importlib

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "SpectralDiversityConfig": (".spectral_diversity_memory", "SpectralDiversityConfig"),
    "SpectralDiversityMemory": (".spectral_diversity_memory", "SpectralDiversityMemory"),
    "SpectralLandscapeMemory": (".spectral_landscape_memory", "SpectralLandscapeMemory"),
    "SpectralFrustrationAnalyzer": (".spectral_frustration", "SpectralFrustrationAnalyzer"),
    "SpectralSignature": (".spectral_signature", "SpectralSignature"),
    "TrapMemoryConfig": (".trap_memory", "TrapMemoryConfig"),
    "TrapSubspaceMemory": (".trap_memory", "TrapSubspaceMemory"),
    "AuditoryPhaseSignature": (".closed_loop_auditory_phase_control", "AuditoryPhaseSignature"),
    "AuditoryPhaseLedger": (".closed_loop_auditory_phase_control", "AuditoryPhaseLedger"),
    "TemporalAuditorySequenceDecision": (".temporal_auditory_sequence_analysis", "TemporalAuditorySequenceDecision"),
    "TemporalAuditorySequenceLedger": (".temporal_auditory_sequence_analysis", "TemporalAuditorySequenceLedger"),
    "TemporalAuditoryPolicyState": (".temporal_auditory_sequence_policy_memory", "TemporalAuditoryPolicyState"),
    "TemporalAuditoryPolicyLedger": (".temporal_auditory_sequence_policy_memory", "TemporalAuditoryPolicyLedger"),
}

__all__ = [
    "SpectralDiversityConfig",
    "SpectralDiversityMemory",
    "SpectralFrustrationAnalyzer",
    "SpectralSignature",
    "SpectralLandscapeMemory",
    "TrapMemoryConfig",
    "TrapSubspaceMemory",
    "AuditoryPhaseLedger",
    "AuditoryPhaseSignature",
    "TemporalAuditorySequenceDecision",
    "TemporalAuditorySequenceLedger",
    "TemporalAuditoryPolicyState",
    "TemporalAuditoryPolicyLedger",
]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        submod, attr = _LAZY_IMPORTS[name]
        module = _importlib.import_module(submod, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    # Fall back to api module for all other public names
    try:
        api = _importlib.import_module(".api", __name__)
    except ImportError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(api, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
