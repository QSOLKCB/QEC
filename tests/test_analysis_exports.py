from __future__ import annotations

from src.qec.analysis import (
    SpectralDiversityConfig,
    SpectralDiversityMemory,
    SpectralFrustrationAnalyzer,
    SpectralLandscapeMemory,
    SpectralSignature,
    TrapMemoryConfig,
    TrapSubspaceMemory,
)


def test_analysis_exports_are_importable() -> None:
    assert SpectralFrustrationAnalyzer is not None
    assert SpectralSignature is not None
    assert TrapSubspaceMemory is not None
    assert SpectralDiversityMemory is not None
    assert SpectralDiversityConfig is not None
    assert TrapMemoryConfig is not None
    assert SpectralLandscapeMemory is not None
