"""Discovery-loop basin switch detection in spectral space."""

from __future__ import annotations

from qec.analysis.spectral_geometry import spectral_distance


class BasinSwitchDetector:
    """Flags basin switches when spectral jumps exceed a threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def detect(self, prev_spectrum, new_spectrum) -> bool:
        return spectral_distance(prev_spectrum, new_spectrum) > self.threshold
