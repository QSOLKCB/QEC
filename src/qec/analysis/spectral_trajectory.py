import json

import numpy as np


class SpectralTrajectoryRecorder:
    """Records spectral state vectors during discovery runs."""

    def __init__(self):
        self.history = []

    def record(self, spectrum):
        self.history.append(np.asarray(spectrum, dtype=np.float64))

    def length(self):
        return len(self.history)

    def as_array(self):
        if not self.history:
            return np.zeros((0, 0), dtype=np.float64)
        return np.vstack(self.history)

    def to_json(self):
        return {
            "spectral_trajectory": [s.tolist() for s in self.history],
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f)
