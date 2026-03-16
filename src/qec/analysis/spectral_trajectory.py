import json

import numpy as np


class SpectralTrajectoryRecorder:
    """Records spectral state vectors during discovery runs."""

    def __init__(self, save_every_n_steps: int = 1):
        save_every_n = int(save_every_n_steps)
        if save_every_n < 1:
            raise ValueError("save_every_n_steps must be >= 1")
        self.save_every_n_steps = save_every_n
        self.step_counter = 0
        self.history = []

    def record(self, spectrum):
        if self.step_counter % self.save_every_n_steps == 0:
            self.history.append(np.asarray(spectrum, dtype=np.float64))
        self.step_counter += 1

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
