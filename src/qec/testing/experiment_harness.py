import hashlib
import json
import platform

import numpy as np


class DeterministicExperimentHarness:
    """Provides deterministic experiment execution for tests."""

    def __init__(self, seed=0):
        self.seed = int(seed)
        np.random.seed(self.seed)

    def environment_metadata(self):
        return {
            "seed": self.seed,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "timestamp": self.seed,
        }

    def experiment_hash(self, result, metadata):
        payload = {"result": result, "metadata": metadata}
        text = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(text).hexdigest()

    def run(self, experiment_fn, *args, **kwargs):
        """Execute experiment deterministically."""
        result = experiment_fn(*args, **kwargs)
        metadata = self.environment_metadata()
        exp_hash = self.experiment_hash(result, metadata)
        return {
            "result": result,
            "metadata": metadata,
            "experiment_hash": exp_hash,
        }
