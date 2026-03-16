import numpy as np


def replay_spectral_trajectory(trajectory):
    """Iterate over recorded spectral states."""
    for state in trajectory:
        yield np.asarray(state, dtype=np.float64)
