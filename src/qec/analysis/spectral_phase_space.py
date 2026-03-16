import numpy as np


def project_2d(trajectory, dims=(0, 1)):
    """Extract 2D projection of spectral trajectory."""
    traj = np.asarray(trajectory, dtype=np.float64)
    return traj[:, dims]


def project_3d(trajectory, dims=(0, 1, 2)):
    """Extract 3D projection of spectral trajectory."""
    traj = np.asarray(trajectory, dtype=np.float64)
    return traj[:, dims]
