import numpy as np


def project_2d(trajectory, dims=(0, 1)):
    """Extract 2D projection of spectral trajectory."""
    traj = np.asarray(trajectory, dtype=np.float64)
    return traj[:, dims]


def project_3d(trajectory, dims=(0, 1, 2)):
    """Extract 3D projection of spectral trajectory."""
    traj = np.asarray(trajectory, dtype=np.float64)
    return traj[:, dims]


def project_basin_centers_2d(basin_centers, dims=(0, 1)):
    """Project basin centers into 2D spectral phase space."""
    centers = np.asarray(basin_centers, dtype=np.float64)
    if centers.ndim != 2:
        raise ValueError("basin_centers must be a 2D array")
    if centers.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return project_2d(centers, dims=dims)


def project_basin_centers_3d(basin_centers, dims=(0, 1, 2)):
    """Project basin centers into 3D spectral phase space."""
    centers = np.asarray(basin_centers, dtype=np.float64)
    if centers.ndim != 2:
        raise ValueError("basin_centers must be a 2D array")
    if centers.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return project_3d(centers, dims=dims)
