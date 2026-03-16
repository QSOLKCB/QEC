import numpy as np

from src.qec.analysis.spectral_sonifier import sonify_spectrum


def interpolate_spectra(a, b, steps: int = 16):
    """Generate intermediate spectral vectors for smooth audio transitions."""
    return np.linspace(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        int(steps),
        dtype=np.float64,
    )


def sonify_trajectory(trajectory):
    audio = []

    for spectrum in trajectory:
        tone = sonify_spectrum(spectrum)
        audio.append(tone)

    if not audio:
        return np.zeros(1, dtype=np.float64)

    return np.concatenate(audio)


def sonify_trajectory_smooth(trajectory, interpolation_steps: int = 16):
    """Sonify a spectral trajectory using smooth interpolated transitions."""
    traj = [np.asarray(spectrum, dtype=np.float64) for spectrum in trajectory]
    if not traj:
        return np.zeros(1, dtype=np.float64)

    if len(traj) == 1:
        return sonify_spectrum(traj[0])

    audio = []
    for a, b in zip(traj[:-1], traj[1:]):
        for spectrum in interpolate_spectra(a, b, steps=interpolation_steps):
            audio.append(sonify_spectrum(spectrum))

    if not audio:
        return np.zeros(1, dtype=np.float64)

    return np.concatenate(audio)
