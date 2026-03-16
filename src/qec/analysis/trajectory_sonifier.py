import numpy as np

from src.qec.analysis.spectral_sonifier import sonify_spectrum


def sonify_trajectory(trajectory):
    audio = []

    for spectrum in trajectory:
        tone = sonify_spectrum(spectrum)
        audio.append(tone)

    if not audio:
        return np.zeros(1, dtype=np.float64)

    return np.concatenate(audio)
