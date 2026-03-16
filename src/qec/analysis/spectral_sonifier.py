import numpy as np


def sonify_spectrum(spectrum, *, sample_rate=8000, duration_s=0.05):
    """Deterministically map a spectral vector to a short audio tone."""
    s = np.asarray(spectrum, dtype=np.float64)
    n = max(1, int(sample_rate * duration_s))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    base_hz = 220.0 + 30.0 * float(np.sum(np.abs(s)))
    tone = np.sin(2.0 * np.pi * base_hz * t)
    amp = 1.0 / (1.0 + float(np.linalg.norm(s)))
    return (amp * tone).astype(np.float64, copy=False)
