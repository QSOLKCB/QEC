"""Deterministic Gaussian approximation model over spectral space."""

from __future__ import annotations

import numpy as np


class BayesianSpectralModel:
    """Kernel-regression style Bayesian spectral model.

    Uses an RBF kernel and deterministic linear solve for posterior mean.
    """

    def __init__(self, *, length_scale: float = 1.0, noise: float = 1e-6) -> None:
        self.centers = np.zeros((0, 0), dtype=np.float64)
        self.weights = np.zeros((0,), dtype=np.float64)
        self.length_scale = float(np.float64(length_scale))
        self.noise = float(np.float64(noise))
        self._gram_inv = np.zeros((0, 0), dtype=np.float64)
        self.is_trained = False

    def _rbf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X64 = np.asarray(X, dtype=np.float64)
        Y64 = np.asarray(Y, dtype=np.float64)
        if X64.size == 0 or Y64.size == 0:
            return np.zeros((X64.shape[0], Y64.shape[0]), dtype=np.float64)
        diff = X64[:, None, :] - Y64[None, :, :]
        sq_norm = np.sum(diff * diff, axis=2, dtype=np.float64)
        denom = 2.0 * float(self.length_scale) * float(self.length_scale)
        if denom <= 0.0:
            denom = 2.0
        return np.exp(-sq_norm / denom).astype(np.float64, copy=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X64 = np.asarray(X, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64).reshape(-1)
        if X64.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X64.shape[0] != y64.shape[0]:
            raise ValueError("X and y must have matching sample counts")

        self.centers = X64
        if X64.shape[0] == 0:
            self.weights = np.zeros((0,), dtype=np.float64)
            self._gram_inv = np.zeros((0, 0), dtype=np.float64)
            self.is_trained = False
            return

        K = self._rbf(X64, X64)
        n = K.shape[0]
        jitter = 1e-8 * np.eye(n, dtype=np.float64)
        K = K + jitter
        K_reg = K + float(self.noise) * np.eye(K.shape[0], dtype=np.float64)
        self.weights = np.linalg.solve(K_reg, y64).astype(np.float64, copy=False)
        self._gram_inv = np.linalg.inv(K_reg).astype(np.float64, copy=False)
        self.is_trained = True

    def predict(self, spectrum: np.ndarray) -> tuple[float, float]:
        if not self.is_trained:
            return 0.0, 1.0

        x = np.asarray(spectrum, dtype=np.float64).reshape(1, -1)
        k_star = self._rbf(self.centers, x).reshape(-1)
        mean = float(np.dot(k_star, self.weights))

        variance = 1.0 - float(k_star @ self._gram_inv @ k_star)
        variance = float(max(variance, 0.0))
        sigma = float(np.sqrt(np.float64(variance)))
        return mean, sigma
