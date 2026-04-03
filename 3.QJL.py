from __future__ import annotations

import math
from typing import Optional

import numpy as np


def make_qjl_matrix(dimension: int, seed: Optional[int] = None) -> np.ndarray:
    """Sample the Gaussian matrix used by QJL."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dimension, dimension))


def qjl_quantize(x: np.ndarray, s_matrix: np.ndarray) -> np.ndarray:
    """QJL forward map: Q_qjl(x) = sign(Sx)."""
    x = np.asarray(x, dtype=np.float64)
    s_matrix = np.asarray(s_matrix, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("x must be a 1D vector.")
    if s_matrix.ndim != 2 or s_matrix.shape[0] != s_matrix.shape[1]:
        raise ValueError("s_matrix must be a square 2D array.")
    if s_matrix.shape[1] != x.shape[0]:
        raise ValueError("The dimension of x must match s_matrix.")

    projected = s_matrix @ x
    signs = np.where(projected >= 0.0, 1.0, -1.0)
    return signs


def qjl_dequantize(z: np.ndarray, s_matrix: np.ndarray) -> np.ndarray:
    """QJL inverse map: Q_qjl^{-1}(z) = sqrt(pi/2) / d * S^T z."""
    z = np.asarray(z, dtype=np.float64)
    s_matrix = np.asarray(s_matrix, dtype=np.float64)

    if z.ndim != 1:
        raise ValueError("z must be a 1D vector.")
    if s_matrix.ndim != 2 or s_matrix.shape[0] != s_matrix.shape[1]:
        raise ValueError("s_matrix must be a square 2D array.")
    if s_matrix.shape[0] != z.shape[0]:
        raise ValueError("The dimension of z must match s_matrix.")

    dimension = s_matrix.shape[0]
    scale = math.sqrt(math.pi / 2.0) / dimension
    return scale * (s_matrix.T @ z)


def main() -> None:
    dimension = 8
    x = np.array([1.0, -0.5, 0.25, 0.0, 0.75, -1.25, 0.5, -0.2], dtype=np.float64)
    x = x / np.linalg.norm(x)
    s_matrix = make_qjl_matrix(dimension=dimension, seed=7)

    z = qjl_quantize(x, s_matrix)
    x_hat = qjl_dequantize(z, s_matrix)
    difference = x_hat - x

    print("x:", np.round(x, 4))
    print("Q_qjl(x):", z.astype(int))
    print("Q_qjl^{-1}(Q_qjl(x)):", np.round(x_hat, 4))
    print("difference:", np.round(difference, 4))


if __name__ == "__main__":
    main()
