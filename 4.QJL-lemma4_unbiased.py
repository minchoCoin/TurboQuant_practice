from __future__ import annotations

import math

import numpy as np


def make_qjl_matrix(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Sample the Gaussian matrix used by QJL."""
    return rng.normal(loc=0.0, scale=1.0, size=(dimension, dimension))


def qjl_quantize(x: np.ndarray, s_matrix: np.ndarray) -> np.ndarray:
    """QJL forward map: Q_qjl(x) = sign(Sx)."""
    projected = s_matrix @ x
    return np.where(projected >= 0.0, 1.0, -1.0)


def qjl_dequantize(z: np.ndarray, s_matrix: np.ndarray) -> np.ndarray:
    """QJL inverse map: Q_qjl^{-1}(z) = sqrt(pi/2) / d * S^T z."""
    dimension = s_matrix.shape[0]
    scale = math.sqrt(math.pi / 2.0) / dimension
    return scale * (s_matrix.T @ z)


def average_qjl_reconstruction(
    x: np.ndarray,
    num_trials: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat QJL -> QJL^{-1} and return all reconstructions plus their mean."""
    rng = np.random.default_rng(seed)
    reconstructions = np.zeros((num_trials, x.shape[0]), dtype=np.float64)

    for trial in range(num_trials):
        s_matrix = make_qjl_matrix(dimension=x.shape[0], rng=rng)
        z = qjl_quantize(x, s_matrix)
        reconstructions[trial] = qjl_dequantize(z, s_matrix)

    mean_reconstruction = reconstructions.mean(axis=0)
    return reconstructions, mean_reconstruction


def main() -> None:
    x = np.array([1.0, -0.5, 0.25, 0.0, 0.75, -1.25, 0.5, -0.2], dtype=np.float64)
    x = x / np.linalg.norm(x)

    num_trials = 10_000
    reconstructions, mean_reconstruction = average_qjl_reconstruction(
        x=x,
        num_trials=num_trials,
        seed=7,
    )
    difference = mean_reconstruction - x
    mean_l2_error = np.linalg.norm(reconstructions - x, axis=1).mean()
    averaged_l2_error = np.linalg.norm(difference)

    print("number of trials:", num_trials)
    print("original x:", np.round(x, 6))
    print("average of Q_qjl^{-1}(Q_qjl(x)):", np.round(mean_reconstruction, 6))
    print("difference (average - original):", np.round(difference, 6))
    print("mean single-trial L2 error:", round(float(mean_l2_error), 6))
    print("L2 error of the averaged reconstruction:", round(float(averaged_l2_error), 6))


if __name__ == "__main__":
    main()
