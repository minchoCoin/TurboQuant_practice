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


def qjl_inner_product_estimate(x: np.ndarray, y: np.ndarray, s_matrix: np.ndarray) -> float:
    """Compute <y, Q_qjl^{-1}(Q_qjl(x))> for one random QJL matrix."""
    z = qjl_quantize(x, s_matrix)
    x_hat = qjl_dequantize(z, s_matrix)
    return float(y @ x_hat)


def estimate_variance(
    x: np.ndarray,
    y: np.ndarray,
    num_trials: int,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Run repeated QJL trials and return samples plus empirical variance."""
    rng = np.random.default_rng(seed)
    samples = np.zeros(num_trials, dtype=np.float64)

    for trial in range(num_trials):
        s_matrix = make_qjl_matrix(dimension=x.shape[0], rng=rng)
        samples[trial] = qjl_inner_product_estimate(x, y, s_matrix)

    empirical_variance = float(np.var(samples))
    return samples, empirical_variance


def main() -> None:
    x = np.array([1.0, -0.5, 0.25, 0.0, 0.75, -1.25, 0.5, -0.2], dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = np.array([0.3, -1.2, 0.8, 0.4, -0.6, 1.0, -0.7, 0.9], dtype=np.float64)

    num_trials = 20_000
    samples, empirical_variance = estimate_variance(
        x=x,
        y=y,
        num_trials=num_trials,
        seed=7,
    )

    dimension = x.shape[0]
    variance_bound = (math.pi / (2.0 * dimension)) * float(np.linalg.norm(y) ** 2)
    true_inner_product = float(y @ x)
    sample_mean = float(np.mean(samples))

    print("number of trials:", num_trials)
    print("dimension d:", dimension)
    print("x:", np.round(x, 6))
    print("y:", np.round(y, 6))
    print("true inner product <y, x>:", round(true_inner_product, 6))
    print("sample mean of <y, Q_qjl^{-1}(Q_qjl(x))>:", round(sample_mean, 6))
    print("empirical variance:", round(empirical_variance, 6))
    print("lemma 4 variance bound:", round(variance_bound, 6))
    print("empirical variance <= bound:", empirical_variance <= variance_bound)


if __name__ == "__main__":
    main()
