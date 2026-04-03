from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np


def make_random_rotation(dimension: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random orthogonal matrix using QR decomposition."""
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(dimension, dimension))
    q_matrix, r_matrix = np.linalg.qr(gaussian)

    signs = np.sign(np.diag(r_matrix))
    signs[signs == 0.0] = 1.0
    return q_matrix @ np.diag(signs)


def sample_sphere_coordinate_distribution(
    dimension: int,
    num_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample one coordinate of a random point on S^{d-1}."""
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(num_samples, dimension))
    normalized = gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)
    return normalized[:, 0]


def initialize_codebook(samples: np.ndarray, num_centroids: int) -> np.ndarray:
    """Initialize codebook using sample quantiles."""
    quantiles = np.linspace(0.0, 1.0, num_centroids + 2)[1:-1]
    codebook = np.quantile(samples, quantiles)
    return np.sort(codebook.astype(np.float64))


def lloyd_max_quantizer(
    samples: np.ndarray,
    num_centroids: int,
    max_iters: int = 100,
    tol: float = 1e-7,
) -> np.ndarray:
    """Fit a 1D codebook by Lloyd-Max iterations on Monte Carlo samples."""
    codebook = initialize_codebook(samples, num_centroids)

    for _ in range(max_iters):
        boundaries = 0.5 * (codebook[:-1] + codebook[1:])
        assignments = np.digitize(samples, boundaries)

        updated = codebook.copy()
        for idx in range(num_centroids):
            bucket = samples[assignments == idx]
            if len(bucket) > 0:
                updated[idx] = bucket.mean()

        updated = np.sort(updated)
        if np.max(np.abs(updated - codebook)) < tol:
            codebook = updated
            break
        codebook = updated

    return codebook


def quantize_with_codebook(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Assign each scalar to the nearest centroid."""
    distances = np.abs(values[:, None] - codebook[None, :])
    return np.argmin(distances, axis=1)


def dequantize_with_codebook(indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Map centroid indices back to scalar values."""
    return codebook[indices]


@dataclass
class TurboQuantMSE:
    dimension: int
    bit_width: int
    rotation: np.ndarray
    codebook: np.ndarray

    @classmethod
    def create(
        cls,
        dimension: int,
        bit_width: int,
        rotation_seed: Optional[int] = None,
        codebook_seed: Optional[int] = None,
        codebook_samples: int = 200_000,
    ) -> "TurboQuantMSE":
        rotation = make_random_rotation(dimension=dimension, seed=rotation_seed)
        coordinate_samples = sample_sphere_coordinate_distribution(
            dimension=dimension,
            num_samples=codebook_samples,
            seed=codebook_seed,
        )
        codebook = lloyd_max_quantizer(
            samples=coordinate_samples,
            num_centroids=2**bit_width,
        )
        return cls(
            dimension=dimension,
            bit_width=bit_width,
            rotation=rotation,
            codebook=codebook,
        )

    def quant(self, x: np.ndarray) -> np.ndarray:
        """Quantize x by rotating and storing nearest-centroid indices."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != self.dimension:
            raise ValueError("x must be a 1D vector with the configured dimension.")

        rotated = self.rotation @ x
        return quantize_with_codebook(rotated, self.codebook)

    def dequant(self, indices: np.ndarray) -> np.ndarray:
        """Dequantize indices by reconstructing rotated coordinates and rotating back."""
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.shape[0] != self.dimension:
            raise ValueError("indices must be a 1D vector with the configured dimension.")

        rotated_reconstruction = dequantize_with_codebook(indices, self.codebook)
        return self.rotation.T @ rotated_reconstruction


def main() -> None:
    dimension = 8
    bit_width = 2
    x = np.array([1.0, -0.5, 0.25, 0.0, 0.75, -1.25, 0.5, -0.2], dtype=np.float64)
    x = x / np.linalg.norm(x)

    turboquant = TurboQuantMSE.create(
        dimension=dimension,
        bit_width=bit_width,
        rotation_seed=7,
        codebook_seed=11,
    )

    indices = turboquant.quant(x)
    x_hat = turboquant.dequant(indices)
    d_mse = float(np.sum((x - x_hat) ** 2))
    per_coordinate_mse = float(np.mean((x - x_hat) ** 2))
    lower_bound = 1.0 / (4.0**bit_width)
    upper_bound = (math.sqrt(3.0) * math.pi / 2.0) * lower_bound

    print("dimension:", dimension)
    print("bit width:", bit_width)
    print("codebook:", np.round(turboquant.codebook, 6))
    print("x:", np.round(x, 6))
    print("indices:", indices)
    print("reconstruction:", np.round(x_hat, 6))
    print("D_mse (sum squared error):", round(d_mse, 6))
    print("per-coordinate MSE:", round(per_coordinate_mse, 6))
    print("lower bound for D_mse:", round(lower_bound, 6))
    print("upper bound for D_mse:", round(upper_bound, 6))
    print("lower bound <= D_mse:", lower_bound <= d_mse)
    print("D_mse <= upper bound:", d_mse <= upper_bound)


if __name__ == "__main__":
    main()
