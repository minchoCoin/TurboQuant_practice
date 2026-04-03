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


def sphere_coordinate_log_density(x: np.ndarray, dimension: int) -> np.ndarray:
    """Stable log-density from Lemma 1 using log-gamma."""
    x = np.asarray(x, dtype=np.float64)
    safe_term = np.maximum(1.0 - x**2, np.finfo(np.float64).tiny)
    log_coefficient = (
        math.lgamma(dimension / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma((dimension - 1.0) / 2.0)
    )
    log_power = ((dimension - 3.0) / 2.0) * np.log(safe_term)
    log_density = log_coefficient + log_power

    outside_support = np.abs(x) > 1.0
    log_density[outside_support] = -np.inf
    return log_density


def sphere_coordinate_density(x: np.ndarray, dimension: int) -> np.ndarray:
    """Stable density from Lemma 1 using log-gamma."""
    return np.exp(sphere_coordinate_log_density(x, dimension))


def initialize_codebook_from_grid(grid: np.ndarray, weights: np.ndarray, num_centroids: int) -> np.ndarray:
    """Initialize codebook using weighted quantiles on a deterministic grid."""
    quantiles = np.linspace(0.0, 1.0, num_centroids + 2)[1:-1]
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]
    codebook = np.interp(quantiles, cumulative, grid)
    return np.sort(codebook.astype(np.float64))


def lloyd_max_quantizer_from_density(
    dimension: int,
    num_centroids: int,
    num_grid_points: int = 200_001,
    max_iters: int = 100,
    tol: float = 1e-7,
) -> np.ndarray:
    """Fit a 1D codebook from the exact density using stable log-gamma formulas."""
    grid = np.linspace(-1.0, 1.0, num_grid_points, dtype=np.float64)
    grid_spacing = grid[1] - grid[0]
    weights = sphere_coordinate_density(grid, dimension) * grid_spacing
    codebook = initialize_codebook_from_grid(grid, weights, num_centroids)

    for _ in range(max_iters):
        boundaries = 0.5 * (codebook[:-1] + codebook[1:])
        assignments = np.digitize(grid, boundaries)

        updated = codebook.copy()
        for idx in range(num_centroids):
            bucket_mask = assignments == idx
            bucket_grid = grid[bucket_mask]
            bucket_weights = weights[bucket_mask]
            weight_sum = bucket_weights.sum()
            if weight_sum > 0.0:
                updated[idx] = float(np.sum(bucket_grid * bucket_weights) / weight_sum)

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
        codebook = lloyd_max_quantizer_from_density(
            dimension=dimension,
            num_centroids=2**bit_width,
            num_grid_points=max(codebook_samples + 1, 50_001),
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
    dimension = 1024
    bit_width = 2
    x = np.array([1.0, -0.5, 0.25, 0.0, 0.75, -1.25, 0.5, -0.2], dtype=np.float64)
    x = x / np.linalg.norm(x)

    turboquant = TurboQuantMSE.create(
        dimension=dimension,
        bit_width=bit_width,
        rotation_seed=7,
        codebook_seed=11,
    )

    x_padded = np.zeros(dimension, dtype=np.float64)
    x_padded[: x.shape[0]] = x
    indices = turboquant.quant(x_padded)
    x_hat = turboquant.dequant(indices)
    d_mse = float(np.sum((x_padded - x_hat) ** 2))
    per_coordinate_mse = float(np.mean((x_padded - x_hat) ** 2))

    print("dimension:", dimension)
    print("bit width:", bit_width)
    print("first codebook entries:", np.round(turboquant.codebook[: min(4, len(turboquant.codebook))], 6))
    print("D_mse (sum squared error):", round(d_mse, 6))
    print("per-coordinate MSE:", round(per_coordinate_mse, 6))


if __name__ == "__main__":
    main()
