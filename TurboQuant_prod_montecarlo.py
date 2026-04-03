from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np

from TurboQuant_mse_montecarlo import TurboQuantMSE


def make_qjl_matrix(dimension: int, seed: Optional[int] = None) -> np.ndarray:
    """Sample the Gaussian matrix used by the QJL residual sketch."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dimension, dimension))


def qjl_quantize(x: np.ndarray, s_matrix: np.ndarray) -> np.ndarray:
    """QJL forward map: sign(Sx)."""
    projected = s_matrix @ x
    return np.where(projected >= 0.0, 1.0, -1.0)


def qjl_dequantize(z: np.ndarray, s_matrix: np.ndarray, gamma: float) -> np.ndarray:
    """Scaled QJL inverse used in TurboQuant_prod."""
    dimension = s_matrix.shape[0]
    scale = math.sqrt(math.pi / 2.0) * gamma / dimension
    return scale * (s_matrix.T @ z)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


@dataclass
class TurboQuantProd:
    dimension: int
    bit_width: int
    turboquant_mse: TurboQuantMSE
    qjl_matrix: np.ndarray

    @classmethod
    def create(
        cls,
        dimension: int,
        bit_width: int,
        rotation_seed: Optional[int] = None,
        codebook_seed: Optional[int] = None,
        qjl_seed: Optional[int] = None,
        codebook_samples: int = 200_000,
    ) -> "TurboQuantProd":
        if bit_width < 1:
            raise ValueError("bit_width must be at least 1.")

        turboquant_mse = TurboQuantMSE.create(
            dimension=dimension,
            bit_width=max(bit_width - 1, 0),
            rotation_seed=rotation_seed,
            codebook_seed=codebook_seed,
            codebook_samples=codebook_samples,
        )
        qjl_matrix = make_qjl_matrix(dimension=dimension, seed=qjl_seed)
        return cls(
            dimension=dimension,
            bit_width=bit_width,
            turboquant_mse=turboquant_mse,
            qjl_matrix=qjl_matrix,
        )

    def quant(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Quantize x using MSE quantization plus a QJL sketch of the residual."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != self.dimension:
            raise ValueError("x must be a 1D vector with the configured dimension.")

        idx = self.turboquant_mse.quant(x)
        x_mse = self.turboquant_mse.dequant(idx)
        residual = x - x_mse
        qjl = qjl_quantize(residual, self.qjl_matrix)
        gamma = float(np.linalg.norm(residual))
        return idx, qjl, gamma

    def dequant(self, idx: np.ndarray, qjl: np.ndarray, gamma: float) -> np.ndarray:
        """Dequantize the MSE part and add the QJL residual correction."""
        idx = np.asarray(idx)
        qjl = np.asarray(qjl, dtype=np.float64)

        if idx.ndim != 1 or idx.shape[0] != self.dimension:
            raise ValueError("idx must be a 1D vector with the configured dimension.")
        if qjl.ndim != 1 or qjl.shape[0] != self.dimension:
            raise ValueError("qjl must be a 1D vector with the configured dimension.")

        x_mse = self.turboquant_mse.dequant(idx)
        x_qjl = qjl_dequantize(qjl, self.qjl_matrix, gamma)
        return x_mse + x_qjl


def main() -> None:
    dimension = 8
    bit_width = 3
    x = np.array([1.0, -0.5, 0.25, 0.0, 0.75, -1.25, 0.5, -0.2], dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = np.array([0.3, -1.2, 0.8, 0.4, -0.6, 1.0, -0.7, 0.9], dtype=np.float64)

    turboquant = TurboQuantProd.create(
        dimension=dimension,
        bit_width=bit_width,
        rotation_seed=7,
        codebook_seed=11,
        qjl_seed=13,
    )

    idx, qjl, gamma = turboquant.quant(x)
    x_hat = turboquant.dequant(idx, qjl, gamma)
    inner_product_true = float(np.dot(y, x))
    inner_product_reconstructed = float(np.dot(y, x_hat))
    d_mse = float(np.sum((x - x_hat) ** 2))
    cosine = cosine_similarity(x, x_hat)

    print("dimension:", dimension)
    print("bit width:", bit_width)
    print("x:", np.round(x, 6))
    print("y:", np.round(y, 6))
    print("idx:", idx)
    print("qjl:", qjl.astype(int))
    print("gamma:", round(gamma, 6))
    print("reconstruction:", np.round(x_hat, 6))
    print("D_mse:", round(d_mse, 6))
    print("cosine similarity cos(x, x_hat):", round(cosine, 6))
    print("true inner product <y, x>:", round(inner_product_true, 6))
    print("reconstructed inner product <y, x_hat>:", round(inner_product_reconstructed, 6))


if __name__ == "__main__":
    main()
