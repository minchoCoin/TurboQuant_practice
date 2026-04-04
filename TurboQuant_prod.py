from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np

from TurboQuant_mse import TurboQuantMSE


def make_qjl_matrix(dimension: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(dimension, dimension))


def qjl_quantize(x: np.ndarray, s_matrix: np.ndarray) -> np.ndarray:
    projection = s_matrix @ x
    signs = np.sign(projection)
    signs[signs == 0.0] = 1.0
    return signs


def qjl_dequantize(z: np.ndarray, s_matrix: np.ndarray, gamma: float) -> np.ndarray:
    dimension = s_matrix.shape[0]
    scale = math.sqrt(math.pi / 2.0) * gamma / dimension
    return scale * (s_matrix.T @ z)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


@dataclass
class TurboQuantProd:
    dimension: int
    bit_width: int
    turboquant_mse: TurboQuantMSE
    s_matrix: np.ndarray

    @classmethod
    def create(
        cls,
        dimension: int,
        bit_width: int,
        rotation_seed: Optional[int] = None,
        codebook_seed: Optional[int] = None,
        qjl_seed: Optional[int] = None,
    ) -> "TurboQuantProd":
        if bit_width < 1:
            raise ValueError("bit_width must be >= 1")

        mse_bits = max(bit_width - 1, 1)
        turboquant_mse = TurboQuantMSE.create(
            dimension=dimension,
            bit_width=mse_bits,
            rotation_seed=rotation_seed,
            codebook_seed=codebook_seed,
        )
        s_matrix = make_qjl_matrix(dimension=dimension, seed=qjl_seed)
        return cls(
            dimension=dimension,
            bit_width=bit_width,
            turboquant_mse=turboquant_mse,
            s_matrix=s_matrix,
        )

    def quant(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dimension,):
            raise ValueError(f"x must have shape ({self.dimension},), got {x.shape}")

        idx = self.turboquant_mse.quant(x)
        x_mse = self.turboquant_mse.dequant(idx)
        residual = x - x_mse
        gamma = float(np.linalg.norm(residual))
        qjl = qjl_quantize(residual, self.s_matrix)
        return idx, qjl, gamma

    def dequant(self, idx: np.ndarray, qjl: np.ndarray, gamma: float) -> np.ndarray:
        x_mse = self.turboquant_mse.dequant(idx)
        x_qjl = qjl_dequantize(qjl, self.s_matrix, gamma)
        return x_mse + x_qjl


def main() -> None:
    dimension = 256
    bit_width = 3
    rng = np.random.default_rng(7)

    x = rng.normal(size=dimension)
    x = x / np.linalg.norm(x)
    y = rng.normal(size=dimension)

    turboquant = TurboQuantProd.create(
        dimension=dimension,
        bit_width=bit_width,
        rotation_seed=7,
        codebook_seed=7,
        qjl_seed=7,
    )

    idx, qjl, gamma = turboquant.quant(x)
    x_hat = turboquant.dequant(idx, qjl, gamma)

    print(f"dimension={dimension}, bit_width={bit_width}")
    print(f"D_mse = {np.sum((x - x_hat) ** 2):.6f}")
    print(f"cos(x, x_hat) = {cosine_similarity(x, x_hat):.6f}")
    print(f"<y, x> = {np.dot(y, x):.6f}")
    print(f"<y, x_hat> = {np.dot(y, x_hat):.6f}")


if __name__ == "__main__":
    main()
