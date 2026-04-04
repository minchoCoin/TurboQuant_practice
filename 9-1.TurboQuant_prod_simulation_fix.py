from __future__ import annotations

from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np

from TurboQuant_prod import TurboQuantProd, cosine_similarity


def sample_unit_vector(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random unit vector from S^{d-1}."""
    vector = rng.normal(size=dimension)
    return vector / np.linalg.norm(vector)


def run_turboquant_prod_trials(
    dimension: int,
    bit_width: int,
    num_trials: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=dimension)
    turboquant = TurboQuantProd.create(
        dimension=dimension,
        bit_width=bit_width,
        rotation_seed=int(rng.integers(0, 2**31 - 1)),
        codebook_seed=int(rng.integers(0, 2**31 - 1)),
        qjl_seed=int(rng.integers(0, 2**31 - 1)),
    )

    d_mse_values = np.zeros(num_trials, dtype=np.float64)
    per_coordinate_mse_values = np.zeros(num_trials, dtype=np.float64)
    d_prod_values = np.zeros(num_trials, dtype=np.float64)
    cosine_values = np.zeros(num_trials, dtype=np.float64)

    for trial in range(num_trials):
        x = sample_unit_vector(dimension, rng)
        idx, qjl, gamma = turboquant.quant(x)
        x_hat = turboquant.dequant(idx, qjl, gamma)

        difference = x - x_hat
        d_mse_values[trial] = float(np.sum(difference**2))
        per_coordinate_mse_values[trial] = float(np.mean(difference**2))
        d_prod_values[trial] = float((np.dot(y, x) - np.dot(y, x_hat)) ** 2)
        cosine_values[trial] = cosine_similarity(x, x_hat)

    y_norm_sq = float(np.linalg.norm(y) ** 2)
    lower_bound = y_norm_sq / (dimension * (4.0**bit_width))
    upper_bound = (math.sqrt(3.0) * math.pi**2 * y_norm_sq) / (dimension * (4.0**bit_width))

    return {
        "d_mse_values": d_mse_values,
        "per_coordinate_mse_values": per_coordinate_mse_values,
        "d_prod_values": d_prod_values,
        "cosine_values": cosine_values,
        "mean_d_mse": float(np.mean(d_mse_values)),
        "mean_per_coordinate_mse": float(np.mean(per_coordinate_mse_values)),
        "mean_d_prod": float(np.mean(d_prod_values)),
        "mean_cosine": float(np.mean(cosine_values)),
        "d_prod_lower_bound": lower_bound,
        "d_prod_upper_bound": upper_bound,
    }


def plot_turboquant_prod_simulation(
    dimensions: list[int],
    bit_width: int,
    num_trials: int,
    output_path: Path,
    seed: int = 7,
) -> None:
    num_dimensions = len(dimensions)
    num_cols = 3
    num_rows = (num_dimensions + num_cols - 1) // num_cols

    fig = plt.figure(figsize=(15, 4 * (4 * num_rows)))
    grid = fig.add_gridspec(4 * num_rows, num_cols, height_ratios=[1.0] * (4 * num_rows))

    mse_axes = [
        fig.add_subplot(grid[row, col])
        for row in range(num_rows)
        for col in range(num_cols)
    ]
    per_coordinate_mse_axes = [
        fig.add_subplot(grid[num_rows + row, col])
        for row in range(num_rows)
        for col in range(num_cols)
    ]
    prod_axes = [
        fig.add_subplot(grid[2 * num_rows + row, col])
        for row in range(num_rows)
        for col in range(num_cols)
    ]
    cosine_axes = [
        fig.add_subplot(grid[3 * num_rows + row, col])
        for row in range(num_rows)
        for col in range(num_cols)
    ]

    for idx, dimension in enumerate(dimensions):
        result = run_turboquant_prod_trials(
            dimension=dimension,
            bit_width=bit_width,
            num_trials=num_trials,
            seed=seed + dimension,
        )

        mse_ax = mse_axes[idx]
        mse_ax.hist(
            result["d_mse_values"],
            bins=80,
            density=True,
            alpha=0.45,
            color="#4C78A8",
        )
        mse_ax.axvline(
            x=float(result["mean_d_mse"]),
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label=f"mean = {result['mean_d_mse']:.4f}",
        )
        mse_ax.set_title(f"$D_{{mse}}$ distribution (d = {dimension})")
        mse_ax.set_xlabel(r"$\|x - \tilde{x}\|_2^2$")
        mse_ax.set_ylabel("Density")
        mse_ax.grid(alpha=0.2)
        mse_ax.legend(frameon=False)

        per_coordinate_mse_ax = per_coordinate_mse_axes[idx]
        per_coordinate_mse_ax.hist(
            result["per_coordinate_mse_values"],
            bins=80,
            density=True,
            alpha=0.45,
            color="#72B7B2",
        )
        per_coordinate_mse_ax.axvline(
            x=float(result["mean_per_coordinate_mse"]),
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label=f"mean = {result['mean_per_coordinate_mse']:.6f}",
        )
        per_coordinate_mse_ax.set_title(f"Per-coordinate MSE distribution (d = {dimension})")
        per_coordinate_mse_ax.set_xlabel(r"$\frac{1}{d}\|x - \tilde{x}\|_2^2$")
        per_coordinate_mse_ax.set_ylabel("Density")
        per_coordinate_mse_ax.grid(alpha=0.2)
        per_coordinate_mse_ax.legend(frameon=False)

        prod_ax = prod_axes[idx]
        prod_ax.hist(
            result["d_prod_values"],
            bins=80,
            density=True,
            alpha=0.45,
            color="#F58518",
        )
        prod_ax.axvline(
            x=float(result["mean_d_prod"]),
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label=f"mean = {result['mean_d_prod']:.4f}",
        )
        prod_ax.axvline(
            x=float(result["d_prod_lower_bound"]),
            color="#72B7B2",
            linestyle="--",
            linewidth=1.8,
            label=f"lower bound = {result['d_prod_lower_bound']:.4f}",
        )
        prod_ax.axvline(
            x=float(result["d_prod_upper_bound"]),
            color="#4C78A8",
            linestyle="--",
            linewidth=1.8,
            label=f"upper bound = {result['d_prod_upper_bound']:.4f}",
        )
        prod_ax.set_title(f"$D_{{prod}}$ distribution (d = {dimension})")
        prod_ax.set_xlabel(r"$\left(\langle y, x \rangle - \langle y, \tilde{x} \rangle\right)^2$")
        prod_ax.set_ylabel("Density")
        prod_ax.grid(alpha=0.2)
        prod_ax.legend(frameon=False)

        cosine_ax = cosine_axes[idx]
        cosine_ax.hist(
            result["cosine_values"],
            bins=80,
            density=True,
            alpha=0.45,
            color="#54A24B",
        )
        cosine_ax.axvline(
            x=float(result["mean_cosine"]),
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label=f"mean = {result['mean_cosine']:.4f}",
        )
        cosine_ax.set_title(f"Cosine similarity distribution (d = {dimension})")
        cosine_ax.set_xlabel(r"$\cos(x, \tilde{x})$")
        cosine_ax.set_ylabel("Density")
        cosine_ax.grid(alpha=0.2)
        cosine_ax.legend(frameon=False)

    for ax in mse_axes[len(dimensions) :]:
        ax.axis("off")

    for ax in per_coordinate_mse_axes[len(dimensions) :]:
        ax.axis("off")

    for ax in prod_axes[len(dimensions) :]:
        ax.axis("off")

    for ax in cosine_axes[len(dimensions) :]:
        ax.axis("off")

    fig.suptitle(
        f"TurboQuant_prod Simulation Across Dimensions (bit width = {bit_width})",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dimensions = [16, 256, 1024]
    bit_width = 3
    num_trials = 10_000
    output_path = Path("results/9-1.TurboQuant_prod_simulation_fix.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_turboquant_prod_simulation(
        dimensions=dimensions,
        bit_width=bit_width,
        num_trials=num_trials,
        output_path=output_path,
        seed=7,
    )

    print(f"Saved plot to: {output_path.resolve()}")
    print(f"Dimensions: {dimensions}")
    print(f"Bit width: {bit_width}")
    print(f"Trials per dimension: {num_trials}")
    print("Visualizes:")
    print("1. D_mse distribution after TurboQuant_prod reconstruction")
    print("2. Per-coordinate MSE distribution")
    print("3. D_prod distribution with lower and upper bounds")
    print("4. Cosine similarity distribution between x and reconstructed x")


if __name__ == "__main__":
    main()
