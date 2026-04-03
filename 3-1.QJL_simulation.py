from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
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


def sample_unit_vector(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random unit vector from S^{d-1}."""
    vector = rng.normal(size=dimension)
    return vector / np.linalg.norm(vector)


def sample_gaussian_vector(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a general Gaussian vector in R^d without normalization."""
    return rng.normal(size=dimension)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def run_qjl_trials(
    dimension: int,
    num_trials: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Run repeated QJL experiments for one dimension."""
    rng = np.random.default_rng(seed)
    y = sample_gaussian_vector(dimension, rng)
    inner_product_estimates = np.zeros(num_trials, dtype=np.float64)
    true_inner_products = np.zeros(num_trials, dtype=np.float64)
    squared_errors = np.zeros(num_trials, dtype=np.float64)
    cosine_similarities = np.zeros(num_trials, dtype=np.float64)

    for trial in range(num_trials):
        x = sample_unit_vector(dimension, rng) 
        s_matrix = make_qjl_matrix(dimension, rng)

        z = qjl_quantize(x, s_matrix)
        x_hat = qjl_dequantize(z, s_matrix)

        true_inner_product = float(np.dot(y, x))
        estimated_inner_product = float(np.dot(y, x_hat))
        squared_error = (estimated_inner_product - true_inner_product) ** 2

        inner_product_estimates[trial] = estimated_inner_product
        true_inner_products[trial] = true_inner_product
        squared_errors[trial] = squared_error
        cosine_similarities[trial] = cosine_similarity(x, x_hat)

    estimator_errors = inner_product_estimates - true_inner_products
    empirical_variance = float(np.var(estimator_errors))
    variance_bound = (math.pi / (2.0 * dimension)) * float(np.linalg.norm(y) ** 2)

    return {
        "true_inner_products": true_inner_products,
        "inner_product_estimates": inner_product_estimates,
        "estimator_errors": estimator_errors,
        "squared_errors": squared_errors,
        "cosine_similarities": cosine_similarities,
        "empirical_variance": empirical_variance,
        "variance_bound": variance_bound,
    }


def plot_qjl_simulation(
    dimensions: list[int],
    num_trials: int,
    output_path: Path,
    seed: int = 7,
) -> None:
    num_dimensions = len(dimensions)
    num_cols = 2
    num_hist_rows = math.ceil(num_dimensions / num_cols)

    fig = plt.figure(figsize=(15, 4 * (2 * num_hist_rows + 1)))
    grid = fig.add_gridspec(2 * num_hist_rows + 1, num_cols, height_ratios=[1.0] * (2 * num_hist_rows) + [1.0])

    error_axes = [
        fig.add_subplot(grid[row, col])
        for row in range(num_hist_rows)
        for col in range(num_cols)
    ]
    mse_axes = [
        fig.add_subplot(grid[num_hist_rows + row, col])
        for row in range(num_hist_rows)
        for col in range(num_cols)
    ]
    cosine_ax = fig.add_subplot(grid[2 * num_hist_rows, :])

    for idx, dimension in enumerate(dimensions):
        result = run_qjl_trials(
            dimension=dimension,
            num_trials=num_trials,
            seed=seed + dimension,
        )

        error_ax = error_axes[idx]
        error_ax.hist(
            result["estimator_errors"],
            bins=80,
            density=True,
            alpha=0.45,
            color="#4C78A8",
        )
        empirical_std = math.sqrt(float(result["empirical_variance"]))
        std_bound = math.sqrt(float(result["variance_bound"]))
        error_ax.axvline(
            x=empirical_std,
            color="#4C78A8",
            linestyle="--",
            linewidth=1.8,
            label=r"$+\sqrt{\mathrm{empirical\ variance}}$",
        )
        error_ax.axvline(
            x=-empirical_std,
            color="#4C78A8",
            linestyle="--",
            linewidth=1.8,
            label=r"$-\sqrt{\mathrm{empirical\ variance}}$",
        )
        error_ax.axvline(
            x=std_bound,
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label=r"$+\sqrt{\mathrm{Var}\ bound}$",
        )
        error_ax.axvline(
            x=-std_bound,
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label=r"$-\sqrt{\mathrm{Var}\ bound}$",
        )
        error_ax.set_title(f"Inner product error distribution (d = {dimension})")
        error_ax.set_xlabel(
            r"$\langle y, Q_{\mathrm{qjl}}^{-1}(Q_{\mathrm{qjl}}(x)) \rangle - \langle y, x \rangle$"
        )
        error_ax.set_ylabel("Density")
        error_ax.grid(alpha=0.2)
        error_ax.legend(frameon=False)

        mse_ax = mse_axes[idx]
        mse_ax.hist(
            result["squared_errors"],
            bins=80,
            density=True,
            alpha=0.45,
            color="#E45756",
        )
        mse_ax.set_title(f"Squared error distribution (d = {dimension})")
        mse_ax.set_xlabel(
            r"$\left(\langle y, Q_{\mathrm{qjl}}^{-1}(Q_{\mathrm{qjl}}(x)) \rangle - \langle y, x \rangle\right)^2$"
        )
        mse_ax.set_ylabel("Density")
        mse_ax.grid(alpha=0.2)

        cosine_ax.hist(
            result["cosine_similarities"],
            bins=80,
            density=True,
            alpha=0.35,
            label=f"d = {dimension}",
        )

    for ax in error_axes[len(dimensions) :]:
        ax.axis("off")

    for ax in mse_axes[len(dimensions) :]:
        ax.axis("off")

    cosine_ax.set_title("Cosine similarity distribution between x and QJL reconstruction")
    cosine_ax.set_xlabel(r"$\cos(x, Q_{\mathrm{qjl}}^{-1}(Q_{\mathrm{qjl}}(x)))$")
    cosine_ax.set_ylabel("Density")
    cosine_ax.grid(alpha=0.2)
    cosine_ax.legend(frameon=False)

    fig.suptitle("QJL Simulation Across Dimensions", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dimensions = [16, 32, 64, 256]
    num_trials = 20_000
    output_path = Path("results/3-1.QJL_simulation.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_qjl_simulation(
        dimensions=dimensions,
        num_trials=num_trials,
        output_path=output_path,
        seed=7,
    )

    print(f"Saved plot to: {output_path.resolve()}")
    print(f"Dimensions: {dimensions}")
    print(f"Trials per dimension: {num_trials}")
    print("Visualizes:")
    print("1. Inner product error distribution")
    print("2. Squared error distribution")
    print("3. Cosine similarity distribution")


if __name__ == "__main__":
    main()
