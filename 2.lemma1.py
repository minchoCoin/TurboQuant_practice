from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sample_uniform_sphere_coordinates(
    dimension: int,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample the first coordinate of points drawn uniformly from S^{d-1}."""
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(num_samples, dimension))
    normalized = gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)
    return normalized[:, 0]


def lemma1_density(x: np.ndarray, dimension: int) -> np.ndarray:
    """Exact coordinate density from Lemma 1."""
    coefficient = math.gamma(dimension / 2) / (
        math.sqrt(math.pi) * math.gamma((dimension - 1) / 2)
    )
    return coefficient * np.maximum(1.0 - x**2, 0.0) ** ((dimension - 3) / 2)


def normal_approx_density(x: np.ndarray, dimension: int) -> np.ndarray:
    """High-dimensional Gaussian approximation N(0, 1/d)."""
    variance = 1.0 / dimension
    coefficient = 1.0 / math.sqrt(2 * math.pi * variance)
    return coefficient * np.exp(-(x**2) / (2 * variance))


def plot_coordinate_distributions(
    dimensions: list[int],
    num_samples: int,
    output_path: Path,
) -> None:
    x = np.linspace(-1.0, 1.0, 800)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for ax, dimension in zip(axes, dimensions):
        samples = sample_uniform_sphere_coordinates(
            dimension=dimension,
            num_samples=num_samples,
            seed=dimension,
        )

        ax.hist(
            samples,
            bins=80,
            range=(-1, 1),
            density=True,
            alpha=0.35,
            color="#4C78A8",
            label="Empirical distribution",
        )
        ax.plot(
            x,
            lemma1_density(x, dimension),
            color="#E45756",
            linewidth=2.0,
            label="Lemma 1 density",
        )
        ax.plot(
            x,
            normal_approx_density(x, dimension),
            color="#54A24B",
            linewidth=2.0,
            linestyle="--",
            label=r"Normal approximation $N(0, 1/d)$",
        )

        ax.set_title(f"Dimension d = {dimension}")
        ax.set_xlim(-1, 1)
        ax.set_xlabel(r"Coordinate value $x_j$")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Coordinate Distribution on the Unit Sphere", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dimensions = [3, 10, 50,100]
    num_samples = 100_000
    output_path = Path("results/2.lemma1_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_coordinate_distributions(
        dimensions=dimensions,
        num_samples=num_samples,
        output_path=output_path,
    )

    print(f"Saved plot to: {output_path.resolve()}")
    print(f"Dimensions: {dimensions}")
    print(f"Samples per dimension: {num_samples}")


if __name__ == "__main__":
    main()
