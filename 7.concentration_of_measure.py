from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sample_sphere_points(
    dimension: int,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points uniformly from the unit sphere S^{d-1}."""
    gaussian = rng.normal(size=(num_samples, dimension))
    return gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)


def estimate_coordinate_concentration(
    points: np.ndarray,
    epsilons: np.ndarray,
) -> np.ndarray:
    """Estimate P(|x_1| <= epsilon) for a list of epsilon values."""
    first_coordinate = np.abs(points[:, 0])
    return np.array([(first_coordinate <= eps).mean() for eps in epsilons], dtype=np.float64)


def plot_concentration_of_measure(
    dimensions: list[int],
    num_samples: int,
    output_path: Path,
    seed: int = 7,
) -> None:
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    histogram_ax = fig.add_subplot(grid[0, :])
    concentration_ax = fig.add_subplot(grid[1, :])

    epsilons = np.linspace(0.0, 0.35, 120)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    for color, dimension in zip(colors, dimensions):
        points = sample_sphere_points(dimension=dimension, num_samples=num_samples, rng=rng)
        first_coordinate = points[:, 0]
        concentration = estimate_coordinate_concentration(points=points, epsilons=epsilons)

        histogram_ax.hist(
            first_coordinate,
            bins=100,
            range=(-0.5, 0.5),
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"d = {dimension}",
        )
        concentration_ax.plot(
            epsilons,
            concentration,
            linewidth=2.0,
            color=color,
            label=f"d = {dimension}",
        )

    histogram_ax.set_title("Coordinate distribution on the unit sphere")
    histogram_ax.set_xlabel(r"First coordinate $x_1$")
    histogram_ax.set_ylabel("Density")
    histogram_ax.grid(alpha=0.2)
    histogram_ax.legend(frameon=False)

    concentration_ax.set_title(r"Concentration near zero: $P(|x_1| \leq \varepsilon)$")
    concentration_ax.set_xlabel(r"$\varepsilon$")
    concentration_ax.set_ylabel("Probability")
    concentration_ax.grid(alpha=0.2)
    concentration_ax.legend(frameon=False)

    fig.suptitle("Concentration of Measure on High-Dimensional Spheres", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dimensions = [3, 10, 100, 1000]
    num_samples = 50_000
    output_path = Path("results/7.concentration_of_measure.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_concentration_of_measure(
        dimensions=dimensions,
        num_samples=num_samples,
        output_path=output_path,
        seed=7,
    )

    print(f"Saved plot to: {output_path.resolve()}")
    print(f"Dimensions: {dimensions}")
    print(f"Samples per dimension: {num_samples}")
    print("Phenomenon: as dimension increases, a coordinate of a random unit vector concentrates near 0.")


if __name__ == "__main__":
    main()
