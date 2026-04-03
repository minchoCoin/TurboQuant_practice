from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def standard_normal_density(x: np.ndarray) -> np.ndarray:
    """Density of N(0, 1)."""
    return np.exp(-(x**2) / 2.0) / math.sqrt(2.0 * math.pi)


def sample_exponential_standardized_means(
    sample_size: int,
    num_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate standardized sample means for an Exponential(1) distribution.

    If X_i ~ Exp(1), then mu = 1 and sigma = 1.
    The CLT says sqrt(n) * (X_bar - mu) / sigma converges to N(0,1).
    """
    samples = rng.exponential(scale=1.0, size=(num_trials, sample_size))
    sample_means = samples.mean(axis=1)
    return np.sqrt(sample_size) * (sample_means - 1.0)


def sample_exponential_means(
    sample_size: int,
    num_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate sample means for an Exponential(1) distribution."""
    samples = rng.exponential(scale=1.0, size=(num_trials, sample_size))
    return samples.mean(axis=1)


def plot_clt_simulation(
    sample_sizes: list[int],
    num_trials: int,
    output_path: Path,
    seed: int = 7,
) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(-4.0, 4.0, 800)

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.9])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    empirical_variances = []

    for ax, sample_size in zip(axes, sample_sizes):
        standardized_means = sample_exponential_standardized_means(
            sample_size=sample_size,
            num_trials=num_trials,
            rng=rng,
        )
        sample_means = sample_exponential_means(
            sample_size=sample_size,
            num_trials=num_trials,
            rng=rng,
        )
        empirical_variances.append(float(np.var(sample_means)))

        ax.hist(
            standardized_means,
            bins=80,
            density=True,
            alpha=0.4,
            color="#4C78A8",
            label="Empirical histogram",
        )
        ax.plot(
            x,
            standard_normal_density(x),
            color="#E45756",
            linewidth=2.0,
            label="Standard normal density",
        )
        ax.set_title(f"Sample size n = {sample_size}")
        ax.set_xlabel(r"$\sqrt{n}(\bar{X}_n - \mu)/\sigma$")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

    variance_ax = fig.add_subplot(grid[2, :])
    theoretical_variances = [1.0 / sample_size for sample_size in sample_sizes]
    variance_ax.plot(
        sample_sizes,
        empirical_variances,
        marker="o",
        linewidth=2.0,
        color="#4C78A8",
        label=r"Empirical Var($\bar{X}_n$)",
    )
    variance_ax.plot(
        sample_sizes,
        theoretical_variances,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#E45756",
        label=r"Theoretical Var($\bar{X}_n$) = \sigma^2/n",
    )
    variance_ax.set_title("Variance of the sample mean decreases with sample size")
    variance_ax.set_xlabel("Sample size n")
    variance_ax.set_ylabel("Variance")
    variance_ax.grid(alpha=0.2)
    variance_ax.legend(frameon=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Central Limit Theorem Simulation", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sample_sizes = [1, 2, 5, 30]
    num_trials = 50_000
    output_path = Path("results/6.central_limit_theorem.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_clt_simulation(
        sample_sizes=sample_sizes,
        num_trials=num_trials,
        output_path=output_path,
        seed=7,
    )

    print(f"Saved plot to: {output_path.resolve()}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Trials per sample size: {num_trials}")
    print("Base distribution: Exponential(1)")
    print("Standardized quantity: sqrt(n) * (X_bar - 1)")
    print("Also plots how Var(X_bar) decreases approximately like 1/n.")


if __name__ == "__main__":
    main()
