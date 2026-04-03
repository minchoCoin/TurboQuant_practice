from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TurboQuant_mse_lgamma import TurboQuantMSE
from TurboQuant_prod_lgamma import TurboQuantProd


def sample_unit_vector(dimension: int, rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=dimension)
    return vector / np.linalg.norm(vector)


def run_method_trials(
    method_name: str,
    dimension: int,
    bit_width: int,
    num_trials: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=dimension)

    d_mse_values = np.zeros(num_trials, dtype=np.float64)
    d_prod_values = np.zeros(num_trials, dtype=np.float64)
    prod_error_values = np.zeros(num_trials, dtype=np.float64)

    for trial in range(num_trials):
        if method_name == "mse":
            quantizer = TurboQuantMSE.create(
                dimension=dimension,
                bit_width=bit_width,
                rotation_seed=int(rng.integers(0, 2**31 - 1)),
                codebook_seed=int(rng.integers(0, 2**31 - 1)),
            )
        elif method_name == "prod":
            quantizer = TurboQuantProd.create(
                dimension=dimension,
                bit_width=bit_width,
                rotation_seed=int(rng.integers(0, 2**31 - 1)),
                codebook_seed=int(rng.integers(0, 2**31 - 1)),
                qjl_seed=int(rng.integers(0, 2**31 - 1)),
            )
        else:
            raise ValueError("method_name must be 'mse' or 'prod'.")
        x = sample_unit_vector(dimension, rng)

        if method_name == "mse":
            idx = quantizer.quant(x)
            x_hat = quantizer.dequant(idx)
        else:
            idx, qjl, gamma = quantizer.quant(x)
            x_hat = quantizer.dequant(idx, qjl, gamma)

        difference = x - x_hat
        d_mse_values[trial] = float(np.sum(difference**2))
        prod_error = float(np.dot(y, x) - np.dot(y, x_hat))
        prod_error_values[trial] = prod_error
        d_prod_values[trial] = float(prod_error**2)

    mse_lower_bound = 1.0 / (4.0**bit_width)
    mse_upper_bound = (math.sqrt(3.0) * math.pi / 2.0) * mse_lower_bound
    y_norm_sq = float(np.linalg.norm(y) ** 2)
    prod_lower_bound = y_norm_sq / (dimension * (4.0**bit_width))
    prod_upper_bound = (math.sqrt(3.0) * math.pi**2 * y_norm_sq) / (dimension * (4.0**bit_width))

    return {
        "d_mse_values": d_mse_values,
        "d_prod_values": d_prod_values,
        "prod_error_values": prod_error_values,
        "mean_d_mse": float(np.mean(d_mse_values)),
        "mean_d_prod": float(np.mean(d_prod_values)),
        "mse_lower_bound": mse_lower_bound,
        "mse_upper_bound": mse_upper_bound,
        "prod_lower_bound": prod_lower_bound,
        "prod_upper_bound": prod_upper_bound,
    }


def plot_final_simulation(
    dimension: int,
    bit_widths: list[int],
    num_trials: int,
    output_path: Path,
    seed: int = 7,
) -> None:
    num_cols = len(bit_widths)
    fig = plt.figure(figsize=(4.2 * num_cols, 20))
    grid = fig.add_gridspec(5, num_cols, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0])

    mse_error_axes = [fig.add_subplot(grid[0, col]) for col in range(num_cols)]
    prod_error_axes = [fig.add_subplot(grid[1, col]) for col in range(num_cols)]
    mse_sq_error_axes = [fig.add_subplot(grid[2, col]) for col in range(num_cols)]
    prod_sq_error_axes = [fig.add_subplot(grid[3, col]) for col in range(num_cols)]
    summary_grid = grid[4, :].subgridspec(1, 2, wspace=0.28)
    prod_summary_ax = fig.add_subplot(summary_grid[0, 0])
    mse_summary_ax = fig.add_subplot(summary_grid[0, 1])

    mse_mean_prod_errors = []
    prod_mean_prod_errors = []
    prod_lower_bounds = []
    prod_upper_bounds = []

    mse_mean_mse = []
    prod_mean_mse = []
    mse_lower_bounds = []
    mse_upper_bounds = []

    for idx, bit_width in enumerate(bit_widths):
        mse_result = run_method_trials(
            method_name="mse",
            dimension=dimension,
            bit_width=bit_width,
            num_trials=num_trials,
            seed=seed + 100 * bit_width,
        )
        prod_result = run_method_trials(
            method_name="prod",
            dimension=dimension,
            bit_width=bit_width,
            num_trials=num_trials,
            seed=seed + 200 * bit_width,
        )

        mse_error_ax = mse_error_axes[idx]
        mse_error_ax.hist(
            mse_result["prod_error_values"],
            bins=80,
            density=True,
            alpha=0.45,
            color="#4C78A8",
        )
        mse_error_ax.axvline(
            x=0.0,
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label="zero error",
        )
        mse_error_ax.set_title(f"MSE quantizer error: bit width = {bit_width}")
        mse_error_ax.set_xlabel(r"$\langle y, x \rangle - \langle y, \tilde{x} \rangle$")
        mse_error_ax.set_ylabel("Density")
        mse_error_ax.grid(alpha=0.2)
        mse_error_ax.legend(frameon=False)

        prod_error_ax = prod_error_axes[idx]
        prod_error_ax.hist(
            prod_result["prod_error_values"],
            bins=80,
            density=True,
            alpha=0.45,
            color="#F58518",
        )
        prod_error_ax.axvline(
            x=0.0,
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label="zero error",
        )
        prod_error_ax.set_title(f"Prod quantizer error: bit width = {bit_width}")
        prod_error_ax.set_xlabel(r"$\langle y, x \rangle - \langle y, \tilde{x} \rangle$")
        prod_error_ax.set_ylabel("Density")
        prod_error_ax.grid(alpha=0.2)
        prod_error_ax.legend(frameon=False)

        mse_sq_error_ax = mse_sq_error_axes[idx]
        mse_sq_error_ax.hist(
            mse_result["d_prod_values"],
            bins=80,
            density=True,
            alpha=0.45,
            color="#72B7B2",
        )
        mse_sq_error_ax.axvline(
            x=float(mse_result["mean_d_prod"]),
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label=f"mean = {mse_result['mean_d_prod']:.4f}",
        )
        mse_sq_error_ax.axvline(
            x=float(prod_result["prod_lower_bound"]),
            color="#54A24B",
            linestyle="--",
            linewidth=1.4,
            label=f"lower = {prod_result['prod_lower_bound']:.4f}",
        )
        mse_sq_error_ax.axvline(
            x=float(prod_result["prod_upper_bound"]),
            color="#F58518",
            linestyle="--",
            linewidth=1.4,
            label=f"upper = {prod_result['prod_upper_bound']:.4f}",
        )
        mse_sq_error_ax.set_title(f"MSE quantizer squared error: bit width = {bit_width}")
        mse_sq_error_ax.set_xlabel(r"$\left(\langle y, x \rangle - \langle y, \tilde{x} \rangle\right)^2$")
        mse_sq_error_ax.set_ylabel("Density")
        mse_sq_error_ax.grid(alpha=0.2)
        mse_sq_error_ax.legend(frameon=False)

        prod_sq_error_ax = prod_sq_error_axes[idx]
        prod_sq_error_ax.hist(
            prod_result["d_prod_values"],
            bins=80,
            density=True,
            alpha=0.45,
            color="#B279A2",
        )
        prod_sq_error_ax.axvline(
            x=float(prod_result["mean_d_prod"]),
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
            label=f"mean = {prod_result['mean_d_prod']:.4f}",
        )
        prod_sq_error_ax.axvline(
            x=float(prod_result["prod_lower_bound"]),
            color="#54A24B",
            linestyle="--",
            linewidth=1.4,
            label=f"lower = {prod_result['prod_lower_bound']:.4f}",
        )
        prod_sq_error_ax.axvline(
            x=float(prod_result["prod_upper_bound"]),
            color="#F58518",
            linestyle="--",
            linewidth=1.4,
            label=f"upper = {prod_result['prod_upper_bound']:.4f}",
        )
        prod_sq_error_ax.set_title(f"Prod quantizer squared error: bit width = {bit_width}")
        prod_sq_error_ax.set_xlabel(r"$\left(\langle y, x \rangle - \langle y, \tilde{x} \rangle\right)^2$")
        prod_sq_error_ax.set_ylabel("Density")
        prod_sq_error_ax.grid(alpha=0.2)
        prod_sq_error_ax.legend(frameon=False)

        mse_mean_prod_errors.append(float(mse_result["mean_d_prod"]))
        prod_mean_prod_errors.append(float(prod_result["mean_d_prod"]))
        prod_lower_bounds.append(float(prod_result["prod_lower_bound"]))
        prod_upper_bounds.append(float(prod_result["prod_upper_bound"]))

        mse_mean_mse.append(float(mse_result["mean_d_mse"]))
        prod_mean_mse.append(float(prod_result["mean_d_mse"]))
        mse_lower_bounds.append(float(mse_result["mse_lower_bound"]))
        mse_upper_bounds.append(float(mse_result["mse_upper_bound"]))

    prod_summary_ax.plot(
        bit_widths,
        mse_mean_prod_errors,
        marker="o",
        linewidth=2.0,
        color="#4C78A8",
        label="MSE quantizer mean inner-product error",
    )
    prod_summary_ax.plot(
        bit_widths,
        prod_mean_prod_errors,
        marker="o",
        linewidth=2.0,
        color="#F58518",
        label="Prod quantizer mean inner-product error",
    )
    prod_summary_ax.plot(
        bit_widths,
        prod_lower_bounds,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#72B7B2",
        label="Lower bound",
    )
    prod_summary_ax.plot(
        bit_widths,
        prod_upper_bounds,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#E45756",
        label="Upper bound",
    )
    prod_summary_ax.set_title("Mean inner-product error vs. bit width")
    prod_summary_ax.set_xlabel("Bit width")
    prod_summary_ax.set_ylabel(r"$\mathbb{E}\left[(\langle y, x \rangle - \langle y, \tilde{x} \rangle)^2\right]$")
    prod_summary_ax.grid(alpha=0.2)
    prod_summary_ax.legend(frameon=False)

    mse_summary_ax.plot(
        bit_widths,
        mse_mean_mse,
        marker="o",
        linewidth=2.0,
        color="#4C78A8",
        label="MSE quantizer mean MSE",
    )
    mse_summary_ax.plot(
        bit_widths,
        prod_mean_mse,
        marker="o",
        linewidth=2.0,
        color="#F58518",
        label="Prod quantizer mean MSE",
    )
    mse_summary_ax.plot(
        bit_widths,
        mse_lower_bounds,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#72B7B2",
        label="Lower bound",
    )
    mse_summary_ax.plot(
        bit_widths,
        mse_upper_bounds,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#E45756",
        label="Upper bound",
    )
    mse_summary_ax.set_title("Mean MSE vs. bit width")
    mse_summary_ax.set_xlabel("Bit width")
    mse_summary_ax.set_ylabel(r"$\mathbb{E}\left[\|x - \tilde{x}\|_2^2\right]$")
    mse_summary_ax.grid(alpha=0.2)
    mse_summary_ax.legend(frameon=False)

    fig.suptitle(f"Final TurboQuant Comparison at dimension d = {dimension}", fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dimension = 256
    bit_widths = [1, 2, 3, 4]
    num_trials = 10_000
    output_path = Path("results/10.TurboQuant_final_simulation.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_final_simulation(
        dimension=dimension,
        bit_widths=bit_widths,
        num_trials=num_trials,
        output_path=output_path,
        seed=7,
    )

    print(f"Saved plot to: {output_path.resolve()}")
    print(f"Dimension: {dimension}")
    print(f"Bit widths: {bit_widths}")
    print(f"Trials per bit width and method: {num_trials}")
    print("Visualizes:")
    print("1. Inner-product distortion distributions for MSE and Prod quantizers")
    print("2. Mean inner-product distortion vs. lower and upper bounds")
    print("3. Mean MSE vs. lower and upper bounds")


if __name__ == "__main__":
    main()
