from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class QuantizationResult:
    codebook_size: int
    codebook: np.ndarray
    image: np.ndarray
    mse: float


def make_simple_image(height: int = 160, width: int = 160) -> np.ndarray:
    """Create a simple image with a small number of flat colors."""
    image = np.ones((height, width, 3), dtype=np.float32)
    image[:] = np.array([0.96, 0.95, 0.90], dtype=np.float32)

    image[10:70, 10:150] = np.array([0.25, 0.55, 0.90], dtype=np.float32)
    image[90:150, 20:70] = np.array([0.92, 0.35, 0.30], dtype=np.float32)
    image[90:150, 90:145] = np.array([0.25, 0.75, 0.42], dtype=np.float32)

    yy, xx = np.ogrid[:height, :width]
    center_y, center_x = 90, 112
    radius = 18
    mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
    image[mask] = np.array([0.99, 0.84, 0.18], dtype=np.float32)

    return image


def initialize_codebook(pixels: np.ndarray, codebook_size: int, rng: np.random.Generator) -> np.ndarray:
    if codebook_size >= len(pixels):
        return pixels.copy()

    indices = rng.choice(len(pixels), size=codebook_size, replace=False)
    return pixels[indices].copy()

'''
run_kmeans is the core algorithm: 
1. calculate distances from each pixel to each codebook color
2. assign each pixel to the nearest codebook color
3. update each codebook color to be the mean of its assigned pixels
4. repeat until convergence or max iterations
'''
def run_kmeans(
    pixels: np.ndarray,
    codebook_size: int,
    max_iters: int = 25,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    codebook = initialize_codebook(pixels, codebook_size, rng)

    for _ in range(max_iters):
        distances = np.sum((pixels[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)

        updated = codebook.copy()
        for idx in range(codebook_size):
            members = pixels[assignments == idx]
            if len(members) == 0:
                updated[idx] = pixels[rng.integers(0, len(pixels))]
            else:
                updated[idx] = members.mean(axis=0)

        if np.allclose(updated, codebook, atol=1e-5):
            codebook = updated
            break
        codebook = updated

    distances = np.sum((pixels[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    assignments = np.argmin(distances, axis=1)
    return codebook, assignments


def quantize_image(image: np.ndarray, codebook_size: int) -> QuantizationResult:
    pixels = image.reshape(-1, 3)
    codebook, assignments = run_kmeans(pixels, codebook_size=codebook_size)
    quantized = codebook[assignments].reshape(image.shape)
    mse = float(np.mean((image - quantized) ** 2))
    return QuantizationResult(codebook_size=codebook_size, codebook=codebook, image=quantized, mse=mse)


def sort_codebook(codebook: np.ndarray) -> np.ndarray:
    brightness = np.sum(codebook, axis=1)
    order = np.argsort(brightness)
    return codebook[order]


def plot_palette(ax: plt.Axes, codebook: np.ndarray) -> None:
    sorted_codebook = sort_codebook(codebook)
    palette = sorted_codebook[np.newaxis, :, :]
    ax.imshow(palette, aspect="auto")
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(sorted_codebook)))
    ax.set_xticklabels([str(idx + 1) for idx in range(len(sorted_codebook))], fontsize=8)
    ax.set_xlabel("Codebook colors", fontsize=9)


def plot_results(original: np.ndarray, results: list[QuantizationResult], output_path: Path) -> None:
    num_cols = len(results) + 1
    fig = plt.figure(figsize=(4 * num_cols, 6))
    grid = fig.add_gridspec(2, num_cols, height_ratios=[5, 1.2])

    original_ax = fig.add_subplot(grid[0, 0])
    original_ax.imshow(original)
    original_ax.set_title("Original")
    original_ax.axis("off")

    original_palette_ax = fig.add_subplot(grid[1, 0])
    unique_colors = np.unique(original.reshape(-1, 3), axis=0)
    plot_palette(original_palette_ax, unique_colors)
    original_palette_ax.set_title(f"Original colors ({len(unique_colors)})", fontsize=10)

    for col, result in enumerate(results, start=1):
        image_ax = fig.add_subplot(grid[0, col])
        image_ax.imshow(result.image)
        image_ax.set_title(f"Codebook={result.codebook_size}\nMSE={result.mse:.5f}")
        image_ax.axis("off")

        palette_ax = fig.add_subplot(grid[1, col])
        plot_palette(palette_ax, result.codebook)
        palette_ax.set_title(f"Palette ({len(result.codebook)})", fontsize=10)

    fig.suptitle("Codebook Size Comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    original = make_simple_image()
    codebook_sizes = [1, 2, 3, 4,5]
    results = [quantize_image(original, size) for size in codebook_sizes]

    output_path = Path("results/1.codebook_comparison.png")
    plot_results(original, results, output_path)

    print(f"Saved comparison image to: {output_path.resolve()}")
    for result in results:
        palette_text = np.round(sort_codebook(result.codebook), 3).tolist()
        print(f"Codebook size {result.codebook_size:>2}: MSE={result.mse:.6f}, palette={palette_text}")


if __name__ == "__main__":
    main()
