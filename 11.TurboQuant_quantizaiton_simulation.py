from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TurboQuant_mse_lgamma import TurboQuantMSE


def sample_unit_vector(dimension: int, rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=dimension)
    return vector / np.linalg.norm(vector)


def sample_unit_matrix(num_vectors: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(num_vectors, dimension))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    m = 0.5 * (p_safe + q_safe)
    kl_pm = np.sum(p_safe * np.log(p_safe / m))
    kl_qm = np.sum(q_safe * np.log(q_safe / m))
    return float(0.5 * (kl_pm + kl_qm))


def quantized_dot_from_indices(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    codebook_product_table: np.ndarray,
) -> float:
    # No dequantization to original space: index pair -> centroid product lookup.
    return float(np.sum(codebook_product_table[idx_a, idx_b]))


def qk_logits_with_quantized_k(
    turboquant: TurboQuantMSE,
    query: np.ndarray,
    keys: np.ndarray,
) -> np.ndarray:
    query_rot = turboquant.rotation @ query
    scale = np.sqrt(turboquant.dimension)
    logits = np.empty(keys.shape[0], dtype=np.float64)

    for key_idx, key in enumerate(keys):
        key_indices = turboquant.quant(key)
        key_rot_quantized = turboquant.codebook[key_indices]
        logits[key_idx] = float(np.dot(query_rot, key_rot_quantized) / scale)

    return logits


def run_simulation(
    dimension: int = 256,
    bit_widths: list[int] | None = None,
    num_pair_samples: int = 3000,
    num_softmax_trials: int = 200,
    num_keys: int = 256,
    seed: int = 7,
) -> dict:
    if bit_widths is None:
        bit_widths = [2, 3, 4]

    rng = np.random.default_rng(seed)
    summary = {}

    for bit_width in bit_widths:
        turboquant = TurboQuantMSE.create(
            dimension=dimension,
            bit_width=bit_width,
            rotation_seed=seed + 31 * bit_width,
            codebook_seed=seed + 67 * bit_width,
        )
        codebook_product_table = np.outer(turboquant.codebook, turboquant.codebook)

        true_sims = np.empty(num_pair_samples, dtype=np.float64)
        quantized_only_sims = np.empty(num_pair_samples, dtype=np.float64)

        for i in range(num_pair_samples):
            x = sample_unit_vector(dimension, rng)
            y = sample_unit_vector(dimension, rng)
            idx_x = turboquant.quant(x)
            idx_y = turboquant.quant(y)

            true_sims[i] = float(np.dot(x, y))
            quantized_only_sims[i] = quantized_dot_from_indices(idx_x, idx_y, codebook_product_table)

        sim_errors = quantized_only_sims - true_sims
        corr = float(np.corrcoef(true_sims, quantized_only_sims)[0, 1])
        sim_mse = float(np.mean(sim_errors**2))

        softmax_js_values = np.empty(num_softmax_trials, dtype=np.float64)
        example_probs_true = None
        example_probs_quant = None

        for trial in range(num_softmax_trials):
            query = sample_unit_vector(dimension, rng)
            keys = sample_unit_matrix(num_keys, dimension, rng)

            logits_true = (keys @ query) / np.sqrt(dimension)
            logits_quant = qk_logits_with_quantized_k(turboquant, query, keys)

            probs_true = softmax(logits_true)
            probs_quant = softmax(logits_quant)

            softmax_js_values[trial] = js_divergence(probs_true, probs_quant)

            if trial == 0:
                # Sort for visual comparison.
                order = np.argsort(probs_true)[::-1]
                example_probs_true = probs_true[order]
                example_probs_quant = probs_quant[order]

        summary[bit_width] = {
            "true_sims": true_sims,
            "quantized_only_sims": quantized_only_sims,
            "sim_errors": sim_errors,
            "sim_corr": corr,
            "sim_mse": sim_mse,
            "softmax_js_values": softmax_js_values,
            "softmax_js_mean": float(np.mean(softmax_js_values)),
            "softmax_js_std": float(np.std(softmax_js_values)),
            "example_probs_true": example_probs_true,
            "example_probs_quant": example_probs_quant,
        }

    return {
        "dimension": dimension,
        "bit_widths": bit_widths,
        "summary": summary,
        "num_pair_samples": num_pair_samples,
        "num_softmax_trials": num_softmax_trials,
        "num_keys": num_keys,
    }


def plot_results(result: dict, output_path: Path) -> None:
    bit_widths = result["bit_widths"]
    summary = result["summary"]
    best_bit = max(bit_widths)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Similarity without dequantization: scatter.
    ax = axes[0, 0]
    true_sims = summary[best_bit]["true_sims"]
    quantized_only_sims = summary[best_bit]["quantized_only_sims"]
    ax.scatter(true_sims, quantized_only_sims, s=8, alpha=0.25, color="#4C78A8")
    min_val = float(min(np.min(true_sims), np.min(quantized_only_sims)))
    max_val = float(max(np.max(true_sims), np.max(quantized_only_sims)))
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="#E45756", linewidth=1.6)
    ax.set_title(
        f"Similarity From Quantized-Only Vectors (b={best_bit})\n"
        f"corr={summary[best_bit]['sim_corr']:.4f}, mse={summary[best_bit]['sim_mse']:.6f}"
    )
    ax.set_xlabel("True similarity: <x, y>")
    ax.set_ylabel("Quantized-only similarity")
    ax.grid(alpha=0.2)

    # (2) Similarity error histogram by bit width.
    ax = axes[0, 1]
    colors = {2: "#4C78A8", 3: "#F58518", 4: "#54A24B"}
    for bit_width in bit_widths:
        ax.hist(
            summary[bit_width]["sim_errors"],
            bins=70,
            density=True,
            alpha=0.35,
            color=colors.get(bit_width, None),
            label=f"b={bit_width}",
        )
    ax.axvline(0.0, linestyle="--", color="#E45756", linewidth=1.6)
    ax.set_title("Similarity Error Distribution (quantized-only - true)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    # (3) Softmax distribution example: Q float, K quantized.
    ax = axes[1, 0]
    x_axis = np.arange(result["num_keys"])
    ax.plot(
        x_axis,
        summary[best_bit]["example_probs_true"],
        color="#4C78A8",
        linewidth=2.0,
        label="Full precision softmax",
    )
    ax.plot(
        x_axis,
        summary[best_bit]["example_probs_quant"],
        color="#F58518",
        linewidth=1.8,
        label=f"Q float, K quantized softmax (b={best_bit})",
    )
    ax.set_title("Softmax Distribution Comparison (single query example)")
    ax.set_xlabel("Key index (sorted by full precision probability)")
    ax.set_ylabel("Probability")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    # (4) JS divergence summary vs bit width.
    ax = axes[1, 1]
    js_means = [summary[b]["softmax_js_mean"] for b in bit_widths]
    js_stds = [summary[b]["softmax_js_std"] for b in bit_widths]
    ax.errorbar(
        bit_widths,
        js_means,
        yerr=js_stds,
        marker="o",
        linewidth=2.0,
        color="#72B7B2",
        capsize=4,
    )
    ax.set_title("Softmax Distribution Gap (JS divergence)")
    ax.set_xlabel("Bit width")
    ax.set_ylabel("JS divergence (lower is more similar)")
    ax.grid(alpha=0.2)

    fig.suptitle(
        f"TurboQuant Quantization Simulation (d={result['dimension']})",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_path = Path("results/11.TurboQuant_quantizaiton_simulation.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_simulation(
        dimension=256,
        bit_widths=[2, 3, 4],
        num_pair_samples=3000,
        num_softmax_trials=200,
        num_keys=256,
        seed=7,
    )
    plot_results(result, output_path)

    print(f"Saved plot to: {output_path.resolve()}")
    print(f"Dimension: {result['dimension']}")
    print(f"Bit widths: {result['bit_widths']}")
    print(f"Pair samples per bit width: {result['num_pair_samples']}")
    print(f"Softmax trials per bit width: {result['num_softmax_trials']}")
    print("Shows:")
    print("1. Similarity estimation from quantized vectors only (no dequantization).")
    print("2. Error distribution of quantized-only similarity.")
    print("3. Softmax distribution similarity for Q(float) x K(quantized).")
    print("4. JS-divergence trend of softmax distributions vs bit width.")


if __name__ == "__main__":
    main()
