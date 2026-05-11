import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

from atlas_ai.transformer.generation import (
    SequenceGenerator,
)


OUTPUT_DIR = Path("benchmarks")
OUTPUT_DIR.mkdir(exist_ok=True)


def benchmark_generation():

    generator = (
        SequenceGenerator()
    )

    latencies = []

    token_counts = [
        1,
        2,
        4,
        8,
        16,
    ]

    throughputs = []

    for num_tokens in token_counts:

        start = time.perf_counter()

        generator.generate(
            prompt_tokens=[1, 2, 3],
            max_new_tokens=num_tokens,
        )

        end = time.perf_counter()

        latency = end - start

        throughput = (
            num_tokens / latency
        )

        latencies.append(latency)

        throughputs.append(
            throughput
        )

        print(
            f"tokens={num_tokens} "
            f"latency={latency:.6f} "
            f"throughput={throughput:.2f}"
        )

    return (
        token_counts,
        latencies,
        throughputs,
    )


def plot_results(
    token_counts,
    latencies,
    throughputs,
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        token_counts,
        latencies,
        marker="o",
        label="Latency",
    )

    plt.plot(
        token_counts,
        throughputs,
        marker="s",
        label="Throughput",
    )

    plt.xlabel(
        "Generated Tokens"
    )

    plt.ylabel("Value")

    plt.title(
        "Transformer Generation Benchmark"
    )

    plt.legend()

    output_path = (
        OUTPUT_DIR
        / "transformer_generation_benchmark.png"
    )

    plt.savefig(output_path)

    print(
        f"\nPlot saved to: "
        f"{output_path}"
    )


def main():

    (
        token_counts,
        latencies,
        throughputs,
    ) = benchmark_generation()

    print(
        "\nAverage Latency:",
        round(
            statistics.mean(
                latencies
            ),
            6,
        ),
    )

    plot_results(
        token_counts,
        latencies,
        throughputs,
    )


if __name__ == "__main__":
    main()
