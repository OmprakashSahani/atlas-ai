from pathlib import Path

import matplotlib.pyplot as plt

from atlas_ai.distributed.distributed_simulator import (
    DistributedTrainingSimulator,
)


OUTPUT_DIR = Path("benchmarks")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    simulator = DistributedTrainingSimulator()

    workers = []
    efficiencies = []
    communication_ratios = []

    for num_workers in [1, 2, 4, 8, 16, 32]:
        results = simulator.simulate_step(
            num_workers=num_workers,
            compute_time_sec=0.8,
            gradient_size_mb=200,
            bandwidth_gbps=10,
        )

        workers.append(num_workers)
        efficiencies.append(results["scaling_efficiency"])
        communication_ratios.append(
            results["communication_ratio"]
        )

        print(results)

    plt.figure(figsize=(8, 5))

    plt.plot(
        workers,
        efficiencies,
        marker="o",
        label="Scaling Efficiency",
    )

    plt.plot(
        workers,
        communication_ratios,
        marker="s",
        label="Communication Ratio",
    )

    plt.xlabel("Number of Workers")
    plt.ylabel("Value")
    plt.title("Distributed Training Scaling Analysis")

    plt.legend()

    output_path = (
        OUTPUT_DIR / "distributed_scaling_analysis.png"
    )

    plt.savefig(output_path)

    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()
