from pathlib import Path

import matplotlib.pyplot as plt

from atlas_ai.distributed.runtime.communication import (
    CommunicationProfiler,
)
from atlas_ai.distributed.runtime.distributed_trainer import (
    DistributedTrainer,
)


OUTPUT_DIR = Path("benchmarks")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():

    profiler = (
        CommunicationProfiler()
    )

    worker_counts = [
        1,
        2,
        4,
        8,
    ]

    runtimes = []

    communication_times = []

    efficiencies = []

    baseline_runtime = None

    for workers in worker_counts:

        trainer = DistributedTrainer(
            num_workers=workers
        )

        results = trainer.run()

        runtime = results[
            "total_runtime_sec"
        ]

        if baseline_runtime is None:
            baseline_runtime = runtime

        communication_time = (
            profiler.estimate_all_reduce_time(
                gradient_size_mb=200,
                bandwidth_gbps=10,
                num_workers=workers,
            )
        )

        efficiency = (
            baseline_runtime
            / (
                workers
                * runtime
            )
        )

        runtimes.append(runtime)

        communication_times.append(
            communication_time
        )

        efficiencies.append(
            efficiency
        )

        print(
            f"workers={workers} "
            f"runtime={runtime} "
            f"comm={communication_time} "
            f"efficiency={efficiency:.4f}"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        worker_counts,
        runtimes,
        marker="o",
        label="Runtime",
    )

    plt.plot(
        worker_counts,
        communication_times,
        marker="s",
        label="Communication Time",
    )

    plt.plot(
        worker_counts,
        efficiencies,
        marker="^",
        label="Scaling Efficiency",
    )

    plt.xlabel("Number of Workers")

    plt.ylabel("Value")

    plt.title(
        "Distributed Scaling Benchmark"
    )

    plt.legend()

    output_path = (
        OUTPUT_DIR
        / "distributed_runtime_scaling.png"
    )

    plt.savefig(output_path)

    print(
        f"\nPlot saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
