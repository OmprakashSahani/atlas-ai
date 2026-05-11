import json
from datetime import UTC, datetime
from pathlib import Path


BENCHMARK_DIR = Path(
    "benchmark_history"
)


class BenchmarkStore:
    """
    Store historical benchmark results.
    """

    def __init__(self):

        BENCHMARK_DIR.mkdir(
            exist_ok=True
        )

    def save_benchmark(
        self,
        benchmark_name: str,
        metrics: dict,
    ):

        timestamp = (
            datetime.now(UTC)
            .isoformat()
            .replace(":", "-")
        )

        filename = (
            f"{timestamp}_"
            f"{benchmark_name}.json"
        )

        path = (
            BENCHMARK_DIR
            / filename
        )

        benchmark_data = {
            "benchmark_name": (
                benchmark_name
            ),
            "timestamp": timestamp,
            "metrics": metrics,
        }

        with open(path, "w") as f:

            json.dump(
                benchmark_data,
                f,
                indent=4,
            )

        return str(path)

    def list_benchmarks(self):

        benchmark_files = list(
            BENCHMARK_DIR.glob("*.json")
        )

        return sorted(
            [
                file.name
                for file in benchmark_files
            ]
        )
