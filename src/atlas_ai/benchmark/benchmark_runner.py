import statistics
import time


class BenchmarkRunner:
    """Simple benchmarking utility."""

    def run(self, func, num_runs: int = 10) -> dict:
        """
        Benchmark a function across multiple runs.
        """

        execution_times = []

        for _ in range(num_runs):
            start = time.perf_counter()

            func()

            end = time.perf_counter()

            execution_times.append(end - start)

        average_time = statistics.mean(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)

        throughput = 1 / average_time if average_time > 0 else 0

        return {
            "num_runs": num_runs,
            "average_latency_sec": round(average_time, 6),
            "min_latency_sec": round(min_time, 6),
            "max_latency_sec": round(max_time, 6),
            "throughput_ops_per_sec": round(throughput, 2),
        }
