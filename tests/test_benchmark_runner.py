from atlas_ai.benchmark.benchmark_runner import BenchmarkRunner


def dummy_workload():
    total = 0

    for i in range(1000):
        total += i

    return total


def test_benchmark_runner():
    runner = BenchmarkRunner()

    results = runner.run(dummy_workload, num_runs=5)

    assert "average_latency_sec" in results
    assert "throughput_ops_per_sec" in results
    assert results["num_runs"] == 5
