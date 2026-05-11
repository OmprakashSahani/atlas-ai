from atlas_ai.benchmark.benchmark_store import (
    BenchmarkStore,
)


def test_benchmark_storage():

    store = BenchmarkStore()

    path = store.save_benchmark(
        benchmark_name="test_benchmark",
        metrics={
            "latency": 0.1,
            "throughput": 100,
        },
    )

    assert path.endswith(".json")


def test_benchmark_listing():

    store = BenchmarkStore()

    benchmarks = (
        store.list_benchmarks()
    )

    assert isinstance(
        benchmarks,
        list,
    )
