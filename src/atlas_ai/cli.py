import argparse

from atlas_ai.benchmark.benchmark_runner import BenchmarkRunner
from atlas_ai.distributed.distributed_simulator import (
    DistributedTrainingSimulator,
)
from atlas_ai.profiler.system_profiler import get_system_profile
from atlas_ai.tracker.experiment_tracker import ExperimentTracker


def show_system_profile() -> None:
    """Display system profile."""

    profile = get_system_profile()

    print("Atlas AI — System Profile")
    print("-" * 32)

    for key, value in profile.items():
        print(f"{key}: {value}")


def create_experiment_run(name: str) -> None:
    """Create a new experiment run."""

    tracker = ExperimentTracker()

    run_id = tracker.create_run(
        name=name,
        config={"framework": "atlas-ai"},
    )

    print("Experiment created successfully")
    print(f"Run ID: {run_id}")


def run_benchmark() -> None:
    """Run a sample benchmark."""

    runner = BenchmarkRunner()

    def workload():
        total = 0

        for i in range(100000):
            total += i

        return total

    results = runner.run(workload, num_runs=10)

    print("Atlas AI — Benchmark Results")
    print("-" * 36)

    for key, value in results.items():
        print(f"{key}: {value}")


def run_distributed_simulation() -> None:
    """Run distributed training simulation."""

    simulator = DistributedTrainingSimulator()

    results = simulator.simulate_step(
        num_workers=8,
        compute_time_sec=0.8,
        gradient_size_mb=200,
        bandwidth_gbps=10,
    )

    print("Atlas AI — Distributed Simulation")
    print("-" * 40)

    for key, value in results.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas AI CLI")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("profile")
    subparsers.add_parser("benchmark")
    subparsers.add_parser("simulate-distributed")

    run_parser = subparsers.add_parser("create-run")
    run_parser.add_argument("--name", required=True)

    args = parser.parse_args()

    if args.command == "profile":
        show_system_profile()

    elif args.command == "create-run":
        create_experiment_run(args.name)

    elif args.command == "benchmark":
        run_benchmark()

    elif args.command == "simulate-distributed":
        run_distributed_simulation()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()