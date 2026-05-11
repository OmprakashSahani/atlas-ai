import argparse

from atlas_ai.benchmark.benchmark_runner import BenchmarkRunner
from atlas_ai.distributed.distributed_simulator import (
    DistributedTrainingSimulator,
)
from atlas_ai.distributed.runtime.distributed_trainer import DistributedTrainer
from atlas_ai.profiler.system_profiler import get_system_profile
from atlas_ai.tracker.experiment_tracker import ExperimentTracker
from atlas_ai.training.mlp_trainer import MLPTrainer


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


def run_distributed_runtime() -> None:
    """Run multiprocessing distributed runtime."""
    trainer = DistributedTrainer(num_workers=4)

    results = trainer.run()

    print("Atlas AI — Distributed Runtime")
    print("-" * 40)

    print(f"num_workers: {results['num_workers']}")
    print(f"total_runtime_sec: {results['total_runtime_sec']}")

    print("\nWorker Results:")
    for worker in results["worker_results"]:
        print(
            f"worker_id={worker['worker_id']} "
            f"execution_time_sec={worker['execution_time_sec']}"
        )


def run_training() -> None:
    """Run MLP training with checkpointing."""
    trainer = MLPTrainer(
        learning_rate=0.05,
        epochs=50,
    )

    results = trainer.train()

    print("Atlas AI — MLP Training Results")
    print("-" * 40)

    for key, value in results.items():
        if key != "loss_history":
            print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas AI CLI")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("profile")
    subparsers.add_parser("benchmark")
    subparsers.add_parser("simulate-distributed")
    subparsers.add_parser("run-distributed")
    subparsers.add_parser("train")

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
    elif args.command == "run-distributed":
        run_distributed_runtime()
    elif args.command == "train":
        run_training()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()