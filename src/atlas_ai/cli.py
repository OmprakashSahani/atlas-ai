import argparse

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atlas AI CLI"
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("profile")

    run_parser = subparsers.add_parser("create-run")
    run_parser.add_argument("--name", required=True)

    args = parser.parse_args()

    if args.command == "profile":
        show_system_profile()

    elif args.command == "create-run":
        create_experiment_run(args.name)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
