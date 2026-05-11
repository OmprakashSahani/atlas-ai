import json
from datetime import UTC, datetime
from pathlib import Path


RUNS_DIR = Path("runs")


class ExperimentTracker:
    """Simple experiment tracking system."""

    def __init__(self) -> None:
        RUNS_DIR.mkdir(exist_ok=True)

    def create_run(self, name: str, config: dict) -> str:
        """Create a new experiment run."""

        timestamp = datetime.now(UTC).isoformat()
        run_id = f"{timestamp}_{name}".replace(":", "-")

        run_data = {
            "run_id": run_id,
            "experiment_name": name,
            "created_at": timestamp,
            "config": config,
            "metrics": {},
        }

        run_path = RUNS_DIR / f"{run_id}.json"

        with open(run_path, "w") as f:
            json.dump(run_data, f, indent=4)

        return run_id

    def log_metric(self, run_id: str, metric_name: str, value: float) -> None:
        """Log a metric to an existing run."""

        run_path = RUNS_DIR / f"{run_id}.json"

        with open(run_path, "r") as f:
            run_data = json.load(f)

        run_data["metrics"][metric_name] = value

        with open(run_path, "w") as f:
            json.dump(run_data, f, indent=4)
