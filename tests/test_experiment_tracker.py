from atlas_ai.tracker.experiment_tracker import ExperimentTracker


def test_create_run():
    tracker = ExperimentTracker()

    run_id = tracker.create_run(
        name="test_experiment",
        config={"learning_rate": 0.001},
    )

    assert isinstance(run_id, str)


def test_log_metric():
    tracker = ExperimentTracker()

    run_id = tracker.create_run(
        name="metric_test",
        config={},
    )

    tracker.log_metric(
        run_id=run_id,
        metric_name="accuracy",
        value=0.95,
    )
