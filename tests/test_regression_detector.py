from atlas_ai.benchmark.regression_detector import (
    RegressionDetector,
)


def test_regression_detection():

    detector = RegressionDetector()

    baseline = {
        "latency": 100,
        "memory": 200,
    }

    current = {
        "latency": 120,
        "memory": 205,
    }

    regressions = (
        detector.compare_metrics(
            baseline_metrics=baseline,
            current_metrics=current,
            threshold_percent=10,
        )
    )

    assert len(regressions) == 1

    assert (
        regressions[0]["metric"]
        == "latency"
    )
