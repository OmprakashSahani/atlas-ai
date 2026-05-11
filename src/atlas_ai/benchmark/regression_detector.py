class RegressionDetector:
    """
    Detect performance regressions
    between benchmark runs.
    """

    def compare_metrics(
        self,
        baseline_metrics: dict,
        current_metrics: dict,
        threshold_percent: float = 10.0,
    ):

        regressions = []

        for metric_name in baseline_metrics:

            if (
                metric_name
                not in current_metrics
            ):
                continue

            baseline = baseline_metrics[
                metric_name
            ]

            current = current_metrics[
                metric_name
            ]

            if baseline == 0:
                continue

            percent_change = (
                (
                    current - baseline
                )
                / baseline
            ) * 100

            if (
                percent_change
                > threshold_percent
            ):

                regressions.append(
                    {
                        "metric": metric_name,
                        "baseline": baseline,
                        "current": current,
                        "percent_change": round(
                            percent_change,
                            2,
                        ),
                    }
                )

        return regressions
