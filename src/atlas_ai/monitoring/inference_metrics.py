import statistics
import time


class InferenceMetrics:
    """
    Track inference request metrics.
    """

    def __init__(self):

        self.request_count = 0

        self.latencies = []

    def start_timer(self):

        return time.perf_counter()

    def stop_timer(self, start_time):

        latency = (
            time.perf_counter()
            - start_time
        )

        self.request_count += 1

        self.latencies.append(latency)

        return latency

    def summary(self):

        if not self.latencies:

            return {
                "request_count": 0,
                "average_latency_sec": 0,
                "min_latency_sec": 0,
                "max_latency_sec": 0,
            }

        return {
            "request_count": (
                self.request_count
            ),
            "average_latency_sec": round(
                statistics.mean(
                    self.latencies
                ),
                6,
            ),
            "min_latency_sec": round(
                min(self.latencies),
                6,
            ),
            "max_latency_sec": round(
                max(self.latencies),
                6,
            ),
        }
