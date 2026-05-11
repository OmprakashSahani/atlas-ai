import math


class DistributedTrainingSimulator:
    """
    Simulate distributed training behavior.
    """

    def simulate_step(
        self,
        num_workers: int,
        compute_time_sec: float,
        gradient_size_mb: float,
        bandwidth_gbps: float,
    ) -> dict:
        """
        Simulate one distributed training step.
        """

        compute_per_worker = compute_time_sec / num_workers

        bandwidth_mb_per_sec = bandwidth_gbps * 1024

        communication_time = (
            gradient_size_mb * math.log2(num_workers)
        ) / bandwidth_mb_per_sec

        total_step_time = (
            compute_per_worker + communication_time
        )

        scaling_efficiency = (
            compute_time_sec /
            (num_workers * total_step_time)
        )

        communication_ratio = (
            communication_time / total_step_time
        )

        bottleneck = (
            "communication-bound"
            if communication_ratio > 0.5
            else "compute-bound"
        )

        return {
            "num_workers": num_workers,
            "compute_time_sec": round(compute_per_worker, 6),
            "communication_time_sec": round(communication_time, 6),
            "total_step_time_sec": round(total_step_time, 6),
            "scaling_efficiency": round(scaling_efficiency, 4),
            "communication_ratio": round(communication_ratio, 4),
            "bottleneck": bottleneck,
        }
