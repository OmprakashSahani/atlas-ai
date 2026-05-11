import math


class CommunicationProfiler:
    """
    Simulate distributed communication cost.
    """

    def estimate_all_reduce_time(
        self,
        gradient_size_mb: float,
        bandwidth_gbps: float,
        num_workers: int,
    ):
        """
        Estimate all-reduce communication time.
        """

        bandwidth_mb_per_sec = (
            bandwidth_gbps * 1024
        )

        communication_time = (
            gradient_size_mb
            * math.log2(num_workers)
        ) / bandwidth_mb_per_sec

        return round(
            communication_time,
            6,
        )

    def communication_efficiency(
        self,
        compute_time_sec: float,
        communication_time_sec: float,
    ):

        total = (
            compute_time_sec
            + communication_time_sec
        )

        efficiency = (
            compute_time_sec / total
        )

        return round(
            efficiency,
            4,
        )
