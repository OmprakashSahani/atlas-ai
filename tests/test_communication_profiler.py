from atlas_ai.distributed.runtime.communication import (
    CommunicationProfiler,
)


def test_all_reduce_time():

    profiler = (
        CommunicationProfiler()
    )

    communication_time = (
        profiler.estimate_all_reduce_time(
            gradient_size_mb=200,
            bandwidth_gbps=10,
            num_workers=8,
        )
    )

    assert communication_time > 0


def test_communication_efficiency():

    profiler = (
        CommunicationProfiler()
    )

    efficiency = (
        profiler.communication_efficiency(
            compute_time_sec=1.0,
            communication_time_sec=0.5,
        )
    )

    assert efficiency < 1.0
