from atlas_ai.distributed.distributed_simulator import (
    DistributedTrainingSimulator,
)


def test_distributed_simulation():
    simulator = DistributedTrainingSimulator()

    results = simulator.simulate_step(
        num_workers=8,
        compute_time_sec=0.8,
        gradient_size_mb=200,
        bandwidth_gbps=10,
    )

    assert results["num_workers"] == 8
    assert "scaling_efficiency" in results
    assert "communication_ratio" in results
    assert "bottleneck" in results
