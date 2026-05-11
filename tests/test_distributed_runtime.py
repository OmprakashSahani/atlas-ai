from atlas_ai.distributed.runtime.distributed_trainer import (
    DistributedTrainer,
)


def test_distributed_runtime():

    trainer = DistributedTrainer(
        num_workers=2,
    )

    results = trainer.run()

    assert results["num_workers"] == 2

    assert (
        len(results["worker_results"])
        == 2
    )

    assert (
        "total_runtime_sec"
        in results
    )
