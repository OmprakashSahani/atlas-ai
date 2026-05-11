from atlas_ai.distributed.runtime.gradient_sync import (
    GradientSynchronizer,
)


def test_gradient_averaging():

    synchronizer = (
        GradientSynchronizer()
    )

    worker_gradients = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ]

    averaged = (
        synchronizer.average_gradients(
            worker_gradients
        )
    )

    assert averaged == [
        2.0,
        3.0,
        4.0,
    ]
