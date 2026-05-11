from atlas_ai.autograd.value import Value
from atlas_ai.optim.optimizers import (
    Momentum,
    SGD,
)


def test_sgd_step():

    parameter = Value(1.0)

    parameter.grad = 0.5

    optimizer = SGD(
        [parameter],
        learning_rate=0.1,
    )

    optimizer.step()

    assert parameter.data == 0.95


def test_momentum_step():

    parameter = Value(1.0)

    parameter.grad = 0.5

    optimizer = Momentum(
        [parameter],
        learning_rate=0.1,
        beta=0.9,
    )

    optimizer.step()

    assert parameter.data < 1.0
