from atlas_ai.autograd.value import Value
from atlas_ai.nn.modules import (
    Layer,
    MLP,
    Neuron,
)


def test_neuron_output():

    neuron = Neuron(3)

    x = [
        Value(1.0),
        Value(2.0),
        Value(3.0),
    ]

    output = neuron(x)

    assert isinstance(output, Value)


def test_layer_output():

    layer = Layer(3, 2)

    x = [
        Value(1.0),
        Value(2.0),
        Value(3.0),
    ]

    output = layer(x)

    assert len(output) == 2


def test_mlp_output():

    mlp = MLP(3, [4, 4, 1])

    x = [
        Value(1.0),
        Value(2.0),
        Value(3.0),
    ]

    output = mlp(x)

    assert isinstance(output, Value)
