import random

from atlas_ai.autograd.value import Value


class Neuron:
    """
    Single fully connected neuron.
    """

    def __init__(self, num_inputs: int):

        self.weights = [
            Value(random.uniform(-1, 1))
            for _ in range(num_inputs)
        ]

        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, inputs):

        activation = sum(
            (
                w * x
                for w, x in zip(
                    self.weights,
                    inputs,
                )
            ),
            self.bias,
        )

        return activation.tanh()

    def parameters(self):

        return self.weights + [self.bias]


class Layer:
    """
    Fully connected neural network layer.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
    ):

        self.neurons = [
            Neuron(num_inputs)
            for _ in range(num_outputs)
        ]

    def __call__(self, inputs):

        outputs = [
            neuron(inputs)
            for neuron in self.neurons
        ]

        return (
            outputs[0]
            if len(outputs) == 1
            else outputs
        )

    def parameters(self):

        params = []

        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params


class MLP:
    """
    Multi-layer perceptron.
    """

    def __init__(self, num_inputs, layer_sizes):

        sizes = [num_inputs] + layer_sizes

        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(layer_sizes))
        ]

    def __call__(self, inputs):

        for layer in self.layers:
            inputs = layer(inputs)

            if not isinstance(inputs, list):
                inputs = [inputs]

        return inputs[0]

    def parameters(self):

        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params
