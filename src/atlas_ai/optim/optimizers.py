class SGD:
    """
    Basic stochastic gradient descent.
    """

    def __init__(
        self,
        parameters,
        learning_rate=0.01,
    ):

        self.parameters = parameters

        self.learning_rate = (
            learning_rate
        )

    def step(self):

        for parameter in self.parameters:

            parameter.data -= (
                self.learning_rate
                * parameter.grad
            )

    def zero_grad(self):

        for parameter in self.parameters:

            parameter.grad = 0.0


class Momentum:
    """
    Momentum-based optimizer.
    """

    def __init__(
        self,
        parameters,
        learning_rate=0.01,
        beta=0.9,
    ):

        self.parameters = parameters

        self.learning_rate = (
            learning_rate
        )

        self.beta = beta

        self.velocities = [
            0.0
            for _ in parameters
        ]

    def step(self):

        for index, parameter in enumerate(
            self.parameters
        ):

            self.velocities[index] = (
                self.beta
                * self.velocities[index]
                + (1 - self.beta)
                * parameter.grad
            )

            parameter.data -= (
                self.learning_rate
                * self.velocities[index]
            )

    def zero_grad(self):

        for parameter in self.parameters:

            parameter.grad = 0.0
