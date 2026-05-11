import math


class PositionalEncoding:
    """
    Sinusoidal positional encoding.
    """

    def __init__(
        self,
        embedding_dim: int,
    ):

        self.embedding_dim = (
            embedding_dim
        )

    def encode(
        self,
        embeddings,
    ):

        encoded = []

        for position, embedding in enumerate(
            embeddings
        ):

            position_vector = []

            for index in range(
                self.embedding_dim
            ):

                denominator = (
                    10000
                    ** (
                        (2 * (index // 2))
                        / self.embedding_dim
                    )
                )

                if index % 2 == 0:

                    value = math.sin(
                        position
                        / denominator
                    )

                else:

                    value = math.cos(
                        position
                        / denominator
                    )

                position_vector.append(
                    value
                )

            encoded_embedding = [
                embedding[i]
                + position_vector[i]
                for i in range(
                    self.embedding_dim
                )
            ]

            encoded.append(
                encoded_embedding
            )

        return encoded
