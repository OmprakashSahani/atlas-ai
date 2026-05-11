import random


class TokenEmbedding:
    """
    Simple token embedding table.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ):

        self.vocab_size = vocab_size

        self.embedding_dim = (
            embedding_dim
        )

        self.embedding_table = [
            [
                random.uniform(-1, 1)
                for _ in range(
                    embedding_dim
                )
            ]
            for _ in range(vocab_size)
        ]

    def forward(
        self,
        token_ids,
    ):

        embeddings = []

        for token_id in token_ids:

            embedding = (
                self.embedding_table[
                    token_id
                ]
            )

            embeddings.append(
                embedding
            )

        return embeddings
