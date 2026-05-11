from atlas_ai.transformer.block import (
    TransformerBlock,
)
from atlas_ai.transformer.embeddings import (
    TokenEmbedding,
)
from atlas_ai.transformer.positional_encoding import (
    PositionalEncoding,
)


class TinyTransformer:
    """
    Simplified transformer model.
    """

    def __init__(
        self,
        vocab_size=100,
        embedding_dim=8,
        num_blocks=2,
    ):

        self.embedding = (
            TokenEmbedding(
                vocab_size,
                embedding_dim,
            )
        )

        self.position_encoder = (
            PositionalEncoding(
                embedding_dim
            )
        )

        self.blocks = [
            TransformerBlock()
            for _ in range(
                num_blocks
            )
        ]

    def forward(
        self,
        token_ids,
    ):

        embeddings = (
            self.embedding.forward(
                token_ids
            )
        )

        encoded = (
            self.position_encoder.encode(
                embeddings
            )
        )

        hidden = encoded

        for block in self.blocks:

            hidden = block.forward(
                hidden
            )

        return hidden
