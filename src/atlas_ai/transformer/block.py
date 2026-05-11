from atlas_ai.transformer.attention import (
    SelfAttention,
)


class FeedForward:
    """
    Simple feedforward transformation.
    """

    def forward(
        self,
        embeddings,
    ):

        outputs = []

        for embedding in embeddings:

            transformed = [
                value * 1.5
                for value in embedding
            ]

            outputs.append(
                transformed
            )

        return outputs


class TransformerBlock:
    """
    Simplified transformer block.
    """

    def __init__(self):

        self.attention = (
            SelfAttention()
        )

        self.feedforward = (
            FeedForward()
        )

    def residual_add(
        self,
        original,
        transformed,
    ):

        outputs = []

        for original_vec, transformed_vec in zip(
            original,
            transformed,
        ):

            combined = [
                a + b
                for a, b in zip(
                    original_vec,
                    transformed_vec,
                )
            ]

            outputs.append(
                combined
            )

        return outputs

    def forward(
        self,
        embeddings,
    ):

        attention_output = (
            self.attention.forward(
                embeddings
            )
        )

        attention_residual = (
            self.residual_add(
                embeddings,
                attention_output,
            )
        )

        feedforward_output = (
            self.feedforward.forward(
                attention_residual
            )
        )

        final_output = (
            self.residual_add(
                attention_residual,
                feedforward_output,
            )
        )

        return final_output
