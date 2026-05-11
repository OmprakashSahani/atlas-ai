from atlas_ai.transformer.model import (
    TinyTransformer,
)


def test_tiny_transformer():

    model = TinyTransformer(
        vocab_size=100,
        embedding_dim=8,
        num_blocks=2,
    )

    token_ids = [1, 5, 10, 15]

    outputs = model.forward(
        token_ids
    )

    assert len(outputs) == 4

    assert len(outputs[0]) == 8
