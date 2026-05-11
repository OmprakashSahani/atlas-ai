from atlas_ai.transformer.block import (
    TransformerBlock,
)


def test_transformer_block():

    block = TransformerBlock()

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]

    outputs = block.forward(
        embeddings
    )

    assert len(outputs) == 3

    assert len(outputs[0]) == 3
