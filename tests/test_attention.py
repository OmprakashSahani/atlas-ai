from atlas_ai.transformer.attention import (
    SelfAttention,
)


def test_attention_output_shape():

    attention = SelfAttention()

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]

    outputs = attention.forward(
        embeddings
    )

    assert len(outputs) == 3

    assert len(outputs[0]) == 3
