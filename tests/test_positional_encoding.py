from atlas_ai.transformer.positional_encoding import (
    PositionalEncoding,
)


def test_positional_encoding():

    encoder = PositionalEncoding(
        embedding_dim=4,
    )

    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]

    encoded = encoder.encode(
        embeddings
    )

    assert len(encoded) == 2

    assert len(encoded[0]) == 4

    assert encoded[0] != embeddings[0]
