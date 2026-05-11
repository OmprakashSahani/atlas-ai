from atlas_ai.transformer.embeddings import (
    TokenEmbedding,
)


def test_token_embedding():

    embedding = TokenEmbedding(
        vocab_size=100,
        embedding_dim=8,
    )

    token_ids = [1, 5, 10]

    outputs = embedding.forward(
        token_ids
    )

    assert len(outputs) == 3

    assert len(outputs[0]) == 8
