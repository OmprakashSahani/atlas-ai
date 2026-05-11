from atlas_ai.transformer.generation import (
    SequenceGenerator,
)


def test_sequence_generation():

    generator = SequenceGenerator()

    results = generator.generate(
        prompt_tokens=[1, 2, 3],
        max_new_tokens=5,
    )

    generated = results[
        "generated_tokens"
    ]

    assert len(generated) == 8

    assert (
        results["kv_cache_size"]
        == 5
    )
