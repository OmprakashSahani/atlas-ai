from atlas_ai.transformer.streaming import (
    StreamingGenerator,
)


def test_stream_generation():

    generator = StreamingGenerator()

    results = generator.stream_generate(
        prompt_tokens=[1, 2],
        max_new_tokens=3,
        delay_sec=0,
    )

    assert (
        len(
            results["generated_tokens"]
        )
        == 5
    )

    assert (
        len(
            results["streamed_tokens"]
        )
        == 3
    )
