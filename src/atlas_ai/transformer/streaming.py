import time

from atlas_ai.transformer.generation import (
    SequenceGenerator,
)


class StreamingGenerator:
    """
    Stream tokens incrementally.
    """

    def __init__(self):

        self.generator = (
            SequenceGenerator()
        )

    def stream_generate(
        self,
        prompt_tokens,
        max_new_tokens=5,
        delay_sec=0.1,
    ):

        generated = list(
            prompt_tokens
        )

        streamed_tokens = []

        for _ in range(
            max_new_tokens
        ):

            results = (
                self.generator.generate(
                    generated,
                    max_new_tokens=1,
                )
            )

            next_token = results[
                "generated_tokens"
            ][-1]

            generated.append(
                next_token
            )

            streamed_tokens.append(
                next_token
            )

            print(
                f"Generated token: "
                f"{next_token}"
            )

            time.sleep(delay_sec)

        return {
            "generated_tokens": (
                generated
            ),
            "streamed_tokens": (
                streamed_tokens
            ),
        }
