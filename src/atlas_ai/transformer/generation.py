import random

from atlas_ai.transformer.kv_cache import (
    KVCache,
)
from atlas_ai.transformer.model import (
    TinyTransformer,
)


class SequenceGenerator:
    """
    Simplified autoregressive generator.
    """

    def __init__(self):

        self.model = TinyTransformer()

        self.cache = KVCache()

    def predict_next_token(
        self,
        token_ids,
    ):

        hidden = self.model.forward(
            token_ids
        )

        last_representation = (
            hidden[-1]
        )

        token_score = sum(
            last_representation
        )

        next_token = int(
            abs(token_score * 100)
        ) % 100

        return next_token

    def generate(
        self,
        prompt_tokens,
        max_new_tokens=5,
    ):

        generated = list(
            prompt_tokens
        )

        for _ in range(
            max_new_tokens
        ):

            next_token = (
                self.predict_next_token(
                    generated
                )
            )

            self.cache.add(
                key=generated[-1],
                value=next_token,
            )

            generated.append(
                next_token
            )

        return {
            "generated_tokens": (
                generated
            ),
            "kv_cache_size": (
                self.cache.size()
            ),
        }
