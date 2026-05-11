import math


class SelfAttention:
    """
    Simplified self-attention mechanism.
    """

    def softmax(self, values):

        exp_values = [
            math.exp(v)
            for v in values
        ]

        total = sum(exp_values)

        return [
            value / total
            for value in exp_values
        ]

    def dot_product(
        self,
        vector_a,
        vector_b,
    ):

        return sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b,
            )
        )

    def forward(
        self,
        embeddings,
    ):

        sequence_length = len(
            embeddings
        )

        attention_outputs = []

        for i in range(sequence_length):

            query = embeddings[i]

            attention_scores = []

            for j in range(
                sequence_length
            ):

                key = embeddings[j]

                score = self.dot_product(
                    query,
                    key,
                )

                score /= math.sqrt(
                    len(query)
                )

                attention_scores.append(
                    score
                )

            attention_weights = (
                self.softmax(
                    attention_scores
                )
            )

            output = [
                0.0
                for _ in range(
                    len(query)
                )
            ]

            for weight, value_vector in zip(
                attention_weights,
                embeddings,
            ):

                for k in range(
                    len(output)
                ):

                    output[k] += (
                        weight
                        * value_vector[k]
                    )

            attention_outputs.append(
                output
            )

        return attention_outputs
