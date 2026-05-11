class TransformerMemoryProfiler:
    """
    Estimate transformer memory usage.
    """

    def estimate_embedding_memory(
        self,
        vocab_size: int,
        embedding_dim: int,
        bytes_per_param: int = 4,
    ):

        total_bytes = (
            vocab_size
            * embedding_dim
            * bytes_per_param
        )

        return self.bytes_to_mb(
            total_bytes
        )

    def estimate_kv_cache_memory(
        self,
        sequence_length: int,
        hidden_size: int,
        num_layers: int,
        bytes_per_value: int = 4,
    ):

        total_bytes = (
            sequence_length
            * hidden_size
            * num_layers
            * 2
            * bytes_per_value
        )

        return self.bytes_to_mb(
            total_bytes
        )

    def estimate_attention_memory(
        self,
        sequence_length: int,
        bytes_per_value: int = 4,
    ):

        total_bytes = (
            sequence_length
            * sequence_length
            * bytes_per_value
        )

        return self.bytes_to_mb(
            total_bytes
        )

    def bytes_to_mb(
        self,
        num_bytes,
    ):

        return round(
            num_bytes / (1024 ** 2),
            4,
        )
