from atlas_ai.monitoring.memory_profiler import (
    TransformerMemoryProfiler,
)


class DashboardMetrics:
    """
    Aggregate serving and
    transformer diagnostics.
    """

    def __init__(self):

        self.memory_profiler = (
            TransformerMemoryProfiler()
        )

    def build_dashboard_summary(
        self,
    ):

        embedding_memory = (
            self.memory_profiler
            .estimate_embedding_memory(
                vocab_size=50000,
                embedding_dim=768,
            )
        )

        kv_cache_memory = (
            self.memory_profiler
            .estimate_kv_cache_memory(
                sequence_length=2048,
                hidden_size=768,
                num_layers=12,
            )
        )

        attention_memory = (
            self.memory_profiler
            .estimate_attention_memory(
                sequence_length=2048,
            )
        )

        return {
            "embedding_memory_mb": (
                embedding_memory
            ),
            "kv_cache_memory_mb": (
                kv_cache_memory
            ),
            "attention_memory_mb": (
                attention_memory
            ),
            "status": "healthy",
        }
