from atlas_ai.monitoring.memory_profiler import (
    TransformerMemoryProfiler,
)


def test_embedding_memory():

    profiler = (
        TransformerMemoryProfiler()
    )

    memory = (
        profiler.estimate_embedding_memory(
            vocab_size=1000,
            embedding_dim=128,
        )
    )

    assert memory > 0


def test_kv_cache_memory():

    profiler = (
        TransformerMemoryProfiler()
    )

    memory = (
        profiler.estimate_kv_cache_memory(
            sequence_length=512,
            hidden_size=256,
            num_layers=12,
        )
    )

    assert memory > 0


def test_attention_memory():

    profiler = (
        TransformerMemoryProfiler()
    )

    memory = (
        profiler.estimate_attention_memory(
            sequence_length=512,
        )
    )

    assert memory > 0
