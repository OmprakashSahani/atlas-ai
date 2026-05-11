from atlas_ai.transformer.kv_cache import (
    KVCache,
)


def test_kv_cache():

    cache = KVCache()

    cache.add(
        key=[1, 2, 3],
        value=[4, 5, 6],
    )

    assert cache.size() == 1

    assert cache.get_keys() == [
        [1, 2, 3]
    ]

    assert cache.get_values() == [
        [4, 5, 6]
    ]

    cache.clear()

    assert cache.size() == 0
