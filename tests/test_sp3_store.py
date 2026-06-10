"""The store holds (word, vector) rows and returns nearest words by cosine DISTANCE
(1 - cosine similarity), ascending. InMemoryVectorStore is the test/dev implementation;
it satisfies the same VectorStore Protocol as the production PgVectorStore.
"""
from htr_sp3 import vectorize
from htr_sp3.store import InMemoryVectorStore


def _populate():
    store = InMemoryVectorStore()
    store.add_many([(w, vectorize.word_to_vector(w)) for w in ["medical", "record", "zebra"]])
    return store


def test_nearest_returns_closest_word_first():
    store = _populate()
    query = vectorize.word_to_vector("medisal")  # typo of "medical"
    results = store.nearest(query, k=3)
    assert results[0][0] == "medical"


def test_nearest_respects_k():
    store = _populate()
    query = vectorize.word_to_vector("medisal")
    assert len(store.nearest(query, k=2)) == 2


def test_nearest_results_sorted_by_ascending_distance():
    store = _populate()
    query = vectorize.word_to_vector("medisal")
    distances = [dist for _, dist in store.nearest(query, k=3)]
    assert distances == sorted(distances)


def test_empty_store_returns_empty():
    assert InMemoryVectorStore().nearest(vectorize.word_to_vector("x"), k=5) == []
