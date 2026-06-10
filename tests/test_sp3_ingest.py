"""ingest_vocabulary builds the vocab, vectorizes each word, and loads it into ANY store. We
inject an InMemoryVectorStore so the test needs no database, and assert the words are queryable.
"""
from htr_sp3 import ingest, vectorize
from htr_sp3.store import InMemoryVectorStore


def test_ingest_populates_store_from_records():
    records = [{"text": "the medical record"}, {"text": "the patient"}]
    store = InMemoryVectorStore()

    count = ingest.ingest_vocabulary(records, store)

    assert count == 4  # the, medical, record, patient
    nearest = store.nearest(vectorize.word_to_vector("medisal"), k=1)
    assert nearest[0][0] == "medical"
