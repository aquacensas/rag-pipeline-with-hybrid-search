"""Sanity check: store a couple of fake chunks and confirm they're
retrievable and correctly tagged by strategy."""

from src.ingestion.chunking import Chunk
from src.ingestion.embeddings import embed_texts
from src.db.vector_store import store_chunks, query_by_strategy, collection_stats

# Two fake chunks, same content, different strategy tags 
# storage + filtering works before we run it on the real 155-doc corpus
fake_chunks = [
    Chunk(chunk_id="test::fixed::0", source_path="test.md", chunk_index=0,
          content="FastAPI is a modern Python web framework.", strategy="fixed"),
    Chunk(chunk_id="test::structural::0", source_path="test.md", chunk_index=0,
          content="FastAPI is a modern Python web framework.", strategy="structural"),
]

vectors = embed_texts([c.content for c in fake_chunks])
store_chunks(fake_chunks, vectors)

print("Stats:", collection_stats())

query_vec = embed_texts(["What is FastAPI?"])[0]
results = query_by_strategy(query_vec, strategy="fixed", top_k=1)
print("\nQuery result (fixed only):", results["documents"])