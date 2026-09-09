from src.ingestion.embeddings import embed_text, embed_texts

# Single embedding
vec = embed_text("FastAPI is a modern web framework for building APIs")
print(f"Single embedding dimension: {len(vec)}")
print(f"First 5 values: {vec[:5]}")

# Batch embedding + a basic sanity check on similarity
import numpy as np

texts = [
    "FastAPI is a modern Python web framework",
    "FastAPI is a fast Python framework for APIs",
    "The Eiffel Tower is located in Paris",
]
vectors = embed_texts(texts)

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

sim_similar = cosine_sim(vectors[0], vectors[1])   # both about FastAPI
sim_different = cosine_sim(vectors[0], vectors[2])  # FastAPI vs Eiffel Tower

print(f"\nSimilarity (both about FastAPI): {sim_similar:.4f}")
print(f"Similarity (FastAPI vs Eiffel Tower): {sim_different:.4f}")
print("\nExpected: the FastAPI/FastAPI similarity should be noticeably higher")