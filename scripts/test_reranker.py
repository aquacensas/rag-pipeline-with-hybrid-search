from src.retrieval.fusion import hybrid_search
from src.retrieval.reranker import rerank

query = "How do I use BackgroundTasks in FastAPI?"

fused = hybrid_search(query, strategy="structural", top_k=20)
print(f"--------- Fusion top 5 (before reranking) ---------")
for i, r in enumerate(fused[:5], 1):
    print(f"{i}. [{r['source_path']}] fused_score={r['fused_score']:.4f}")

reranked = rerank(query, fused, top_k=5)
print(f"\n---------- Reranked top 5 ----------")
for i, r in enumerate(reranked, 1):
    print(f"{i}. [{r['source_path']}] rerank_score={r['rerank_score']}")
    print(r["content"][:150])
    print("...\n")