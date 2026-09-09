'''Running hybrid search on a query that should benefit from both methods 
and see how the fused ranking compares to either method alone'''

from src.retrieval.fusion import hybrid_search
results=hybrid_search('How do I use BackgroundTasks in FastAPI?',strategy='structural',top_k=5)

for i, r in enumerate(results,start=1):
    print(f"{i}. [{r['source_path']}] fused_score={r['fused_score']:.4f}")
    print(r["content"][:150])
    print("...\n")
    