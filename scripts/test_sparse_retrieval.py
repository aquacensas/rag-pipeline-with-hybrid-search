from src.retrieval.sparse import sparse_search

results = sparse_search("BackgroundTasks", strategy="structural", top_k=5)

for i, r in enumerate(results, 1):
    print(f"{i}. [{r['source_path']}] score={r['score']:.4f}")
    print(r["content"][:150])
    print("...\n")