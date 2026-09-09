from src.retrieval.dense import dense_search

results=dense_search('How do I create a path operations in FastAPI ?', strategy='structural',top_k=5)

for i,r in enumerate(results,1):
    print(f"{i}. [{r['source_path']}] score={r['score']:.4f}")
    print(r["content"][:150])
    print("...\n")
