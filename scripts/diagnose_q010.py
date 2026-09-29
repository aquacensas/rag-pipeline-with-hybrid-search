'''Diagnostic: reruns q010 in isolation with full visibility into what was
retrieved, what was generated, and why citations failed — to root-cause
the 0.000 correctness/faithfulness score from the full eval run.'''

from src.retrieval.fusion import hybrid_search
from src.retrieval.reranker import rerank
from src.generation.generate import generate_answer
from src.generation.citations import verify_citations

question = "How do middleware functions in FastAPI interact with background tasks added during request handling?"
expected_sources = ["tutorial/middleware.md", "tutorial/background-tasks.md"]

print("=" * 70)
print("STEP 1: What did raw hybrid search + reranking retrieve?")
print("=" * 70)

fused = hybrid_search(question, strategy="structural", top_k=20)
reranked = rerank(question, fused, top_k=5)

for i, r in enumerate(reranked, 1):
    print(f"\n{i}. [{r['source_path']}] rerank_score={r.get('rerank_score')}")
    print(r["content"][:200])

retrieved_sources = {r["source_path"] for r in reranked}
print(f"\n--- Expected sources: {expected_sources}")
print(f"--- Actually retrieved sources: {sorted(retrieved_sources)}")
for src in expected_sources:
    print(f"    {'✓' if src in retrieved_sources else '✗ MISSING'}: {src}")

print("\n" + "=" * 70)
print("STEP 2: What did the full pipeline actually generate?")
print("=" * 70)

result = generate_answer(question, strategy="structural")
print(f"\nRetrieval confidence: {result.retrieved_confidence:.3f}")
print(f"Insufficient context: {result.insufficient_context}")
print(f"\nGenerated answer:\n{result.answer}")

print("\n" + "=" * 70)
print("STEP 3: Why did citation verification score 0.000?")
print("=" * 70)

verification = verify_citations(result.answer, result.retrieved_chunks)
coverage_str = f"{verification.coverage:.1%}" if verification.coverage is not None else "N/A"
print(f"\nCoverage: {coverage_str} ({verification.verified_citations}/{verification.total_citations})\n")

for claim in verification.claims:
    print(f"Claim: {claim.claim_text}")
    for num, is_verified in claim.verified.items():
        status = "✓ VERIFIED" if is_verified else "✗ NOT SUPPORTED"
        print(f"  [{num}] {status} — {claim.reasoning[num]}")
    print()