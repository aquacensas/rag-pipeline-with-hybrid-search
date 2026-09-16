from src.generation.generate import generate_answer
from src.generation.citations import verify_citations

result = generate_answer("How do I use BackgroundTasks in FastAPI?", strategy="structural")

print(f"Answer:\n{result.answer}\n")

verification = verify_citations(result.answer, result.retrieved_chunks)

print(f"--------------- Citation Verification ---------------")
print(f"Coverage: {verification.coverage:.1%} ({verification.verified_citations}/{verification.total_citations})\n")

for claim in verification.claims:
    print(f"Claim: {claim.claim_text}")
    for num, is_verified in claim.verified.items():
        status = "✓ VERIFIED" if is_verified else "✗ NOT SUPPORTED"
        print(f"  [{num}] {status} — {claim.reasoning[num]}")
    print()