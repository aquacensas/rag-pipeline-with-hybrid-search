"""Sanity check: ask a real question and confirm we get a grounded answer
with citations, plus a low-signal question to confirm the confidence gate
correctly refuses to answer rather than hallucinating."""

from src.generation.generate import generate_answer

print("----------- Test 1: Well-covered question -----------")
result = generate_answer("How do I use BackgroundTasks in FastAPI?", strategy="structural")
print(f"Confidence: {result.retrieved_confidence:.3f}")
print(f"Insufficient context: {result.insufficient_context}")
print(f"\nAnswer:\n{result.answer}\n")

print("----------- Test 2: Question with no real answer in this corpus -----------  ")
result2 = generate_answer("What is the capital of France?", strategy="structural")
print(f"Confidence: {result2.retrieved_confidence:.3f}")
print(f"Insufficient context: {result2.insufficient_context}")
print(f"\nAnswer:\n{result2.answer}\n")